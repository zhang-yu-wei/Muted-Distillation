"""Backend-independent asynchronous OfficeQA agent loop."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from officeqa_rl.actions import parse_fenced_command
from officeqa_rl.backends.protocol import GenerationBackend
from officeqa_rl.full_corpus_env import FullCorpusEnvironment
from officeqa_rl.prompts import build_system_prompt, format_error_observation
from officeqa_rl.token_usage import episode_usage
from officeqa_rl.types import (
    AgentConfig,
    AgentResult,
    FormatError,
    GeneratedTurn,
    LimitsExceeded,
    Message,
    Submitted,
)


def _text_message(role: str, text: str, **extra: Any) -> Message:
    message: Message = {"role": role, "content": text}
    if extra:
        message["extra"] = extra
    return message


def _step_message(step_output: Any) -> Message:
    if step_output.image is None:
        return _text_message("user", step_output.output)
    return {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": step_output.image,
                "alt_text": "Rendered PDF page",
            },
            {"type": "text", "text": step_output.output},
        ],
    }


def _trace_safe_messages(messages: list[Message]) -> list[Message]:
    """Copy messages while replacing in-memory images with JSON-safe metadata."""
    if not any(isinstance(message.get("content"), list) for message in messages):
        return messages
    safe_messages: list[Message] = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            safe_content: list[Any] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image = item.get("image")
                    size = getattr(image, "size", None)
                    mode = getattr(image, "mode", None)
                    safe_item = dict(item)
                    safe_item["image"] = {
                        "kind": "rendered_pdf_page",
                        "size": list(size) if size is not None else None,
                        "mode": mode,
                    }
                    safe_content.append(safe_item)
                else:
                    safe_content.append(item)
            copied["content"] = safe_content
        safe_messages.append(copied)
    return safe_messages


class OfficeQAAgent:
    """Run the OfficeQA command protocol using any generation backend."""

    def __init__(
        self,
        backend: GenerationBackend,
        config: AgentConfig,
        *,
        sampling_params: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.sampling_params = dict(sampling_params or {})
        # None keeps the OfficeQA prompt; benchmark variants (SpreadsheetBench)
        # pass their own prompt while reusing the command protocol unchanged.
        self.system_prompt = system_prompt

    async def run(self, env: FullCorpusEnvironment) -> AgentResult:
        """Run until submission, the step limit, or an unrecoverable error."""
        messages: list[Message] = [
            _text_message(
                "system", self.system_prompt or build_system_prompt(self.config)
            ),
            env.get_task_message(),
        ]
        generated_turns: list[GeneratedTurn] = []
        submission: str | None = None
        status: str | None = None
        current_step = 0

        try:
            for current_step in range(1, self.config.step_limit + 1):
                turn = await self.backend.generate(
                    messages,
                    dict(self.sampling_params),
                )
                generated_turns.append(turn)
                if (
                    turn.finish_reason == "length"
                    and not turn.output_ids
                    and not turn.text
                ):
                    # The context window is exhausted before this turn could
                    # generate anything: end the episode like the step limit
                    # does. A format-error retry could never shrink the
                    # prompt, and aborting would discard the episode's
                    # completed turns.
                    raise LimitsExceeded(
                        f"Context window exhausted after {current_step - 1} steps"
                    )
                assistant_message = _text_message("assistant", turn.text)
                if turn.reasoning_content is not None:
                    # Kimi requires the complete assistant message, including
                    # reasoning_content, on the next multi-turn request. Keep
                    # it top-level so the backend can replay it unchanged and
                    # raw traces retain the SFT reasoning target.
                    assistant_message["reasoning_content"] = turn.reasoning_content
                messages.append(assistant_message)

                try:
                    command = parse_fenced_command(turn.text)
                except FormatError as error:
                    messages.append(
                        _text_message(
                            "user",
                            format_error_observation(self.config, error),
                            error={
                                "type": type(error).__name__,
                                "message": str(error),
                            },
                        )
                    )
                    continue

                step_output = await env.execute(command)
                messages.append(_step_message(step_output))

            raise LimitsExceeded(f"Step limit reached: {self.config.step_limit}")
        except Submitted as terminal:
            submission = terminal.submission
            status = "submitted"
            messages.append(
                _text_message(
                    "exit",
                    f"Submitted answer: {submission}",
                    submission=submission,
                    exit_status=status,
                )
            )
        except LimitsExceeded as terminal:
            submission = "TIMEOUT"
            status = "limits_exceeded"
            messages.append(
                _text_message(
                    "exit",
                    str(terminal),
                    submission=submission,
                    exit_status=status,
                )
            )
        except Exception as error:
            submission = "ERROR"
            status = f"error: {type(error).__name__}"
            messages.append(
                _text_message(
                    "exit",
                    f"Agent crashed with error: {error}",
                    submission=submission,
                    exit_status=status,
                )
            )

        return AgentResult(
            submission=submission or "TIMEOUT",
            status=status or "limits_exceeded",
            messages=_trace_safe_messages(messages),
            steps=current_step,
            metadata={
                "run_id": env.run_id,
                "environment": env.get_info(),
                "sampling_params": dict(self.sampling_params),
                "token_usage": episode_usage(generated_turns),
                "generated_turns": [asdict(turn) for turn in generated_turns],
            },
        )


__all__ = ["OfficeQAAgent"]
