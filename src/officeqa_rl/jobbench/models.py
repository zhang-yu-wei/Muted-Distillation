"""mini-swe-agent ``Model`` implementations backed by this repository's backends.

Upstream ships litellm/OpenRouter model classes; none of them fit here. Offline
evaluation goes through ``officeqa_rl.backends`` (vLLM, Gemini, an
OpenAI-compatible server), and RL rollouts go through the slime/SGLang backend
that returns exact prompt and output token ids. Both are asynchronous, while
``DefaultAgent`` is synchronous, so ``LoopBridge`` hands each blocking
``query()`` back to the event loop that owns the backend.

Everything else follows upstream's text-based model contract exactly: one
fenced command per turn parsed by ``parse_regex_actions``, a ``FormatError``
carrying the error observation when the count is not one, and observations
rendered by ``format_observation_messages``.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from officeqa_rl.backends.protocol import GenerationBackend
from officeqa_rl.jobbench._msa import (
    FormatError,
    format_observation_messages,
    parse_regex_actions,
)
from officeqa_rl.types import GeneratedTurn, Message

#: Upstream's text-based action format (``minisweagent.models.litellm_textbased_model``).
DEFAULT_ACTION_REGEX = r"```mswea_bash_command\s*\n(.*?)\n```"

DEFAULT_FORMAT_ERROR_TEMPLATE = """Format error:

<error>
{{error}}
</error>

Provide EXACTLY ONE action in a single ```mswea_bash_command block; found {{actions|length}}.

<response_example>
Your reasoning about what to do next.

```mswea_bash_command
ls -la
```
</response_example>

To finish, run `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` on its own."""

#: The environment already truncates long output, so this stays simple.
DEFAULT_OBSERVATION_TEMPLATE = (
    "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n"
    "{% endif %}<returncode>{{output.returncode}}</returncode>\n"
    "<output>\n{{output.output}}</output>"
)


class GenerationTimeoutError(TimeoutError):
    """Infrastructure timeout, not a scored policy termination."""


class LoopBridge:
    """Run coroutines from a worker thread on the event loop that owns them.

    ``DefaultAgent.run()`` is synchronous and gets executed with
    ``asyncio.to_thread``; the generation backends it must call are async and
    belong to the caller's loop. Constructed without a loop (tests, single
    episode scripts), each call gets its own ``asyncio.run``.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.loop = loop

    @classmethod
    def from_running_loop(cls) -> LoopBridge:
        return cls(asyncio.get_running_loop())

    def run(self, coro: Any, *, timeout: float | None = None) -> Any:
        if self.loop is None:
            try:
                async def bounded():
                    return await asyncio.wait_for(coro, timeout=timeout)
                return asyncio.run(bounded())
            except TimeoutError as error:
                raise GenerationTimeoutError(f"Generation timed out after {timeout}s") from error
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout)
        except FutureTimeoutError as error:
            future.cancel()
            raise GenerationTimeoutError(f"Generation timed out after {timeout}s") from error


@dataclass
class JobBenchModelConfig:
    """Model-side knobs of one JobBench episode."""

    model_name: str = "officeqa-backend"
    action_regex: str = DEFAULT_ACTION_REGEX
    format_error_template: str = DEFAULT_FORMAT_ERROR_TEMPLATE
    observation_template: str = DEFAULT_OBSERVATION_TEMPLATE
    sampling_params: dict[str, Any] = field(default_factory=dict)
    generate_timeout_sec: float | None = None

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        """Mirror the pydantic method upstream calls when serializing."""
        del mode
        return asdict(self)


class BackendModel:
    """Adapter exposing a ``GenerationBackend`` as a mini-swe-agent model."""

    def __init__(
        self,
        backend: GenerationBackend,
        config: JobBenchModelConfig | None = None,
        *,
        bridge: LoopBridge | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or JobBenchModelConfig()
        self.bridge = bridge or LoopBridge()
        #: One entry per model call, in order, including malformed turns.
        self.generated_turns: list[GeneratedTurn] = []
        #: Prompt actually sent for each call, so training can rebuild samples
        #: even when a malformed assistant turn is dropped from the history.
        self.turn_prompts: list[list[Message]] = []
        self.n_calls = 0

    # ------------------------------------------------------------ Model protocol

    def query(self, messages: list[dict[str, Any]], **kwargs) -> dict:
        prompt = _prepare_messages(messages)
        sampling_params = {**self.config.sampling_params, **kwargs}
        turn: GeneratedTurn = self.bridge.run(
            self.backend.generate(prompt, dict(sampling_params)),
            timeout=self.config.generate_timeout_sec,
        )
        self.n_calls += 1
        self.generated_turns.append(turn)
        self.turn_prompts.append(prompt)

        extra: dict[str, Any] = {
            "cost": 0.0,
            "timestamp": time.time(),
            "turn_index": self.n_calls - 1,
            "finish_reason": turn.finish_reason,
            "usage": turn.usage,
        }
        try:
            actions = parse_regex_actions(
                turn.text or "",
                action_regex=self.config.action_regex,
                format_error_template=self.config.format_error_template,
                template_kwargs={"finish_reason": turn.finish_reason},
            )
        except FormatError as error:
            # Upstream contract: the response and its cost must survive on the
            # error message, because the assistant turn never reaches history.
            error.messages[0]["extra"].update(extra)
            raise

        message: dict[str, Any] = {"role": "assistant", "content": turn.text or ""}
        if turn.reasoning_content is not None:
            message["reasoning_content"] = turn.reasoning_content
        message["extra"] = {"actions": actions, **extra}
        return message

    def format_message(self, **kwargs) -> dict:
        return dict(kwargs)

    def format_observation_messages(
        self,
        message: dict,
        outputs: list[dict],
        template_vars: dict | None = None,
    ) -> list[dict]:
        del message
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {"model_name": self.config.model_name, **kwargs}

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": {
                        "model_name": self.config.model_name,
                        "action_regex": self.config.action_regex,
                        "sampling_params": dict(self.config.sampling_params),
                    },
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "model_stats": {"api_calls": self.n_calls},
            }
        }


def _prepare_messages(messages: Sequence[dict[str, Any]]) -> list[Message]:
    """Strip harness bookkeeping so backends see plain chat messages.

    ``extra`` holds parsed actions, raw command output, and timestamps; sending
    it upstream would change the prompt tokens a training rollout scores.
    """
    prepared: list[Message] = []
    for message in messages:
        role = message.get("role", "user")
        if role == "exit":  # terminal marker, never part of a prompt
            continue
        prepared_message: Message = {"role": role, "content": message.get("content", "")}
        if message.get("reasoning_content") is not None:
            prepared_message["reasoning_content"] = message["reasoning_content"]
        prepared.append(prepared_message)
    return prepared


__all__ = [
    "BackendModel",
    "DEFAULT_ACTION_REGEX",
    "DEFAULT_FORMAT_ERROR_TEMPLATE",
    "DEFAULT_OBSERVATION_TEMPLATE",
    "JobBenchModelConfig",
    "LoopBridge",
]
