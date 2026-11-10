"""Vertex AI rawPredict generation backend for self-deployed model servers.

The chat template is applied locally with a HuggingFace tokenizer. The
historical/default protocol sends the rendered prompt to SGLang ``/generate``
as ``text``. An opt-in vLLM mode sends the same prompt to OpenAI-compatible
``/v1/completions`` as ``prompt``. Vertex passes request/response bytes
through unchanged, so ``base_url`` may also point at a plain local server —
no auth is attempted for non-https URLs, which is also how the tests run.

Auth is Application Default Credentials (``gcloud auth application-default
login`` or a service-account key). Tokens expire after one hour; the refresh
is guarded by a thread lock because the evaluator calls ``generate`` from
many daemon threads concurrently. ``google-auth`` is imported lazily so the
module imports without it when only non-https URLs are used.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

from officeqa_rl.backends._errors import is_context_length_error
from officeqa_rl.backends._reasoning import strip_gemma_reasoning
from officeqa_rl.backends._worker import run_sync_in_daemon_thread
from officeqa_rl.types import GeneratedTurn, Message

logger = logging.getLogger(__name__)

# Ceiling on one retry's wait before jitter, matching the OPD teacher client.
_MAX_BACKOFF_SEC = 60.0

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_GEMMA_THINK_OPEN = "<|channel>thought"
_GEMMA_THINK_CLOSE = "<channel|>"
_TOKEN_REFRESH_MARGIN = timedelta(minutes=5)
_ADC_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# Must stay identical to _INSTRUCTION_BLOCK in
# officeqa_rl.training.vertex_teacher_client (not imported from there: that
# module needs aiohttp, which eval-only environments may not have).
_INSTRUCTION_BLOCK = "<|im_start|>system\n{}<|im_end|>\n"


def render_instruction_block(text: str) -> str:
    """Wrap an instruction in its own system block unless already rendered.

    Same rule as the OPD teacher client's ``_render_instruction``: during
    training the low/high-resource instruction is prepended to the teacher's
    scoring context as a separate leading system block, before the agent's
    own system prompt. Evaluating a teacher with the same block reproduces
    the exact context layout it sees when scoring.
    """
    return text if "<|im_start|>" in text else _INSTRUCTION_BLOCK.format(text)


class VertexAPIError(RuntimeError):
    """Sanitized HTTP/API failure carrying the status code for retry logic."""

    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def rawpredict_url(project: str, location: str, endpoint_id: str) -> str:
    host = (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )
    return (
        f"https://{host}/v1/projects/{project}/locations/{location}"
        f"/endpoints/{endpoint_id}:rawPredict"
    )


def _split_inline_reasoning(content: str) -> tuple[str | None, str]:
    """Split Qwen ``<think>`` or Gemma ``thought`` channel from visible text.

    A truncated turn with an explicitly opened but unclosed reasoning block is
    all reasoning: leaking it as visible text could execute a fenced command
    the model was still deliberating about. Qwen generation prompts usually
    contain the opening tag, so their generated text can start mid-reasoning;
    Gemma emits its channel opener in the generated text when thinking is on.
    """
    if _GEMMA_THINK_OPEN in content or _GEMMA_THINK_CLOSE in content:
        return strip_gemma_reasoning(content)
    for opening, closing in (
        (_GEMMA_THINK_OPEN, _GEMMA_THINK_CLOSE),
        (_THINK_OPEN, _THINK_CLOSE),
    ):
        reasoning, closed, text = content.partition(closing)
        if closed:
            reasoning = reasoning.strip()
            if reasoning.startswith(opening):
                reasoning = reasoning[len(opening) :].strip()
            return reasoning, text.strip()
        stripped = content.lstrip()
        if stripped.startswith(opening):
            return stripped[len(opening) :].strip(), ""
    return None, content


class VertexSGLangGenerationBackend:
    """Call an SGLang or vLLM server through Vertex ``rawPredict``.

    ``protocol`` defaults to ``sglang`` so existing launchers retain their
    exact request and response behavior.
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str,
        protocol: str = "sglang",
        tokenizer_path: str | None = None,
        tokenizer: Any | None = None,
        max_tokens: int = 8192,
        max_retries: int = 3,
        quota_max_retries: int = 10,
        backoff_base_sec: float = 2.0,
        request_timeout_sec: float = 600.0,
        serialize_calls: bool = False,
        system_instruction: str | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        if tokenizer is None and tokenizer_path is None:
            raise ValueError("VertexSGLangGenerationBackend needs tokenizer or tokenizer_path")
        protocol = protocol.strip().lower()
        if protocol not in {"sglang", "vllm"}:
            raise ValueError("Vertex protocol must be 'sglang' or 'vllm'")
        self.model_name = model_name
        self.model_path = model_name
        self.base_url = base_url
        self.protocol = protocol
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.quota_max_retries = quota_max_retries
        self.backoff_base_sec = float(backoff_base_sec)
        self.request_timeout_sec = request_timeout_sec
        self.enable_thinking = enable_thinking
        self._is_gemma = "gemma" in f"{model_name} {tokenizer_path or ''}".lower()
        self._chat_template_kwargs = (
            {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        )
        # Prepended verbatim to every rendered prompt; empty by default so
        # existing launchers keep byte-identical prompts.
        self._instruction_prefix = (
            render_instruction_block(system_instruction) if system_instruction else ""
        )
        self._tokenizer = tokenizer
        self._tokenizer_path = tokenizer_path
        self._tokenizer_lock = threading.Lock()
        self._generate_lock = threading.Lock() if serialize_calls else None
        self._creds: Any = None
        self._creds_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._n_calls = 0
        self._usage_metadata: list[dict[str, Any]] = []

    async def generate(
        self,
        messages: list[Message],
        sampling_params: dict[str, Any],
    ) -> GeneratedTurn:
        """Call the synchronous HTTP client without blocking the event loop."""
        return await run_sync_in_daemon_thread(
            self._generate_with_optional_lock,
            messages,
            sampling_params,
            thread_name="officeqa-vertex-generate",
        )

    def _generate_with_optional_lock(
        self,
        messages: list[Message],
        sampling_params: dict[str, Any],
    ) -> GeneratedTurn:
        if self._generate_lock is None:
            return self._generate_sync(messages, sampling_params)
        with self._generate_lock:
            return self._generate_sync(messages, sampling_params)

    def _generate_sync(
        self,
        messages: list[Message],
        sampling_params: dict[str, Any],
    ) -> GeneratedTurn:
        prompt = self._render_prompt(messages)
        if self.protocol == "sglang":
            payload = {
                "text": prompt,
                "sampling_params": self._sglang_sampling_params(sampling_params),
            }
        else:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                **sampling_params,
            }

        last_error: Exception | None = None
        transient_attempts = 0
        capacity_attempts = 0
        while True:
            try:
                with self._stats_lock:
                    self._n_calls += 1
                response = self._post_json(payload)
                return self._parse_response(response)
            except Exception as error:
                last_error = error
                # Retrying cannot make a deterministic over-length prompt fit.
                if is_context_length_error(error):
                    break
                if self._is_capacity_error(error):
                    capacity_attempts += 1
                    if capacity_attempts > self.quota_max_retries:
                        break
                    sleep_seconds = self._retry_delay(capacity_attempts)
                    label = "capacity retry"
                else:
                    transient_attempts += 1
                    if transient_attempts > self.max_retries:
                        break
                    sleep_seconds = self._retry_delay(transient_attempts)
                    label = "transient retry"
                logger.warning(
                    "Vertex %s call failed (%s: %s); %s in %.1fs",
                    self.protocol,
                    type(error).__name__,
                    error,
                    label,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        if last_error is not None and is_context_length_error(last_error):
            raise RuntimeError(
                f"Vertex {self.protocol} request exceeds context length: {last_error}"
            ) from last_error
        raise RuntimeError(
            f"Vertex {self.protocol} call failed after retries: {last_error}"
        ) from last_error

    def _sglang_sampling_params(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {"max_new_tokens": self.max_tokens}
        # Gemma's thought boundaries are special tokens. SGLang strips them by
        # default, leaving literal ``thought\n...`` mixed into visible content
        # and making it impossible to separate deliberation from the command.
        # Preserve them in BOTH Gemma ablation arms: if an ostensibly no-think
        # generation reopens a channel, its output must remain auditable.
        if self.enable_thinking or self._is_gemma:
            params.update(
                {
                    "skip_special_tokens": False,
                    "spaces_between_special_tokens": False,
                }
            )
        for key, value in sampling_params.items():
            if key == "max_tokens":
                params["max_new_tokens"] = value
            else:
                params[key] = value
        return params

    def _parse_response(self, response: dict[str, Any]) -> GeneratedTurn:
        if self.protocol == "vllm":
            return self._parse_vllm_response(response)
        content = response.get("text")
        if not isinstance(content, str):
            raise VertexAPIError(None, "Server response contained no text")
        meta = response.get("meta_info") or {}
        finish_reason = meta.get("finish_reason")
        if isinstance(finish_reason, dict):
            finish_reason = finish_reason.get("type")
        usage: dict[str, Any] = {}
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            if isinstance(meta.get(key), int):
                usage[key] = meta[key]
        if usage:
            with self._stats_lock:
                self._usage_metadata.append(usage)
        reasoning, text = _split_inline_reasoning(content)
        return GeneratedTurn(
            text=text,
            reasoning_content=reasoning,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            usage=usage or None,
        )

    def _parse_vllm_response(self, response: dict[str, Any]) -> GeneratedTurn:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise VertexAPIError(None, "vLLM response contained no choices")
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("text"), str):
            raise VertexAPIError(None, "vLLM response contained no text")
        usage = response.get("usage")
        if isinstance(usage, dict):
            with self._stats_lock:
                self._usage_metadata.append(usage)
        reasoning, text = _split_inline_reasoning(choice["text"])
        finish_reason = choice.get("finish_reason")
        return GeneratedTurn(
            text=text,
            reasoning_content=reasoning,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            usage=usage if isinstance(usage, dict) else None,
        )

    def _render_prompt(self, messages: list[Message]) -> str:
        chat: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "exit":
                continue
            if role not in {"system", "user", "assistant"}:
                raise TypeError(f"Unsupported chat message role: {role!r}")
            content = message.get("content", "")
            if not isinstance(content, str):
                raise TypeError(
                    "The Vertex SGLang backend is text-only; multimodal "
                    "content is not supported by the /generate route"
                )
            if self._is_gemma and role == "assistant":
                # Defend against raw/repeated thought blocks in saved history
                # too; never rely on markerless decoding or template version.
                _, content = strip_gemma_reasoning(content)
            chat.append({"role": role, "content": content})
        rendered = self._get_tokenizer().apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **self._chat_template_kwargs,
        )
        return self._instruction_prefix + rendered

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            with self._tokenizer_lock:
                if self._tokenizer is None:
                    from transformers import AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self._tokenizer_path, trust_remote_code=True
                    )
        return self._tokenizer

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        token = self._access_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.request_timeout_sec
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise VertexAPIError(error.code, self._error_message(body)) from error
        except urllib.error.URLError as error:
            raise VertexAPIError(None, f"Server network error: {error.reason}") from error
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as error:
            raise VertexAPIError(None, "Server returned invalid JSON") from error
        if not isinstance(parsed, dict):
            raise VertexAPIError(None, "Server returned a non-object JSON response")
        return parsed

    def _access_token(self) -> str | None:
        """ADC bearer token for https URLs; plain-http servers get none."""
        if not self.base_url.startswith("https://"):
            return None
        with self._creds_lock:
            if self._creds is None or self._token_stale():
                import google.auth
                import google.auth.transport.requests

                if self._creds is None:
                    self._creds, _ = google.auth.default(scopes=[_ADC_SCOPE])
                self._creds.refresh(google.auth.transport.requests.Request())
            return self._creds.token

    def _token_stale(self) -> bool:
        if not self._creds.valid:
            return True
        expiry = getattr(self._creds, "expiry", None)  # naive UTC in google-auth
        if expiry is None:
            return False
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        return now + _TOKEN_REFRESH_MARGIN >= expiry

    @staticmethod
    def _error_message(body: str) -> str:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return body[:1000]
        error = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        return str(error or parsed)[:1000]

    @staticmethod
    def _is_capacity_error(error: Exception) -> bool:
        return getattr(error, "status_code", None) in {429, 503}

    def _retry_delay(self, attempt: int) -> float:
        """Truncated exponential backoff with jitter, as in the OPD client.

        Jitter (0.5x-1.5x) matters with a pool: without it every in-flight
        episode that a restarting model server rejected would come back at the
        same instant and knock it over again.
        """
        delay = min(_MAX_BACKOFF_SEC, self.backoff_base_sec * 2 ** max(0, attempt - 1))
        return delay * (0.5 + random.random())

    def get_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            usage_by_call = list(self._usage_metadata)
            n_calls = self._n_calls
        totals: dict[str, int] = {}
        for usage in usage_by_call:
            for key, value in usage.items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value
        return {
            "n_calls": n_calls,
            "usage_totals": totals,
            "usage_by_call": usage_by_call,
        }


__all__ = ["VertexAPIError", "VertexSGLangGenerationBackend", "rawpredict_url"]
