"""OpenAI-compatible generation backend for self-hosted servers (vLLM/SGLang)."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from itertools import count
from typing import Any

from officeqa_rl.backends._errors import is_context_length_error
from officeqa_rl.backends._worker import run_sync_in_daemon_thread
from officeqa_rl.types import GeneratedTurn, Message


logger = logging.getLogger(__name__)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class _OpenAIServerEpisodeBackend:
    """Bind all turns of one episode to one vLLM data-parallel rank."""

    def __init__(
        self,
        backend: "OpenAIServerGenerationBackend",
        data_parallel_rank: int,
    ) -> None:
        self._backend = backend
        self._data_parallel_rank = data_parallel_rank

    async def generate(
        self,
        messages: list[Message],
        sampling_params: dict[str, Any],
    ) -> GeneratedTurn:
        return await self._backend._generate_on_rank(
            messages,
            sampling_params,
            data_parallel_rank=self._data_parallel_rank,
        )


class OpenAIServerAPIError(RuntimeError):
    """Sanitized HTTP/API failure that never includes an authorization key."""

    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenAIServerGenerationBackend:
    """Call a locally served model through its Chat Completions endpoint.

    Servers launched without a reasoning parser (e.g. ``vllm serve`` without
    ``--reasoning-parser``) return thinking inline in ``content`` terminated by
    ``</think>``; that reasoning is split into ``reasoning_content`` so the
    agent's single-fenced-command parser only ever sees the visible answer.
    Prior-turn reasoning is not replayed: reasoning-model chat templates drop
    it, and replaying it would inflate the context instead.
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        max_tokens: int = 8192,
        max_retries: int = 3,
        quota_max_retries: int = 10,
        request_timeout_sec: float = 600.0,
        serialize_calls: bool = False,
        api_key: str | None = None,
        data_parallel_size: int = 1,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_name
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.quota_max_retries = quota_max_retries
        self.request_timeout_sec = request_timeout_sec
        # Self-hosted servers usually run without authentication; send a
        # bearer token only when one is configured.
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._generate_lock = threading.Lock() if serialize_calls else None
        self._stats_lock = threading.Lock()
        self._n_calls = 0
        self._usage_metadata: list[dict[str, Any]] = []
        self._data_parallel_size = max(1, int(data_parallel_size))
        self._next_data_parallel_rank = count()

    def for_episode(
        self, affinity_key: str
    ) -> OpenAIServerGenerationBackend | _OpenAIServerEpisodeBackend:
        """Return a view that pins the episode to one server-side DP rank."""
        if self._data_parallel_size == 1:
            return self
        del affinity_key
        rank = next(self._next_data_parallel_rank) % self._data_parallel_size
        return _OpenAIServerEpisodeBackend(self, rank)

    async def generate(
        self,
        messages: list[Message],
        sampling_params: dict[str, Any],
    ) -> GeneratedTurn:
        """Call the synchronous HTTP client without blocking the event loop."""
        return await self._generate_on_rank(messages, sampling_params)

    async def _generate_on_rank(
        self,
        messages: list[Message],
        sampling_params: dict[str, Any],
        *,
        data_parallel_rank: int | None = None,
    ) -> GeneratedTurn:
        return await run_sync_in_daemon_thread(
            self._generate_with_optional_lock,
            messages,
            sampling_params,
            data_parallel_rank,
            thread_name="officeqa-openai-server-generate",
        )

    def _generate_with_optional_lock(
        self,
        messages: list[Message],
        sampling_params: dict[str, Any],
        data_parallel_rank: int | None = None,
    ) -> GeneratedTurn:
        if self._generate_lock is None:
            return self._generate_sync(
                messages,
                sampling_params,
                data_parallel_rank=data_parallel_rank,
            )
        with self._generate_lock:
            return self._generate_sync(
                messages,
                sampling_params,
                data_parallel_rank=data_parallel_rank,
            )

    def _generate_sync(
        self,
        messages: list[Message],
        sampling_params: dict[str, Any],
        *,
        data_parallel_rank: int | None = None,
    ) -> GeneratedTurn:
        payload = {
            "model": self.model_name,
            "messages": self._to_chat_messages(messages),
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
                response = (
                    self._post_json(payload)
                    if data_parallel_rank is None
                    else self._post_json(
                        payload,
                        data_parallel_rank=data_parallel_rank,
                    )
                )
                usage = response.get("usage")
                if isinstance(usage, dict):
                    with self._stats_lock:
                        self._usage_metadata.append(usage)
                choices = response.get("choices") or []
                if not choices:
                    raise OpenAIServerAPIError(None, "Server response contained no choices")
                choice = choices[0]
                message = choice.get("message") or {}
                content = message.get("content")
                if not isinstance(content, str):
                    raise OpenAIServerAPIError(None, "Server response contained no text content")
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                text = content
                if not isinstance(reasoning, str):
                    reasoning, text = self._split_inline_reasoning(content)
                return GeneratedTurn(
                    text=text,
                    reasoning_content=reasoning if isinstance(reasoning, str) else None,
                    finish_reason=choice.get("finish_reason"),
                    usage=usage if isinstance(usage, dict) else None,
                )
            except Exception as error:
                last_error = error
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
                    "OpenAI-compatible server call failed (%s: %s); %s in %.1fs",
                    type(error).__name__,
                    error,
                    label,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        if last_error is not None and is_context_length_error(last_error):
            raise RuntimeError(
                f"OpenAI-compatible server request exceeds context length: {last_error}"
            ) from last_error
        raise RuntimeError(
            f"OpenAI-compatible server call failed after retries: {last_error}"
        ) from last_error

    @staticmethod
    def _split_inline_reasoning(content: str) -> tuple[str | None, str]:
        """Split ``[<think>]...</think>answer`` content from a server that has
        no reasoning parser. A truncated turn with an unclosed ``<think>`` is
        all reasoning: leaking it as text could execute a fenced command the
        model was still deliberating about."""
        reasoning, closed, text = content.partition(_THINK_CLOSE)
        if not closed:
            if content.lstrip().startswith(_THINK_OPEN):
                return content.lstrip()[len(_THINK_OPEN):].strip(), ""
            return None, content
        reasoning = reasoning.strip()
        if reasoning.startswith(_THINK_OPEN):
            reasoning = reasoning[len(_THINK_OPEN):].strip()
        return reasoning, text.strip()

    def _post_json(
        self,
        payload: dict[str, Any],
        *,
        data_parallel_rank: int | None = None,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if data_parallel_rank is not None:
            headers["X-data-parallel-rank"] = str(data_parallel_rank)
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.request_timeout_sec,
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise OpenAIServerAPIError(error.code, self._error_message(body)) from error
        except urllib.error.URLError as error:
            raise OpenAIServerAPIError(None, f"Server network error: {error.reason}") from error
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as error:
            raise OpenAIServerAPIError(None, "Server returned invalid JSON") from error
        if not isinstance(parsed, dict):
            raise OpenAIServerAPIError(None, "Server returned a non-object JSON response")
        return parsed

    @staticmethod
    def _error_message(body: str) -> str:
        """Extract the provider message without logging request headers."""
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

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        return min(60.0, 2.0 ** max(0, attempt - 1))

    @staticmethod
    def _to_chat_messages(messages: list[Message]) -> list[dict[str, Any]]:
        chat_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "exit":
                continue
            if role not in {"system", "user", "assistant"}:
                raise TypeError(f"Unsupported chat message role: {role!r}")
            content = message.get("content", "")
            if isinstance(content, list):
                converted: list[dict[str, Any]] = []
                for item in content:
                    if not isinstance(item, dict):
                        raise TypeError(
                            "Multimodal message items must be dictionaries"
                        )
                    if item.get("type") == "text":
                        converted.append(
                            {"type": "text", "text": str(item.get("text", ""))}
                        )
                    elif item.get("type") == "image":
                        converted.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": OpenAIServerGenerationBackend._image_data_url(
                                        item.get("image")
                                    )
                                },
                            }
                        )
                    else:
                        raise TypeError(
                            f"Unsupported multimodal item type: {item.get('type')!r}"
                        )
                content = converted
            elif not isinstance(content, str):
                raise TypeError(
                    "Message content must be a string or multimodal content list"
                )
            chat_messages.append({"role": role, "content": content})
        return chat_messages

    @staticmethod
    def _image_data_url(image: Any) -> str:
        if image is None or not hasattr(image, "save"):
            raise TypeError("Multimodal image item must contain a Pillow image")
        buffer = io.BytesIO()
        converted = image.convert("RGB") if hasattr(image, "convert") else image
        converted.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def get_stats(self) -> dict[str, Any]:
        """Aggregate usage without revealing any configured key material."""
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


__all__ = ["OpenAIServerAPIError", "OpenAIServerGenerationBackend"]
