"""Moonshot Kimi OpenAI-compatible generation backend."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from officeqa_rl.backends._errors import is_context_length_error
from officeqa_rl.backends._worker import run_sync_in_daemon_thread
from officeqa_rl.types import GeneratedTurn, Message


logger = logging.getLogger(__name__)


class KimiAPIError(RuntimeError):
    """Sanitized HTTP/API failure that never includes an authorization key."""

    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class KimiGenerationBackend:
    """Call Kimi K3 with one Moonshot API key."""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "https://api.moonshot.ai/v1",
        max_tokens: int = 8192,
        reasoning_effort: str = "max",
        max_retries: int = 3,
        quota_max_retries: int = 10,
        request_timeout_sec: float = 300.0,
        serialize_calls: bool = False,
        api_key: str | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("MOONSHOT_API_KEY")
        if not api_key:
            raise RuntimeError("Kimi requires MOONSHOT_API_KEY")
        if reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("reasoning_effort must be one of: low, high, max")

        self.model_name = model_name
        self.model_path = model_name
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.quota_max_retries = quota_max_retries
        self.request_timeout_sec = request_timeout_sec
        self._api_key = api_key
        self._generate_lock = threading.Lock() if serialize_calls else None
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
            thread_name="officeqa-kimi-generate",
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
        # K3 fixes temperature/top_p/n/penalties server-side. Deliberately
        # ignore generic runner sampling parameters rather than sending an
        # incompatible temperature=0 default.
        del sampling_params
        payload = {
            "model": self.model_name,
            "messages": self._to_chat_messages(messages),
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_tokens,
        }

        last_error: Exception | None = None
        transient_attempts = 0
        capacity_attempts = 0
        while True:
            try:
                with self._stats_lock:
                    self._n_calls += 1
                response = self._post_json(self._api_key, payload)
                usage = response.get("usage")
                if isinstance(usage, dict):
                    with self._stats_lock:
                        self._usage_metadata.append(usage)
                choices = response.get("choices") or []
                if not choices:
                    raise KimiAPIError(None, "Kimi response contained no choices")
                choice = choices[0]
                message = choice.get("message") or {}
                content = message.get("content")
                if not isinstance(content, str):
                    raise KimiAPIError(None, "Kimi response contained no text content")
                reasoning = message.get("reasoning_content")
                return GeneratedTurn(
                    text=content,
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
                    "Kimi call failed (%s: %s); %s in %.1fs",
                    type(error).__name__,
                    error,
                    label,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        if last_error is not None and is_context_length_error(last_error):
            raise RuntimeError(
                f"Kimi request exceeds context length: {last_error}"
            ) from last_error
        raise RuntimeError(f"Kimi call failed after retries: {last_error}") from last_error

    def _post_json(self, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
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
            raise KimiAPIError(error.code, self._error_message(body)) from error
        except urllib.error.URLError as error:
            raise KimiAPIError(None, f"Kimi network error: {error.reason}") from error
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as error:
            raise KimiAPIError(None, "Kimi returned invalid JSON") from error
        if not isinstance(parsed, dict):
            raise KimiAPIError(None, "Kimi returned a non-object JSON response")
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
                raise TypeError(f"Unsupported Kimi message role: {role!r}")
            content = message.get("content", "")
            if not isinstance(content, str):
                raise TypeError("Kimi backend is text-only; message content must be a string")
            item: dict[str, Any] = {"role": role, "content": content}
            reasoning = message.get("reasoning_content")
            if role == "assistant" and isinstance(reasoning, str):
                item["reasoning_content"] = reasoning
            chat_messages.append(item)
        return chat_messages

    def get_stats(self) -> dict[str, Any]:
        """Aggregate usage without revealing which keys served the calls."""
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


__all__ = ["KimiAPIError", "KimiGenerationBackend"]
