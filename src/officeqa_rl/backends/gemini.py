"""Gemini API generation backend for OfficeQA inference."""

from __future__ import annotations

import importlib
import io
import logging
import random
import re
import threading
import time
from typing import Any, Sequence

from officeqa_rl.backends._errors import is_context_length_error
from officeqa_rl.backends._worker import run_sync_in_daemon_thread
from officeqa_rl.types import GeneratedTurn, Message


logger = logging.getLogger(__name__)

# Server-suggested wait from a 429 body, e.g. "'retryDelay': '38s'".
_RETRY_DELAY_PATTERN = re.compile(r"retryDelay[\"']?\s*[:=]\s*[\"']?(\d+(?:\.\d+)?)s")


class GeminiGenerationBackend:
    """Convert OfficeQA messages to Gemini ``generate_content`` calls."""

    def __init__(
        self,
        model_name: str,
        *,
        api_version: str = "v1",
        max_tokens: int = 2048,
        max_retries: int = 3,
        quota_max_retries: int = 10,
        thinking_budget: int | None = 0,
        include_thoughts: bool = False,
        serialize_calls: bool = False,
        api_keys: Sequence[str] | None = None,
    ) -> None:
        try:
            genai = importlib.import_module("google.genai")
            types = importlib.import_module("google.genai.types")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "google-genai is required for GeminiGenerationBackend; install "
                "the 'officeqa-rl[gemini]' optional dependencies"
            ) from exc

        self.model_path = model_name
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.quota_max_retries = quota_max_retries
        self.thinking_budget = thinking_budget
        self.include_thoughts = include_thoughts
        # Gemini 3.6 uses provider-managed sampling/thinking defaults. The API
        # rejects the legacy temperature/top-p/top-k and thinking-budget
        # fields accepted by earlier Gemini generations.
        self._uses_managed_sampling = model_name.startswith("gemini-3.6")
        self._types = types
        self._generate_lock = threading.Lock() if serialize_calls else None
        self._stats_lock = threading.Lock()
        self._n_calls = 0
        self._usage_metadata: list[dict[str, Any]] = []
        self._response_metadata_log: list[dict[str, Any]] = []
        self._warned_thinking_config = False
        http_options = types.HttpOptions(api_version=api_version)
        keys = [key for key in (api_keys or []) if key]
        if keys:
            self._clients = [
                genai.Client(api_key=key, http_options=http_options) for key in keys
            ]
            logger.info("Gemini backend rotating across %d API keys", len(keys))
        else:
            # Single client picking up GOOGLE_API_KEY / GEMINI_API_KEY from env.
            self._clients = [genai.Client(http_options=http_options)]
        self.client = self._clients[0]
        self._client_cursor = 0
        self._client_lock = threading.Lock()

    async def generate(
        self,
        messages: list[Message],
        sampling_params: dict[str, Any],
    ) -> GeneratedTurn:
        """Call the synchronous Gemini SDK without blocking the event loop."""
        return await run_sync_in_daemon_thread(
            self._generate_with_optional_lock,
            messages,
            sampling_params,
            thread_name="officeqa-gemini-generate",
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
        system_instruction, contents = self._to_gemini_contents(messages)
        config = self._generate_content_config(system_instruction, sampling_params)

        last_error: Exception | None = None
        transient_attempts = 0
        quota_attempts = 0
        client = self._next_client()
        while True:
            try:
                with self._stats_lock:
                    self._n_calls += 1
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                usage = getattr(response, "usage_metadata", None)
                usage_dict = self._model_dump(usage) if usage is not None else None
                response_metadata = self._response_metadata(response)
                with self._stats_lock:
                    if usage_dict is not None:
                        self._usage_metadata.append(usage_dict)
                    self._response_metadata_log.append(response_metadata)
                candidates = getattr(response, "candidates", None) or []
                finish_reason = (
                    self._jsonable(getattr(candidates[0], "finish_reason", None))
                    if candidates
                    else None
                )
                return GeneratedTurn(
                    text=self._response_text(response),
                    finish_reason=str(finish_reason) if finish_reason is not None else None,
                    usage=usage_dict,
                )
            except Exception as error:
                last_error = error
                if is_context_length_error(error):
                    break
                if self._is_capacity_error(error):
                    # Quota exhaustion (429) and load shedding (503) are
                    # guaranteed-transient: wait out the rate window or demand
                    # spike (server-suggested when available) with its own,
                    # larger retry budget.
                    quota_attempts += 1
                    if quota_attempts > self.quota_max_retries:
                        break
                    client = self._next_client()
                    if len(self._clients) > 1 and quota_attempts % len(self._clients):
                        # Other projects have independent quotas: try the next
                        # key almost immediately, and only honor the full rate
                        # window once every key in the pool has hit 429.
                        sleep_seconds = random.uniform(0.5, 2.0)
                        label = (
                            f"capacity retry {quota_attempts}/{self.quota_max_retries}"
                            " on next API key"
                        )
                    else:
                        sleep_seconds = self._quota_retry_delay(error, quota_attempts)
                        label = f"capacity retry {quota_attempts}/{self.quota_max_retries}"
                else:
                    transient_attempts += 1
                    if transient_attempts > self.max_retries:
                        break
                    sleep_seconds = min(30, 2 ** (transient_attempts - 1))
                    label = f"retry {transient_attempts}/{self.max_retries}"
                logger.warning(
                    "Gemini call failed (%s: %s); %s in %.1fs",
                    type(error).__name__,
                    error,
                    label,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        if last_error is not None and is_context_length_error(last_error):
            raise RuntimeError(
                f"Gemini request exceeds context length: {last_error}"
            ) from last_error
        raise RuntimeError(f"Gemini call failed after retries: {last_error}") from last_error

    def _next_client(self) -> Any:
        """Round-robin over the key pool so concurrent trajectories spread
        their request rate across projects."""
        with self._client_lock:
            client = self._clients[self._client_cursor % len(self._clients)]
            self._client_cursor += 1
            return client

    @staticmethod
    def _is_capacity_error(error: Exception) -> bool:
        """429 quota exhaustion, or 503 load shedding under demand spikes.

        Both mean "the service refused this request right now, try again",
        so both take the rotate-across-keys path with the quota retry budget
        rather than the short transient-error budget.
        """
        code = getattr(error, "code", None) or getattr(error, "status_code", None)
        if code in (429, 503):
            return True
        status = getattr(error, "status", None)
        if status in ("RESOURCE_EXHAUSTED", "UNAVAILABLE"):
            return True
        text = str(error)
        return "RESOURCE_EXHAUSTED" in text or "503 UNAVAILABLE" in text

    @staticmethod
    def _quota_retry_delay(error: Exception, attempt: int) -> float:
        """Honor the 429 body's retryDelay when present; jitter avoids all
        concurrent trajectories retrying in the same instant."""
        match = _RETRY_DELAY_PATTERN.search(str(error))
        base = float(match.group(1)) if match else min(60.0, 5.0 * 2 ** (attempt - 1))
        return base + random.uniform(0, 0.25 * base)

    def _generate_content_config(
        self,
        system_instruction: str,
        sampling_params: dict[str, Any],
    ) -> Any:
        extra_params = dict(sampling_params)
        temperature = extra_params.pop("temperature", 0.0)
        max_output_tokens = extra_params.pop(
            "max_tokens",
            extra_params.pop("max_output_tokens", self.max_tokens),
        )
        if self._uses_managed_sampling:
            extra_params.pop("top_p", None)
            extra_params.pop("top_k", None)
        kwargs = {
            "system_instruction": system_instruction or None,
            "max_output_tokens": max_output_tokens,
            **extra_params,
        }
        if not self._uses_managed_sampling:
            kwargs["temperature"] = temperature

        if (
            not self._uses_managed_sampling
            and self.thinking_budget is not None
            and hasattr(self._types, "ThinkingConfig")
        ):
            try:
                kwargs["thinking_config"] = self._types.ThinkingConfig(
                    thinking_budget=self.thinking_budget,
                    include_thoughts=self.include_thoughts,
                )
            except TypeError:
                try:
                    kwargs["thinking_config"] = self._types.ThinkingConfig(
                        thinking_budget=self.thinking_budget,
                    )
                except TypeError:
                    kwargs.pop("thinking_config", None)

        try:
            return self._types.GenerateContentConfig(**kwargs)
        except TypeError:
            if "thinking_config" not in kwargs:
                raise
            if not self._warned_thinking_config:
                logger.warning(
                    "Gemini SDK/model rejected thinking_config; continuing without it"
                )
                self._warned_thinking_config = True
            kwargs.pop("thinking_config", None)
            return self._types.GenerateContentConfig(**kwargs)

    def _to_gemini_contents(self, messages: list[Message]) -> tuple[str, list[Any]]:
        system_parts: list[str] = []
        contents: list[Any] = []
        for message in messages:
            role = message.get("role", "user")
            if role == "exit":
                continue
            if role == "system":
                text = self._content_text(message.get("content", ""))
                if text:
                    system_parts.append(text)
                continue

            gemini_role = "model" if role == "assistant" else "user"
            parts = self._content_parts(message.get("content", ""))
            if not parts:
                continue
            if contents and contents[-1].role == gemini_role:
                contents[-1].parts.extend(parts)
            else:
                contents.append(self._types.Content(role=gemini_role, parts=parts))
        return "\n\n".join(system_parts), contents

    def _content_parts(self, content: Any) -> list[Any]:
        if isinstance(content, str):
            return [self._types.Part.from_text(text=content)]
        if not isinstance(content, list):
            return [self._types.Part.from_text(text=str(content))]

        parts: list[Any] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                image = item.get("image")
                if self._is_pil_image(image):
                    parts.append(
                        self._types.Part.from_bytes(
                            data=self._image_to_png_bytes(image),
                            mime_type="image/png",
                        )
                    )
                elif image:
                    parts.append(self._types.Part.from_text(text=str(image)))
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(self._types.Part.from_text(text=str(item.get("text", ""))))
            elif isinstance(item, dict):
                parts.append(self._types.Part.from_text(text=str(item)))
            else:
                parts.append(self._types.Part.from_text(text=str(item)))
        return parts

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif not (isinstance(item, dict) and item.get("type") == "image"):
                text_parts.append(str(item))
        return "\n".join(part for part in text_parts if part)

    @staticmethod
    def _is_pil_image(value: Any) -> bool:
        try:
            image_module = importlib.import_module("PIL.Image")
        except ModuleNotFoundError:
            return False
        return isinstance(value, image_module.Image)

    @staticmethod
    def _image_to_png_bytes(image: Any) -> bytes:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()

    def _response_metadata(self, response: Any) -> dict[str, Any]:
        candidates = []
        for candidate in getattr(response, "candidates", None) or []:
            candidates.append(
                {
                    "finish_reason": self._jsonable(
                        getattr(candidate, "finish_reason", None)
                    ),
                    "finish_message": self._jsonable(
                        getattr(candidate, "finish_message", None)
                    ),
                    "avg_logprobs": self._jsonable(
                        getattr(candidate, "avg_logprobs", None)
                    ),
                }
            )
        return {
            "text_len": len(self._response_text(response)),
            "candidates": candidates,
            "prompt_feedback": self._jsonable(
                getattr(response, "prompt_feedback", None)
            ),
        }

    def _jsonable(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        name = getattr(value, "name", None)
        return name if name is not None else str(value)

    @staticmethod
    def _model_dump(value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return dict(value)

    @staticmethod
    def _response_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return text
        chunks = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(part_text)
        return "\n".join(chunks)

    def get_stats(self) -> dict[str, Any]:
        """Aggregate usage and response metadata across API attempts."""
        with self._stats_lock:
            usage_by_call = list(self._usage_metadata)
            response_metadata = list(self._response_metadata_log)
            n_calls = self._n_calls
        totals: dict[str, int] = {}
        for usage in usage_by_call:
            for key, value in usage.items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value
        return {
            "n_calls": n_calls,
            "cost": 0.0,
            "usage_totals": totals,
            "usage_by_call": usage_by_call,
            "response_metadata": response_metadata,
        }


__all__ = ["GeminiGenerationBackend"]
