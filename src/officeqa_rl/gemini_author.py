"""Gemini client used to author synthetic QA pairs."""

from __future__ import annotations

import importlib
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


logger = logging.getLogger(__name__)
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


def gemini_api_keys_from_env() -> list[str] | None:
    """Read authoring keys from a file or comma/whitespace-separated list."""
    keys_file = os.environ.get("GEMINI_API_KEYS_FILE")
    if keys_file:
        lines = Path(keys_file).read_text().splitlines()
        keys = [line.strip() for line in lines]
        keys = [key for key in keys if key and not key.startswith("#")]
        if not keys:
            raise RuntimeError(f"GEMINI_API_KEYS_FILE {keys_file} contains no keys")
        return keys
    raw = os.environ.get("GEMINI_API_KEYS", "")
    keys = [key for key in re.split(r"[,\s]+", raw) if key]
    return keys or None


@dataclass
class GeminiAuthor:
    """Generate authored QA JSON while rotating across Gemini API projects."""

    model: str = DEFAULT_GEMINI_MODEL
    # Gemini thinking tokens draw from this same budget; a truncated response
    # is recorded as an author_parse_error by the synthesis script.
    max_output_tokens: int = 16384
    api_keys: Sequence[str] | None = None
    quota_max_retries: int = 10

    def __post_init__(self) -> None:
        genai = importlib.import_module("google.genai")
        self._types = importlib.import_module("google.genai.types")
        http_options = self._types.HttpOptions(api_version="v1beta")
        keys = [key for key in (self.api_keys or gemini_api_keys_from_env() or []) if key]
        if keys:
            self._clients = [
                genai.Client(api_key=key, http_options=http_options) for key in keys
            ]
            logger.info("Gemini author rotating across %d API keys", len(keys))
        else:
            # Preserve google-genai's normal GEMINI_API_KEY / GOOGLE_API_KEY
            # environment resolution for a single-key run.
            self._clients = [genai.Client(http_options=http_options)]
        self._client_cursor = 0
        self._client_lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> str:
        last_error: Exception | None = None
        capacity_attempts = 0
        client = self._next_client()
        while True:
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=self._types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens or self.max_output_tokens,
                    ),
                )
                break
            except Exception as error:  # noqa: BLE001 - provider-specific SDK errors
                last_error = error
                if not self._is_capacity_error(error):
                    raise
                capacity_attempts += 1
                if capacity_attempts > self.quota_max_retries:
                    raise RuntimeError(
                        f"Gemini authoring failed after retries: {last_error}"
                    ) from last_error
                client = self._next_client()
                if len(self._clients) > 1 and capacity_attempts % len(self._clients):
                    sleep_seconds = random.uniform(0.5, 2.0)
                    label = "next API key"
                else:
                    sleep_seconds = min(60, 5 * 2 ** (capacity_attempts - 1))
                    label = "full key-pool cycle"
                logger.warning(
                    "Gemini authoring failed (%s); retrying on %s in %.1fs",
                    error,
                    label,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        text = getattr(response, "text", None)
        if text:
            return text
        chunks = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "thought", False):
                    continue
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(part_text)
        return "\n".join(chunks)

    def _next_client(self) -> Any:
        """Spread concurrent authoring jobs round-robin across API projects."""
        with self._client_lock:
            client = self._clients[self._client_cursor % len(self._clients)]
            self._client_cursor += 1
            return client

    @staticmethod
    def _is_capacity_error(error: Exception) -> bool:
        code = getattr(error, "code", None) or getattr(error, "status_code", None)
        if code in (429, 503):
            return True
        status = getattr(error, "status", None)
        if status in ("RESOURCE_EXHAUSTED", "UNAVAILABLE"):
            return True
        message = str(error)
        return "RESOURCE_EXHAUSTED" in message or "503 UNAVAILABLE" in message


__all__ = ["DEFAULT_GEMINI_MODEL", "GeminiAuthor", "gemini_api_keys_from_env"]
