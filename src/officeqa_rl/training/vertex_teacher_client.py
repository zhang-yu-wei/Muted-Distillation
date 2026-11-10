"""Async client scoring fixed token sequences against the served OPD teacher.

RL+OPD needs the teacher's log-probs of the student's sampled tokens on every
training sample; slime's train side then applies the reverse-KL advantage tilt
(``apply_opd_kl_to_advantages``). Instead of hosting a second local server, the
teacher is a Vertex AI self-deployed SGLang or vLLM container. Vertex
``:rawPredict`` passes request/response bytes through unmodified, so the client
speaks the serving container's native scoring protocol:

* SGLang ``/generate``: post ``input_ids`` with ``return_logprob: true`` and
  read ``meta_info.input_token_logprobs``;
* vLLM ``/v1/completions``: post the exact token IDs as ``prompt`` with
  ``echo: true``, ``logprobs: 1``, and ``max_tokens: 0``; read
  ``choices[0].logprobs.token_logprobs``.

To avoid changing existing launchers, protocol selection defaults to
SGLang-first auto detection. The first successful format is cached for the
life of the client. A local URL ending in ``/generate`` or
``/v1/completions`` is known without probing. Authentication is attempted only
for https URLs.

Client rules:

- its own concurrency semaphore, far smaller than slime's rollout semaphore,
  so Vertex backpressure can never starve local policy generation;
- a hard ``asyncio.wait_for`` timeout on every request;
- truncated exponential backoff with jitter, retrying the HTTP call — never
  the trajectory;
- self-managed ADC bearer token: Google credentials expire after one hour and
  the documented refresher is not coroutine-safe, so refresh happens behind a
  double-checked ``asyncio.Lock`` in a worker thread, proactively once less
  than five minutes of lifetime remain;
- one client per rollout worker: ``get_teacher_client`` caches per config and
  every call shares one ``aiohttp`` session/connection pool.

``google-auth`` is imported lazily inside the refresh path: the training
container ships it, the repo venv does not, and tests exercise the client
against plain-http fakes that need no credentials.

COPD low-resource conditioning (``officeqa_opd_instruction``): when set, a
reasoning-effort instruction is prepended (as a system block) to every scored
sequence, so the teacher's log-probs — and therefore the reverse-KL advantage
tilt — prefer tokens compatible with efficient, low-tool-call behaviour. This
is the single-context variant of Contrastive On-Policy Distillation
(COPD): one teacher pass, the
existing tilt machinery, no loss changes. The student never sees the
instruction; only the teacher is conditioned on it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class ScoredToken:
    """One scored position when ``score_tokens(..., top_logprobs=k)`` is used.

    ``top_log_probs`` holds the server's top-k log-probs at this position in
    descending order (None when the server returned none for the position).
    ``top_token_ids`` aligns with it where the protocol carries ids (SGLang);
    vLLM's completion echo keys top entries by token text, so ids are None.
    """

    log_prob: float
    top_log_probs: list[float] | None = None
    top_token_ids: list[int] | None = None

_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
# Refresh ahead of the 1-hour expiry so a token cannot lapse mid-request.
_TOKEN_REFRESH_MARGIN = timedelta(minutes=5)
_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_MAX_BACKOFF_SEC = 60.0


class TeacherScoringError(RuntimeError):
    """Unrecoverable teacher-scoring failure (bad config or exhausted retries)."""


class _RetryableHTTPError(Exception):
    def __init__(self, status: int, snippet: str):
        super().__init__(f"HTTP {status}: {snippet}")
        self.status = status


class _ProtocolHTTPError(TeacherScoringError):
    """An HTTP response that commonly means the request wire format is wrong."""

    def __init__(self, status: int, snippet: str):
        super().__init__(f"HTTP {status}: {snippet}")
        self.status = status


class _ProtocolMismatchError(TeacherScoringError):
    """A successful HTTP response that does not match the selected protocol."""


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


class VertexTeacherClient:
    """Scores exact token sequences with the teacher behind a Vertex endpoint."""

    def __init__(
        self,
        *,
        teacher_url: str,
        backend: str = "auto",
        model: str | None = None,
        max_concurrent: int = 16,
        timeout_sec: float = 180.0,
        max_attempts: int = 5,
        backoff_base_sec: float = 1.0,
        instruction_token_ids: list[int] | None = None,
        heavy_instruction_token_ids: list[int] | None = None,
        contrast_clip: float = 10.0,
        bos_token_id: int | None = None,
    ):
        # Round-robin over multiple endpoints serving the SAME model:
        # ``teacher_url`` accepts a comma-separated list. Every scoring call
        # may carry an affinity key (the episode's run_id): the first call of
        # an episode picks the next endpoint in rotation and all later calls
        # of that episode stick to it, so that endpoint's radix/KV cache keeps
        # hitting across the episode's turns while load still spreads evenly.
        self._urls = [u.strip() for u in teacher_url.split(",") if u.strip()]
        if not self._urls:
            raise ValueError("teacher_url must contain at least one URL")
        self.teacher_url = self._urls[0]
        backend = backend.strip().lower()
        if backend not in {"auto", "sglang", "vllm"}:
            raise ValueError(
                f"unsupported OPD teacher backend {backend!r}; expected auto, "
                "sglang, or vllm"
            )
        self.teacher_backend = backend
        self.teacher_model = model or None
        # All endpoints serve the same deployment, so one detected protocol
        # applies to every URL.
        self._resolved_backend = (
            backend if backend != "auto" else _backend_from_url(self._urls[0])
        )
        self._protocol_lock = asyncio.Lock()
        self._affinity: dict[str, int] = {}
        self._rr_next = 0
        # COPD conditioning (contrastive_on_policy_distillation.pdf): the
        # instruction tokens are silently prepended to every scored sequence
        # so the teacher's log-probs are conditioned on a reasoning-effort
        # instruction the student never saw. Two modes:
        #   * light only (§3.3.5 single-context): score once under the light
        #     instruction; score_tokens returns teacher log-probs as usual.
        #   * light + heavy (§2.2 paired scoring): score TWICE per call and
        #     return the per-token contrast clip(l_LT - l_HT, ±contrast_clip)
        #     (§2.3, paper uses clip 10). The slime loss consumes it directly
        #     when officeqa_opd_contrast is set.
        # Either way the caller-visible contract of score_tokens is unchanged
        # — same window, same result length — so per-turn/per-sample alignment
        # code never needs to know. The constant prefixes are shared across
        # all requests, so the server's radix cache keeps hitting.
        self._instruction_ids = list(instruction_token_ids or [])
        self._heavy_ids = list(heavy_instruction_token_ids or [])
        self._contrast_clip = float(contrast_clip)
        if bos_token_id is not None and int(bos_token_id) < 0:
            raise ValueError(f"bos_token_id must be non-negative, got {bos_token_id}")
        # Some serving stacks do not add a model-required BOS. Keep this
        # opt-in so existing Qwen requests remain byte-for-byte unchanged.
        # Request assembly puts BOS before any teacher-only instruction.
        self._leading_bos_token_id = (
            int(bos_token_id) if bos_token_id is not None else None
        )
        self._timeout_sec = float(timeout_sec)
        self._max_attempts = int(max_attempts)
        self._backoff_base_sec = float(backoff_base_sec)
        # Concurrency protects each server, so the cap is per endpoint:
        # total in-flight = max_concurrent * n_endpoints.
        self._semaphores = [
            asyncio.Semaphore(int(max_concurrent)) for _ in self._urls
        ]
        self._refresh_lock = asyncio.Lock()
        self._creds: Any = None
        self._session: aiohttp.ClientSession | None = None

    @property
    def instruction_prefix_len(self) -> int:
        """Number of light-instruction tokens prepended per call (0 = off)."""
        return len(self._instruction_ids)

    @property
    def heavy_prefix_len(self) -> int:
        """Number of heavy-instruction tokens (0 = single-context mode)."""
        return len(self._heavy_ids)

    @property
    def contrast_enabled(self) -> bool:
        """True when score_tokens returns COPD light-minus-heavy contrasts."""
        return bool(self._heavy_ids)

    @property
    def bos_token_id(self) -> int | None:
        """Optional token forced to the very beginning of every request."""
        return self._leading_bos_token_id

    @property
    def resolved_backend(self) -> str | None:
        """Detected scoring protocol, or ``None`` before an auto probe succeeds."""
        return self._resolved_backend

    @property
    def endpoint_urls(self) -> tuple[str, ...]:
        """Every configured teacher endpoint URL (>=1)."""
        return tuple(self._urls)

    def _endpoint_index(self, affinity_key: str | None) -> int:
        """Sticky round-robin: an episode keeps its endpoint for cache reuse."""
        if len(self._urls) == 1:
            return 0
        if affinity_key is not None:
            index = self._affinity.get(affinity_key)
            if index is not None:
                return index
        index = self._rr_next
        self._rr_next = (self._rr_next + 1) % len(self._urls)
        if affinity_key is not None:
            # Entries outlive their episodes; the map only grows by one small
            # entry per episode over a whole run, but cap it defensively.
            if len(self._affinity) >= 100_000:
                self._affinity.clear()
            self._affinity[affinity_key] = index
        return index

    async def score_tokens(
        self,
        input_ids: list[int],
        *,
        logprob_start_len: int = 0,
        affinity_key: str | None = None,
        endpoint_index: int | None = None,
        top_logprobs: int = 0,
    ) -> list[float] | list[ScoredToken]:
        """Return the teacher's log-probs for ``input_ids[logprob_start_len + 1:]``.

        ``affinity_key`` (typically the episode's run_id) pins all of an
        episode's calls to one endpoint when several are configured, keeping
        that endpoint's prefix cache warm; ``endpoint_index`` overrides the
        selection explicitly (preflight/diagnostics). With a single endpoint
        both are no-ops.

        The first token of a causal-LM scoring window is unscored, so the
        result holds ``len(input_ids) - logprob_start_len - 1`` values.
        SGLang applies ``logprob_start_len`` server-side. vLLM scores the full
        prompt and the client slices the same window from its echoed prompt
        log-probs.

        In contrast mode (a heavy instruction is configured) each value is
        instead the clipped COPD contrast ``clip(l_LT - l_HT, ±contrast_clip)``
        for the same token — same window, same count, two teacher passes.

        ``top_logprobs=k`` (analysis paths only) additionally requests the
        server's top-k log-probs per scored position and changes the return
        type to ``list[ScoredToken]``. Training never sets it, and it cannot
        be combined with contrast mode.
        """
        if top_logprobs and self._heavy_ids:
            raise ValueError("top_logprobs is not supported in COPD contrast mode")
        if not 0 <= logprob_start_len < len(input_ids):
            raise ValueError(
                f"logprob_start_len={logprob_start_len} out of range for "
                f"{len(input_ids)} input tokens"
            )
        expected = len(input_ids) - logprob_start_len - 1
        url_index = (
            endpoint_index
            if endpoint_index is not None
            else self._endpoint_index(affinity_key)
        )
        if not 0 <= url_index < len(self._urls):
            raise ValueError(
                f"endpoint_index={url_index} out of range for "
                f"{len(self._urls)} endpoints"
            )
        if not self._heavy_ids:
            return await self._score_with_prefix(
                self._instruction_ids,
                input_ids,
                logprob_start_len,
                expected,
                url_index,
                top_logprobs=top_logprobs,
            )
        # COPD paired scoring (§2.2): the same window under both instructions,
        # concurrently (each pass retries independently; one terminal failure
        # fails the call — same fatal semantics as single-context scoring).
        # Returned values are the clipped per-token contrast A_t (§2.3), which
        # the slime loss adds to advantages when officeqa_opd_contrast is set.
        # Both passes share the episode's endpoint so both instructed contexts
        # stay cached there across the episode's turns.
        light, heavy = await asyncio.gather(
            self._score_with_prefix(
                self._instruction_ids, input_ids, logprob_start_len, expected, url_index
            ),
            self._score_with_prefix(
                self._heavy_ids, input_ids, logprob_start_len, expected, url_index
            ),
        )
        clip = self._contrast_clip
        return [max(-clip, min(clip, lt - ht)) for lt, ht in zip(light, heavy)]

    async def _score_with_prefix(
        self,
        prefix_ids: list[int],
        input_ids: list[int],
        logprob_start_len: int,
        expected: int,
        url_index: int,
        top_logprobs: int = 0,
    ) -> list[float] | list[ScoredToken]:
        # Instruction conditioning and an optional model-required BOS are
        # prepended while shifting the window by the same amount, so exactly
        # the caller's tokens are scored under the added context. If an older
        # caller already supplied BOS, move it ahead of the instruction rather
        # than sending either ``instruction + BOS`` or two BOS tokens.
        sent_ids, start_shift = self._with_context_prefix(prefix_ids, input_ids)
        sent_start = logprob_start_len + start_shift
        backend = self._resolved_backend
        if backend is not None:
            return await self._score_protocol(
                backend, sent_ids, sent_start, expected, url_index, top_logprobs
            )

        # Vertex rawPredict URLs do not reveal the serving container's route.
        # Probe the historical SGLang format once, then fall back only for a
        # clear schema mismatch. The lock prevents a rollout burst from
        # issuing the same failed probe on every concurrent episode.
        async with self._protocol_lock:
            if self._resolved_backend is not None:
                return await self._score_protocol(
                    self._resolved_backend,
                    sent_ids,
                    sent_start,
                    expected,
                    url_index,
                    top_logprobs,
                )
            try:
                scores = await self._score_protocol(
                    "sglang", sent_ids, sent_start, expected, url_index, top_logprobs
                )
            except (_ProtocolHTTPError, _ProtocolMismatchError) as sglang_error:
                logger.info(
                    "OPD teacher did not accept the SGLang scoring protocol; "
                    "trying vLLM once: %s",
                    sglang_error,
                )
                try:
                    scores = await self._score_protocol(
                        "vllm", sent_ids, sent_start, expected, url_index, top_logprobs
                    )
                except Exception as vllm_error:
                    raise TeacherScoringError(
                        "OPD teacher protocol auto-detection failed: "
                        f"SGLang={sglang_error}; vLLM={vllm_error}"
                    ) from vllm_error
                self._resolved_backend = "vllm"
                return scores
            self._resolved_backend = "sglang"
            return scores

    def _with_context_prefix(
        self, prefix_ids: list[int], input_ids: list[int]
    ) -> tuple[list[int], int]:
        """Return ``BOS + teacher prefix + caller ids`` and window shift.

        The shift maps a caller-relative ``logprob_start_len`` into the sent
        sequence. With no configured BOS this is the historical
        ``prefix + input`` behavior. A caller-supplied leading BOS is reused
        and relocated ahead of the teacher-only prefix.
        """
        prefix = list(prefix_ids)
        inputs = list(input_ids)
        bos = self._leading_bos_token_id
        if bos is None:
            return prefix + inputs, len(prefix)

        # Be forgiving if either tokenizer/caller already emitted BOS.
        if prefix and prefix[0] == bos:
            prefix = prefix[1:]
        input_had_bos = bool(inputs and inputs[0] == bos)
        if input_had_bos:
            inputs = inputs[1:]
        sent_ids = [bos, *prefix, *inputs]
        # Reusing the caller's BOS does not grow its sequence; otherwise the
        # inserted BOS shifts every caller-relative position by one.
        shift = len(prefix) + (0 if input_had_bos else 1)
        return sent_ids, shift

    async def _score_protocol(
        self,
        backend: str,
        input_ids: list[int],
        logprob_start_len: int,
        expected: int,
        url_index: int,
        top_logprobs: int = 0,
    ) -> list[float] | list[ScoredToken]:
        payload = self._scoring_payload(backend, input_ids, logprob_start_len, top_logprobs)
        response = await self._request_json(payload, url_index)
        return _parse_scored_log_probs(
            response,
            backend=backend,
            input_ids=input_ids,
            logprob_start_len=logprob_start_len,
            expected=expected,
            top_logprobs=top_logprobs,
        )

    async def _request_json(self, payload: dict[str, Any], url_index: int) -> Any:
        """POST with the shared retry/backoff/auth machinery; parsed JSON out.
        Protocol errors (400/422) propagate immediately; retryable statuses
        and transport errors go through the exponential-backoff loop."""
        url = self._urls[url_index]
        force_refresh = False
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            if attempt:
                delay = min(_MAX_BACKOFF_SEC, self._backoff_base_sec * 2 ** (attempt - 1))
                await asyncio.sleep(delay * (0.5 + random.random()))
            try:
                token = await self._ensure_token(force=force_refresh, url=url)
                force_refresh = False
                async with self._semaphores[url_index]:
                    return await asyncio.wait_for(
                        self._post(payload, token, url), timeout=self._timeout_sec
                    )
            except _RetryableHTTPError as error:
                # 401 usually means the token lapsed mid-flight; refresh first.
                force_refresh = error.status == 401
                last_error = error
            except (asyncio.TimeoutError, aiohttp.ClientError) as error:
                last_error = error
            logger.warning(
                "OPD teacher scoring attempt %d/%d failed: %r",
                attempt + 1,
                self._max_attempts,
                last_error,
            )
        raise TeacherScoringError(
            f"teacher scoring failed after {self._max_attempts} attempts: {last_error!r}"
        ) from last_error

    async def generate_tokens(
        self,
        input_ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        affinity_key: str | None = None,
        endpoint_index: int | None = None,
    ) -> list[int]:
        """Generate a continuation of the exact token prefix with the teacher.

        Used by analysis (teacher-action consistency probes) and by the
        teacher-think scoring mode, which lets the teacher write its own
        reasoning before judging the student's command. SGLang protocol only:
        the completion echo cannot return generated token ids. Any configured
        light instruction is prepended, matching the scoring path; contrast
        mode is rejected. ``stop`` strings end generation early (the match is
        kept in the output — ``no_stop_trim`` — so callers can cut on it).
        Returns the generated token ids (the server stops at a stop string,
        its EOS, or the cap).
        """
        if self._heavy_ids:
            raise ValueError("generation is not supported in COPD contrast mode")
        backend = self._resolved_backend
        if backend is None:
            # Resolve the wire protocol with a minimal scoring probe.
            await self.score_tokens(
                list(input_ids[:2]) or [0, 0],
                affinity_key=affinity_key,
                endpoint_index=endpoint_index,
            )
            backend = self._resolved_backend
        if backend != "sglang":
            raise TeacherScoringError(
                "teacher generation requires the SGLang protocol; the resolved "
                f"backend is {backend!r}"
            )
        url_index = (
            endpoint_index
            if endpoint_index is not None
            else self._endpoint_index(affinity_key)
        )
        sent_ids, _ = self._with_context_prefix(self._instruction_ids, input_ids)
        payload = {
            "input_ids": sent_ids,
            "sampling_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": int(max_new_tokens),
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                "no_stop_trim": True,
                **({"stop": list(stop)} if stop else {}),
            },
            "return_logprob": True,
        }
        response = await self._request_json(payload, url_index)
        meta = response.get("meta_info") if isinstance(response, dict) else None
        pairs = meta.get("output_token_logprobs") if isinstance(meta, dict) else None
        if not isinstance(pairs, list):
            raise TeacherScoringError(
                "teacher generation response lacks meta_info.output_token_logprobs"
            )
        return [int(pair[1]) for pair in pairs]

    def _scoring_payload(
        self,
        backend: str,
        input_ids: list[int],
        logprob_start_len: int,
        top_logprobs: int = 0,
    ) -> dict[str, Any]:
        if backend == "sglang":
            payload: dict[str, Any] = {
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 0,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
                "logprob_start_len": logprob_start_len,
            }
            if top_logprobs:
                payload["top_logprobs_num"] = int(top_logprobs)
            return payload
        payload = {
            "prompt": input_ids,
            "temperature": 0,
            "max_tokens": 0,
            "echo": True,
            "logprobs": max(1, int(top_logprobs)),
        }
        if self.teacher_model:
            payload["model"] = self.teacher_model
        return payload

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _post(self, payload: dict[str, Any], token: str | None, url: str) -> Any:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        session = self._get_session()
        async with session.post(url, json=payload, headers=headers) as response:
            body = await response.read()
            if response.status != 200:
                snippet = body[:500].decode("utf-8", errors="replace")
                if response.status == 401 or response.status in _RETRYABLE_STATUSES:
                    raise _RetryableHTTPError(response.status, snippet)
                if response.status in {400, 422}:
                    raise _ProtocolHTTPError(response.status, snippet)
                raise TeacherScoringError(
                    f"teacher endpoint returned HTTP {response.status}: {snippet}"
                )
            return json.loads(body)

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # No session-level total timeout: the per-call wait_for owns it.
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        return self._session

    @staticmethod
    def _needs_auth(url: str) -> bool:
        return url.startswith("https://")

    def _token_stale(self) -> bool:
        creds = self._creds
        if creds is None or not creds.valid:
            return True
        expiry = getattr(creds, "expiry", None)  # naive UTC in google-auth
        if expiry is None:
            return False
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        return now + _TOKEN_REFRESH_MARGIN >= expiry

    async def _ensure_token(self, *, force: bool, url: str) -> str | None:
        if not self._needs_auth(url):
            return None
        if force or self._token_stale():
            async with self._refresh_lock:
                if force or self._token_stale():  # re-check inside the lock
                    await asyncio.to_thread(self._refresh_credentials)
        return self._creds.token

    def _refresh_credentials(self) -> None:
        # Blocking google-auth work; always runs in a worker thread.
        import google.auth
        import google.auth.transport.requests

        if self._creds is None:
            self._creds, _ = google.auth.default(scopes=[_SCOPE])
        self._creds.refresh(google.auth.transport.requests.Request())


def _backend_from_url(url: str) -> str | None:
    path = url.rstrip("/").lower()
    if path.endswith("/v1/completions"):
        return "vllm"
    if path.endswith("/generate"):
        return "sglang"
    return None


def _parse_scored_log_probs(
    response: Any,
    *,
    backend: str,
    input_ids: list[int],
    logprob_start_len: int,
    expected: int,
    top_logprobs: int = 0,
) -> list[float] | list[ScoredToken]:
    if backend == "vllm":
        return _parse_vllm_log_probs(
            response,
            input_ids=input_ids,
            logprob_start_len=logprob_start_len,
            expected=expected,
            top_logprobs=top_logprobs,
        )
    return _parse_sglang_log_probs(
        response, expected=expected, top_logprobs=top_logprobs
    )


def _parse_sglang_log_probs(
    response: Any, *, expected: int, top_logprobs: int = 0
) -> list[float] | list[ScoredToken]:
    meta = response.get("meta_info") if isinstance(response, dict) else None
    items = meta.get("input_token_logprobs") if isinstance(meta, dict) else None
    if not items:
        raise _ProtocolMismatchError(
            "teacher response lacks SGLang meta_info.input_token_logprobs"
        )
    # items[0] is the unscored first token of the logprob window; every later
    # entry is [log_prob, token_id, ...].
    log_probs = [float(item[0]) for item in items[1:]]
    if len(log_probs) != expected:
        raise TeacherScoringError(
            f"teacher scored {len(log_probs)} tokens but {expected} were expected; "
            "the token stream or logprob_start_len does not match the server's view"
        )
    if not top_logprobs:
        return log_probs
    # input_top_logprobs mirrors input_token_logprobs position for position;
    # each entry is a descending list of [log_prob, token_id, ...] rows (or
    # None where the server has nothing, e.g. the unscored first token). A
    # server without top-k support simply yields ScoredTokens with None tops.
    tops = meta.get("input_top_logprobs")
    if not isinstance(tops, list) or len(tops) != len(items):
        tops = [None] * len(items)
    scored: list[ScoredToken] = []
    for log_prob, top in zip(log_probs, tops[1:], strict=True):
        if top:
            scored.append(
                ScoredToken(
                    log_prob=log_prob,
                    top_log_probs=[float(row[0]) for row in top],
                    top_token_ids=[int(row[1]) for row in top],
                )
            )
        else:
            scored.append(ScoredToken(log_prob=log_prob))
    return scored


def _parse_vllm_log_probs(
    response: Any,
    *,
    input_ids: list[int],
    logprob_start_len: int,
    expected: int,
    top_logprobs: int = 0,
) -> list[float] | list[ScoredToken]:
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise _ProtocolMismatchError("teacher response lacks vLLM choices[0]")
    choice = choices[0]
    logprobs = choice.get("logprobs")
    items = logprobs.get("token_logprobs") if isinstance(logprobs, dict) else None
    if not isinstance(items, list):
        raise _ProtocolMismatchError(
            "teacher response lacks vLLM choices[0].logprobs.token_logprobs"
        )
    returned_ids = choice.get("prompt_token_ids")
    if returned_ids is not None and returned_ids != input_ids:
        raise TeacherScoringError(
            "vLLM teacher returned prompt token IDs that differ from the exact "
            "training token sequence"
        )
    if len(items) < len(input_ids):
        raise TeacherScoringError(
            f"vLLM teacher returned {len(items)} prompt log-probs for "
            f"{len(input_ids)} input tokens"
        )
    window = items[logprob_start_len + 1 : len(input_ids)]
    if any(value is None for value in window):
        raise TeacherScoringError(
            "vLLM teacher returned an unscored token inside the requested window"
        )
    try:
        values = [float(value) for value in window]
    except (TypeError, ValueError) as error:
        raise TeacherScoringError(
            "vLLM teacher returned a non-numeric prompt log-prob"
        ) from error
    if len(values) != expected:
        raise TeacherScoringError(
            f"teacher scored {len(values)} tokens but {expected} were expected; "
            "the token stream or logprob_start_len does not match the server's view"
        )
    if not top_logprobs:
        return values
    # The completion echo keys top entries by token text, so only the top
    # log-prob values survive (descending); ids stay None.
    tops_all = logprobs.get("top_logprobs")
    if isinstance(tops_all, list) and len(tops_all) >= len(input_ids):
        window_tops = tops_all[logprob_start_len + 1 : len(input_ids)]
    else:
        window_tops = [None] * expected
    scored: list[ScoredToken] = []
    for log_prob, top in zip(values, window_tops, strict=True):
        if isinstance(top, dict) and top:
            scored.append(
                ScoredToken(
                    log_prob=log_prob,
                    top_log_probs=sorted(
                        (float(value) for value in top.values()), reverse=True
                    ),
                )
            )
        else:
            scored.append(ScoredToken(log_prob=log_prob))
    return scored


def _teacher_url_from_args(args: Any) -> str:
    """Resolve the teacher URL(s); comma-separated values configure the
    round-robin pool (officeqa_opd_teacher_url with full URLs for arbitrary
    mixes, or officeqa_opd_endpoint_id with several ids sharing one
    project/location)."""
    url = getattr(args, "officeqa_opd_teacher_url", None)
    if url:
        return url
    project = getattr(args, "officeqa_opd_project", None)
    location = getattr(args, "officeqa_opd_location", None)
    endpoint_id = getattr(args, "officeqa_opd_endpoint_id", None)
    if not (project and location and endpoint_id):
        raise ValueError(
            "OPD teacher scoring needs either officeqa_opd_teacher_url or all of "
            "officeqa_opd_project / officeqa_opd_location / officeqa_opd_endpoint_id; "
            "set them in the --custom-config-path YAML"
        )
    return ",".join(
        rawpredict_url(project, location, single.strip())
        for single in str(endpoint_id).split(",")
        if single.strip()
    )


# The instruction is wrapped as a standalone system block in Qwen chat markup
# so the teacher reads it as a directive rather than stray text before
# <|im_start|>. Configs that need full control can embed their own markers —
# any text already containing <|im_start|> is tokenized verbatim.
_INSTRUCTION_BLOCK = "<|im_start|>system\n{}<|im_end|>\n"


def _render_instruction(text: str) -> str:
    return text if "<|im_start|>" in text else _INSTRUCTION_BLOCK.format(text)


def _instruction_token_ids(args: Any, text: str) -> list[int]:
    """Tokenize the low-resource instruction with the run's own tokenizer
    (student and teacher share the Qwen3.5 vocabulary — required anyway for
    token-id scoring to make sense)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    return list(tokenizer(_render_instruction(text), add_special_tokens=False)["input_ids"])


_CLIENTS: dict[tuple, VertexTeacherClient] = {}


def get_teacher_client(args: Any) -> VertexTeacherClient:
    """One client per rollout worker and config, so all concurrent episodes
    share the ADC credential, connection pool, and concurrency semaphore."""
    config = (
        _teacher_url_from_args(args),
        str(getattr(args, "officeqa_opd_teacher_backend", "auto") or "auto"),
        str(getattr(args, "officeqa_opd_teacher_model", "") or ""),
        int(getattr(args, "officeqa_opd_concurrency", 16)),
        float(getattr(args, "officeqa_opd_timeout_sec", 180.0)),
        int(getattr(args, "officeqa_opd_attempts", 5)),
        float(getattr(args, "officeqa_opd_backoff_sec", 1.0)),
        str(getattr(args, "officeqa_opd_instruction", "") or ""),
        str(getattr(args, "officeqa_opd_instruction_high", "") or ""),
        float(getattr(args, "officeqa_opd_contrast_clip", 10.0)),
        (
            int(getattr(args, "officeqa_opd_bos_token_id"))
            if getattr(args, "officeqa_opd_bos_token_id", None) not in (None, "")
            else None
        ),
    )
    client = _CLIENTS.get(config)
    if client is None:
        (
            teacher_url,
            teacher_backend,
            teacher_model,
            max_concurrent,
            timeout_sec,
            max_attempts,
            backoff_base_sec,
            instruction,
            instruction_high,
            contrast_clip,
            bos_token_id,
        ) = config
        client = VertexTeacherClient(
            teacher_url=teacher_url,
            backend=teacher_backend,
            model=teacher_model or None,
            max_concurrent=max_concurrent,
            timeout_sec=timeout_sec,
            max_attempts=max_attempts,
            backoff_base_sec=backoff_base_sec,
            instruction_token_ids=(
                _instruction_token_ids(args, instruction) if instruction else None
            ),
            heavy_instruction_token_ids=(
                _instruction_token_ids(args, instruction_high) if instruction_high else None
            ),
            contrast_clip=contrast_clip,
            bos_token_id=bos_token_id,
        )
        _CLIENTS[config] = client
    return client


__all__ = [
    "ScoredToken",
    "TeacherScoringError",
    "VertexTeacherClient",
    "get_teacher_client",
    "rawpredict_url",
]
