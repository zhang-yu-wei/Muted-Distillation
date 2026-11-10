"""Rubric judging and reward for JobBench episodes.

A port of the scoring half of JobBench's ``eval/judge.py``, kept faithful where
the score is concerned:

- one judge call per rubric item, all of that item's sub-criteria in the call,
- the item passes only if **every** sub-criterion passes,
- item score is its full weight or zero,
- the task score is ``sum(scores) / sum(weights)``, which is also the reward.

Differences from upstream, all operational rather than scoring:

- async (``openai.AsyncOpenAI``) with a per-task concurrency cap, because a
  rollout grades many episodes at once,
- an episode whose ``output/`` is empty scores zero without calling the judge,
- judge failures are counted on the result (``judge_failures``) instead of only
  being folded into the score as failed items, so a rollout can discard a
  sample the judge could not grade rather than train on a false zero.

The judge is an LLM, so this reward is neither stationary nor
hard to game: a deliverable that asserts the right sentences without doing the
work can score. Keep that in mind before reading too much into a training
curve.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import threading
import urllib.parse
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from officeqa_rl.jobbench.dataset import JobBenchTask
from officeqa_rl.jobbench.extraction import (
    DocumentReaderUnavailableError,
    collect_image_attachments,
    extract_all_file_contents,
    rubric_needs_vision,
)

logger = logging.getLogger("officeqa_rl.jobbench.judge")

DEFAULT_JUDGE_API_BASE = "https://api.x.ai/v1"

# A judge served through Vertex AI's OpenAI-compatible endpoint authenticates
# with short-lived ADC bearer tokens instead of a static key. Tokens expire
# after one hour; ``judge_episode`` builds a fresh client per attempt, so
# refreshing at client construction (with a margin) keeps every grading burst
# inside a valid token. Mirrors officeqa_rl.backends.vertex.
_ADC_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_ADC_REFRESH_MARGIN_SEC = 300.0
_adc_lock = threading.Lock()
_adc_creds = None


def is_vertex_judge_api_base(api_base: str) -> bool:
    host = urllib.parse.urlsplit(api_base).hostname or ""
    return host == "aiplatform.googleapis.com" or host.endswith(
        "-aiplatform.googleapis.com"
    )



# Retrying a 400 or a 404 just burns the budget and delays the episode; only
# transport, rate-limit and server-side faults can succeed on a second try.
_RETRYABLE_MARKERS = (
    "429", "500", "502", "503", "504",
    "rate limit", "ratelimit", "quota", "overloaded", "unavailable",
    "timeout", "timed out", "connection", "temporarily",
)
_PERMANENT_MARKERS = ("400", "401", "403", "404", "invalid_argument", "permission")
# Vertex's OpenAI-compatible endpoint has returned bare "400 INVALID_ARGUMENT"
# (no detail) in bursts at the start of an eval (63 episodes x N rubrics fired
# at once; 4 cases on 2026-09-14, different tasks and rubric items, seconds
# apart, never reproduced on the same item). A genuine bad request costs one
# extra call per retry below; a transient one saves the episode.
INVALID_ARGUMENT_RETRIES = 2
INVALID_ARGUMENT_RETRY_DELAY_SEC = 5.0


class JudgeResponseTruncatedError(ValueError):
    """The judge spent its completion budget before returning a verdict."""


def _is_retryable(error: Exception) -> bool:
    # Repeating an oversized rubric at the same budget can consume hours
    # without ever producing JSON. Preserve it as ungraded for an explicit
    # budget override, rather than mistaking truncation for a transport fault.
    if isinstance(error, JudgeResponseTruncatedError):
        return False
    text = f"{type(error).__name__}: {error}".lower()
    # The judge occasionally emits truncated or malformed JSON.  That is a
    # sampling failure, not a permanent request error, and a fresh call often
    # succeeds.  Treat parser failures explicitly before the generic marker
    # logic (their exception class contains the word ``error``).
    if isinstance(error, json.JSONDecodeError) or "no json object" in text:
        return True
    if any(m in text for m in _PERMANENT_MARKERS):
        # A permanent rejection, unless it also carries a transient signal.
        return any(m in text for m in ("429", "503", "overloaded", "rate limit"))
    return any(m in text for m in _RETRYABLE_MARKERS) or "error" not in text


def _is_invalid_argument(error: Exception) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return "invalid_argument" in text or "invalid argument" in text


def _request_shape(prompt: str, images: list[tuple[str, str]]) -> str:
    """One-line description of a judge request, for diagnosing 400s."""
    image_bytes = sum(len(url) for _, url in images)
    return (
        f"prompt_chars={len(prompt)} images={len(images)} image_url_chars={image_bytes}"
        + (f" image_names={[name for name, _ in images]}" if images else "")
    )


def _adc_token() -> str:
    global _adc_creds
    with _adc_lock:
        import google.auth
        import google.auth.transport.requests

        if _adc_creds is None:
            _adc_creds, _ = google.auth.default(scopes=[_ADC_SCOPE])
        expiry = getattr(_adc_creds, "expiry", None)  # naive UTC in google-auth
        stale = not _adc_creds.valid or (
            expiry is not None
            and (expiry - datetime.utcnow()).total_seconds() < _ADC_REFRESH_MARGIN_SEC
        )
        if stale:
            _adc_creds.refresh(google.auth.transport.requests.Request())
        return _adc_creds.token

JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge. You must return valid JSON only, with no "
    "markdown formatting or extra text."
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class JudgeConfig:
    """Where the judge runs and how hard it tries."""

    model: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", "grok-4.3"))
    api_base: str = field(
        default_factory=lambda: os.getenv("JUDGE_API_BASE", DEFAULT_JUDGE_API_BASE)
    )
    api_key: str = field(default_factory=lambda: os.getenv("JUDGE_API_KEY", ""))
    timeout_sec: float = 300.0
    # The judge runs on the same Vertex infrastructure as the OPD teacher,
    # which needs 18 attempts at 20s to ride out its 503s. Three attempts at
    # 2s was far too thin: a rubric that never grades still counts its weight
    # in the denominator, so every judge failure biases that attempt's score
    # DOWN rather than merely losing a data point.
    max_retries: int = 8
    retry_delay_sec: float = 4.0
    retry_max_delay_sec: float = 60.0
    max_concurrent_rubrics: int = 4
    max_completion_tokens: int = 8192
    temperature: float = 0.0
    attach_images: bool = True
    # Optional explicit effort for the first request; empty preserves the
    # provider default used by existing training/evaluation campaigns.
    reasoning_effort: str = ""
    # After a truncated verdict (finish_reason=length with the budget spent on
    # reasoning), retry the rubric ONCE with this reasoning_effort ("minimal",
    # "low", "medium", "high", "max" on the Vertex OpenAI-compatible endpoint;
    # empty = no fallback, the item stays ungraded). Gemini 3.5 Flash loops on
    # some rubric items at full effort regardless of budget (15.7k reasoning
    # tokens of 16k, 31.5k of 32k, same task); a capped effort answers.
    truncation_fallback_reasoning_effort: str = ""

    def require_key(self) -> str:
        if not self.api_key:
            if is_vertex_judge_api_base(self.api_base):
                return _adc_token()
            raise RuntimeError(
                "No judge API key. Set JUDGE_API_KEY (and JUDGE_API_BASE / "
                "JUDGE_MODEL for a non-default endpoint)."
            )
        return self.api_key


@dataclass
class RubricResult:
    """Outcome of one rubric item."""

    index: int
    rubric: str
    weight: int
    passed: bool
    score: int
    criteria_count: int
    criteria_passed: int
    criteria_results: list[dict[str, Any]] = field(default_factory=list)
    overall_reasoning: str = ""
    vision_used: bool = False
    error: str = ""


@dataclass
class JobBenchScore:
    """Weighted rubric score for one episode. ``normalized_score`` is the reward."""

    uid: str
    total_score: int
    max_score: int
    normalized_score: float
    pass_rate: float
    passed_count: int
    total_count: int
    judge_model: str
    judge_failures: int = 0
    n_output_files: int = 0
    reason: str = ""
    rubrics: list[RubricResult] = field(default_factory=list)

    @property
    def reward(self) -> float:
        return self.normalized_score

    @property
    def graded(self) -> bool:
        """False when the judge could not score every rubric item."""
        return self.judge_failures == 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reward"] = self.reward
        payload["graded"] = self.graded
        return payload


def _client(config: JudgeConfig):
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=config.require_key(),
        base_url=config.api_base,
        timeout=config.timeout_sec,
        max_retries=0,  # retries are handled here so failures can be counted
    )


def _load_judge_json(text: str) -> dict[str, Any]:
    # Judges sometimes put literal LaTeX/path backslashes in explanation
    # strings. Repair ONLY the invalid escape identified by the JSON parser;
    # never guess missing verdicts, truncate the reply, or alter valid escapes.
    for _ in range(33):
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            if error.msg != "Invalid \\escape" or text[error.pos] != "\\":
                raise
            text = text[:error.pos] + "\\" + text[error.pos:]
    raise ValueError("Judge JSON contains too many invalid string escapes")


def parse_judge_json(raw: str) -> dict[str, Any]:
    """Parse the judge reply, tolerating a fenced or prefixed JSON object."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return _load_judge_json(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return _load_judge_json(match.group(0))
    raise ValueError("Judge reply contained no JSON object")


def build_rubric_prompt(rubric: dict[str, Any], file_contents: str, *, vision_used: bool) -> str:
    """Upstream's per-rubric judge prompt, reproduced verbatim in structure."""
    criteria = rubric.get("criterion", [])
    if isinstance(criteria, str):
        criteria = [criteria]
    criteria_list_text = "\n".join(
        f"Criterion {index}: {criterion}" for index, criterion in enumerate(criteria)
    )
    image_clause = " and the attached images" if vision_used else ""
    return f"""You are an evaluation judge. Your task is to evaluate ALL criteria for a single rubric.

## Rubric Description
{rubric.get("rubric", "")}

## Criteria to Evaluate (Judge ALL of them)
{criteria_list_text}

## Output Files Content
The following are the contents of all output files to evaluate:

{file_contents}

## Evaluation Rules
- Evaluate EACH criterion listed above independently
- For each criterion: determine if it PASSES or FAILS
- Semantic matching is acceptable (you don't need exact wording match)
- Binary judgment for each criterion: PASS or FAIL only
- The rubric passes ONLY if ALL criteria pass

## Output Format
Return your judgment as a JSON object with EXACTLY this structure (no markdown, no extra text):
{{
  "criteria_results": [
    {{"index": 0, "passed": true/false, "reasoning": "...", "evidence": "..."}},
    {{"index": 1, "passed": true/false, "reasoning": "...", "evidence": "..."}}
  ],
  "rubric_passed": true/false,
  "overall_reasoning": "Summary of why the rubric passed or failed"
}}

IMPORTANT:
- criteria_results array must have exactly {len(criteria)} items (one for each criterion)
- rubric_passed should be true ONLY if ALL criteria passed
- Include specific evidence from the output files{image_clause}
"""


def _user_content(prompt: str, images: list[tuple[str, str]]) -> Any:
    if not images:
        return prompt
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    plural = "s" if len(images) != 1 else ""
    content.append(
        {"type": "text", "text": f"\n## Attached Images ({len(images)} file{plural})"}
    )
    for position, (name, url) in enumerate(images, start=1):
        content.append({"type": "text", "text": f"Image {position}: {name}"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


async def judge_rubric(
    client: Any,
    index: int,
    rubric: dict[str, Any],
    file_contents: str,
    config: JudgeConfig,
    *,
    image_attachments: list[tuple[str, str]] | None = None,
) -> RubricResult:
    """Judge one rubric item, retrying transport and parse failures."""
    criteria = rubric.get("criterion", [])
    if isinstance(criteria, str):
        criteria = [criteria]
    weight = int(rubric.get("weight", 0))
    images = (
        list(image_attachments)
        if image_attachments and config.attach_images and rubric_needs_vision(rubric)
        else []
    )
    prompt = build_rubric_prompt(rubric, file_contents, vision_used=bool(images))

    last_error: Exception | None = None
    # Upstream sends max_completion_tokens. Some OpenAI-compatible servers only
    # accept the older max_tokens and reject the request outright, which would
    # otherwise burn every retry on the same rejection.
    token_param = "max_completion_tokens"
    reasoning_effort: str | None = config.reasoning_effort or None
    invalid_argument_retries = 0
    for attempt in range(config.max_retries):
        try:
            # A rubric with long retries can outlive the token used when its
            # episode client was constructed. Refresh ADC before every request
            # (the credential helper caches still-valid tokens).
            if not config.api_key and is_vertex_judge_api_base(config.api_base):
                client.api_key = await asyncio.to_thread(config.require_key)
            response = await client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": _user_content(prompt, images)},
                ],
                temperature=config.temperature,
                timeout=config.timeout_sec,
                **{token_param: config.max_completion_tokens},
                **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
            )
            choice = response.choices[0]
            # A reply can carry no message at all (observed once on the
            # low-effort fallback retry: choices[0].message is None). That is
            # a sampling failure like malformed JSON -- "no JSON object" makes
            # it retryable -- not an AttributeError to be logged as permanent.
            message = getattr(choice, "message", None)
            content = (getattr(message, "content", None) or "") if message is not None else ""
            try:
                parsed = parse_judge_json(content)
            except ValueError as error:
                if getattr(choice, "finish_reason", None) == "length":
                    usage = getattr(response, "usage", None)
                    details = getattr(usage, "completion_tokens_details", None)
                    raise JudgeResponseTruncatedError(
                        "judge_response_truncated: finish_reason=length; "
                        f"max_completion_tokens={config.max_completion_tokens}; "
                        f"reasoning_tokens={getattr(details, 'reasoning_tokens', None)}; "
                        "no complete JSON verdict"
                    ) from error
                raise
            model_criteria = parsed.get("criteria_results", [])
            enriched = []
            for position, criterion in enumerate(criteria):
                item = (
                    model_criteria[position]
                    if position < len(model_criteria) and isinstance(model_criteria[position], dict)
                    else {}
                )
                enriched.append(
                    {
                        "index": position,
                        "criterion": criterion,
                        "passed": bool(item.get("passed", False)),
                        "reasoning": item.get("reasoning", ""),
                        "evidence": item.get("evidence", ""),
                    }
                )
            passed = bool(parsed.get("rubric_passed", False))
            return RubricResult(
                index=index,
                rubric=str(rubric.get("rubric", "")),
                weight=weight,
                passed=passed,
                score=weight if passed else 0,
                criteria_count=len(criteria),
                criteria_passed=sum(1 for item in enriched if item["passed"]),
                criteria_results=enriched,
                overall_reasoning=str(parsed.get("overall_reasoning", "")),
                vision_used=bool(images),
            )
        except Exception as error:  # transport, rate limit, or malformed JSON
            last_error = error
            if (
                not isinstance(error, JudgeResponseTruncatedError)
                and token_param == "max_completion_tokens"
                and "max_completion_tokens" in str(error)
            ):
                token_param = "max_tokens"
                continue  # not a real attempt: the request never reached the model
            if (
                isinstance(error, JudgeResponseTruncatedError)
                and config.truncation_fallback_reasoning_effort
                and reasoning_effort is None
            ):
                reasoning_effort = config.truncation_fallback_reasoning_effort
                logger.warning(
                    "Judge rubric %d truncated (%s); retrying once with reasoning_effort=%s",
                    index, error, reasoning_effort,
                )
                continue
            if not _is_retryable(error):
                if (
                    _is_invalid_argument(error)
                    and invalid_argument_retries < INVALID_ARGUMENT_RETRIES
                    and attempt < config.max_retries - 1
                ):
                    invalid_argument_retries += 1
                    logger.warning(
                        "Judge rubric %d got %s (%s); retry %d/%d after %.0fs",
                        index, error, _request_shape(prompt, images),
                        invalid_argument_retries, INVALID_ARGUMENT_RETRIES,
                        INVALID_ARGUMENT_RETRY_DELAY_SEC,
                    )
                    await asyncio.sleep(INVALID_ARGUMENT_RETRY_DELAY_SEC * (0.5 + random.random()))
                    continue
                logger.warning(
                    "Judge rubric %d failed permanently, not retrying: %s (%s)",
                    index, error, _request_shape(prompt, images),
                )
                break
            if attempt < config.max_retries - 1:
                # Exponential with jitter: many rubrics run concurrently and a
                # shared 503 would otherwise retry them in lockstep.
                delay = min(
                    config.retry_delay_sec * (2 ** attempt), config.retry_max_delay_sec
                )
                await asyncio.sleep(delay * (0.5 + random.random()))

    # Report attempts actually made, not the configured ceiling: a permanent
    # rejection breaks out after one try, and logging max_retries there reads
    # as exhaustion and sends the reader hunting a capacity problem.
    logger.warning(
        "Judge failed on rubric %d after %d attempt(s) of %d: %s",
        index, attempt + 1, config.max_retries, last_error,
    )
    return _failed_result(index, rubric, weight, criteria, f"judge_error: {last_error}")


def _failed_result(
    index: int, rubric: dict[str, Any], weight: int, criteria: list[str], reason: str
) -> RubricResult:
    return RubricResult(
        index=index,
        rubric=str(rubric.get("rubric", "")),
        weight=weight,
        passed=False,
        score=0,
        criteria_count=len(criteria),
        criteria_passed=0,
        criteria_results=[
            {
                "index": position,
                "criterion": criterion,
                "passed": False,
                "reasoning": reason,
                "evidence": "",
            }
            for position, criterion in enumerate(criteria)
        ],
        overall_reasoning=reason,
        error=reason,
    )


def build_score(
    uid: str, results: list[RubricResult], *, judge_model: str, n_output_files: int, reason: str = ""
) -> JobBenchScore:
    """Aggregate rubric outcomes exactly as upstream's scorecard does."""
    total_score = sum(result.score for result in results)
    max_score = sum(result.weight for result in results)
    passed_count = sum(1 for result in results if result.passed)
    total_count = len(results)
    return JobBenchScore(
        uid=uid,
        total_score=total_score,
        max_score=max_score,
        normalized_score=round(total_score / max_score, 4) if max_score > 0 else 0.0,
        pass_rate=round(passed_count / total_count, 4) if total_count > 0 else 0.0,
        passed_count=passed_count,
        total_count=total_count,
        judge_model=judge_model,
        judge_failures=sum(1 for result in results if result.error),
        n_output_files=n_output_files,
        reason=reason,
        rubrics=results,
    )


async def judge_episode(
    task: JobBenchTask,
    output_dir: Path,
    config: JudgeConfig,
    *,
    client: Any | None = None,
) -> JobBenchScore:
    """Score one episode's ``output/`` directory against the task's rubrics."""
    rubrics = task.rubric_dicts()
    output_files = (
        [path for path in output_dir.rglob("*") if path.is_file()]
        if output_dir.is_dir()
        else []
    )
    if not output_files:
        results = [
            _failed_result(
                index,
                rubric,
                int(rubric.get("weight", 0)),
                list(rubric.get("criterion", [])),
                "no_output_files",
            )
            for index, rubric in enumerate(rubrics)
        ]
        score = build_score(
            task.uid,
            results,
            judge_model=config.model,
            n_output_files=0,
            reason="no_output_files",
        )
        score.judge_failures = 0  # a genuine zero, not a grading failure
        return score

    try:
        file_contents = await asyncio.to_thread(extract_all_file_contents, output_dir)
    except DocumentReaderUnavailableError as error:
        reason = f"document_extraction_unavailable: {error}"
        return build_score(
            task.uid,
            [_failed_result(index, rubric, int(rubric.get("weight", 0)),
                            list(rubric.get("criterion", [])), reason)
             for index, rubric in enumerate(rubrics)],
            judge_model=config.model, n_output_files=len(output_files), reason=reason,
        )
    images: list[tuple[str, str]] = []
    if config.attach_images and any(rubric_needs_vision(rubric) for rubric in rubrics):
        images = await asyncio.to_thread(collect_image_attachments, output_dir)

    owns_client = client is None
    client = client or _client(config)
    semaphore = asyncio.Semaphore(max(1, config.max_concurrent_rubrics))

    async def judge_one(index: int, rubric: dict[str, Any]) -> RubricResult:
        async with semaphore:
            return await judge_rubric(
                client, index, rubric, file_contents, config, image_attachments=images
            )

    try:
        results = await asyncio.gather(
            *(judge_one(index, rubric) for index, rubric in enumerate(rubrics))
        )
    finally:
        if owns_client:
            await client.close()

    return build_score(
        task.uid,
        list(results),
        judge_model=config.model,
        n_output_files=len(output_files),
    )


__all__ = [
    "DEFAULT_JUDGE_API_BASE",
    "JobBenchScore",
    "JudgeConfig",
    "RubricResult",
    "build_rubric_prompt",
    "build_score",
    "judge_episode",
    "judge_rubric",
    "parse_judge_json",
]
