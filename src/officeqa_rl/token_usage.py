"""Backend-independent token accounting for one agent episode.

Cost-per-task comparisons need the three classes that providers bill
separately: fresh input tokens, cached input tokens, and output tokens. Every
backend reports them under its own names, so this module normalizes them into
one schema:

``prompt_tokens``
    Total context sent to the model, summed over turns. In an agent loop the
    whole transcript is resent each turn, so this grows roughly quadratically
    in the number of turns and usually dominates the bill.
``cached_input_tokens``
    The part of ``prompt_tokens`` served from a prefix/context cache (vLLM
    prefix caching, OpenAI ``cached_tokens``, Gemini ``cached_content``).
``input_tokens``
    ``prompt_tokens - cached_input_tokens``: the part billed at the full input
    rate.
``output_tokens``
    Generated tokens, including reasoning tokens when the provider bills them
    as output.
``reasoning_tokens``
    The reasoning subset of ``output_tokens`` where the provider reports it.

``source`` records where the numbers came from: ``provider`` when the backend
returned a usage payload, ``token_ids`` when they were counted from the token
ids on the turn, and ``mixed`` when both contributed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from officeqa_rl.types import GeneratedTurn

_PROMPT_KEYS = ("prompt_tokens", "prompt_token_count", "input_tokens")
_OUTPUT_KEYS = (
    "completion_tokens",
    "candidates_token_count",
    "output_tokens",
    "output_token_count",
)
_CACHED_KEYS = (
    "cached_tokens",
    "cached_content_token_count",
    "cache_read_input_tokens",
    "prompt_cache_hit_tokens",
    "num_cached_tokens",
)
_REASONING_KEYS = (
    "reasoning_tokens",
    "thoughts_token_count",
    "reasoning_token_count",
)
_TOTAL_KEYS = ("total_tokens", "total_token_count")


def _first_int(usage: Mapping[str, Any], keys: Iterable[str]) -> int | None:
    """Find the first key present anywhere in a possibly nested usage payload."""
    for key in keys:
        value = usage.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    for value in usage.values():
        if isinstance(value, Mapping):
            nested = _first_int(value, keys)
            if nested is not None:
                return nested
    return None


def turn_usage(turn: GeneratedTurn) -> dict[str, int | str]:
    """Normalize one turn's usage, falling back to token-id lengths."""
    usage = turn.usage if isinstance(turn.usage, Mapping) else {}
    prompt = _first_int(usage, _PROMPT_KEYS)
    output = _first_int(usage, _OUTPUT_KEYS)
    cached = _first_int(usage, _CACHED_KEYS)
    reasoning = _first_int(usage, _REASONING_KEYS)

    from_ids = False
    if prompt is None and turn.prompt_ids is not None:
        prompt, from_ids = len(turn.prompt_ids), True
    if output is None and turn.output_ids is not None:
        output, from_ids = len(turn.output_ids), True
    if output is None and prompt is not None:
        total = _first_int(usage, _TOTAL_KEYS)
        if total is not None:
            output = max(total - prompt, 0)

    prompt = max(prompt or 0, 0)
    output = max(output or 0, 0)
    # A provider that reports no cache is not the same as a full cache miss,
    # but for accounting both mean "billed at the full input rate".
    cached = min(max(cached or 0, 0), prompt)
    if usage and from_ids:
        source = "mixed"
    elif usage:
        source = "provider"
    elif from_ids:
        source = "token_ids"
    else:
        source = "none"
    return {
        "prompt_tokens": prompt,
        "cached_input_tokens": cached,
        "input_tokens": prompt - cached,
        "output_tokens": output,
        "reasoning_tokens": max(reasoning or 0, 0),
        "source": source,
    }


def episode_usage(turns: Iterable[GeneratedTurn]) -> dict[str, Any]:
    """Sum per-turn usage over one episode.

    ``final_prompt_tokens`` is the context length of the last model call, which
    bounds how close the episode came to the context window.
    """
    totals = {
        "prompt_tokens": 0,
        "cached_input_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }
    sources: set[str] = set()
    final_prompt = 0
    calls = 0
    for turn in turns:
        counted = turn_usage(turn)
        calls += 1
        for key in totals:
            totals[key] += int(counted[key])
        final_prompt = int(counted["prompt_tokens"])
        sources.add(str(counted["source"]))
    sources.discard("none")
    return {
        **totals,
        "model_calls": calls,
        "final_prompt_tokens": final_prompt,
        "source": sources.pop() if len(sources) == 1 else ("mixed" if sources else "none"),
    }


def add_usage(total: dict[str, Any] | None, episode: Mapping[str, Any]) -> dict[str, Any]:
    """Accumulate one episode's usage into a running total."""
    running = dict(total or {})
    for key, value in episode.items():
        if key in {"source", "final_prompt_tokens"} or not isinstance(value, int):
            continue
        running[key] = running.get(key, 0) + value
    running["episodes"] = running.get("episodes", 0) + 1
    running["max_final_prompt_tokens"] = max(
        running.get("max_final_prompt_tokens", 0),
        int(episode.get("final_prompt_tokens", 0)),
    )
    return running


def per_task_means(total: Mapping[str, Any], tasks: int) -> dict[str, float]:
    """Per-task averages, the form used for cost-per-task comparisons."""
    if tasks <= 0:
        return {}
    return {
        f"{key}_per_task": round(value / tasks, 1)
        for key, value in total.items()
        if isinstance(value, int) and key not in {"episodes", "max_final_prompt_tokens"}
    }


__all__ = ["add_usage", "episode_usage", "per_task_means", "turn_usage"]
