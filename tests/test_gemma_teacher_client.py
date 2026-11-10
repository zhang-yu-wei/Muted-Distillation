"""Wire contract of the Gemma prompt_logprobs teacher client.

The values here are the live fleet's own response shape (probed 2026-08-28),
so a change in the deployment's protocol shows up as a test failure rather
than as silently wrong KL during training.
"""

from __future__ import annotations

import pytest

from officeqa_rl.training.gemma_teacher_client import (
    GEMMA_BOS_TOKEN_ID,
    MAX_PROMPT_LOGPROBS,
    GemmaVertexTeacherClient,
    _parse_prompt_logprobs,
)
from officeqa_rl.training.vertex_teacher_client import (
    TeacherScoringError,
    _ProtocolMismatchError,
)

# "The capital of France is" -> these are the real ids and the real logprobs
# the fleet returned for them.
IDS = [818, 5279, 529, 7001, 563]


def _entry(token_id: int, logprob: float, extra: dict | None = None) -> dict:
    entry = {str(token_id): {"logprob": logprob, "rank": 1, "decoded_token": "x"}}
    entry.update(extra or {})
    return entry


def _response(logprobs: list[float]) -> dict:
    # position 0 is null (nothing precedes BOS); the rest align to IDS.
    entries = [None] + [_entry(tid, lp) for tid, lp in zip(IDS, logprobs)]
    return {"choices": [{"prompt_logprobs": entries}]}


def test_payload_prepends_bos_and_uses_generating_max_tokens():
    client = GemmaVertexTeacherClient.__new__(GemmaVertexTeacherClient)
    client._bos_token_id = GEMMA_BOS_TOKEN_ID
    client.teacher_model = "google/gemma-4-31B-it"

    payload = client._scoring_payload("gemma", IDS, 0)
    assert payload["prompt"] == [GEMMA_BOS_TOKEN_ID, *IDS], "BOS must lead every call"
    assert payload["max_tokens"] == 1, "max_tokens=0 is rejected by this route"
    assert payload["temperature"] == 0
    assert payload["model"] == "google/gemma-4-31B-it"
    assert "echo" not in payload and "logprobs" not in payload


def test_payload_rejects_top_logprobs_above_the_served_cap():
    client = GemmaVertexTeacherClient.__new__(GemmaVertexTeacherClient)
    client._bos_token_id = GEMMA_BOS_TOKEN_ID
    client.teacher_model = None
    with pytest.raises(ValueError, match="max-logprobs"):
        client._scoring_payload("gemma", IDS, 0, top_logprobs=MAX_PROMPT_LOGPROBS + 1)


def test_parse_returns_the_window_after_logprob_start_len():
    lps = [-9.96, -13.92, -0.63, -0.03, -0.51]
    scores = _parse_prompt_logprobs(
        _response(lps), input_ids=IDS, logprob_start_len=0, expected=len(IDS) - 1
    )
    # The first token of a causal window is unscored, and the BOS offset must
    # not shift the alignment: token i is read from entry i+1.
    assert scores == pytest.approx(lps[1:])


def test_parse_honours_a_nonzero_start():
    lps = [-9.96, -13.92, -0.63, -0.03, -0.51]
    scores = _parse_prompt_logprobs(
        _response(lps), input_ids=IDS, logprob_start_len=2, expected=len(IDS) - 3
    )
    assert scores == pytest.approx(lps[3:])


def test_parse_collects_top_alternatives_in_descending_order():
    entries = [None] + [_entry(tid, -1.0) for tid in IDS]
    entries[2] = _entry(IDS[1], -13.92, {"999": {"logprob": -0.14, "rank": 1}})
    scored = _parse_prompt_logprobs(
        {"choices": [{"prompt_logprobs": entries}]},
        input_ids=IDS,
        logprob_start_len=0,
        expected=len(IDS) - 1,
        top_logprobs=2,
    )
    first = scored[0]
    assert first.log_prob == pytest.approx(-13.92)
    assert first.top_log_probs == pytest.approx([-0.14, -13.92])
    assert first.top_token_ids == [999, IDS[1]]


def test_parse_rejects_a_length_mismatch():
    # A response one entry short means the server did not see our BOS.
    entries = [_entry(tid, -1.0) for tid in IDS]
    with pytest.raises(_ProtocolMismatchError, match="entries for"):
        _parse_prompt_logprobs(
            {"choices": [{"prompt_logprobs": entries}]},
            input_ids=IDS,
            logprob_start_len=0,
            expected=len(IDS) - 1,
        )


def test_parse_rejects_a_response_from_another_protocol():
    with pytest.raises(_ProtocolMismatchError, match="prompt_logprobs"):
        _parse_prompt_logprobs(
            {"choices": [{"logprobs": {"token_logprobs": [None, -1.0]}}]},
            input_ids=IDS,
            logprob_start_len=0,
            expected=len(IDS) - 1,
        )


def test_parse_rejects_an_entry_missing_the_scored_token():
    entries = [None] + [_entry(tid, -1.0) for tid in IDS]
    entries[3] = {"424242": {"logprob": -0.1}}  # some other token only
    with pytest.raises(TeacherScoringError, match="no logprob for id"):
        _parse_prompt_logprobs(
            {"choices": [{"prompt_logprobs": entries}]},
            input_ids=IDS,
            logprob_start_len=0,
            expected=len(IDS) - 1,
        )
