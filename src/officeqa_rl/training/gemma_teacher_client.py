"""OPD teacher client for the Gemma scoring fleet (vLLM ``/v1/completions``).

The Gemma scoring endpoints speak a third scoring protocol, distinct from the
two :class:`~officeqa_rl.training.vertex_teacher_client.VertexTeacherClient`
already auto-detects:

* SGLang ``/generate``: ``input_ids`` + ``return_logprob`` ->
  ``meta_info.input_token_logprobs``;
* vLLM completion echo: ``echo: true``, ``logprobs: 1``, ``max_tokens: 0`` ->
  ``choices[0].logprobs.token_logprobs``;
* **this module**: ``prompt_logprobs: K``, ``max_tokens: 1`` ->
  ``choices[0].prompt_logprobs``.

Three things make it worth a dedicated class rather than a third branch in the
base client's auto-detection, all verified against the live fleet on
2026-08-28:

1. **A BOS token must be prepended on every call.** The deployment does not add
   it, and neither does the tokenizer: ``AutoTokenizer.encode`` on Gemma
   returns no BOS even with ``add_special_tokens=True``, and the server's
   ``add_bos_token`` config is ignored by vLLM's ``/v1/completions`` route.
   Without it the model degenerates and the log-probs are meaningless -- on
   ``"The capital of France is"`` the token ``" capital"`` scores -13.92
   (rank 1032) with BOS and -20.16 (rank 34965) without, and the top-ranked
   continuation becomes an unrelated CJK token. This is silent: the response is
   well-formed either way, so nothing downstream would notice.
2. **``max_tokens`` must be >= 1.** The echo protocol's ``max_tokens: 0``
   trick is rejected outright, so a scoring call always generates (and
   discards) one token.
3. **The scored value lives under the token's own id**, in a per-position dict
   that also carries the top-K alternatives:
   ``prompt_logprobs[i][str(token_id)]["logprob"]``, with ``null`` at
   position 0.

Token ids are sent rather than text: the fleet accepts an integer ``prompt``
and returns log-probs identical to the equivalent ``"<bos>..."`` string, so
scoring the student's exact ids costs nothing and avoids any
detokenize/retokenize drift.
"""

from __future__ import annotations

import logging
from typing import Any

from officeqa_rl.training.vertex_teacher_client import (
    ScoredToken,
    TeacherScoringError,
    VertexTeacherClient,
    _ProtocolMismatchError,
)

logger = logging.getLogger(__name__)

#: Gemma's ``<bos>``. Asserted against the tokenizer in the client's
#: constructor when one is supplied, so a vocabulary change cannot silently
#: reintroduce the degenerate-scoring bug this constant exists to prevent.
GEMMA_BOS_TOKEN_ID = 2

#: The fleet is served with ``--max-logprobs=64``; larger K is rejected.
MAX_PROMPT_LOGPROBS = 64


class GemmaVertexTeacherClient(VertexTeacherClient):
    """Score a student's tokens with the Gemma vLLM ``prompt_logprobs`` fleet.

    Drop-in for :class:`VertexTeacherClient`: same constructor, same
    ``score_tokens`` contract (log-probs for ``input_ids[start + 1:]``), same
    retry/auth/affinity machinery. Only the wire payload and the response
    parsing differ, and protocol auto-detection is disabled -- this deployment
    answers neither of the base client's two probes, so probing would waste a
    round trip per worker and could resolve wrongly.
    """

    def __init__(self, *args: Any, bos_token_id: int = GEMMA_BOS_TOKEN_ID, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._bos_token_id = int(bos_token_id)
        # Never let the base class probe: it would try SGLang, then the echo
        # protocol, and both fail on this route.
        self._resolved_backend = "gemma-prompt-logprobs"

    # -- wire format ---------------------------------------------------------

    def _scoring_payload(
        self,
        backend: str,
        input_ids: list[int],
        logprob_start_len: int,
        top_logprobs: int = 0,
    ) -> dict[str, Any]:
        if top_logprobs > MAX_PROMPT_LOGPROBS:
            raise ValueError(
                f"top_logprobs={top_logprobs} exceeds the fleet's "
                f"--max-logprobs={MAX_PROMPT_LOGPROBS}"
            )
        payload: dict[str, Any] = {
            # BOS is prepended here, on every call, and stripped back out when
            # the response is parsed. See the module docstring for why.
            "prompt": [self._bos_token_id, *input_ids],
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": max(0, int(top_logprobs)),
        }
        if self.teacher_model:
            payload["model"] = self.teacher_model
        return payload

    # -- response parsing ----------------------------------------------------

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
        return _parse_prompt_logprobs(
            response,
            input_ids=input_ids,
            logprob_start_len=logprob_start_len,
            expected=expected,
            top_logprobs=top_logprobs,
        )


def _parse_prompt_logprobs(
    response: Any,
    *,
    input_ids: list[int],
    logprob_start_len: int,
    expected: int,
    top_logprobs: int = 0,
) -> list[float] | list[ScoredToken]:
    """Pull the scored window out of ``choices[0].prompt_logprobs``.

    The request prepended BOS, so entry ``i + 1`` scores ``input_ids[i]``;
    entry 0 is ``null`` (nothing precedes BOS).
    """
    try:
        choices = response["choices"]
        entries = choices[0]["prompt_logprobs"]
    except (KeyError, IndexError, TypeError) as error:
        raise _ProtocolMismatchError(
            "Gemma teacher response lacks choices[0].prompt_logprobs "
            f"(got keys {sorted(response) if isinstance(response, dict) else type(response)})"
        ) from error

    if not isinstance(entries, list):
        raise _ProtocolMismatchError(
            f"prompt_logprobs is {type(entries).__name__}, expected a list"
        )
    # One entry per sent token, and BOS was prepended to the caller's ids.
    if len(entries) != len(input_ids) + 1:
        raise _ProtocolMismatchError(
            f"prompt_logprobs has {len(entries)} entries for "
            f"{len(input_ids) + 1} sent tokens (BOS + {len(input_ids)})"
        )

    scores: list[float] | list[ScoredToken] = []
    for position in range(logprob_start_len + 1, len(input_ids)):
        token_id = input_ids[position]
        entry = entries[position + 1]  # +1 for the prepended BOS
        if not isinstance(entry, dict):
            raise TeacherScoringError(
                f"prompt_logprobs entry at token {position} is {entry!r}, expected a dict"
            )
        own = entry.get(str(token_id))
        if own is None or "logprob" not in own:
            # vLLM always includes the sampled token's own entry; its absence
            # means the response is not aligned with the ids we sent.
            raise TeacherScoringError(
                f"prompt_logprobs entry at token {position} has no logprob for "
                f"id {token_id} (keys: {sorted(entry)[:8]})"
            )
        log_prob = float(own["logprob"])
        if not top_logprobs:
            scores.append(log_prob)  # type: ignore[arg-type]
            continue
        ranked = sorted(
            ((int(tid), float(v["logprob"])) for tid, v in entry.items() if "logprob" in v),
            key=lambda pair: pair[1],
            reverse=True,
        )
        scores.append(  # type: ignore[arg-type]
            ScoredToken(
                log_prob=log_prob,
                top_log_probs=[lp for _, lp in ranked],
                top_token_ids=[tid for tid, _ in ranked],
            )
        )

    if len(scores) != expected:
        raise TeacherScoringError(
            f"Gemma teacher returned {len(scores)} scores, expected {expected}"
        )
    return scores


__all__ = [
    "GEMMA_BOS_TOKEN_ID",
    "MAX_PROMPT_LOGPROBS",
    "GemmaVertexTeacherClient",
]
