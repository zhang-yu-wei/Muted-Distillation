"""Exact-token SGLang generation backend for slime training rollouts.

Unlike the vLLM/Gemini inference backends, this backend exists to train on its
own output: every turn preserves the exact sampled token ids and per-token
rollout log probabilities from the current slime actor served through SGLang.
Decoded text is produced only so the agent can parse the action; token ids are
the training source of truth and are never reconstructed by re-tokenizing text.

The request/abort protocol mirrors the vendored slime adapter
(``third_party/slime/slime/agent/adapters/common.py``): render ``input_ids``
with the actor tokenizer, POST with ``return_logprob=True``, read ids and log
probabilities from ``meta_info.output_token_logprobs``, and free the SGLang
slot through ``/abort_request`` if the coroutine is cancelled or times out.

This module deliberately does not import slime: the Task 9 rollout hook
converts the recorded session turns into ``slime.agent.trajectory.TurnRecord``
objects inside the training environment.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

import aiohttp

from officeqa_rl.types import GeneratedTurn, Message

# Token-exact serving defaults shared with slime's adapters: keep special
# tokens and stop sequences in the output so served tokens match trained ones.
SAMPLING_DEFAULTS = {
    "skip_special_tokens": False,
    "spaces_between_special_tokens": False,
    "no_stop_trim": True,
}


@dataclass(frozen=True)
class SessionTurn:
    """One recorded generation turn: the exact prompt rendered for it and its
    exact-token result. The rollout hook replays these into slime's
    ``TrajectoryManager`` when the agent session ends."""

    prompt_messages: list[Message]
    turn: GeneratedTurn


class SlimeSGLangBackend:
    """Generate agent turns from the current slime actor served through SGLang.

    One instance is one rollout session: it enforces the session's total
    context budget and records every turn for exact-token trajectory export.
    Do not share an instance across concurrent agent runs.
    """

    def __init__(
        self,
        *,
        base_url: str,
        tokenizer: Any,
        max_context_len: int,
        routing_key: str | None = None,
        sock_read_timeout_sec: float = 900.0,
        chat_template_kwargs: dict[str, Any] | None = None,
        think_budget: int = 0,
        think_open_token_id: int | None = None,
        think_close_token_id: int | None = None,
        think_budget_single_record: bool = False,
    ) -> None:
        if max_context_len <= 0:
            raise ValueError("max_context_len must be positive")
        if think_budget and (think_open_token_id is None or think_close_token_id is None):
            raise ValueError("think_budget requires think_open_token_id and think_close_token_id")
        self.base_url = base_url.rstrip("/")
        self.tokenizer = tokenizer
        self.max_context_len = max_context_len
        self.routing_key = routing_key
        self.sock_read_timeout_sec = sock_read_timeout_sec
        # Extra kwargs for apply_chat_template. Gemma-4's template opens AND
        # closes the thought channel unless enable_thinking=True is passed, so
        # without this a gemma run silently trains a non-reasoning policy.
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        # Hard cap on the thought channel per turn, 0 = off. When the channel
        # is still open at the cap the backend closes it and continues into
        # the action, so a turn always acts. The forced close token is never
        # sampled and never trained: see _generate_with_think_budget.
        self.think_budget = int(think_budget or 0)
        self.think_open_token_id = think_open_token_id
        self.think_close_token_id = think_close_token_id
        # Record a forced-close turn as ONE session record (thought + close +
        # action) instead of two. Right for every mode whose loss spans are
        # cut by the fence/tail planners (-aod, -aod2op, -tail, -maskthink):
        # they locate ``</think>`` in the turn's OUTPUT, so a second record
        # whose close sits in its prompt has no visible region and is
        # discarded -- measured on 14 of 18 forced closes in one step -- while
        # the planners already exclude the close token from the trained span.
        # Off (two records) for losses that train the whole output.
        self.think_budget_single_record = bool(think_budget_single_record)
        self.forced_close_turns = 0
        self.session_turns: list[SessionTurn] = []
        # Optional observer fired with the turn's prompt_ids just before the
        # actor request is issued. OPD's teacher-think scoring uses it to
        # start the teacher's own deliberation concurrently with the
        # student's generation (the deliberation depends only on the prompt).
        self.on_turn_prompt: Any = None
        # Prefix-cache accounting across the session's requests, reported by
        # SGLang per request; the rollout hook copies the totals onto one
        # emitted sample so slime's rollout/prefix_cache_hit_rate is real.
        self.cached_prompt_tokens = 0
        self.total_prompt_tokens = 0

    async def generate(
        self,
        messages: list[Message],
        sampling_params: dict[str, Any],
    ) -> GeneratedTurn:
        prompt_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ]
        encoded = self.tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            **self.chat_template_kwargs,
        )
        # Newer transformers return a BatchEncoding dict rather than the ids.
        if hasattr(encoded, "__getitem__") and "input_ids" in encoded:
            encoded = encoded["input_ids"]
        prompt_ids = list(encoded)

        remaining_context = self.max_context_len - len(prompt_ids)
        if remaining_context <= 0:
            # Same contract as slime's adapter: an exhausted context yields an
            # empty "length" turn instead of an oversized request.
            return self._record(prompt_messages, prompt_ids, [], [], "length")

        params = {**SAMPLING_DEFAULTS, **sampling_params}
        requested = params.get("max_new_tokens")
        total_budget = (
            remaining_context
            if requested is None
            else min(int(requested), remaining_context)
        )

        if self.on_turn_prompt is not None:
            self.on_turn_prompt(prompt_ids)

        if self.think_budget <= 0:
            params["max_new_tokens"] = total_budget
            output_ids, output_log_probs, finish_reason = await self._request(
                prompt_ids, params
            )
            return self._record(
                prompt_messages, prompt_ids, output_ids, output_log_probs, finish_reason
            )
        return await self._generate_with_think_budget(
            prompt_messages, prompt_ids, params, total_budget
        )

    def _channel_open(self, prompt_ids: list[int], output_ids: list[int]) -> bool:
        """Is the thought channel still open once ``output_ids`` has been generated?

        Decided by the LAST channel marker seen across the prompt and the
        output, because who opens the channel depends on the template:

        * gemma: the model emits ``<|channel>thought`` itself, so the open
          marker is in the output.
        * Qwen3.5: the chat template ends the generation prompt with
          ``<think>\n``, so the model starts inside the thought and its
          output carries only ``</think>``. The open marker is in the PROMPT
          (second-to-last token, before the newline) and never in the output.

        The pre-2026-09-09 test looked for the open marker in the output
        alone, so it never recognised a Qwen channel and the forced close
        below was unreachable on Qwen: a thought that hit the budget just
        resumed to the response cap. History turns re-rendered from text
        carry no markers, and any that do are balanced, so a whole-prompt
        scan reduces to the generation prompt's own opener.
        """
        state = None
        for token in prompt_ids:
            if token == self.think_open_token_id:
                state = "open"
            elif token == self.think_close_token_id:
                state = "closed"
        for token in output_ids:
            if token == self.think_open_token_id:
                state = "open"
            elif token == self.think_close_token_id:
                state = "closed"
        return state == "open"

    async def _generate_with_think_budget(
        self,
        prompt_messages: list[Message],
        prompt_ids: list[int],
        params: dict[str, Any],
        total_budget: int,
    ) -> GeneratedTurn:
        """Two-phase generation with a hard cap on the thought channel.

        Phase one runs at most ``think_budget`` new tokens and stops on the
        channel-close token. If the model closed the channel itself, or ended
        the turn, or hit the cap while writing visible text with no channel
        open, generation simply resumes and the turn stays ONE record --
        identical to what the single-shot path produces. Only when the channel
        is still open at the cap does the backend intervene: it appends the
        close token itself and generates the action from there.

        That forced close was never sampled, so it must carry no loss. It is
        placed in the SECOND session record's prompt rather than in any
        record's output, so it can only ever enter a sample as context
        (loss_mask=0). The two records share prompt_messages, so the
        trajectory manager forks them into two samples -- the same per-turn
        fan-out every agent turn already gets -- with the thought trained in
        the first and the action in the second, each exactly once, and the
        forced close in neither. No change to slime is needed.
        """
        close = self.think_close_token_id
        phase1 = dict(params)
        phase1["max_new_tokens"] = min(self.think_budget, total_budget)
        phase1["stop_token_ids"] = [*(params.get("stop_token_ids") or []), close]
        ids1, lps1, fr1 = await self._request(prompt_ids, phase1)
        closed = bool(ids1) and ids1[-1] == close
        remaining = total_budget - len(ids1)

        if not closed and fr1 != "length":
            # Turn ended (eos) before any close: no channel, or a short one.
            return self._record(prompt_messages, prompt_ids, ids1, lps1, fr1)

        if closed or not self._channel_open(prompt_ids, ids1):
            # Natural close (no_stop_trim keeps the sampled close token), or
            # the cap fell inside visible text: resume on-policy, one record.
            if remaining <= 0:
                return self._record(prompt_messages, prompt_ids, ids1, lps1, "length")
            phase2 = dict(params)
            phase2["max_new_tokens"] = remaining
            ids2, lps2, fr2 = await self._request(prompt_ids + ids1, phase2)
            return self._record(
                prompt_messages, prompt_ids, ids1 + ids2, lps1 + lps2, fr2
            )

        # Channel open at the cap: force the close and continue into the action.
        # The thought record carries its own finish reason rather than
        # "length": the trajectory manager flags a chain as truncated when its
        # last turn finished on "length", and this turn did not run out of
        # room -- it ran into the budget and still produced an action.
        self.forced_close_turns += 1
        single = self.think_budget_single_record
        if not single:
            self._record(
                prompt_messages, prompt_ids, ids1, lps1, "think_budget",
                usage={"think_budget_forced": True},
            )
        prompt2 = prompt_ids + ids1 + [close]
        remaining2 = min(remaining - 1, self.max_context_len - len(prompt2))
        if remaining2 <= 0:
            if single:
                return self._record(
                    prompt_messages, prompt_ids, ids1 + [close], lps1 + [0.0], "length",
                    usage={"think_budget_forced": True},
                )
            return GeneratedTurn(
                text=self.tokenizer.decode(ids1 + [close], skip_special_tokens=True),
                prompt_ids=prompt_ids,
                output_ids=ids1 + [close],
                output_log_probs=lps1 + [0.0],
                finish_reason="length",
            )
        phase2 = dict(params)
        phase2["max_new_tokens"] = remaining2
        ids2, lps2, fr2 = await self._request(prompt2, phase2)
        if single:
            # One record, the whole turn: the tail/fence planners cut the
            # trained span at the close they find in this output, so the
            # thought and the forced close (log-prob 0.0) carry no loss.
            return self._record(
                prompt_messages, prompt_ids, ids1 + [close] + ids2, lps1 + [0.0] + lps2, fr2,
                usage={"think_budget_forced": True},
            )
        self._record(prompt_messages, prompt2, ids2, lps2, fr2)
        # The agent sees the whole turn; training sees the two records above.
        return GeneratedTurn(
            text=self.tokenizer.decode(ids1 + [close] + ids2, skip_special_tokens=True),
            prompt_ids=prompt_ids,
            output_ids=ids1 + [close] + ids2,
            output_log_probs=lps1 + [0.0] + lps2,
            finish_reason=fr2,
        )

    def _record(
        self,
        prompt_messages: list[Message],
        prompt_ids: list[int],
        output_ids: list[int],
        output_log_probs: list[float],
        finish_reason: str,
        *,
        usage: dict[str, Any] | None = None,
    ) -> GeneratedTurn:
        turn = GeneratedTurn(
            text=self.tokenizer.decode(output_ids, skip_special_tokens=True),
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            output_log_probs=output_log_probs,
            finish_reason=finish_reason,
            usage=usage,
        )
        self.session_turns.append(SessionTurn(prompt_messages, turn))
        return turn

    async def _request(
        self, prompt_ids: list[int], params: dict[str, Any]
    ) -> tuple[list[int], list[float], str]:
        """One /generate call -> (output_ids, output_log_probs, finish_reason).

        An engine context-length rejection comes back as an empty "length"
        result: the engine's admissible input is slightly smaller than
        max_context_len (scheduler reserve), so a prompt can pass the guard in
        generate() and still be refused. Ending the episode gracefully keeps
        its already-generated turns as valid training samples.
        """
        rid = uuid.uuid4().hex
        payload = {
            "rid": rid,
            "input_ids": prompt_ids,
            "sampling_params": params,
            "return_logprob": True,
        }
        headers = {"X-SMG-Routing-Key": self.routing_key} if self.routing_key else None
        timeout = aiohttp.ClientTimeout(total=None, sock_read=self.sock_read_timeout_sec)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.base_url}/generate",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status >= 400:
                        detail = await response.text()
                        if (
                            response.status == 400
                            and "exceeds the maximum allowed length" in detail
                        ):
                            return [], [], "length"
                        raise RuntimeError(
                            f"SGLang /generate returned {response.status}: {detail[:400]}"
                        )
                    data = await response.json(content_type=None)
        except (asyncio.CancelledError, asyncio.TimeoutError, aiohttp.ClientError):
            # Free the SGLang slot eagerly; an orphaned generation keeps
            # occupying KV cache until its own length cap otherwise.
            await self._abort_request(rid)
            raise

        meta_info = data.get("meta_info") or {}
        self.cached_prompt_tokens += int(meta_info.get("cached_tokens") or 0)
        self.total_prompt_tokens += int(meta_info.get("prompt_tokens") or len(prompt_ids))
        pairs = meta_info.get("output_token_logprobs") or []
        output_ids = [pair[1] for pair in pairs]
        output_log_probs = [float(pair[0]) for pair in pairs]
        finish_reason = (meta_info.get("finish_reason") or {}).get("type") or "stop"
        return output_ids, output_log_probs, finish_reason

    async def _abort_request(self, rid: str) -> None:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                await session.post(
                    f"{self.base_url}/abort_request",
                    json={"rid": rid},
                )
        except Exception:
            pass  # aborting is best-effort; the original error is re-raised


__all__ = ["SAMPLING_DEFAULTS", "SessionTurn", "SlimeSGLangBackend"]
