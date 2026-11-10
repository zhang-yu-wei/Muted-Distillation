"""Per-turn OPD teacher scoring rollout hook (scores concurrently with the episode).

Alternative ``--custom-generate-function-path`` entry point:
``officeqa_rl.training.opd_rollout.generate``. The default hook
(:mod:`officeqa_rl.training.rollout`) scores each finished training sample's
full token sequence after the episode ends, which serializes all teacher work
behind generation and re-prefills every fan-out sibling's duplicated context
head. This hook instead scores each agent turn the moment the backend records
it, while the episode is still running:

- one request per turn — ``turn.prompt_ids + turn.output_ids`` with
  ``logprob_start_len = len(prompt_ids) - 1`` — returning exactly the
  output-token log-probs;
- consecutive turn requests extend each other, so the teacher's radix cache
  prefills each context region once per episode instead of once per fan-out
  sibling;
- teacher work overlaps sandbox execution and later turns, so the
  post-rollout scoring drain disappears.

Correctness relies on how ``slime.agent.trajectory`` builds samples: every
sample is assembled from the exact served per-turn token ids, so a turn's
live-scored log-probs are conditioned on precisely the context its trained
span keeps in whichever sample it lands in. Turns whose output is re-emitted
as untrained context (REALIGN) simply have their scores discarded during
assembly, and observation/prompt positions get 0.0 (inert under the loss
mask). Failure semantics match the per-sample path: a terminal scoring
failure is appended to ``officeqa_opd_failure_log`` and then re-raised, so a
broken teacher stops the run loudly.

``officeqa_opd_action_only`` (yaml; launcher env ``OPD_ACTION_ONLY=1``)
switches to action-only distillation (AOD): each turn with a valid fenced
command becomes its own single-turn training sample whose token sequence is
``context + <think>\\n\\n</think> + visible-tail`` — the student's reasoning
for that turn is replaced by the canonical empty think block — with the loss
mask restricted to the fenced command span. The teacher scores exactly that
sequence, so student and teacher condition on the SAME reasoning-free
prefix. Training the original sequence against a reasoning-free teacher
would instead teach the student to act against reasoning it just produced
(an unfaithfulness pressure); ablating the reasoning on both sides removes
that. Turns with no, several, or malformed fences (or a never-closed think
block) contribute no sample. Under ``officeqa_opd_pure`` the only gradient
in the run is therefore the distillation term on tool-call tokens.

``officeqa_opd_mask_think`` (yaml; launcher env ``OPD_MASK_THINK=1``) is the
alternative that keeps the reasoning: the teacher scores the full turn exactly
as it normally would, the sample keeps every token, and only the loss mask
narrows to the fenced command. The student is therefore trained on its action
tokens under a teacher that read its reasoning — where action-only
distillation asks what the teacher would do without it.

``officeqa_opd_teacher_no_think`` (yaml; launcher env
``OPD_TEACHER_NO_THINK=1``) is the third cell of that matrix and the only one
whose two sides disagree on purpose: the TEACHER scores the reasoning-free
request AOD builds, while the student's log-probs come from the sequence it
actually generated, reasoning ahead of the action. The fenced-command tokens
are identical in both, so the teacher's scores transfer onto them positionally;
the loss mask keeps only that span. AOD makes the pair faithful by ablating the
reasoning on both sides and pays for it by training on a sequence the student
never sampled; this keeps the student on-policy and accepts the mismatch
instead. All three modes are mutually exclusive.

``officeqa_opd_mask_to_think`` (yaml; launcher env ``OPD_MASK_TO_THINK=1``)
is the fourth alternative in that group: the teacher scores the student's
REAL rollout with the reasoning in place (the same request as pure OPD and
mask-think), and the loss mask narrows to the THINK region — the tokens from
the start of the think content through ``</think>`` inclusive. Where
mask-think asks "match the teacher's action distribution", this asks "match
the teacher's deliberation distribution" and leaves actions untrained.
``officeqa_opd_think_span`` (``all`` | ``first`` | ``last``, default ``all``)
selects what carries the loss. ``all`` is the think region, and a turn whose
think block never closes is dropped while a fence-less turn with a closed
think KEEPS its loss (deliberation is what is being trained). ``first`` and
``last`` are POSITIONAL over the whole turn response — the first / last
``officeqa_opd_think_span_tokens`` tokens, no think parsing, no turn dropped.
Since the chat template opens ``<think>`` in the prompt, ``first`` lands on
the opening deliberation (where teacher entropy concentrates), while ``last``
lands on whatever ends the turn — normally the fenced command, not reasoning.
Not composable with the tail/prose/teacher-think modifiers.

``officeqa_opd_tail_think_tokens`` (yaml; launcher env
``OPD_TAIL_THINK_TOKENS=N``, default 0 = off, name suffix ``-tailthink<N>``)
modifies the tail loss under the on-policy base: the trained span is pulled
back into the last N tokens of the turn's thought (never past its start,
never into an empty thought), and the teacher request keeps the student's
sequence intact minus the head of the thought: ``context + <think> +
"[... earlier reasoning omitted ...]" + thought-ending + </think> + visible
tail``. The judge therefore deliberates on the truncated thought as its own,
told up front that the head is missing, and scores the visible tail after
it. The close marker carries no loss. (A first build scored the ending as
visible prose behind the empty think block — the no-think judge rated it
20x less likely than a command and the arm collapsed; that variant was
removed.) Built for gemma, whose tail
arm trains a bare command in 83-97% of turns because it writes no prose
between thought and fence; the plan for the command is stated in the
thought's last sentences (last two sentences: median 35-39 tokens).

``officeqa_opd_tail_loss`` (yaml; launcher env ``OPD_TAIL_LOSS=1``) modifies
``officeqa_opd_action_only`` or ``officeqa_opd_teacher_no_think`` (invalid
alone or with mask-think): the trained span widens from the fenced command to
the WHOLE visible region — every token after ``</think>``, prose included —
and a fence-less turn keeps its loss instead of being dropped. Motivation:
offline pass@8 evals of aod2 iter-139 showed the command-only mask leaks —
visible prose carries no loss, and in failing episodes at sampling
temperature the deliberation migrates there (3.4k prose chars/turn vs pure
OPD's 2.2k, commands identical), erasing the median token advantage in the
mean. Only a never-closed think block still discards the turn.

``officeqa_opd_prose_as_think`` (yaml; launcher env ``OPD_PROSE_AS_THINK=1``)
is the opposite treatment of the same prose, for the same two modes and
mutually exclusive with tail loss: the visible prose is reclassified as
REASONING and ablated with the think block. The kept tail shrinks to exactly
the fenced command, so the teacher request becomes ``context +
<think>\n\n</think>\n\n + command`` — the teacher conditions on no
deliberation at all, closing the leak where prose-borne reasoning reached a
nominally reasoning-blind teacher — and under action_only the trained sample
shrinks the same way. The loss span is unchanged: the command. Turn
eligibility matches plain AOD (exactly one well-formed visible fence).

``officeqa_opd_teacher_think`` (yaml; launcher env ``OPD_TEACHER_THINK=1``)
modifies ``officeqa_opd_teacher_no_think`` only (invalid elsewhere, and not
composable with prose-as-think): before scoring a turn, the teacher first
GENERATES its own think block from the same context (greedy by default,
capped at ``officeqa_opd_teacher_think_max_tokens``, default 512), and the
scored request becomes ``context + <think>teacher's deliberation</think> +
student's visible tail``. The judge deliberates before valuing the student's
action; the student stays fully on-policy, since scores still transfer
positionally onto its real rollout. Alone, the loss covers the fenced command
(the ``-aod2op-tthink`` arms); with ``officeqa_opd_tail_loss`` the whole
visible tail — prose and command — follows the teacher's deliberation in the
request and carries the loss (the ``-aod2op-tail-tthink`` arm): the same
request shape, a wider scored slice, and a fence-less turn keeps its loss.
``officeqa_opd_teacher_think_temperature`` (default 0.0) samples the think —
nonzero values add think-sampling variance to the advantages. A failed
generation degrades that turn to the empty-think request instead of aborting
the episode. Throughput note: the extra decode is the dominant cost; the cap
bounds it, and the episode's endpoint affinity lets the scoring call reuse
the prefill the generation just warmed.

``officeqa_opd_train_turns`` (yaml; launcher env ``OPD_TRAIN_TURNS=N``) caps
the TRAINING episode at N turns — the horizon curriculum. It caps the agent,
not ``officeqa_step_limit``, which is what keeps the experiment readable:
commands still execute so turns 2..N see genuine observations, and evaluation
delegates below at the full step limit, so eval measures the same task for
every rung and stays comparable to the uncapped arms. The episode ends by
hitting the cap, so its samples are TRUNCATED and its reward is usually 0 —
this is a distillation curriculum and the launcher requires
``officeqa_opd_pure``.

When ``evaluation`` is true or OPD is disabled this module delegates to
:func:`officeqa_rl.training.rollout.generate` unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, replace
from typing import Any

from slime.agent.trajectory import TrajectoryManager, TurnRecord
from slime.utils.types import Sample

from officeqa_rl.actions import _FENCED_COMMAND_RE
from officeqa_rl.agent import OfficeQAAgent
from officeqa_rl.dataset import task_from_slime_fields
from officeqa_rl.full_corpus_env import FullCorpusEnvironment
from officeqa_rl.scoring import score_answer
from officeqa_rl.training import rollout as default_rollout
from officeqa_rl.training.rollout import (
    _STATUS_BY_EXIT,
    _abort,
    _create_backend,
    _episode_output_tokens,
    _forced_close_turns,
    _log_teacher_failure,
    _make_run_id,
    _teacher_scoring_enabled,
    _tolerance,
    agent_config_from_args,
    build_training_samples,
    environment_config_from_args,
)

logger = logging.getLogger(__name__)

_WATCH_INTERVAL_SEC = 0.5

# Canonical Qwen "non-thinking" assistant opening: replaces the student's
# think block in action-only teacher requests AND training samples, keeping
# both policies on-distribution (an assistant turn never starts with a bare
# command). The student's own text after ``</think>`` supplies the
# following newlines.
_EMPTY_THINK = "<think>\n\n</think>"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class _ThinkDialect:
    """One chat template's reasoning-block markers.

    ``empty`` is the canonical zero-content block exactly as the template
    itself renders it (qwen keeps a blank line inside, gemma does not), and
    ``opener`` is the text that opens a block ready to receive content —
    they are not derivable from ``open``/``close`` alone, which is why all
    four are spelled out per family.
    """

    open: str
    close: str
    empty: str
    opener: str


_QWEN_DIALECT = _ThinkDialect(
    open=_THINK_OPEN, close=_THINK_CLOSE, empty=_EMPTY_THINK,
    opener=_THINK_OPEN + "\n",
)
_GEMMA_DIALECT = _ThinkDialect(
    open="<|channel>thought", close="<channel|>",
    empty="<|channel>thought\n<channel|>", opener="<|channel>thought\n",
)


def _think_dialect(tokenizer: Any) -> _ThinkDialect:
    """The reasoning-block markers this tokenizer's chat template uses.

    Read off the template source rather than the model name so a template
    swap (or a family this code has never seen) fails toward the qwen
    default instead of toward a silent misparse of known gemma markers.
    """
    template = getattr(tokenizer, "chat_template", None) or ""
    if _GEMMA_DIALECT.open in template:
        return _GEMMA_DIALECT
    return _QWEN_DIALECT


@dataclass(frozen=True)
class _ScorePlan:
    """Shared shape of one turn's teacher request and training sample (AOD).

    ``tail_start`` is the first output-token index kept (the visible region
    after ``</think>``); ``None`` discards the whole turn — no request, no
    training sample. ``prefix_ids`` are inserted between the context and the
    tail — the canonical empty think block, or empty when the turn had no
    think block. ``spans`` holds the single fenced-command token range
    (relative to the turn's output): the only positions whose loss mask
    stays 1 in the training sample.
    """

    tail_start: int | None
    prefix_ids: tuple[int, ...] = ()
    spans: tuple[tuple[int, int], ...] = ()
    # First output-token index NOT kept; None keeps everything from
    # ``tail_start`` on. Prose-as-think sets this to the fence end so the
    # kept tail is exactly the command block.
    tail_end: int | None = None
    # Tail-think: the first span starts at ``tail_start`` inside the thought
    # and ends before the close marker; the visible tail is the second span.
    # The request is always the contiguous slice ``output[tail_start:tail_end]``.
    # Fence delimiters (the ```<lang>\n opener and the closing ```) kept out of
    # the loss mask but still in the request: officeqa_opd_mask_fence_delims.
    # Ranges are relative to the turn's output.
    masked: tuple[tuple[int, int], ...] = ()

    def request_ids(self, output_ids: list[int]) -> list[int]:
        """The turn's output tokens as the teacher request carries them."""
        return output_ids[self.tail_start : self.tail_end]

    def score_index(self, offset: int) -> int:
        """Index into the teacher's log-probs for output token ``offset``."""
        return offset - self.tail_start

    def is_masked(self, offset: int) -> bool:
        """Whether output token ``offset`` is a fence delimiter kept out of the loss."""
        return any(lo <= offset < hi for lo, hi in self.masked)

    @property
    def tail_think(self) -> bool:
        """Whether the plan pulled thought tokens into the trained span."""
        return (
            len(self.spans) > 1
            and self.spans[0][0] == self.tail_start
            and self.spans[0][1] < self.spans[1][0]
        )


_DISCARD = _ScorePlan(tail_start=None)


def _token_starts(
    tokenizer: Any, output_ids: list[int], text: str
) -> list[int] | None:
    """Char offset of each token in ``text`` (+ final sentinel), or None.

    Per-token decoding is exact unless a multi-byte character crosses a token
    boundary; the fast-tokenizer offset mapping covers that case. When
    neither reproduces ``text``/``output_ids`` exactly the caller discards
    the turn rather than risking a misaligned fence span.
    """
    pieces = [tokenizer.decode([t], skip_special_tokens=False) for t in output_ids]
    if "".join(pieces) == text:
        starts, position = [], 0
        for piece in pieces:
            starts.append(position)
            position += len(piece)
        starts.append(position)
        return starts
    try:
        encoded = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        if list(encoded["input_ids"]) == list(output_ids):
            starts = [start for start, _ in encoded["offset_mapping"]]
            starts.append(len(text))
            return starts
    except Exception:  # noqa: BLE001 — any tokenizer quirk means "unmappable"
        pass
    return None


def _empty_think_text(tokenizer: Any, prompt_ids: list[int]) -> str:
    """Text that renders the turn's think block as the canonical empty one.

    What must be inserted depends on what the chat template already wrote.
    Qwen3.5 ends a generation prompt with ``<think>\\n`` (thinking mode), so
    the block is already open and only its remainder belongs here — emitting
    a whole ``<think>...</think>`` would nest a second opener and put both
    policies on a sequence no chat template ever produces. A template that
    renders the empty block itself (``enable_thinking=false``) needs nothing,
    and one that opens no block at all needs the full opening.
    """
    # Generous window: a template writes its think marker within a few tokens
    # of the generation prompt's end, and missing it would nest an opener.
    d = _think_dialect(tokenizer)
    tail = tokenizer.decode(prompt_ids[-32:], skip_special_tokens=False)
    open_at, close_at = tail.rfind(d.open), tail.rfind(d.close)
    if open_at < 0 and close_at < 0:
        return d.empty
    if close_at > open_at:
        return ""
    written = tail[open_at + len(d.open) :]
    remainder = d.empty[len(d.open) :]
    return remainder[len(written) :] if remainder.startswith(written) else d.close


def _think_generation_input(
    tokenizer: Any, prompt_ids: list[int]
) -> list[int] | None:
    """Token prefix from which the teacher writes its own think block.

    Mirrors :func:`_empty_think_text`'s template inspection: with the block
    already open (Qwen3.5 thinking mode ends the generation prompt with
    ``<think>\n``) the teacher continues it directly; with no block it opens
    one. A template that already closed the block leaves nowhere to put a
    deliberation — the caller falls back to empty-think scoring.
    """
    d = _think_dialect(tokenizer)
    tail = tokenizer.decode(prompt_ids[-32:], skip_special_tokens=False)
    open_at, close_at = tail.rfind(d.open), tail.rfind(d.close)
    if open_at >= 0 and close_at < open_at:
        return list(prompt_ids)
    if open_at < 0 and close_at < 0:
        return list(prompt_ids) + list(
            tokenizer.encode(d.opener, add_special_tokens=False)
        )
    return None


def _think_prefix_text(tokenizer: Any, prompt_ids: list[int], think_text: str) -> str:
    """Counterpart of :func:`_empty_think_text` with a teacher-written body.

    Completes the turn's think block so it reads
    ``<think>\n{think_text}\n</think>`` given whatever the chat template
    already wrote. A template that wrote something other than whitespace
    inside the block (or already closed it) cannot host the deliberation;
    the block is closed bare and the caller effectively degrades to the
    empty-think request.
    """
    d = _think_dialect(tokenizer)
    tail = tokenizer.decode(prompt_ids[-32:], skip_special_tokens=False)
    open_at, close_at = tail.rfind(d.open), tail.rfind(d.close)
    if open_at < 0 and close_at < 0:
        return f"{d.opener}{think_text}\n{d.close}"
    if close_at > open_at:
        return ""
    written = tail[open_at + len(d.open) :]
    if written.strip() == "":
        separator = "" if written.endswith("\n") else "\n"
        return f"{separator}{think_text}\n{d.close}"
    return d.close


def _fence_span(
    tokenizer: Any, output_ids: list[int]
) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    """Where one turn's visible region and fenced command start, in tokens.

    Mirrors :func:`officeqa_rl.actions.parse_fenced_command`: the turn must
    contain exactly one fenced command block, and it must lie in the visible
    region (after ``</think>``). Anything else — no fence, several fences, a
    fence only inside the think block, or an unmappable tokenization — returns
    None, and the caller drops that turn's distillation signal. Shared by both
    modes: action-only scoring cuts the turn at the visible region, while
    think-masked scoring keeps the whole turn and only trains the span.
    """
    d = _think_dialect(tokenizer)
    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    close = text.find(d.close)
    visible_from = close + len(d.close) if close >= 0 else 0
    fences = list(_FENCED_COMMAND_RE.finditer(text))
    if len(fences) != 1 or fences[0].start() < visible_from:
        return None
    starts = _token_starts(tokenizer, output_ids, text)
    if starts is None:
        return None
    count = len(output_ids)
    tail_start = next(
        (i for i in range(count) if starts[i + 1] > visible_from), count
    )
    fence_lo, fence_hi = fences[0].start(), fences[0].end()
    span_lo = next((i for i in range(count) if starts[i + 1] > fence_lo), count)
    span_hi = next((i for i in range(count) if starts[i] >= fence_hi), count)
    span_lo = max(span_lo, tail_start)
    if span_lo >= span_hi:
        return None
    return tail_start, ((span_lo, span_hi),)


def _fence_delimiter_ranges(
    tokenizer: Any, output_ids: list[int]
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Token ranges of the visible fence's opener and closer, or None.

    The opener is the ```<lang>\n up to the first command byte, the closer
    the newline and ``` after the last. Both are environment syntax, not
    content: on the SSB1 q38 arms the teacher (a Qwen3.8 whose own tool
    format is <tool_call>) put the opening fence at top-1 in only 10 of 24
    real student turns and a prose continuation in 14, giving that single
    token a reverse-KL push of ~1.6 nats -- 20x the average tail token --
    pointing away from opening a command; the student's next-best opener was
    its own <tool_call> token, which the parser rejects. Same fence rule as
    :func:`_fence_span` (exactly one, in the visible region); anything else
    returns None and the caller masks nothing.
    """
    d = _think_dialect(tokenizer)
    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    close = text.find(d.close)
    visible_from = close + len(d.close) if close >= 0 else 0
    fences = list(_FENCED_COMMAND_RE.finditer(text))
    if len(fences) != 1 or fences[0].start() < visible_from:
        return None
    starts = _token_starts(tokenizer, output_ids, text)
    if starts is None:
        return None
    count = len(output_ids)

    def token_range(char_lo: int, char_hi: int) -> tuple[int, int]:
        lo = next((i for i in range(count) if starts[i + 1] > char_lo), count)
        hi = next((i for i in range(count) if starts[i] >= char_hi), count)
        return lo, max(lo, hi)

    fence = fences[0]
    return token_range(fence.start(), fence.start(1)), token_range(fence.end(1), fence.end())


def _tail_span(
    tokenizer: Any, output_ids: list[int]
) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    """The whole visible region as one loss span (the ``-tail`` variants).

    Where :func:`_fence_span` keeps only the fenced command, this keeps every
    token after ``</think>`` — prose included. The offline iter-139 evals
    showed why the command-only mask leaks: visible prose carries no loss, and
    in failing episodes at sampling temperature the deliberation migrates
    there (aod2 flailers averaged 3.4k prose chars/turn vs pure OPD's 2.2k,
    with identical command lengths), erasing the median token advantage in
    the mean. Training the tail pulls prose toward the teacher's terse style,
    and a fence-less rambling turn now carries loss instead of being dropped.
    A turn whose think block never closes has no visible region and is
    discarded, as is a turn that ends at ``</think>``.
    """
    d = _think_dialect(tokenizer)
    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    close = text.find(d.close)
    if close < 0:
        return None
    visible_from = close + len(d.close)
    starts = _token_starts(tokenizer, output_ids, text)
    if starts is None:
        return None
    count = len(output_ids)
    tail_start = next(
        (i for i in range(count) if starts[i + 1] > visible_from), count
    )
    if tail_start >= count:
        return None
    return tail_start, ((tail_start, count),)


def _tail_think_span(
    tokenizer: Any, output_ids: list[int], think_tokens: int
) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    """Tail span pulled back into the last ``think_tokens`` of the thought.

    The ``-tailthink<N>`` modifier of the tail loss. Gemma's tail arm trains
    a bare command in 83-97% of turns because gemma writes no prose between
    the thought and the fence; the last sentences of its thought are where
    the plan for the command is stated. Those tokens (at most N, never past
    the thought's start, never an empty thought) join the trained span. The
    request is the contiguous slice from ``tail_start`` — thought ending,
    the student's own close marker, visible tail — behind a prefix that
    opens the think block and says the head was omitted; the marker itself
    stays out of the loss spans.

    Returns ``(tail_start, spans)``; with a single span (empty thought, or
    nothing pulled in) the result equals :func:`_tail_span`'s.
    """
    found = _tail_span(tokenizer, output_ids)
    if found is None:
        return None
    tail_start, spans = found
    if think_tokens <= 0:
        return tail_start, spans
    d = _think_dialect(tokenizer)
    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    close = text.find(d.close)
    starts = _token_starts(tokenizer, output_ids, text)
    if close < 0 or starts is None:
        return tail_start, spans
    count = len(output_ids)
    open_at = text.find(d.open)
    if 0 <= open_at < close:
        body_from = open_at + len(d.open)
        if text.startswith(d.opener, open_at):
            body_from = open_at + len(d.opener)
    else:
        body_from = 0  # the template opened the block inside the prompt
    if not text[body_from:close].strip():
        return tail_start, spans
    close_lo = next((i for i in range(count) if starts[i + 1] > close), count)
    close_hi = next(
        (i for i in range(count) if starts[i] >= close + len(d.close)), count
    )
    body_lo = next((i for i in range(count) if starts[i + 1] > body_from), count)
    think_lo = max(body_lo, close_lo - think_tokens)
    if think_lo >= close_lo or close_hi > tail_start:
        return tail_start, spans
    new_spans = ((think_lo, close_lo),) + tuple(
        (max(lo, close_hi), hi) for lo, hi in spans if hi > close_hi
    )
    return think_lo, new_spans


# Tail-think: the teacher's think block opens with this line so the judge
# knows it is reading the END of a deliberation, not one that starts
# mid-sentence. Teacher-side only (it lives in the request prefix); the
# student's own tokens and the loss spans are untouched.
_TAIL_THINK_OMITTED = "[... earlier reasoning omitted ...]\n"


def _think_opener_text(tokenizer: Any, prompt_ids: list[int]) -> str | None:
    """Text that (re)opens the think block for a tail-think request.

    Mirrors :func:`_think_generation_input`: nothing when the template already
    opened the block (Qwen3.5 thinking mode), the opener when it opened none
    (gemma with thinking on), None when it pre-closed the block — then no
    thought can be hosted and the caller falls back to the plain tail.
    """
    d = _think_dialect(tokenizer)
    tail = tokenizer.decode(prompt_ids[-32:], skip_special_tokens=False)
    open_at, close_at = tail.rfind(d.open), tail.rfind(d.close)
    if open_at >= 0 and close_at < open_at:
        return ""
    if open_at < 0 and close_at < 0:
        return d.opener
    return None


def _distill_span(
    tokenizer: Any,
    output_ids: list[int],
    span: str = "all",
    span_tokens: int = 0,
) -> tuple[tuple[int, int], ...] | None:
    """The loss span for think distillation, in token offsets within a turn.

    ``span="all"`` is the THINK REGION: from the first think content token
    through the token containing ``</think>`` INCLUSIVE — closing the block is
    part of the deliberation policy being distilled (the temp-0 pathology
    showed the close/continue balance is exactly what whole-turn distillation
    moves). A template-written ``<think>`` opener inside the output is
    excluded. Returns None — dropping that turn's loss — for a think block
    that never closes, an empty region, or an unmappable tokenization.

    ``span="first"``/``"last"`` are POSITIONAL over the whole response
    instead: the first / last ``span_tokens`` tokens of the turn, clamped to
    its length, with no think parsing at all. Because the chat template opens
    ``<think>`` in the prompt, a turn's response begins in the think block, so
    ``first`` lands on the opening deliberation (where teacher entropy
    concentrates); ``last`` lands on whatever ends the turn, normally the
    fenced command rather than reasoning. Turns are never dropped for lacking
    a closed think block in these two modes.
    """
    count = len(output_ids)
    if not count:
        return None
    if span == "first":
        span_hi = min(span_tokens, count)
        return ((0, span_hi),) if span_hi > 0 else None
    if span == "last":
        span_lo = max(0, count - span_tokens)
        return ((span_lo, count),) if span_lo < count else None

    d = _think_dialect(tokenizer)
    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    close = text.find(d.close)
    if close < 0:
        return None
    region_end_char = close + len(d.close)
    open_at = text.find(d.open)
    content_start_char = (
        open_at + len(d.open) if 0 <= open_at < close else 0
    )
    starts = _token_starts(tokenizer, output_ids, text)
    if starts is None:
        return None
    span_lo = next(
        (i for i in range(count) if starts[i + 1] > content_start_char), count
    )
    span_hi = next(
        (i for i in range(count) if starts[i] >= region_end_char), count
    )
    if span_lo >= span_hi:
        return None
    return ((span_lo, span_hi),)


def _action_only_plan(
    tokenizer: Any,
    prompt_ids: list[int],
    output_ids: list[int],
    *,
    tail_loss: bool = False,
    prose_as_think: bool = False,
    tail_think_tokens: int = 0,
    mask_fence_delims: bool = False,
    teacher_think_text: str = "",
) -> _ScorePlan:
    """Teacher-request/training-sample shape for one turn under AOD.

    ``teacher_think_text`` puts a FIXED sentence into the teacher's think
    block (``<think>\n{text}\n</think>``) instead of leaving it empty, right
    before the scored tail. Measured on 24 real q38-scored turns: a system
    block prepended to the whole context (officeqa_opd_instruction) leaves
    the teacher's fence-slot log-prob unchanged (-0.70 -> -0.84..-0.95), but
    "One sentence, then the ```bash command block." in the think block
    raises it from -0.70 to -0.50 (median -0.50 -> -0.25) and makes ``` its
    top-1 token in 22/24 turns instead of 19/24. The student never sees it.

    ``prose_as_think`` reclassifies the visible prose as reasoning: the kept
    tail shrinks to exactly the fenced command, so prose is ablated from the
    teacher request (and from AOD's trained sequence) together with the think
    block instead of riding along as context. Because nothing then separates
    ``</think>`` from the fence — a juncture neither policy has ever seen —
    the inserted prefix carries the canonical ``\\n\\n`` the student's own
    tail normally supplies. The loss span is unchanged: the command.
    """
    think_prefix: str | None = None
    if tail_loss and tail_think_tokens > 0 and not prose_as_think:
        opener = _think_opener_text(tokenizer, prompt_ids)
        found_think = (
            _tail_think_span(tokenizer, output_ids, tail_think_tokens)
            if opener is not None  # a pre-closed block cannot host a thought
            else _tail_span(tokenizer, output_ids)
        )
        if found_think is None:
            return _DISCARD
        tail_start, spans = found_think
        if opener is not None and len(spans) > 1:
            think_prefix = opener + _TAIL_THINK_OMITTED
    else:
        found = (
            _tail_span(tokenizer, output_ids)
            if tail_loss
            else _fence_span(tokenizer, output_ids)
        )
        if found is None:
            return _DISCARD
        tail_start, spans = found
        if teacher_think_text:
            think_prefix = _think_prefix_text(tokenizer, prompt_ids, teacher_think_text)
    prefix_text = (
        think_prefix
        if think_prefix is not None
        else _empty_think_text(tokenizer, prompt_ids)
    )
    masked: tuple[tuple[int, int], ...] = ()
    if mask_fence_delims:
        ranges = _fence_delimiter_ranges(tokenizer, output_ids)
        if ranges is not None:
            masked = tuple((lo, hi) for lo, hi in ranges if lo < hi)
    if prose_as_think:
        (span_lo, span_hi) = spans[0]
        prefix_text += "\n\n"
        prefix_ids = tuple(tokenizer.encode(prefix_text, add_special_tokens=False))
        return _ScorePlan(span_lo, prefix_ids, spans, tail_end=span_hi, masked=masked)
    prefix_ids = (
        tuple(tokenizer.encode(prefix_text, add_special_tokens=False))
        if prefix_text
        else ()
    )
    return _ScorePlan(tail_start, prefix_ids, spans, masked=masked)


@dataclass(frozen=True)
class PerTurnModes:
    """Parsed per-turn scoring knobs, shared by every benchmark rollout.

    OfficeQA reads these in :func:`generate`; the SSB1 rollout reads the same
    yaml keys through :func:`parse_per_turn_modes` so the hill-climb arms mean
    the same thing on both benchmarks.
    """

    action_only: bool = False
    mask_think: bool = False
    mask_to_think: bool = False
    think_span: str = "all"
    think_span_tokens: int = 0
    teacher_no_think: bool = False
    tail_loss: bool = False
    prose_as_think: bool = False
    tail_think_tokens: int = 0
    teacher_think: bool = False
    think_max_tokens: int = 512
    think_temperature: float = 0.0
    noaction_penalty: float = 0.0
    noaction_window: int = 0
    # Keep the fenced command's opener/closer out of the loss (yaml
    # officeqa_opd_mask_fence_delims, launcher OPD_MASK_FENCE_DELIMS=1, name
    # suffix -nodelim). See _fence_delimiter_ranges.
    mask_fence_delims: bool = False
    # Fixed text for the teacher's think block on the empty-think request
    # (yaml officeqa_opd_teacher_think_text, launcher OPD_TEACHER_THINK_TEXT,
    # name suffix -tthinktext<preset>). See _action_only_plan.
    teacher_think_text: str = ""

    @property
    def command_only_request(self) -> bool:
        """Whether the teacher request is the reasoning-free AOD shape."""
        return self.action_only or self.teacher_no_think


def parse_per_turn_modes(args: Any) -> PerTurnModes:
    """Read and validate the ``officeqa_opd_*`` mode knobs.

    Raises ``ValueError`` whose message matches the launcher-side validation;
    callers abort the sample under the ``invalid_rollout_config:`` prefix.
    """
    modes = PerTurnModes(
        action_only=bool(getattr(args, "officeqa_opd_action_only", False)),
        mask_think=bool(getattr(args, "officeqa_opd_mask_think", False)),
        mask_to_think=bool(getattr(args, "officeqa_opd_mask_to_think", False)),
        think_span=str(
            getattr(args, "officeqa_opd_think_span", "all") or "all"
        ).strip().lower(),
        think_span_tokens=int(
            getattr(args, "officeqa_opd_think_span_tokens", 0) or 0
        ),
        teacher_no_think=bool(
            getattr(args, "officeqa_opd_teacher_no_think", False)
        ),
        tail_loss=bool(getattr(args, "officeqa_opd_tail_loss", False)),
        prose_as_think=bool(getattr(args, "officeqa_opd_prose_as_think", False)),
        tail_think_tokens=int(
            getattr(args, "officeqa_opd_tail_think_tokens", 0) or 0
        ),
        teacher_think=bool(getattr(args, "officeqa_opd_teacher_think", False)),
        think_max_tokens=int(
            getattr(args, "officeqa_opd_teacher_think_max_tokens", 512) or 512
        ),
        think_temperature=float(
            getattr(args, "officeqa_opd_teacher_think_temperature", 0.0) or 0.0
        ),
        noaction_penalty=float(
            getattr(args, "officeqa_opd_noaction_penalty", 0.0) or 0.0
        ),
        noaction_window=int(
            getattr(args, "officeqa_opd_noaction_window", 0) or 0
        ),
        mask_fence_delims=bool(
            getattr(args, "officeqa_opd_mask_fence_delims", False)
        ),
        teacher_think_text=str(
            getattr(args, "officeqa_opd_teacher_think_text", "") or ""
        ),
    )
    if sum((
        modes.action_only,
        modes.mask_think,
        modes.teacher_no_think,
        modes.mask_to_think,
    )) > 1:
        raise ValueError(
            "officeqa_opd_action_only, "
            "officeqa_opd_mask_think, officeqa_opd_teacher_no_think and "
            "officeqa_opd_mask_to_think are "
            "alternatives — they differ in whether the reasoning reaches the "
            "teacher, the training sequence, and which region carries loss"
        )
    if modes.mask_to_think and modes.think_span not in ("all", "first", "last"):
        raise ValueError(
            "officeqa_opd_think_span must be one of "
            "all, first, last"
        )
    if (
        modes.mask_to_think
        and modes.think_span in ("first", "last")
        and modes.think_span_tokens <= 0
    ):
        raise ValueError(
            "officeqa_opd_think_span=first/last needs "
            "officeqa_opd_think_span_tokens > 0"
        )
    if modes.mask_to_think and (modes.tail_loss or modes.prose_as_think):
        raise ValueError(
            "officeqa_opd_mask_to_think trains the think "
            "region; the tail/prose modifiers reshape the visible region of a "
            "command-only mode — they do not compose"
        )
    if modes.prose_as_think and not modes.command_only_request:
        raise ValueError(
            "officeqa_opd_prose_as_think modifies a command-only mode — set "
            "officeqa_opd_action_only or officeqa_opd_teacher_no_think "
            "alongside it"
        )
    if modes.tail_loss and not (modes.command_only_request or modes.mask_think):
        raise ValueError(
            "officeqa_opd_tail_loss widens a command-planned loss to the whole "
            "visible tail — set officeqa_opd_action_only, "
            "officeqa_opd_teacher_no_think or officeqa_opd_mask_think "
            "alongside it"
        )
    if modes.tail_loss and modes.prose_as_think:
        raise ValueError(
            "officeqa_opd_tail_loss trains the visible "
            "prose while officeqa_opd_prose_as_think ablates it — opposite "
            "treatments, pick one"
        )
    if modes.teacher_think_text and not (modes.action_only or modes.teacher_no_think):
        raise ValueError(
            "officeqa_opd_teacher_think_text fills the teacher's empty think "
            "block — set officeqa_opd_teacher_no_think or "
            "officeqa_opd_action_only alongside it"
        )
    if modes.teacher_think_text and (modes.teacher_think or modes.tail_think_tokens > 0):
        raise ValueError(
            "officeqa_opd_teacher_think_text and officeqa_opd_teacher_think / "
            "officeqa_opd_tail_think_tokens all want the teacher's think block — pick one"
        )
    if modes.mask_fence_delims and not (modes.action_only or modes.teacher_no_think):
        raise ValueError(
            "officeqa_opd_mask_fence_delims drops the fence delimiters from a "
            "fence-planned loss — set officeqa_opd_action_only or "
            "officeqa_opd_teacher_no_think alongside it"
        )
    if modes.tail_think_tokens > 0 and not modes.tail_loss:
        raise ValueError(
            "officeqa_opd_tail_think_tokens extends the TAIL loss into the "
            "thought's last tokens — set officeqa_opd_tail_loss alongside it"
        )
    if modes.tail_think_tokens > 0 and modes.teacher_think:
        raise ValueError(
            "officeqa_opd_tail_think_tokens puts the student's thought ending "
            "in the teacher's think block; officeqa_opd_teacher_think wants "
            "that block for the teacher's own deliberation — pick one"
        )
    if modes.tail_think_tokens > 0 and not modes.teacher_no_think:
        raise ValueError(
            "officeqa_opd_tail_think_tokens trains the student's real "
            "sequence against a request that keeps its truncated thought; "
            "only the on-policy base can host that — set "
            "officeqa_opd_teacher_no_think, not officeqa_opd_action_only"
        )
    if modes.teacher_think and not modes.teacher_no_think:
        raise ValueError(
            "officeqa_opd_teacher_think replaces the "
            "empty think block in the teacher request with a teacher-written "
            "one and transfers scores onto the real rollout — it requires the "
            "on-policy base, set officeqa_opd_teacher_no_think alongside it"
        )
    if modes.teacher_think and modes.prose_as_think:
        raise ValueError(
            "officeqa_opd_teacher_think composes with the plain or the tail "
            "request, not with prose-as-think — drop "
            "officeqa_opd_prose_as_think"
        )
    if modes.noaction_window < 0:
        raise ValueError(
            "officeqa_opd_noaction_window is a token count — it must be >= 0"
        )
    if modes.noaction_window and not modes.noaction_penalty:
        raise ValueError(
            "officeqa_opd_noaction_window has no effect without "
            "officeqa_opd_noaction_penalty > 0"
        )
    if modes.noaction_penalty < 0.0:
        raise ValueError(
            "officeqa_opd_noaction_penalty is a magnitude — it must be >= 0"
        )
    if modes.noaction_penalty and not modes.teacher_no_think:
        raise ValueError(
            "officeqa_opd_noaction_penalty replaces a turn's teacher scores, "
            "which only exist on the on-policy base — set "
            "officeqa_opd_teacher_no_think alongside it"
        )
    if modes.noaction_penalty and not bool(
        getattr(args, "officeqa_opd_pure", False)
    ):
        raise ValueError(
            "officeqa_opd_noaction_penalty is a second, non-distillation term; "
            "outside officeqa_opd_pure its tokens would also carry the "
            "estimator's reward-derived advantage — set officeqa_opd_pure"
        )
    if modes.noaction_penalty and bool(
        getattr(args, "officeqa_opd_contrast", False)
    ):
        raise ValueError(
            "officeqa_opd_noaction_penalty marks tokens with a POSITIVE "
            "teacher log-prob, a value COPD's contrast channel uses "
            "legitimately — they cannot share teacher_log_probs"
        )
    return modes


class _TurnScorer:
    """Fires one teacher-scoring task per recorded turn as the episode runs."""

    def __init__(
        self,
        client: Any,
        affinity_key: str | None = None,
        tokenizer: Any = None,
        modes: PerTurnModes | None = None,
    ):
        modes = modes or PerTurnModes()
        self._client = client
        # The teacher request is the same reasoning-free shape for AOD and its
        # on-policy variant; only what gets trained differs, downstream.
        action_only = modes.command_only_request
        tail_loss = modes.tail_loss
        prose_as_think = modes.prose_as_think
        self._tail_think_tokens = modes.tail_think_tokens
        self._mask_fence_delims = modes.mask_fence_delims
        self._teacher_think_text = modes.teacher_think_text
        self._teacher_think = modes.teacher_think
        self._think_max_tokens = modes.think_max_tokens
        self._think_temperature = modes.think_temperature
        # Teacher deliberations started while the student was still
        # generating the same turn, keyed by the turn's exact prompt ids.
        self._prestarted: dict[tuple[int, ...], asyncio.Task] = {}
        # Pins every scoring call of this episode to one teacher endpoint
        # when the client has several, so that endpoint's prefix cache stays
        # warm across the episode's turns (no-op with a single endpoint).
        self._affinity_key = affinity_key
        self._tokenizer = tokenizer
        self._action_only = action_only
        self._tail_loss = tail_loss
        self._prose_as_think = prose_as_think
        if action_only and tokenizer is None:
            raise ValueError("action-only scoring requires the actor tokenizer")
        # One entry per session turn, in order:
        # (output_ids, task or None, plan or None). ``plan`` is None on the
        # legacy full-turn path.
        self.scored_turns: list[
            tuple[list[int], asyncio.Task | None, _ScorePlan | None]
        ] = []

    def fire_new(self, session_turns: list[Any]) -> None:
        for session_turn in session_turns[len(self.scored_turns) :]:
            turn = session_turn.turn
            prompt_ids = list(turn.prompt_ids or [])
            output_ids = list(turn.output_ids or [])
            task: asyncio.Task | None = None
            plan: _ScorePlan | None = None
            if output_ids and prompt_ids:
                if self._action_only:
                    plan = _action_only_plan(
                        self._tokenizer,
                        prompt_ids,
                        output_ids,
                        tail_loss=self._tail_loss,
                        prose_as_think=self._prose_as_think,
                        tail_think_tokens=self._tail_think_tokens,
                        mask_fence_delims=self._mask_fence_delims,
                        teacher_think_text=self._teacher_think_text,
                    )
                    if plan.tail_start is not None:
                        if self._teacher_think:
                            task = asyncio.create_task(
                                self._score_with_teacher_think(
                                    prompt_ids, output_ids, plan
                                )
                            )
                        else:
                            prefix_ids = list(plan.prefix_ids)
                            task = asyncio.create_task(
                                self._client.score_tokens(
                                    prompt_ids
                                    + prefix_ids
                                    + plan.request_ids(output_ids),
                                    logprob_start_len=len(prompt_ids)
                                    + len(prefix_ids)
                                    - 1,
                                    affinity_key=self._affinity_key,
                                )
                            )
                else:
                    task = asyncio.create_task(
                        self._client.score_tokens(
                            prompt_ids + output_ids,
                            logprob_start_len=len(prompt_ids) - 1,
                            affinity_key=self._affinity_key,
                        )
                    )
            self.scored_turns.append((output_ids, task, plan))

    def prestart_think(self, prompt_ids: list[int]) -> None:
        """Start the teacher's deliberation while the student generates.

        Wired as the backend's ``on_turn_prompt`` observer: the deliberation
        depends only on the turn's prompt, so it can decode concurrently with
        the student's own turn and be ready by scoring time. Failures resolve
        to ``None`` inside the task — the scorer then degrades exactly as the
        non-overlapped path would.
        """
        if not self._teacher_think:
            return
        key = tuple(prompt_ids)
        if key not in self._prestarted:
            self._prestarted[key] = asyncio.create_task(
                self._generate_think_prefix(list(prompt_ids))
            )

    async def _generate_think_prefix(
        self, prompt_ids: list[int]
    ) -> list[int] | None:
        """The teacher's deliberation as prefix ids, or None to fall back."""
        generation_input = _think_generation_input(self._tokenizer, prompt_ids)
        if generation_input is None:
            return None
        try:
            think_ids = await self._client.generate_tokens(
                generation_input,
                max_new_tokens=self._think_max_tokens,
                temperature=self._think_temperature,
                stop=[_think_dialect(self._tokenizer).close],
                affinity_key=self._affinity_key,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 — degrade, don't die
            logger.warning(
                "teacher think generation failed (%s: %s); scoring with "
                "the empty think block instead",
                type(error).__name__,
                error,
            )
            return None
        think_text = self._tokenizer.decode(think_ids, skip_special_tokens=False)
        close = think_text.find(_think_dialect(self._tokenizer).close)
        if close >= 0:
            think_text = think_text[:close]
        think_text = think_text.strip("\n")
        if not think_text:
            return None
        prefix_text = _think_prefix_text(self._tokenizer, prompt_ids, think_text)
        return list(self._tokenizer.encode(prefix_text, add_special_tokens=False))

    async def _score_with_teacher_think(
        self,
        prompt_ids: list[int],
        output_ids: list[int],
        plan: _ScorePlan,
    ) -> list[float]:
        """Let the teacher write its own think block, then score the tail.

        The scored sequence becomes ``context + <think>teacher's
        deliberation</think> + student's visible tail`` — the plan's tail
        slice (the fenced command alone, or the whole visible region under
        tail loss) and the same score alignment as the empty-think request, because
        ``logprob_start_len`` is computed from the prefix actually sent. The
        deliberation is usually already in flight (``prestart_think``); a
        turn without one generates it here. A failed generation degrades this
        turn to the empty-think request rather than killing the episode: a
        worse judge beats a dead run.
        """
        task = self._prestarted.pop(tuple(prompt_ids), None)
        generated = await (
            task if task is not None else self._generate_think_prefix(prompt_ids)
        )
        prefix_ids = list(plan.prefix_ids) if generated is None else generated
        return await self._client.score_tokens(
            prompt_ids + prefix_ids + plan.request_ids(output_ids),
            logprob_start_len=len(prompt_ids) + len(prefix_ids) - 1,
            affinity_key=self._affinity_key,
        )

    async def watch(self, backend: Any) -> None:
        while True:
            self.fire_new(backend.session_turns)
            await asyncio.sleep(_WATCH_INTERVAL_SEC)

    async def cancel(self) -> None:
        for _, task, _ in self.scored_turns:
            if task is not None and not task.done():
                task.cancel()
        for task in self._prestarted.values():
            if not task.done():
                task.cancel()
        for _, task, _ in self.scored_turns:
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        for task in self._prestarted.values():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._prestarted.clear()

    async def results(
        self,
    ) -> list[tuple[list[int], list[float], _ScorePlan | None]]:
        """Await every task; raise the first failure only after all settled."""
        scored: list[tuple[list[int], list[float], _ScorePlan | None]] = []
        first_error: BaseException | None = None
        for output_ids, task, plan in self.scored_turns:
            if task is None:
                scored.append((output_ids, [], plan))
                continue
            try:
                scored.append((output_ids, await task, plan))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
        # Deliberations prestarted for turns that never became scorable
        # (discarded plans, truncated episodes) have no consumer; reap them
        # so they stop holding teacher slots.
        for task in self._prestarted.values():
            if not task.done():
                task.cancel()
        for task in self._prestarted.values():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._prestarted.clear()
        return scored


def build_scored_samples(
    modes: PerTurnModes,
    *,
    session_turns: list[Any],
    scored_turns: list[tuple[list[int], list[float], _ScorePlan | None]],
    base_sample: Sample,
    reward: float,
    tokenizer: Any,
    extra_metadata: dict[str, Any],
    max_sample_tokens: int,
) -> tuple[list[Sample], str | None]:
    """Turn one scored episode into training samples under the mode knobs.

    Returns ``(samples, abort_reason)``: an empty sample list always carries
    the abort reason the caller should record. Alignment failures raise, like
    a scoring failure — silently mis-attached teacher log-probs would train
    on the wrong signal.
    """
    if modes.action_only:
        samples = _build_action_only_samples(
            session_turns,
            scored_turns,
            base_sample=base_sample,
            reward=reward,
            tokenizer=tokenizer,
            extra_metadata=extra_metadata,
            max_sample_tokens=max_sample_tokens,
        )
        if not samples:
            return [], "no_action_turns: no turn carried a scorable fenced command"
        return samples, None
    samples = build_training_samples(
        session_turns=session_turns,
        base_sample=base_sample,
        reward=reward,
        tokenizer=tokenizer,
        extra_metadata=extra_metadata,
        max_sample_tokens=max_sample_tokens,
    )
    if not samples:
        return [], "empty_trajectory: no trained model tokens were recorded"
    if modes.teacher_no_think:
        _apply_action_scores(
            samples,
            scored_turns,
            tokenizer=tokenizer,
            noaction_penalty=modes.noaction_penalty,
            noaction_window=modes.noaction_window,
        )
    else:
        _fill_teacher_log_probs(samples, scored_turns)
        if modes.mask_think:
            _mask_to_actions(
                samples, scored_turns, tokenizer, tail=modes.tail_loss
            )
        elif modes.mask_to_think:
            _mask_to_think(
                samples,
                scored_turns,
                tokenizer,
                modes.think_span,
                modes.think_span_tokens,
            )
    if modes.mask_think or modes.teacher_no_think or modes.mask_to_think:
        samples = [sample for sample in samples if any(sample.loss_mask)]
        if not samples:
            reason = (
                "no_think_turns: no turn carried a closed think block"
                if modes.mask_to_think or (modes.mask_think and modes.tail_loss)
                else "no_action_turns: no turn carried a scorable fenced command"
            )
            return [], reason
    return samples, None


def _walk_trained_spans(
    samples: list[Sample],
    scored_turns: list[tuple[list[int], list[float], _ScorePlan | None]],
):
    """Yield ``(sample, position, span_length, turn)`` per trained turn span.

    Trained (loss-mask-1) spans appear in turn order across the samples, one
    contiguous span per trained turn. Turns whose output never appears as a
    trained span (re-emitted as context by REALIGN) are skipped. A span cut
    short by ``max_sample_tokens`` yields its matching prefix. Any alignment
    mismatch raises: silently misaligned spans would attach the wrong
    teacher log-probs, or mask the wrong tokens.

    The walk reads a snapshot of each loss mask, so a consumer may rewrite
    ``sample.loss_mask`` while iterating.
    """
    queue_index = 0
    for sample in samples:
        response_length = int(sample.response_length or 0)
        tokens = list(sample.tokens or [])
        mask = list(sample.loss_mask or [])
        response_tokens = tokens[len(tokens) - response_length :]
        position = 0
        while position < response_length:
            if mask[position] != 1:
                position += 1
                continue
            match: tuple[tuple[list[int], list[float], _ScorePlan | None], int] | None = None
            while queue_index < len(scored_turns):
                turn = scored_turns[queue_index]
                output_ids = turn[0]
                queue_index += 1
                if not output_ids:
                    continue
                span = response_tokens[position : position + len(output_ids)]
                span_is_trained = all(
                    m == 1 for m in mask[position : position + len(span)]
                )
                full_match = span == output_ids and span_is_trained
                truncated_tail_match = (
                    len(span) < len(output_ids)
                    and position + len(span) == response_length
                    and span == output_ids[: len(span)]
                    and span_is_trained
                )
                if full_match or truncated_tail_match:
                    match = (turn, len(span))
                    break
            if match is None:
                raise ValueError(
                    "cannot align per-turn teacher scores to the sample's trained spans"
                )
            turn, span_length = match
            yield sample, position, span_length, turn
            position += span_length


def _fill_teacher_log_probs(
    samples: list[Sample],
    scored_turns: list[tuple[list[int], list[float], _ScorePlan | None]],
) -> None:
    """Map per-turn output log-probs into each sample's response layout."""
    filled = {id(s): [0.0] * int(s.response_length or 0) for s in samples}
    for sample, position, span_length, turn in _walk_trained_spans(
        samples, scored_turns
    ):
        output_ids, scores, plan = turn
        assert plan is None, "action-only turns fill samples directly"
        if len(scores) < span_length:
            raise ValueError(
                f"teacher returned {len(scores)} log-probs for a "
                f"{len(output_ids)}-token turn"
            )
        filled[id(sample)][position : position + span_length] = scores[:span_length]
    for sample in samples:
        sample.teacher_log_probs = filled[id(sample)]


def _mask_to_actions(
    samples: list[Sample],
    scored_turns: list[tuple[list[int], list[float], _ScorePlan | None]],
    tokenizer: Any,
    *,
    tail: bool = False,
) -> None:
    """Keep the loss on each turn's fenced command; drop it everywhere else.

    Unlike action-only distillation this leaves the sequence itself alone:
    the teacher scored the whole turn with the student's reasoning in place,
    and ``teacher_log_probs`` still cover every position. Only the loss mask
    narrows, so reasoning tokens keep informing the teacher's view of the
    action without being trained toward it. Turns with no usable fence lose
    their loss entirely.

    With ``tail`` (``officeqa_opd_mask_think`` + ``officeqa_opd_tail_loss``,
    the ``-maskthink-tail`` arm) the kept region is the whole visible tail
    after the think block instead — prose, fence and turn end, exactly what
    the ``-tail`` variants train under the reasoning-free request — while the
    teacher's scores still come from the full turn with the student's
    reasoning in view. A fence-less turn then keeps its loss; only a turn
    whose think block never closes loses it.
    """
    masks = {id(s): list(s.loss_mask or []) for s in samples}
    for sample, position, span_length, turn in _walk_trained_spans(
        samples, scored_turns
    ):
        found = (
            _tail_span(tokenizer, turn[0])
            if tail
            else _fence_span(tokenizer, turn[0])
        )
        keep: set[int] = set()
        if found is not None:
            for span_lo, span_hi in found[1]:
                keep.update(range(span_lo, min(span_hi, span_length)))
        mask = masks[id(sample)]
        for offset in range(span_length):
            if offset not in keep:
                mask[position + offset] = 0
    for sample in samples:
        sample.loss_mask = masks[id(sample)]


def _mask_to_think(
    samples: list[Sample],
    scored_turns: list[tuple[list[int], list[float], _ScorePlan | None]],
    tokenizer: Any,
    span: str,
    span_tokens: int,
) -> None:
    """Keep the loss on each turn's distilled span; drop it everywhere else.

    The mirror image of :func:`_mask_to_actions`: the teacher scored the
    whole real rollout, ``teacher_log_probs`` cover every position, and only
    the loss mask narrows — here to the deliberation (``span="all"``) or to a
    positional window of the response (``first``/``last``). Under ``all``,
    fence-less turns with a closed think block KEEP their loss and turns whose
    think never closes lose it; the positional windows keep every turn.
    """
    masks = {id(s): list(s.loss_mask or []) for s in samples}
    for sample, position, span_length, turn in _walk_trained_spans(
        samples, scored_turns
    ):
        found = _distill_span(tokenizer, turn[0], span, span_tokens)
        keep: set[int] = set()
        if found is not None:
            for span_lo, span_hi in found:
                keep.update(range(span_lo, min(span_hi, span_length)))
        mask = masks[id(sample)]
        for offset in range(span_length):
            if offset not in keep:
                mask[position + offset] = 0
    for sample in samples:
        sample.loss_mask = masks[id(sample)]


def _apply_action_scores(
    samples: list[Sample],
    scored_turns: list[tuple[list[int], list[float], _ScorePlan | None]],
    *,
    tokenizer: Any = None,
    noaction_penalty: float = 0.0,
    noaction_window: int = 0,
) -> None:
    """Attach reasoning-free teacher scores to the student's REAL sequence.

    The teacher scored ``context + empty-think + visible-tail`` (the AOD
    request), so its log-probs are indexed from ``plan.tail_start``. The
    samples here are the unmodified rollout, reasoning included, so the
    fenced-command tokens are the same ids in both and their teacher scores
    transfer positionally. Everything outside the fence loses its loss mask
    and keeps a 0.0 teacher log-prob, which is inert under a 0 mask.

    This is the deliberately mismatched pairing: student log-probs come from
    the on-policy sequence it actually generated (reasoning ahead of the
    action), teacher log-probs from the same action without that reasoning.

    ``noaction_penalty`` (default 0.0, off) makes a turn that issued no
    command carry a fixed negative advantage over its whole length INSTEAD of
    teacher scores. ``_fence_span`` returning None is the test, and it covers
    both kinds: a turn whose think block never closed (no visible region at
    all, otherwise dropped from the objective entirely) and one that closed it
    but produced no runnable command (otherwise trained normally against the
    teacher). Both are wasted steps, and on the SSB1 q38 arms they grow from
    ~3% of turns to 45-65%.

    ``noaction_window`` (tokens, 0 = whole turn) bounds WHERE the penalty
    lands: the last N tokens of the turn, each carrying the full magnitude.
    Both earlier builds coupled the per-token magnitude to the turn's length
    and the policy escaped along that axis, in opposite directions:

    * constant value on EVERY token -> a turn's total cost is proportional to
      its length, so the cheapest response is a shorter actionless turn.
      Measured on ``noact0.1``: actionless turns fell 1,947 -> 514 tokens in
      nine steps while their share rose 6.5% -> 64.7%.
    * magnitude spread over the whole turn (constant total) -> the per-token
      value falls as 1/length, and policy gradient acts per token, so the
      cheapest response is a LONGER actionless turn. Measured on
      ``noactpt20``: actionless turns grew 498 -> 3,396 tokens by step 24,
      3.4x the acting turns, with the share at 58.9% against ~46% for the
      same arm with no penalty at all.

    A fixed window at fixed value closes both: the token count does not depend
    on the turn's length, so neither truncating nor inflating changes what an
    actionless turn is charged. The one residual is a turn SHORTER than the
    window, which pays only ``span * penalty``; with a window well below the
    observed turn lengths that requires collapsing the turn, a change large
    enough to show up in mean turn length immediately.

    The per-token value is written into ``teacher_log_probs`` as a strictly
    POSITIVE value — a marker, not a log-prob, since real log-probs are <= 0
    and no other path can emit one. ``apply_opd_kl_to_advantages`` reads it as
    a fixed reverse KL, giving each windowed token the constant advantage
    ``-opd_kl_coef * noaction_penalty``.
    """
    filled = {id(s): [0.0] * int(s.response_length or 0) for s in samples}
    masks = {id(s): list(s.loss_mask or []) for s in samples}
    # Counters land in sample metadata whenever the penalty is on: the walk is
    # the one stage a dump cannot otherwise show, and an earlier build failed
    # silently there.
    walked = no_action_turns = penalised = penalised_tokens = 0
    tail_think_turns = tail_think_tokens = 0
    delim_masked_tokens = 0
    for sample, position, span_length, turn in _walk_trained_spans(
        samples, scored_turns
    ):
        output_ids, scores, plan = turn
        walked += 1
        no_action = (
            noaction_penalty > 0.0
            and tokenizer is not None
            and span_length > 0
            and _fence_span(tokenizer, output_ids) is None
        )
        if no_action:
            no_action_turns += 1
        keep: set[int] = set()
        if no_action:
            # Fixed window at the turn's end, full magnitude on each token, so
            # the charge depends on neither the turn's length nor where the
            # model chooses to stop. window=0 keeps the whole turn.
            window = span_length
            if noaction_window > 0:
                window = min(noaction_window, span_length)
            for offset in range(span_length - window, span_length):
                filled[id(sample)][position + offset] = noaction_penalty
                keep.add(offset)
            penalised += 1
            penalised_tokens += window
        elif plan is not None and plan.tail_start is not None:
            if plan.tail_think:
                tail_think_turns += 1
                tail_think_tokens += max(0, min(plan.spans[0][1], span_length) - plan.tail_start)
            for span_lo, span_hi in plan.spans:
                for offset in range(span_lo, min(span_hi, span_length)):
                    if plan.is_masked(offset):
                        delim_masked_tokens += 1
                        continue
                    score_index = plan.score_index(offset)
                    if not 0 <= score_index < len(scores):
                        raise ValueError(
                            f"teacher returned {len(scores)} log-probs; fence "
                            f"token {offset} maps outside the scored tail"
                        )
                    filled[id(sample)][position + offset] = scores[score_index]
                    keep.add(offset)
        mask = masks[id(sample)]
        for offset in range(span_length):
            if offset not in keep:
                mask[position + offset] = 0
    for sample in samples:
        sample.teacher_log_probs = filled[id(sample)]
        sample.loss_mask = masks[id(sample)]
        # Written unconditionally: the value of ``penalty`` here is what the
        # function actually received, so an arm that expected a penalty and
        # reads 0.0 has a delivery problem, while a missing key means this
        # function never ran. Both were indistinguishable when the block was
        # gated on the penalty being on.
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        metadata["fence_delims_masked"] = delim_masked_tokens
        metadata["noaction"] = {
            "penalty": noaction_penalty,
            "turns": len(scored_turns),
            "walked": walked,
            "no_action": no_action_turns,
            "penalised": penalised,
            # Per-turn charging means the marked VALUE varies with span, so
            # the token count is the only way a dump shows how much of the
            # batch the penalty covers.
            "penalised_tokens": penalised_tokens,
            "window": noaction_window,
        }
        # Tail-think coverage: how many trained turns pulled thought tokens in
        # and how many thought tokens that was (0/0 when the knob is off or
        # every thought was empty).
        metadata["tail_think"] = {
            "turns": tail_think_turns,
            "tokens": tail_think_tokens,
        }
        sample.metadata = metadata


def _build_action_only_samples(
    session_turns: list[Any],
    scored_turns: list[tuple[list[int], list[float], _ScorePlan | None]],
    *,
    base_sample: Sample,
    reward: float,
    tokenizer: Any,
    extra_metadata: dict[str, Any] | None,
    max_sample_tokens: int,
) -> list[Sample]:
    """One single-turn training sample per fence-bearing turn (AOD).

    Each sample's sequence is ``context + empty-think + visible-tail`` — the
    SAME modified sequence the teacher scored — so the student's action
    tokens are trained under the reasoning-free conditioning, never against
    it. The loss mask keeps only the fenced-command span; the empty think
    block sits in the prompt region (never trained). Turns whose plan was
    discarded contribute nothing. ``rollout_log_probs`` keep the generation-
    time values of the tail tokens (sampled under the ORIGINAL reasoning-full
    context): a train-infer gap by construction, inert unless
    ``--use-rollout-logprobs``/TIS is enabled (it must not be).
    """
    samples: list[Sample] = []
    manager = TrajectoryManager()
    for index, (session_turn, (output_ids, scores, plan)) in enumerate(
        zip(session_turns, scored_turns, strict=True)
    ):
        if plan is None or plan.tail_start is None:
            continue
        if plan.tail_think:
            raise ValueError(
                "tail-think plans keep the student's truncated thought in the "
                "request and only exist on the on-policy path; the "
                "ablated-sample builder cannot host them"
            )
        turn = session_turn.turn
        prompt_ids = list(turn.prompt_ids or []) + list(plan.prefix_ids)
        tail_ids = output_ids[plan.tail_start : plan.tail_end]
        tail_log_probs = (
            list(turn.output_log_probs)[plan.tail_start : plan.tail_end]
            if turn.output_log_probs
            else []
        )
        (span_lo, span_hi) = plan.spans[0]
        sid = f"turn-{index}"
        try:
            manager.record_turn(
                sid,
                turn=TurnRecord(
                    prompt_ids=prompt_ids,
                    output_ids=tail_ids,
                    finish_reason=turn.finish_reason or "stop",
                    output_log_probs=tail_log_probs,
                ),
                prompt_messages=session_turn.prompt_messages,
                response_message={
                    "role": "assistant",
                    "content": tokenizer.decode(tail_ids, skip_special_tokens=False),
                },
            )
            turn_samples = manager.get_trajectory(
                sid,
                base_sample=base_sample,
                reward=reward,
                extra_metadata={
                    **(extra_metadata or {}),
                    "action_turn": index,
                    # AOD samples drop the turn's reasoning, so their token
                    # count no longer reflects what the student generated;
                    # metrics read this for the episode's true output cost.
                    "generated_tokens": len(output_ids),
                },
                max_sample_tokens=max_sample_tokens,
            )
        finally:
            manager.drop_session(sid)
        for sample in turn_samples:
            response_length = int(sample.response_length or 0)
            if len(scores) < response_length:
                raise ValueError(
                    f"teacher returned {len(scores)} log-probs for a "
                    f"{len(tail_ids)}-token action-only tail"
                )
            # Restrict the loss to the fenced command (span is relative to
            # the full turn output; the sample response is the tail).
            fence_lo = span_lo - plan.tail_start
            fence_hi = span_hi - plan.tail_start
            sample.loss_mask = [
                mask
                if fence_lo <= position < fence_hi
                and not plan.is_masked(position + plan.tail_start)
                else 0
                for position, mask in enumerate(sample.loss_mask)
            ]
            if not any(sample.loss_mask):
                continue  # truncation cut the fence away entirely
            sample.teacher_log_probs = list(scores[:response_length])
            sample.response = (
                tokenizer.decode(
                    sample.tokens[-response_length:], skip_special_tokens=False
                )
                if response_length and sample.tokens
                else ""
            )
            samples.append(sample)
    return samples


async def generate(
    args: Any,
    base_sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    """Run one OfficeQA rollout with per-turn teacher scoring."""
    if evaluation or not _teacher_scoring_enabled(args):
        return await default_rollout.generate(
            args, base_sample, sampling_params, evaluation=evaluation
        )

    from officeqa_rl.training.vertex_teacher_client import get_teacher_client

    try:
        task = task_from_slime_fields(
            base_sample.prompt, base_sample.label, base_sample.metadata
        )
        if task.answer is None:
            raise ValueError(f"sample {task.uid} has no label to grade")
        tolerance = _tolerance(base_sample)
    except (TypeError, ValueError) as error:
        return _abort(base_sample, f"invalid_task: {error}")

    run_id = _make_run_id(base_sample)
    routing_key = (
        base_sample.session_id
        if getattr(args, "router_policy", None) == "consistent_hashing"
        else None
    )
    try:
        backend = _create_backend(args, evaluation=evaluation, routing_key=routing_key)
        agent_config = agent_config_from_args(args)
        # Horizon curriculum: train on the first N turns of the real episode.
        # The cap lives on the agent, not on officeqa_step_limit, so it applies
        # to training rollouts only — evaluation delegates to the default hook
        # above and keeps the full step limit, staying comparable across arms.
        # Commands still execute, so turns 2..N see genuine observations.
        train_turns = int(getattr(args, "officeqa_opd_train_turns", 0) or 0)
        if train_turns > 0:
            agent_config = replace(agent_config, step_limit=train_turns)
        agent = OfficeQAAgent(
            backend, agent_config, sampling_params=sampling_params
        )
        environment_config = environment_config_from_args(args)
        client = get_teacher_client(args)
    except (TypeError, ValueError) as error:
        return _abort(base_sample, f"invalid_rollout_config: {error}", run_id=run_id)

    try:
        modes = parse_per_turn_modes(args)
    except ValueError as error:
        return _abort(
            base_sample, f"invalid_rollout_config: {error}", run_id=run_id
        )
    scorer = _TurnScorer(
        client,
        affinity_key=run_id,
        tokenizer=backend.tokenizer,
        modes=modes,
    )
    if modes.teacher_think:
        backend.on_turn_prompt = scorer.prestart_think
    watch_task = asyncio.create_task(scorer.watch(backend))
    try:
        try:
            async with FullCorpusEnvironment(
                environment_config, task, run_id=run_id
            ) as env:
                result = await agent.run(env)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # sandbox/workspace failures must not spread
            await scorer.cancel()
            return _abort(
                base_sample,
                f"environment_failure: {type(error).__name__}: {error}",
                run_id=run_id,
            )
    finally:
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watch_task

    # Turns recorded after the watcher's last poll (always including the
    # final turn) get their scoring fired here.
    scorer.fire_new(backend.session_turns)

    terminal_status = _STATUS_BY_EXIT.get(result.status)
    if terminal_status is None:
        await scorer.cancel()
        return _abort(base_sample, f"agent_failure: {result.status}", run_id=run_id)

    if result.status == "submitted":
        reward = score_answer(
            ground_truth=task.answer,
            predicted=result.submission,
            tolerance=tolerance,
        )
    else:
        reward = 0.0

    extra_metadata = {
        **(base_sample.metadata or {}),
        "uid": task.uid,
        "run_id": run_id,
        "submission": result.submission,
        "exit_status": result.status,
        "steps": result.steps,
        "episode_output_tokens": _episode_output_tokens(backend.session_turns),
        "forced_close_turns": _forced_close_turns(backend.session_turns),
    }
    # Teacher-scoring failures are deliberately fatal, exactly as in the
    # per-sample path: log the reason, then stop the run.
    try:
        scored_turns = await scorer.results()
    except asyncio.CancelledError:
        raise
    except Exception as error:
        await _log_teacher_failure(args, base_sample, run_id, error)
        raise

    # Score/sample misalignment is as fatal as a scoring failure: log it
    # to the failure log, then stop the run.
    try:
        samples, abort_reason = build_scored_samples(
            modes,
            session_turns=backend.session_turns,
            scored_turns=scored_turns,
            base_sample=base_sample,
            reward=reward,
            tokenizer=backend.tokenizer,
            extra_metadata=extra_metadata,
            max_sample_tokens=backend.max_context_len,
        )
    except Exception as error:
        await _log_teacher_failure(args, base_sample, run_id, error)
        raise
    if abort_reason is not None:
        return _abort(base_sample, abort_reason, run_id=run_id)

    for sample in samples:
        sample.status = (
            Sample.Status.TRUNCATED
            if sample.metadata.get("truncated")
            else terminal_status
        )

    samples[0].prefix_cache_info.cached_tokens = backend.cached_prompt_tokens
    samples[0].prefix_cache_info.total_prompt_tokens = backend.total_prompt_tokens

    return samples[0] if len(samples) == 1 else samples


__all__ = [
    "PerTurnModes",
    "build_scored_samples",
    "generate",
    "parse_per_turn_modes",
]
