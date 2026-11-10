# GRPO variable fan-out: rollout crash and silently wrong advantage normalization

## TL;DR

The OfficeQA GRPO baseline produces a **variable number of training samples per
rollout execution** (fan-out). Slime's batch-assembly pipeline does not handle
this correctly in two independent places:

| # | Bug | Effect | Visible? |
|---|-----|--------|----------|
| 1 | Flattening infers the batch shape from the **first element only** | `TypeError: 'Sample' object is not iterable` — training crashed at step 104 | Loud (crash) |
| 2 | Default reward normalization falls back to **one global group** and counts each rollout once **per sibling sample** | Every GRPO advantage since step 0 was computed incorrectly | Silent |

**Severity: critical.** Bug 1 stopped the run. Bug 2 invalidates all 103
completed optimizer updates: the run did not train GRPO, it trained something
closer to REINFORCE with a global batch baseline, weighted by fan-out size.

**Recovery: do not resume from any checkpoint. Restart from the base model
after both fixes land** (details at the end).

## Affected run

- Log: `results/grpo_baseline_20260715_071612.log`
- W&B run ID: `nyqmudip`
- Config: `rollout_batch_size = 8`, `n_samples_per_prompt = 8`
  (nominal 64 rollout executions per batch), `advantage_estimator = grpo`,
  `rewards_normalization = True`, `custom_reward_post_process_path = None`
- Last completed training update: step 103
- Last intact checkpoint: `iter_0000099`
- Terminal failure: 2026-07-17 15:17:20 UTC

The repeated `TorchMemorySaver::malloc return OOM` messages in the log are
**not** the terminal error; training and rollouts continued normally after
them.

---

## Background: why one rollout becomes many training samples

To understand both bugs you first need the fan-out mechanism. This section has
no bug in it — it is the intended behavior of our setup.

One OfficeQA rollout execution is a multi-turn agent episode: up to 120 turns
of the model emitting a command, the sandbox returning an observation, repeat.
A training sample, however, must be **one contiguous token sequence** (tokens +
loss mask + logprobs). Whether one episode fits into one contiguous sequence
depends on whether each turn's prompt starts with the previous turn's exact
tokens ("prefix stability").

With Qwen3.5 it does not. Its chat template deletes previous turns' `<think>`
blocks every time it renders the next prompt:

```
what the model SAMPLED at turn 1:        what the template RENDERS as turn 2's prompt:

<|im_start|>assistant                    <|im_start|>assistant
<think>                                  ls corpus/            <- <think> block gone
I should list the files first...         <|im_end|>
</think>                                 <|im_start|>user
ls corpus/                               bulletin_2019.md ...
<|im_end|>                               <|im_end|>
                                         <|im_start|>assistant
                                         <think> ...
```

Turn 1's sampled tokens never reappear inside turn 2's prompt, so the two
turns cannot be joined into one token sequence with a valid loss mask. Slime's
`TrajectoryManager` (`slime/agent/trajectory.py`) handles this by emitting
**one training sample per contiguous token chain** — in practice roughly one
per assistant turn:

```
1 rollout execution (one agent episode, ~40 turns)
        │
        ▼  chat template breaks the token prefix at every turn
┌───────────────────────────────────────────────┐
│ sample 1: [prompt₁ | turn-1 output]           │
│ sample 2: [prompt₂ | turn-2 output]           │   "siblings":
│ sample 3: [prompt₃ | turn-3 output]           │   same rollout_id,
│   ...                                         │   same reward,
│ sample K: [promptK | turn-K output]           │   K varies per episode
└───────────────────────────────────────────────┘
```

K observed in this run: between 1 and 50. So a nominal batch of
8 prompts × 8 rollouts = 64 executions actually produces ~2,000–2,300 flat
training samples, and the number is different every batch.

Our rollout hook (`officeqa_rl.training.rollout.generate`) returns these
samples to slime as:

```python
return samples[0] if len(samples) == 1 else samples   # Sample OR list[Sample]
```

and its abort paths return a bare `Sample`. Both shapes are explicitly legal:
slime's own `generate_and_rm` is typed `-> Sample | list[Sample]`
(`slime/rollout/sglang_rollout.py:229`). Everything below is slime's core
failing to honor that contract downstream.

---

## Bug 1: the crash — flattening decides the shape from the first element

### What the code does

After all rollouts in a batch finish, slime flattens the nested result in
`RolloutManager._get_rollout_data` (`slime/ray/rollout.py:640`):

```python
while isinstance(data[0], list):                 # looks ONLY at data[0]
    data = list(itertools.chain.from_iterable(data))  # iterates EVERY element
```

The check looks at one element; the flattening iterates all of them. That is
only safe if every element has the same shape.

### Worked example

Take a tiny batch: 3 rollout executions, where execution 2 produced a
single-sample trajectory (or aborted) and therefore came back as a bare
`Sample`:

```
data = [                       ← the batch (1 prompt group shown)
  [                            ← group
    [S, S, S],                 ← execution 1: 3 siblings (list)
    S,                         ← execution 2: 1 sample  (bare Sample)
    [S, S],                    ← execution 3: 2 siblings (list)
  ]
]

pass 1:  data[0] is a list → flatten one level
data = [ [S,S,S],  S,  [S,S] ]

pass 2:  data[0] is a list → flatten again
         chain.from_iterable iterates each element:
           [S,S,S] → fine (it is a list)
           S       → TypeError: 'Sample' object is not iterable   ← CRASH
```

Reordering does not save you — it just changes the failure mode:

```
data = [ S,  [S,S,S],  [S,S] ]

pass 2:  data[0] is a Sample → loop exits immediately
result:  a "flat" list that still contains nested lists.
         Downstream code (rewards, tokens, lengths) reads Sample attributes
         off a list object and fails later, further from the cause.
```

The shape validator that runs just before flattening
(`_validate_rollout_id_annotated`, `slime/ray/rollout.py:897`) has the same
blind spot: `if node and isinstance(node[0], Sample): ...` — it also inspects
only the first element.

### Why it took 103 steps to trigger

The mixed shape requires at least one execution in the batch to return a bare
`Sample` — i.e. a trajectory that produced exactly one sample, or an abort.
In every one of the 103 saved batches (~6,650 executions), all 64 executions
produced multi-sample lists, so the shape was uniformly `list[list[list]]` and
the loop worked. Step 104 contained the run's first single-sample execution
(the log shows zero abort warnings, so most likely a legitimately short
episode), which made the batch heterogeneous and crashed it:

```text
File "slime/ray/rollout.py", line 641, in _get_rollout_data
    data = list(itertools.chain.from_iterable(data))
TypeError: 'Sample' object is not iterable
```

---

## Bug 2: the silent one — GRPO statistics computed over the wrong group

This bug does not crash anything. It affected **every** batch from step 0, and
it is the reason the completed checkpoints are unusable.

### What GRPO is supposed to compute

GRPO's advantage for a rollout is its reward compared to **the other rollouts
of the same prompt**:

```
for each prompt:                              8 rollouts of the SAME prompt
    advantage_i = (r_i − mean(r₁..r₈)) / std(r₁..r₈)
```

The prompt group is the whole point: a hard prompt where everything scores 0
produces zero advantage (no gradient), and an easy prompt where everything
scores 1 also produces zero advantage. Only *contrast within a prompt* trains.

### What the default post-processor actually did

`_post_process_rewards` (`slime/ray/rollout.py:668-687`) receives the **flat**
sample list — fan-out already applied, structure already discarded — and tries
to recover the prompt groups by reshaping:

```python
rewards = torch.tensor(raw_rewards)                     # shape: (2291,)
if rewards.shape[-1] == n_samples_per_prompt * rollout_batch_size:   # 2291 == 64? no
    rewards = rewards.reshape(-1, n_samples_per_prompt)  # correct branch, never taken
else:
    rewards = rewards.view(-1, rewards.shape[-1])        # ← view(-1, 2291) = ONE group
mean = rewards.mean(dim=-1, keepdim=True)                # mean over the whole batch
```

`view(-1, 2291)` on a 2,291-element vector produces shape `(1, 2291)`: **one
group containing every sample of every prompt**. No error is raised. Two
distinct distortions follow.

### Distortion A: statistics cross prompt boundaries

Miniature example — 2 prompts, 2 rollouts each, no fan-out yet:

```
                     reward     correct GRPO group     correct advantage
prompt A, rollout 1    1.0      A: mean 0.5            +1  (scaled by 1/std)
prompt A, rollout 2    0.0      A: mean 0.5            −1
prompt B, rollout 1    0.0      B: mean 0.0             0   ← no contrast,
prompt B, rollout 2    0.0      B: mean 0.0             0     no gradient

what the fallback computed (one global group, mean = 0.25):
prompt A, rollout 1   +0.75
prompt A, rollout 2   −0.25
prompt B, rollout 1   −0.25   ← hard prompt now actively pushed DOWN
prompt B, rollout 2   −0.25     even though nothing distinguishes its rollouts
```

Every all-zero prompt group (very common early in this run: most episodes
failed) stops being neutral and instead contributes a uniform negative push on
everything it sampled.

### Distortion B: rollouts are weighted by how many siblings they emitted

This is a **new miniature example**, not a continuation of Distortion A above.
All four rollout executions below belong to one prompt; using only one prompt
removes the cross-prompt error so that sibling weighting can be examined by
itself. First, consider the calculation GRPO is supposed to perform:

```
prompt C, rollout 1: reward 1.0, 3 siblings
prompt C, rollout 2: reward 0.0, 1 sibling
prompt C, rollout 3: reward 0.0, 2 siblings
prompt C, rollout 4: reward 0.0, 2 siblings

correct input to the statistics: [1.0, 0.0, 0.0, 0.0]
correct mean:                    1/4 = 0.25

centered advantage of rollout 1:    1.0 − 0.25 = +0.75
centered advantage of rollouts 2–4: 0.0 − 0.25 = −0.25
```

Each rollout contributes exactly one reward to the mean and standard
deviation. After normalization, the resulting rollout advantage is broadcast
to all of that rollout's siblings. The sibling count does not participate in
the calculation: changing rollout 1 from 3 siblings to 50 must leave every
advantage unchanged.

The buggy calculation instead operates on the flat sample list. Fan-out has
copied each rollout's reward onto every sibling, so the actual input is:

```
buggy flat input: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

buggy mean: 3/8 = 0.375

centered advantage of rollout 1:    1.0 − 0.375 = +0.625
centered advantage of rollouts 2–4: 0.0 − 0.375 = −0.375
```

In other words, the buggy mean is sibling-count weighted:

```
             Σ_e K_e R_e
buggy mean = ───────────
               Σ_e K_e
```

The standard deviation is distorted by the same weighting. If rollout 1 emits
50 siblings instead of 3, with all rewards and policy behavior unchanged, the
flat list contains 50 copies of `1.0` and five copies of `0.0`:

```
buggy mean = 50/55 ≈ 0.909

centered advantage of rollout 1:    1.0 − 0.909 ≈ +0.091
centered advantage of rollouts 2–4: 0.0 − 0.909 ≈ −0.909
```

Identical policy behavior, wildly different gradient, purely because one
episode was longer. In this run sibling counts ranged from 1 to 50, so this is
not a corner case — it is the operating regime.

### What the run actually optimized (the math)

Notation for one batch:

```
P = 8   prompts,  G = 8  rollout executions per prompt,  E = P·G = 64 executions
R_e     reward of execution e            (shared by all of its siblings)
K_e     number of sibling samples of e   (1 to 50 in this run)
N       = Σ_e K_e  flat samples          (~2,291, varies per batch)
T_e     total loss-masked tokens of e    (summed over its siblings)
```

**Intended GRPO advantage** — statistics over the G executions of the same
prompt p(e):

```
                R_e − μ_p(e)                    1
A_e  =  ─────────────────────────,   μ_p  =  ─────  Σ      R_e' ,   σ_p = std over the same G rewards
                σ_p(e) + ε                      G    e'∈ p
```

**What the code computed** — statistics over the flat sample list, i.e. one
global group in which execution e appears K_e times:

```
         1                                Σ_e K_e (R_e − μ̂)²                R_e − μ̂
μ̂  =  ─────  Σ_e  K_e R_e ,     σ̂² ≈  ─────────────────────── ,    Â_e = ───────────
         N                                        N                          σ̂ + ε
```

The loss side, in contrast, was **correct** (verified in the code): slime's
per-rollout reducer divides every sibling's token losses by the rollout's
total token count `T_e` (`rollout_mask_sums`, `slime/ray/rollout.py:747-756`),
so each execution enters the gradient with weight 1 regardless of K_e.
Ignoring PPO clipping, the applied update per batch was therefore:

```
                                1
g  =  Σ_e  Â_e  ·  ─────  Σ  ∇ log π_θ(y_t | context)
                               T_e   t∈e

               1
   =  ─────  Σ_e  (R_e − μ̂) · ∇ log π_θ(τ_e)        (token-mean per execution)
              σ̂
```

At the policy-gradient level, this is **REINFORCE with a shared scalar
baseline** `b = μ̂` and a global scale `1/σ̂` — equivalently, "GRPO with a
single group containing the entire batch, where each execution's reward
enters the baseline statistics K_e times." It is not GRPO, because the
baseline is not conditioned on the prompt.

It also has the broad shape of **plain REINFORCE++**: it is critic-free, uses
a PPO-style clipped policy loss, and centers/scales advantages using global
batch statistics. It is **not REINFORCE++ w/ Baseline**, despite the similar
name. That variant first subtracts the mean of the rollouts for each prompt,
then normalizes the resulting advantages using global batch statistics;
Distortion A is precisely the absence of that prompt-local subtraction here.
Nor was this a correct implementation of plain REINFORCE++: the global
statistics counted trajectory fragments rather than rollout executions, so
they were unintentionally weighted by `K_e`. The most precise description is
therefore a **malformed, fan-out-weighted REINFORCE++-like estimator**, whose
policy-gradient core reduces to REINFORCE with the shared baseline above.

### Why eval improved anyway

REINFORCE with a baseline is a valid policy-gradient method. For any baseline
`b` that does not depend on the trajectory:

```
E[ (R_e − b) ∇log π(τ_e) ]  =  E[ R_e ∇log π(τ_e) ]  =  ∇ E[R]
```

because `E[ b · ∇log π(τ_e) ] = b · ∇ Σ_τ π(τ) = b · ∇1 = 0`. So the expected
update direction still points up the true reward gradient, and the W&B eval
curve rising during the run is consistent with the bug, not evidence against
it. What the bug changes is not *whether* learning happens but *which
estimator* produced it:

- **Lost: prompt-conditional variance reduction.** An all-zero prompt group
  gets advantage `−μ̂/σ̂` on every sample instead of 0. In expectation that
  term is zero-mean (it is baseline noise, not bias), but it is pure gradient
  noise that GRPO exists to remove.
- **Gained: two small uncontrolled bias terms.** `μ̂` includes `R_e` itself
  (an O(1/E) self-inclusion bias GRPO also has, per-group), and `K_e` is
  correlated with episode length and hence with trajectory content, so the
  weights of the baseline are not independent of the outcomes it baselines.
- **The reported metrics are wrong for comparison.** Any number from this run
  labeled "GRPO baseline" actually measures this other estimator.

Training curves stayed plausible for 103 steps precisely because the run was
optimizing *a* coherent policy-gradient objective — just not the one on the
label.

---

## Evidence from the failed run

Verified directly from the saved debug rollouts
(`results/debug_rollout/qwen3.5-4B-grpo-full/rollout_*.pt`):

| Rollout | Flat samples | Executions | Per-prompt sample counts | Max siblings |
|---:|---:|---:|---|---:|
| 0 | 2,291 | 64 | 222, 224, 260, 268, 287, 322, 342, 366 | 50 |
| 97 | 2,037 | 64 | 123, 126, 235, 235, 293, 307, 342, 376 | 50 |
| 99 | 2,064 | 64 | 178, 198, 200, 208, 273, 322, 324, 361 | 50 |
| 103 | 2,180 | 64 | 163, 198, 263, 278, 289, 324, 328, 337 | 50 |

Additional checks: every sample carries a non-null `rollout_id`; all siblings
of one execution agree on their reward; the run's printed config confirms
`rewards_normalization=True` and no custom post-processor. Rollout 0 already
shows 2,291 ≠ 64, proving the normalization fallback was active from the very
first update.

## Why upstream never caught this

At the time of the failed run, the vendored slime was unmodified (single
vendor commit `8567b10`). Comparing all agentic examples that shipped with it:

| Example | Return shape | Bug 1 (crash) | Bug 2 (normalization) |
|---|---|---|---|
| search-r1, retool, tau-bench, strands_sglang, geo3k_vlm_multi_turn | always one `Sample` | immune | immune |
| coding_agent_rl | always `list[Sample]` (aborts wrapped as `[sample]`) | dodged | **present** |
| multi_agent | always `list[Sample]` | dodged | **present** |
| **ours (OfficeQA)** | **mixes `Sample` and `list[Sample]`** | **hit** | **present** |

The first five never re-render the conversation through the chat template —
they append tokens onto one growing stream, so one rollout is one sample by
construction and neither code path is exercised. The two upstream fan-out
examples uniformly return lists, which happens to dodge the first-element
flattening trap, but both run GRPO with the default post-processor on batches
larger than nominal — their advantages are wrong in the same way ours were.
Ours is the only implementation that exercises the scalar-or-list contract the
type signature promises, which is why the crash first appeared here.

### Upstream status (checked 2026-07-17)

- **Bug 2, cross-prompt half: known and fixed after our vendor snapshot.**
  Upstream issue [#1414](https://github.com/THUDM/slime/issues/1414) reports
  exactly the `view(-1, shape[-1])` global-group fallback; it was closed by
  [PR #1655](https://github.com/THUDM/slime/pull/1655), which normalizes per
  `group_index`. Our vendored copy predates that fix.
- **Bug 2, fan-out-weighting half: still unfixed upstream.** PR #1655 computes
  the per-group statistics over the *flat sample list*, so a 50-sibling
  execution still enters its group's mean/std 50 times. Distortion B in this
  report survives the upstream fix.
- **Bug 1 (flattening crash): no upstream issue exists.** The closest is
  [#378](https://github.com/THUDM/slime/issues/378) (open feature request for
  ragged `list[Sample]` batches), which does not mention the crash.

Both remaining upstream problems are drafted for submission in
`bugs/slime-upstream-issue-drafts.md`.

---

## Implemented fixes

Both fixes are in slime's core pipeline, where the contract violation lives.
Changing the OfficeQA hook to always return a list is **not** a fix: it hides
Bug 1 without repairing the documented `Sample | list[Sample]` contract, can
still be defeated by scalar abort paths, and does nothing about Bug 2.

### Fix 1: structure-safe flattening

`RolloutManager._get_rollout_data` now passes the complete nested result to
`_flatten_rollout_samples` instead of repeatedly inspecting `data[0]` and
calling `chain.from_iterable`:

```python
data = _flatten_rollout_samples(data)
```

The helper recursively walks the result tree. Its behavior for each node is:

- A `Sample` is a leaf and becomes a one-element result.
- A non-empty `list` is visited element by element, recursively, regardless of
  the type of its first element.
- An empty list is rejected rather than failing later with `data[0]`.
- Any other node type is rejected immediately.
- All collected leaves are returned in their original left-to-right order as
  one `list[Sample]`.

Errors include the exact structural path, such as `data[0][1]`, so a malformed
rollout can be traced back to its prompt and execution rather than failing in
an unrelated downstream attribute access.

The walker also validates fan-out semantics before erasing the nesting. In
slime's normal shape, `data[prompt][execution]` is a `Sample`. A compact agent
rollout adds one level, so `data[prompt][execution]` is a `list[Sample]` whose
members are sibling fragments of that execution. When such a list contains
more than one sample, every sibling must have a non-null `rollout_id`, and all
IDs must be identical. For example:

```
data[0][0] = Sample(...)                         accepted: scalar execution
data[0][1] = [Sample(rid=7)]                     accepted: one-sample list
data[0][2] = [Sample(rid=8), Sample(rid=8)]      accepted: two siblings
data[0][3] = [Sample(rid=9), Sample(rid=None)]   rejected at data[0][3]
data[0][4] = [Sample(rid=10), Sample(rid=11)]    rejected at data[0][4]
```

This makes mixed scalar/list batches safe in either order, including a bare
aborted `Sample` next to multi-sample trajectories. The old
`_validate_rollout_id_annotated` entry point remains as a compatibility wrapper
around the same complete walker, so it no longer has a first-element blind
spot either.

### Fix 2: per-rollout GRPO normalization

The default `_post_process_rewards` now calls `_normalize_rewards_per_rollout`
for GRPO-like estimators when reward normalization is enabled. No OfficeQA
custom reward hook is required. A configured custom post-processor still takes
precedence, preserving the existing extension contract.

The new calculation is:

```
flat samples
   │ group by group_index                      (recover the 8 prompts)
   ▼
per prompt: group by rollout_id                (recover the 8 executions)
   │ assert all siblings of one rollout agree on reward
   │ take ONE reward per rollout
   ▼
GRPO mean/std over the 8 rollout rewards       (statistics over executions,
   │                                            never over the flat list)
   ▼
broadcast each rollout's normalized advantage
back onto all of its siblings
```

Concretely, the helper preserves the flat input order while building two
temporary maps:

1. It requires `group_index` on every sample and groups positions by that key.
   `group_index` is assigned once per source prompt, so this prevents
   Distortion A.
2. Inside each prompt, it groups samples with the same explicit `rollout_id`.
   A sample without an explicit ID is treated as its own execution, preserving
   the legacy one-sample-per-rollout behavior.
3. It verifies that all siblings in an explicit rollout group carry exactly
   the same reward. A disagreement is a data-contract error and is reported
   with both `group_index` and `rollout_id`.
4. It extracts one reward from each execution, computes the prompt mean and
   (when requested by GRPO/GSPO/CISPO) standard deviation over those unique
   execution rewards, and computes one advantage per execution.
5. It writes that advantage back to every original sibling position. The
   returned list therefore still aligns one-to-one with the flat training
   samples expected downstream.

For prompt `p`, the implemented calculation is:

```
E_p = the set of rollout executions for prompt p

      1
μ_p = ─────  Σ    R_e
      |E_p|  e∈E_p

      R_e − μ_p
A_e = ───────────
       σ_p + ε

advantage(sample) = A_e  for every sibling sample of execution e
```

There is no `K_e` term: one execution contributes once whether it emitted 1,
3, or 50 training samples. A prompt containing only one execution has no
within-prompt contrast and receives advantage 0 instead of the `NaN` produced
by an undefined singleton sample standard deviation.

The same grouping helper also covers `reinforce_plus_plus_baseline`. For that
estimator it performs the prompt-local mean subtraction once per rollout but
does not apply a local standard deviation; slime's later global advantage
normalization remains responsible for the second REINFORCE++ normalization
stage.

The rollout-aware loss reducer already ensures siblings collectively count as
one rollout during optimization; this fix makes the advantage side consistent
with it. That reducer and the rollout-aware training-step splitter were not
changed.

Note: porting upstream PR #1655 is **not** sufficient. It fixes only the
cross-prompt grouping (Distortion A); its statistics are still computed over
the flat sample list, so the sibling weighting (Distortion B) remains.

## Regression tests

CPU regression coverage is in `third_party/slime/tests/test_rollout_fanout.py`:

1. One group containing scalar, one-element-list, multi-element-list, and
   aborted-scalar results — flattens without error.
2. Nested output whose first element is scalar and a later element is a list
   (and the reverse) — both flatten correctly.
3. Missing or inconsistent sibling `rollout_id` — raises a clear validation
   error.
4. **Invariance (the key test):** changing one rollout from 1 to 50
   identical-reward siblings changes no rollout's normalized advantage.
5. Equivalence with existing GRPO behavior when every rollout emits exactly
   one sample.
6. Prompt groups remain isolated even when their flat samples are interleaved.
7. Siblings with disagreeing rewards and samples missing `group_index` fail
   with explicit contract errors.
8. `reinforce_plus_plus_baseline` centers once per rollout without applying a
   prompt-local standard deviation.
9. A singleton prompt group produces zero advantage.

The existing GPU end-to-end fan-out test,
`third_party/slime/tests/test_qwen2.5_0.5B_fanout_short.py`, was also changed to
remove its custom `grpo_normalize_by_group_index` workaround. It now exercises
the default core normalizer with uneven fan-out, so a variable-fan-out batch
must flatten, normalize, save debug data, and complete optimizer updates
through the fixed production path.

Verification performed locally:

- 14/14 new fan-out CPU regression tests passed;
- 50/50 focused slime rollout/sample/scheduling/reducer tests passed;
- 6/6 OfficeQA trajectory-mask tests passed.

The GPU end-to-end test was updated but not executed in this local verification
environment.

## Recovery: restart from the base model

**Do not resume from `iter_0000099`.** Rollout 0 already shows the global
normalization fallback active (2,291 samples in one group), so every optimizer
update in the run — including everything baked into every checkpoint — used
invalid GRPO advantages. Resuming carries that forward.

What survives from the failed run:

- **Nothing usable as a GRPO baseline number.** The algorithm actually trained
  was not GRPO.
- **Eval scores are real measurements** of the (mistrained) policy — evidence
  the task gives learnable signal, but not attributable to GRPO.
- **All infrastructure findings stand** (optimizer offload, throughput,
  fan-out statistics, sandbox stability).

After both fixes and their regression tests pass: start a new baseline from
the original base model under a new experiment/checkpoint directory, and keep
run `nyqmudip` only as a debugging artifact.
