# Upstream issue drafts for THUDM/slime

Two ready-to-paste GitHub issues. Line numbers below refer to our vendored
snapshot; verify against upstream `main` before submitting (function names are
stable). Related upstream context: #1414 (closed by PR #1655) fixed the
cross-prompt half of the normalization problem; #378 is an open feature
request for ragged batches. Neither covers the two problems below.

---

## Issue 1

**Title:** `_get_rollout_data` crashes with `TypeError: 'Sample' object is not iterable` when a custom generate function mixes `Sample` and `list[Sample]` returns

**Labels:** bug

### Body

`generate_and_rm` is typed to return `Sample | list[Sample]`
(`slime/rollout/sglang_rollout.py`), and `generate_and_rm_group` preserves
whichever shape each async task produced. A custom
`--custom-generate-function-path` hook may therefore legally return a bare
`Sample` for some rollouts (e.g. single-sample trajectories, abort paths) and
a `list[Sample]` for others, producing a heterogeneous group such as:

```python
[[Sample, Sample, Sample], Sample, [Sample, Sample]]
```

But the flattening in `RolloutManager._get_rollout_data`
(`slime/ray/rollout.py`) infers the shape of the whole batch from its first
element while iterating every element:

```python
while isinstance(data[0], list):
    data = list(itertools.chain.from_iterable(data))
```

Two failure modes, depending on element order:

1. First element is a list, a later element is a bare `Sample`:
   `chain.from_iterable` tries to iterate the `Sample` and raises
   `TypeError: 'Sample' object is not iterable`. The training job dies.
2. First element is a bare `Sample`, a later element is a list: the loop exits
   immediately and nested lists escape into the "flat" data, failing later in
   reward/token processing, far from the cause.

`_validate_rollout_id_annotated` (same file) has the same first-element blind
spot (`if node and isinstance(node[0], Sample): ...`), so a mixed group can
also bypass sibling `rollout_id` validation.

#### Reproduction

Any agentic rollout function of the form used by several shipped examples:

```python
async def generate(args, sample, sampling_params):
    ...
    return samples[0] if len(samples) == 1 else samples
```

runs fine for as long as every rollout in a batch happens to emit ≥ 2 samples,
then crashes on the first batch containing a single-sample trajectory or a
scalar abort. In our training run this took 103 successful steps (~6,650
rollout executions) before the first mixed batch appeared and killed the job:

```text
File "slime/ray/rollout.py", line 641, in _get_rollout_data
    data = list(itertools.chain.from_iterable(data))
TypeError: 'Sample' object is not iterable
```

Note the shipped `examples/coding_agent_rl` deliberately wraps even aborted
rollouts as `[sample]` to sidestep this — which suggests the sharp edge is
known but the contract is not actually safe for third-party hooks.

#### Suggested fix

Replace the first-element-driven loop with a recursive walker that accepts
`Sample` leaves and arbitrarily nested lists, visits every element, returns a
flat `list[Sample]`, and raises a path-aware error on unexpected nodes. Apply
the same every-element traversal to `_validate_rollout_id_annotated`. Happy to
send a PR with regression tests (mixed scalar/list groups in both orders,
nested lists, abort paths).

---

## Issue 2

**Title:** GRPO advantage statistics weight each rollout by its sample fan-out; follow-up to #1414 / PR #1655

**Labels:** bug

### Body

PR #1655 (closing #1414) correctly changed `_post_process_rewards` to
normalize per `group_index` instead of falling back to one global group when
sample counts are unequal. However, the per-group statistics are still
computed over the **flat sample list**, and that list is fan-out-expanded:
when a custom generate function returns `list[Sample]` (multiple training
samples split out of one rollout execution, all sharing one `rollout_id` and
one reward), every sibling carries a copy of the rollout's reward.

The group mean/std therefore weights each rollout execution by its sibling
count, while slime's own loss side (`rollout_mask_sums`, the per-rollout
reducer) deliberately counts each rollout exactly once. The two halves of the
pipeline disagree about what one unit of experience is.

#### Worked example

One prompt group, `n_samples_per_prompt = 2`:

```
rollout 1: reward 1.0, split into K siblings
rollout 2: reward 0.0, split into 1 sample

K = 3  →  group mean over flat samples = 3/4 = 0.75  →  advantages ≈ +0.25 / −0.75
K = 50 →  group mean over flat samples = 50/51 ≈ 0.98 →  advantages ≈ +0.02 / −0.98
```

Identical policy behavior, identical rewards — but the gradient scale changes
by an order of magnitude depending on how many samples one episode happened to
split into. Correct GRPO semantics (mean over the 2 rollout executions) give
+0.5 / −0.5 in both cases.

This is not a corner case for agentic training: any multi-turn setup whose
chat template is not prefix-stable (e.g. Qwen3-family templates deleting
previous `<think>` blocks) splits every episode into roughly one sample per
turn, with highly variable counts. In our runs, 64 executions produced
2,000–2,300 flat samples with 1–50 siblings each. The shipped
`examples/coding_agent_rl` and `examples/multi_agent` both return variable
`list[Sample]` and both run GRPO with the default post-processor, so they are
affected as well.

#### Suggested fix

In the default `_post_process_rewards` (and as guidance for custom
post-processors), when samples carry `rollout_id`:

1. group by `group_index`;
2. within each group, reduce to one reward per unique `rollout_id`
   (asserting siblings agree);
3. compute GRPO mean/std over those per-rollout rewards;
4. broadcast each rollout's normalized advantage back to all of its siblings.

This makes the advantage side consistent with the per-rollout loss reducer. A
useful invariance test: changing one rollout from 1 to 50 identical-reward
siblings must not change any rollout's normalized advantage. Happy to send a
PR including that test.
