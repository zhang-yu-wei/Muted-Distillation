# Vendored Slime Snapshot

This directory is a vendored source snapshot of slime. It is tracked as ordinary
files by the `officeqa-rl` repository; it is not a Git submodule and does not retain
the source repository's `.git` directory.

## Provenance

- Upstream repository: <https://github.com/THUDM/slime.git>
- Base commit: `680824dd5e01a2e83750bf87fc366ec6fa98766c`
- Vendored on: 2026-07-13
- Source checkout: `/mnt/slime`
- Update policy: no automatic upstream synchronization

## Snapshot state

The snapshot was copied from the source checkout's current working tree, not from a
clean `git archive`. It therefore includes the local tracked modifications and
non-ignored source additions that were present at vendoring time.

Tracked files modified relative to the base commit:

- `examples/tau-bench/generate_with_tau.py`
- `examples/tau-bench/run_qwen3_4B.sh`
- `examples/tau-bench/trainable_agents.py`
- `slime/backends/megatron_utils/arguments.py`
- `slime/backends/megatron_utils/loss.py`
- `slime/backends/megatron_utils/model.py`
- `slime/utils/data.py`
- `slime/utils/ppo_utils.py`

Non-ignored additions present in the source checkout included:

- `bugs/`
- `examples/tau-bench/docker_run.sh`
- `examples/tau-bench/repro_nan_rollout1.sh`
- `examples/tau-bench/run_qwen3.5_4B.sh`
- `tools/convert_hf_to_torch_dist_qwen3_5.py`
- `uv.lock`

Generated or machine-local content was intentionally excluded, including `.git/`,
`.venv/`, Python bytecode and caches, generated egg metadata, and
`tensorboard_log/`.

The `officeqa-rl` Git history is the authoritative record of all changes made after
this vendoring point. Preserve slime's included license and attribution files when
redistributing the repository.
