# Vendored mini-swe-agent Snapshot

This directory is a vendored source snapshot of mini-swe-agent. It is tracked as
ordinary files by the `officeqa-rl` repository; it is not a Git submodule and does
not retain the source repository's `.git` directory.

## Provenance

- Upstream repository: <https://github.com/SWE-agent/mini-swe-agent>
- Base commit: `25941c89cfbc91eb40b3f8756348c91d9977d57e` (2026-08-17)
- Released version: `2.4.6`
- License: MIT (`LICENSE.md`)
- Vendored on: 2026-08-20
- Update policy: no automatic upstream synchronization

## Snapshot state

Copied from a clean `main` tarball, unmodified. Only the parts needed to run the
harness are vendored:

- `src/minisweagent/` — the package (agents, environments, models, config, utils)
- `pyproject.toml` — kept for dependency provenance; the package is **not**
  installed from here
- `LICENSE.md`

Docs, tests, and the upstream CI configuration are not vendored.

## How this repository uses it

`officeqa_rl.jobbench` imports `minisweagent` directly. The package is made
importable by `officeqa_rl/jobbench/_msa.py`, which appends this directory's
`src/` to `sys.path` when `minisweagent` is not already installed, so no
`pip install` step is required on cluster nodes. That module also sets
`MSWEA_SILENT_STARTUP=1` (suppresses the upstream startup banner) and
`MSWEA_GLOBAL_CONFIG_DIR` (keeps the upstream global config out of `$HOME`)
before the import.

Only these upstream pieces are used:

| Piece | Used for |
|---|---|
| `minisweagent.agents.default.DefaultAgent` | the agent control flow, unmodified |
| `minisweagent.exceptions` | `Submitted`, `LimitsExceeded`, `FormatError` |
| `minisweagent.models.utils.actions_text` | one-fenced-command parsing and observation rendering |
| `minisweagent.utils.serialize.recursive_merge` | template-variable merging |
| `minisweagent.Model` / `Environment` protocols | the interfaces implemented in `officeqa_rl.jobbench` |

`minisweagent.models.litellm_model` and the other upstream model backends are
never imported: `officeqa_rl.jobbench.models` implements the `Model` protocol on
top of this repository's own generation backends, so `litellm` is not a
dependency. The upstream `DockerEnvironment` is likewise not used;
`officeqa_rl.jobbench.environment` implements the `Environment` protocol against
the same Docker-outside-of-Docker sandbox the OfficeQA and SpreadsheetBench
environments use.

## Updating

Re-vendor by replacing `src/minisweagent/`, `LICENSE.md`, and `pyproject.toml`
from a fresh tarball, then update the base commit and version above. Check
upstream changes to `agents/default.py` (control flow), the `Model`/`Environment`
protocols in `__init__.py`, and `models/utils/actions_text.py` (action regex and
observation template contract) — those are the surfaces this repository depends
on.
