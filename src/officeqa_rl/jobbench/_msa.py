"""Import shim for the vendored mini-swe-agent harness.

``third_party/mini-swe-agent`` is a source snapshot, not an installed package
(see its ``UPSTREAM.md``). Importing this module makes ``minisweagent``
importable without a ``pip install`` step on cluster nodes, and pins two
environment variables that upstream reads at import time:

``MSWEA_SILENT_STARTUP``
    Suppresses the version banner ``minisweagent/__init__.py`` prints to stdout,
    which would otherwise appear once per rollout worker.
``MSWEA_GLOBAL_CONFIG_DIR``
    Upstream creates this directory and loads a ``.env`` from it at import time.
    Defaulting it under the repository keeps containers from writing into
    ``$HOME``, which is often read-only or shared under DooD.

Import from here rather than importing ``minisweagent`` directly, so the
environment is configured before upstream's module body runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDORED_SRC = _REPO_ROOT / "third_party" / "mini-swe-agent" / "src"

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
os.environ.setdefault(
    "MSWEA_GLOBAL_CONFIG_DIR", str(_REPO_ROOT / "results" / "mini_swe_agent_config")
)

try:  # an installed copy wins, so a pinned wheel can override the snapshot
    import minisweagent  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - exercised by the vendored path
    if not (_VENDORED_SRC / "minisweagent" / "__init__.py").is_file():
        raise ModuleNotFoundError(
            "minisweagent is neither installed nor vendored at "
            f"{_VENDORED_SRC}. Re-vendor it (see "
            "third_party/mini-swe-agent/UPSTREAM.md) or `pip install mini-swe-agent`."
        ) from None
    sys.path.insert(0, str(_VENDORED_SRC))
    import minisweagent  # noqa: F401

from minisweagent import Environment, Model  # noqa: E402
from minisweagent.agents.default import AgentConfig as MiniAgentConfig  # noqa: E402
from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.exceptions import (  # noqa: E402
    FormatError,
    InterruptAgentFlow,
    LimitsExceeded,
    Submitted,
)
from minisweagent.models.utils.actions_text import (  # noqa: E402
    format_observation_messages,
    parse_regex_actions,
)
from minisweagent.utils.serialize import recursive_merge  # noqa: E402

MINI_SWE_AGENT_VERSION = minisweagent.__version__

__all__ = [
    "DefaultAgent",
    "Environment",
    "FormatError",
    "InterruptAgentFlow",
    "LimitsExceeded",
    "MINI_SWE_AGENT_VERSION",
    "MiniAgentConfig",
    "Model",
    "Submitted",
    "format_observation_messages",
    "parse_regex_actions",
    "recursive_merge",
]
