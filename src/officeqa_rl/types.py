"""Contracts for OfficeQA agents, generation backends, and sandbox runtimes.

This module intentionally contains no runner flags or heavyweight runtime
dependencies. CLI entry points translate their arguments into these types before
calling reusable library code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from PIL.Image import Image


Message = dict[str, Any]
SandboxRuntime = Literal["docker", "local"]


class AgentError(Exception):
    """Base exception for expected agent control flow."""


class FormatError(AgentError):
    """Raised when a model response does not contain exactly one action."""


class LimitsExceeded(AgentError):
    """Raised when an agent reaches its configured step budget."""


class Submitted(AgentError):
    """Signal that the environment received a final answer."""

    def __init__(self, submission: str):
        super().__init__(submission)
        self.submission = submission


@dataclass
class StepOutput:
    """Observation returned after an environment action."""

    output: str
    image: Image | None = None


@dataclass(frozen=True)
class OfficeQATask:
    """One benchmark task, with the answer kept separate for grading only."""

    uid: str
    question: str
    answer: str | None
    source_files: str = ""
    difficulty: str | None = None


@dataclass(frozen=True)
class AgentConfig:
    """Stable configuration for one OfficeQA agent run."""

    step_limit: int
    action_block_language: str = "bash"
    web_search_instruction: bool = False
    # Which corpus the filename hints in the system prompt describe. "v1" is the
    # Treasury Bulletin corpus and the default, so every existing launcher keeps
    # a byte-identical prompt; OfficeQA v2 ships a different corpus and needs "v2".
    corpus_profile: str = "v1"


@dataclass(frozen=True)
class EnvironmentConfig:
    """Docker-first settings for hosting an OfficeQA sandbox."""

    corpus_dir: Path
    pdf_dir: Path | None
    workspace_root: Path
    sandbox_image: str | None
    command_timeout_sec: int
    max_observation_chars: int
    sandbox_runtime: SandboxRuntime = "docker"
    enable_visual_pdf_tool: bool = False
    max_concurrent_sandbox_boots: int = 4
    page_text_dir: Path | None = None
    pdf_render_dpi: int = 144


@dataclass
class GeneratedTurn:
    """One backend generation plus optional training-token provenance."""

    text: str
    reasoning_content: str | None = None
    prompt_ids: list[int] | None = None
    output_ids: list[int] | None = None
    output_log_probs: list[float] | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


@dataclass
class AgentResult:
    """Terminal agent state returned to inference or training callers."""

    submission: str
    status: str
    messages: list[Message]
    steps: int
    metadata: dict[str, Any]


__all__ = [
    "AgentConfig",
    "AgentError",
    "AgentResult",
    "EnvironmentConfig",
    "FormatError",
    "GeneratedTurn",
    "LimitsExceeded",
    "Message",
    "OfficeQATask",
    "SandboxRuntime",
    "StepOutput",
    "Submitted",
]
