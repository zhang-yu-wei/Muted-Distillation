"""Full-corpus environment shared by inference and RL rollouts."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from officeqa_rl.types import (
    EnvironmentConfig,
    FormatError,
    Message,
    OfficeQATask,
    StepOutput,
    Submitted,
)


# Semaphores must be shared across environments but bound to one event loop,
# so they are cached per running loop instead of created at import time.
_SANDBOX_BOOT_SEMAPHORES: WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[int, asyncio.Semaphore]
] = WeakKeyDictionary()

_PAGE_MARKER_RE = re.compile(r"(?m)^--- PAGE (\d+) ---[ \t]*$")
# Document names as they appear in search output ("corpus/name.txt:12:...") and,
# for commands, the same with shell glob characters left in place.
_DOCUMENT_NAME_RE = re.compile(r"([A-Za-z0-9_.-]+)\.txt\b")
_DOCUMENT_PATTERN_RE = re.compile(r"([A-Za-z0-9_.\-*?\[\]]+)\.txt\b")
# Ranking budget for one automatic retrieval. A broad search can match hundreds
# of documents; scoring every page of all of them would cost more than the
# command that triggered it.
_RETRIEVAL_MAX_DOCUMENTS = 8
_RETRIEVAL_MAX_PAGES = 1_500
_SEARCH_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SEARCH_STOPWORDS = frozenset(
    {
        "a",
        "according",
        "add",
        "all",
        "amount",
        "an",
        "and",
        "answer",
        "are",
        "as",
        "assume",
        "at",
        "be",
        "between",
        "billion",
        "billions",
        "both",
        "bracket",
        "brackets",
        "breakdown",
        "by",
        "calculate",
        "calculated",
        "calculation",
        "calculations",
        "category",
        "comma",
        "commas",
        "comparable",
        "containing",
        "corresponding",
        "decimal",
        "determine",
        "did",
        "difference",
        "do",
        "does",
        "dollar",
        "dollars",
        "each",
        "expressed",
        "fit",
        "find",
        "first",
        "for",
        "four",
        "from",
        "full",
        "geometric",
        "had",
        "has",
        "have",
        "highest",
        "how",
        "hundredth",
        "hundredths",
        "if",
        "in",
        "inclusive",
        "inside",
        "integer",
        "into",
        "is",
        "it",
        "its",
        "just",
        "lambda",
        "least",
        "linear",
        "many",
        "mean",
        "million",
        "millions",
        "multiply",
        "nearest",
        "nominal",
        "not",
        "number",
        "numbers",
        "numeric",
        "of",
        "on",
        "only",
        "ordinary",
        "or",
        "outcome",
        "page",
        "per",
        "period",
        "perform",
        "place",
        "places",
        "predictor",
        "published",
        "regression",
        "report",
        "reported",
        "represent",
        "resulting",
        "return",
        "rounding",
        "rounded",
        "same",
        "second",
        "separated",
        "slope",
        "spent",
        "spending",
        "specifically",
        "square",
        "squares",
        "sum",
        "take",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "third",
        "to",
        "total",
        "transformed",
        "treasury",
        "u",
        "untransformed",
        "us",
        "use",
        "using",
        "value",
        "values",
        "was",
        "were",
        "what",
        "when",
        "which",
        "whole",
        "with",
        "without",
        "would",
        "adjustments",
        "basic",
        "data",
        "intercept",
        "need",
        "postyear",
        "predict",
        "predicted",
        "produce",
        "time",
        "treat",
        "variable",
        "year",
        "years",
        "you",
        "your",
    }
)


def _sandbox_boot_semaphore(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphores = _SANDBOX_BOOT_SEMAPHORES.setdefault(loop, {})
    if limit not in semaphores:
        semaphores[limit] = asyncio.Semaphore(limit)
    return semaphores[limit]


class FullCorpusEnvironment:
    """Give one OfficeQA task an isolated shell workspace and corpus access.

    The answer remains on ``task`` for the caller to grade after execution. It
    is never copied into the agent-visible workspace or process environment.
    """

    def __init__(
        self,
        config: EnvironmentConfig,
        task: OfficeQATask,
        *,
        run_id: str,
    ) -> None:
        if config.sandbox_runtime not in {"local", "docker"}:
            raise ValueError(f"Unknown sandbox runtime: {config.sandbox_runtime}")
        if config.command_timeout_sec <= 0:
            raise ValueError("command_timeout_sec must be positive")
        if config.max_observation_chars <= 0:
            raise ValueError("max_observation_chars must be positive")
        if config.max_concurrent_sandbox_boots <= 0:
            raise ValueError("max_concurrent_sandbox_boots must be positive")
        if config.enable_visual_pdf_tool and config.pdf_render_dpi <= 0:
            raise ValueError("pdf_render_dpi must be positive")
        if not run_id.strip():
            raise ValueError("run_id must not be empty")

        self.config = config
        self.task = task
        self.run_id = run_id
        self.corpus_dir = config.corpus_dir.expanduser().resolve()
        self.pdf_dir = (
            config.pdf_dir.expanduser().resolve() if config.pdf_dir is not None else None
        )
        self.page_text_dir = (
            config.page_text_dir.expanduser().resolve()
            if config.page_text_dir is not None
            else self.corpus_dir
        )
        if config.enable_visual_pdf_tool and self.pdf_dir is None:
            raise ValueError("pdf_dir is required when enable_visual_pdf_tool is true")
        self.workspace_root = config.workspace_root.expanduser().resolve()
        self.workspace = self.workspace_root / self._workspace_name(task.uid, run_id)

        self.command_container_id: str | None = None
        self.command_history: list[dict[str, Any]] = []
        self.automatic_page_retrievals: list[dict[str, Any]] = []
        self._container_name = self._docker_container_name(task.uid, run_id)
        self._active = False

    @property
    def is_active(self) -> bool:
        """Whether the environment has been entered and not yet cleaned up."""
        return self._active

    async def __aenter__(self) -> FullCorpusEnvironment:
        if self._active:
            raise RuntimeError("Environment is already active")
        try:
            self._prepare_workspace()
            if self.config.sandbox_runtime == "docker":
                async with _sandbox_boot_semaphore(
                    self.config.max_concurrent_sandbox_boots
                ):
                    await self._start_command_container()
            self._active = True
            return self
        except BaseException:
            await self.cleanup()
            raise

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        await self.cleanup()
        return False

    @staticmethod
    def _safe_component(value: str, *, fallback: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or fallback

    @classmethod
    def _workspace_name(cls, uid: str, run_id: str) -> str:
        safe_uid = cls._safe_component(uid, fallback="question")
        safe_run_id = cls._safe_component(run_id, fallback="run")
        digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:10]
        return f"{safe_uid}--{safe_run_id[:64]}--{digest}"

    @classmethod
    def _docker_container_name(cls, uid: str, run_id: str) -> str:
        safe_uid = cls._safe_component(uid, fallback="question").lower()[:30]
        safe_run_id = cls._safe_component(run_id, fallback="run").lower()[:30]
        digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:10]
        return f"officeqa-{safe_uid}-{safe_run_id}-{digest}"

    def _prepare_workspace(self) -> None:
        if not self.corpus_dir.is_dir():
            raise FileNotFoundError(f"Corpus directory does not exist: {self.corpus_dir}")
        if self.config.enable_visual_pdf_tool:
            if not self.page_text_dir.is_dir():
                raise FileNotFoundError(
                    f"Page-text directory does not exist: {self.page_text_dir}"
                )
            if self.pdf_dir is None or not self.pdf_dir.is_dir():
                raise FileNotFoundError(f"PDF directory does not exist: {self.pdf_dir}")

        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=False, exist_ok=False)
        (self.workspace / "question.txt").write_text(
            self.task.question + "\n",
            encoding="utf-8",
        )
        (self.workspace / "source_files.txt").write_text(
            self.task.source_files.strip() + "\n",
            encoding="utf-8",
        )

        if self.config.sandbox_runtime == "local":
            (self.workspace / "officeqa_corpus").symlink_to(
                self.corpus_dir,
                target_is_directory=True,
            )
            if self.config.enable_visual_pdf_tool:
                (self.workspace / "officeqa_pages").symlink_to(
                    self.page_text_dir,
                    target_is_directory=True,
                )

        manifest = {
            "uid": self.task.uid,
            "question": self.task.question,
            "source_files": self.task.source_files,
            "run_id": self.run_id,
            "workspace": str(self.workspace),
            "corpus_dir": str(self.corpus_dir),
            "workspace_corpus_path": "officeqa_corpus",
            "sandbox_runtime": self.config.sandbox_runtime,
            "pdf_dir": str(self.pdf_dir) if self.pdf_dir is not None else None,
            "visual_pdf_tool_enabled": self.config.enable_visual_pdf_tool,
        }
        if self.config.enable_visual_pdf_tool:
            manifest.update(
                {
                    "page_text_dir": str(self.page_text_dir),
                    "pdf_render_dpi": self.config.pdf_render_dpi,
                }
            )
        (self.workspace / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    def get_task_message(self) -> Message:
        """Build the agent-visible task message without exposing the answer."""
        self._require_active()
        display_workspace = (
            "/workspace"
            if self.config.sandbox_runtime == "docker"
            else str(self.workspace)
        )
        text = (
            f"Question: {self.task.question}\n\n"
            f"Workspace: {display_workspace}\n"
            "Useful files: question.txt, source_files.txt, manifest.json.\n"
            "Corpus path: officeqa_corpus/\n\n"
            "Search and read the parsed Treasury Bulletin corpus, calculate as needed, "
            "then submit the final answer."
        )
        if self.config.enable_visual_pdf_tool:
            text += (
                "\n\nPage-indexed parsed text is available at `officeqa_pages/`. "
                "Continue searching the text corpus with ordinary `rg` or `grep` "
                "commands. After a successful search in `officeqa_corpus/` or "
                "`officeqa_pages/`, the environment automatically attaches the "
                "best matching PDF page from the documents your search matched, "
                "as an image with its parsed page text. Finding the right "
                "document remains your job. "
                "Each page begins with a marker such as `--- PAGE 13 ---`. If a "
                "different or additional page is needed, use "
                "`view_page <document> <PDF-page-number>`. For example: "
                "`view_page treasury_bulletin_1954_01 13`."
            )
        message: Message = {
            "role": "user",
            "content": text,
            "extra": {
                "task": {
                    "uid": self.task.uid,
                    "run_id": self.run_id,
                    "workspace": str(self.workspace),
                    "corpus_dir": str(self.corpus_dir),
                    "sandbox_runtime": self.config.sandbox_runtime,
                }
            },
        }
        return message

    def _retrieval_documents(self, command: str, observation: str) -> list[str]:
        """Choose which documents the automatic retrieval may draw pages from.

        Only documents the model's own search matched are eligible, read off
        the observation it received and, when the search printed no file names,
        off the paths named in the command. The task's ``source_files`` must
        never be consulted here: it names the document holding the answer,
        which is the model's job to work out.
        """
        limit = _RETRIEVAL_MAX_DOCUMENTS
        documents = self._document_stems(observation)[:limit]
        if documents:
            return documents

        available: list[str] | None = None
        for token in self._document_stems(command, keep_patterns=True)[: limit * 4]:
            if not any(character in token for character in "*?["):
                if token not in documents:
                    documents.append(token)
            else:
                if available is None:
                    available = sorted(
                        path.stem for path in self.page_text_dir.glob("*.txt")
                    )
                documents.extend(
                    stem
                    for stem in available
                    if fnmatch(stem, token) and stem not in documents
                )
            if len(documents) >= limit:
                break
        return documents[:limit]

    @staticmethod
    def _document_stems(text: str, keep_patterns: bool = False) -> list[str]:
        """Return ``.txt`` document stems in first-appearance order."""
        pattern = _DOCUMENT_PATTERN_RE if keep_patterns else _DOCUMENT_NAME_RE
        stems: list[str] = []
        for match in pattern.finditer(text):
            stem = match.group(1)
            if stem and stem not in stems:
                stems.append(stem)
        return stems

    def _retrieve_pdf_page_after_search(
        self,
        search_query: str,
        documents: Sequence[str],
    ) -> tuple[Any, str, int, str, float] | None:
        """Rank pages of the documents a model text search matched."""
        started = time.monotonic()
        candidates: list[tuple[str, int, str]] = []
        considered: list[str] = []
        for document in documents:
            if not document or not re.fullmatch(r"[A-Za-z0-9_.-]+", document):
                continue
            text_path = self.page_text_dir / f"{document}.txt"
            if not text_path.is_file():
                continue
            try:
                pages = [
                    (document, page_number, page_text)
                    for page_number, page_text in self._read_marked_pages(text_path)
                    if page_text
                ]
            except (OSError, UnicodeError):
                continue
            if not pages:
                continue
            considered.append(document)
            candidates.extend(pages)
            if len(candidates) >= _RETRIEVAL_MAX_PAGES:
                break
        if not candidates:
            return None

        retrieval_query = f"{search_query} {search_query} {self.task.question}"
        scores = self._rank_pages(retrieval_query, [item[2] for item in candidates])
        best_index = max(range(len(candidates)), key=lambda index: scores[index])
        retrieval_reason = "lexical_match"
        explicit_index = self._select_explicit_page(
            self.task.question,
            candidates,
            scores,
        )
        if explicit_index is not None:
            best_index = explicit_index
            retrieval_reason = "explicit_printed_page"
        else:
            referenced_index = self._resolve_contents_reference(
                retrieval_query,
                candidates,
                best_index,
            )
            if referenced_index is not None:
                best_index = referenced_index
                retrieval_reason = "contents_reference"
        best_score = scores[best_index]
        if best_score <= 0 and retrieval_reason == "lexical_match":
            return None
        document, page_number, page_text = candidates[best_index]
        pdf_path = (self.pdf_dir / f"{document}.pdf") if self.pdf_dir is not None else None
        if pdf_path is None or not pdf_path.is_file():
            return None
        try:
            image = self._render_pdf_page(
                pdf_path,
                page_number,
                self.config.pdf_render_dpi,
            )
        except Exception as exc:
            self.automatic_page_retrievals.append(
                {
                    "document": document,
                    "page_number": page_number,
                    "retrieval_score": best_score,
                    "retrieval_reason": retrieval_reason,
                    "search_query": search_query,
                    "candidate_documents": list(considered),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return None

        record = {
            "document": document,
            "page_number": page_number,
            "retrieval_score": best_score,
            "retrieval_reason": retrieval_reason,
            "search_query": search_query,
            "candidate_documents": list(considered),
        }
        self.automatic_page_retrievals.append(record)
        self.command_history.append(
            {
                "command": "automatic_pdf_page_retrieval",
                "runtime": "host",
                "returncode": 0,
                "duration": time.monotonic() - started,
                "timed_out": False,
                **record,
            }
        )
        return image, document, page_number, page_text, best_score

    @staticmethod
    def _search_query_from_command(command: str) -> str | None:
        """Extract the pattern from an ordinary corpus ``rg``/``grep`` command."""
        corpus_targets = (
            "officeqa_corpus",
            "officeqa_pages",
            "$OFFICEQA_CORPUS_DIR",
            "$OFFICEQA_PAGE_TEXT_DIR",
        )
        if not any(target in command for target in corpus_targets):
            return None
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None

        programs = {"rg", "grep", "egrep", "fgrep"}
        options_with_value = {
            "-A",
            "--after-context",
            "-B",
            "--before-context",
            "-C",
            "--context",
            "-g",
            "--glob",
            "-m",
            "--max-count",
            "--exclude",
            "--exclude-dir",
            "--include",
        }
        controls = {"|", "||", "&&", ";"}
        for program_index, token in enumerate(tokens):
            if Path(token).name not in programs:
                continue
            if any(
                option in tokens[program_index + 1 :]
                for option in {"--files", "--type-list"}
            ):
                return None
            skip_next = False
            end_of_options = False
            for candidate_index in range(program_index + 1, len(tokens)):
                candidate = tokens[candidate_index]
                if candidate in controls:
                    break
                if skip_next:
                    skip_next = False
                    continue
                if candidate == "--":
                    end_of_options = True
                    continue
                if not end_of_options and candidate in {"-e", "--regexp"}:
                    if candidate_index + 1 < len(tokens):
                        return tokens[candidate_index + 1]
                    return None
                if not end_of_options and candidate.startswith("--regexp="):
                    return candidate.partition("=")[2] or None
                if not end_of_options and candidate in options_with_value:
                    skip_next = True
                    continue
                if not end_of_options and candidate.startswith("-"):
                    continue
                return candidate
        return None

    def _attach_pdf_to_search_observation(
        self,
        command: str,
        observation: str,
    ) -> StepOutput:
        search_query = self._search_query_from_command(command)
        if search_query is None:
            return StepOutput(observation)
        documents = self._retrieval_documents(command, observation)
        if not documents:
            return StepOutput(observation)
        retrieved = self._retrieve_pdf_page_after_search(search_query, documents)
        if retrieved is None:
            return StepOutput(observation)

        image, document, page_number, page_text, _ = retrieved
        combined = self._truncate_output(
            f"{observation.rstrip()}\n\n"
            "Automatically rendered PDF evidence for the text search:\n"
            f"Document: {document}\n"
            f"PDF page: {page_number}\n\n"
            f"Parsed page text:\n{page_text}"
        ).rstrip()
        return StepOutput(output=combined, image=image)

    @classmethod
    def _select_explicit_page(
        cls,
        question: str,
        candidates: list[tuple[str, int, str]],
        scores: list[float],
    ) -> int | None:
        """Resolve questions that explicitly name a printed bulletin page."""
        match = re.search(r"\bpage\s+(\d{1,4})\b", question, flags=re.IGNORECASE)
        if match is None:
            return None
        requested_page = int(match.group(1))
        printed_matches = [
            index
            for index, (_, _, text) in enumerate(candidates)
            if cls._printed_page_number(text) == requested_page
        ]
        if printed_matches:
            return max(printed_matches, key=lambda index: scores[index])

        pdf_matches = [
            index
            for index, (_, pdf_page, _) in enumerate(candidates)
            if pdf_page == requested_page
        ]
        if pdf_matches:
            return max(pdf_matches, key=lambda index: scores[index])
        return None

    @staticmethod
    def _printed_page_number(page_text: str) -> int | None:
        """Read the printed page number near the top of parsed page text."""
        nonempty_lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        for line in nonempty_lines[:8]:
            match = re.match(r"^(\d{1,4})(?:\s|$)", line)
            if match is not None:
                return int(match.group(1))
        return None

    @classmethod
    def _resolve_contents_reference(
        cls,
        question: str,
        candidates: list[tuple[str, int, str]],
        selected_index: int,
    ) -> int | None:
        """Follow a relevant table-of-contents entry to its actual PDF page."""
        document, _, selected_text = candidates[selected_index]
        if "contents" not in selected_text[:500].lower():
            return None

        printed_pages = {
            printed_page: index
            for index, (candidate_document, _, text) in enumerate(candidates)
            if candidate_document == document
            and (printed_page := cls._printed_page_number(text)) is not None
        }
        reference_lines: list[str] = []
        reference_targets: list[int] = []
        for line in selected_text.splitlines():
            match = re.search(r"(\d{1,4})(?:\.0+)?\s*\|?\s*$", line)
            if match is None:
                continue
            target = printed_pages.get(int(match.group(1)))
            if target is not None:
                reference_lines.append(line)
                reference_targets.append(target)
        if not reference_lines:
            return None

        line_scores = cls._rank_pages(question, reference_lines)
        best_line = max(range(len(reference_lines)), key=lambda index: line_scores[index])
        if line_scores[best_line] <= 0:
            return None
        return reference_targets[best_line]

    @staticmethod
    def _search_tokens(text: str) -> list[str]:
        return [
            token
            for token in _SEARCH_TOKEN_RE.findall(text.lower())
            if token not in _SEARCH_STOPWORDS and (len(token) > 1 or token.isdigit())
        ]

    @classmethod
    def _rank_pages(cls, question: str, pages: list[str]) -> list[float]:
        """Return deterministic lexical scores for page text.

        BM25 supplies the base score. Exact query phrases and matches near the
        beginning of a page receive small boosts because Treasury Bulletin
        table titles and column headings usually appear before the table body.
        """
        if not pages:
            return []
        query_terms = Counter(cls._search_tokens(question))
        if not query_terms:
            return [0.0] * len(pages)

        page_terms = [cls._search_tokens(page) for page in pages]
        term_counts = [Counter(terms) for terms in page_terms]
        document_frequency = {
            term: sum(term in counts for counts in term_counts) for term in query_terms
        }
        average_length = sum(len(terms) for terms in page_terms) / len(page_terms) or 1.0
        page_count = len(pages)
        inverse_frequencies = {
            term: math.log(1.0 + (page_count - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }
        k1 = 1.5
        # Parsed tables are much longer than prose pages; strong BM25 length
        # normalization would otherwise prefer short introductions over the
        # table containing the requested row or column.
        b = 0.2
        scores: list[float] = []
        query_token_sequence = cls._search_tokens(question)
        query_phrases = {
            tuple(query_token_sequence[start : start + size])
            for size in (2, 3, 4)
            for start in range(len(query_token_sequence) - size + 1)
        }
        for page, terms, counts in zip(pages, page_terms, term_counts, strict=True):
            length_norm = 1.0 - b + b * len(terms) / average_length
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                frequency_weight = frequency * (k1 + 1.0) / (
                    frequency + k1 * length_norm
                )
                inverse_frequency = inverse_frequencies[term]
                score += inverse_frequency**1.5 * frequency_weight * (
                    1.0 + math.log(query_frequency)
                )

            heading_counts = Counter(cls._search_tokens(page[:1_500]))
            for term in query_terms:
                if heading_counts.get(term, 0):
                    score += 1.5 * inverse_frequencies[term]

            page_phrases = {
                tuple(terms[start : start + size])
                for size in (2, 3, 4)
                for start in range(len(terms) - size + 1)
            }
            for phrase in query_phrases.intersection(page_phrases):
                score += 0.75 * sum(
                    inverse_frequencies[term]
                    for term in phrase
                )

            positions = {term: [] for term in query_terms}
            for position, term in enumerate(terms):
                if term in positions:
                    positions[term].append(position)
            for left, right in set(
                zip(query_token_sequence, query_token_sequence[1:], strict=False)
            ):
                distances = (
                    right_position - left_position
                    for left_position in positions[left]
                    for right_position in positions[right]
                    if 0 < right_position - left_position <= 20
                )
                closest = min(distances, default=None)
                if closest is not None:
                    score += (
                        0.6
                        * (inverse_frequencies[left] + inverse_frequencies[right])
                        / math.sqrt(closest)
                    )
            scores.append(score)
        return scores

    @staticmethod
    @lru_cache(maxsize=128)
    def _read_marked_pages(path: Path) -> tuple[tuple[int, str], ...]:
        text = path.read_text(encoding="utf-8")
        markers = list(_PAGE_MARKER_RE.finditer(text))
        pages: list[tuple[int, str]] = []
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            pages.append((int(marker.group(1)), text[marker.end() : end].strip()))
        return tuple(pages)

    async def execute(self, action: str | dict[str, str]) -> StepOutput:
        """Execute one shell command or process the terminal ``submit`` action."""
        self._require_active()
        if isinstance(action, dict):
            if action.get("type") != "command":
                raise FormatError(f"Unsupported action type: {action.get('type')}")
            command = action.get("command", "")
        else:
            command = action
        command = command.strip()
        if not command:
            raise FormatError("Empty command.")

        name = command.split(None, 1)[0]
        if name in {"submit", "view_page"}:
            try:
                parts = shlex.split(command)
            except ValueError as exc:
                return StepOutput(f"Command parse error: {exc}")
        else:
            parts = []

        if name == "submit":
            submission = " ".join(parts[1:]).strip()
            if not submission:
                return StepOutput(
                    'Submit requires an answer, for example: submit "123.45"'
                )
            raise Submitted(submission)

        if name == "view_page":
            if not self.config.enable_visual_pdf_tool:
                return StepOutput(
                    "Unknown command. This environment is text-only; use shell commands "
                    'or `submit "<answer>"`.'
                )
            return await self._view_page(parts)

        return await self._execute_command(command)

    async def _view_page(self, parts: list[str]) -> StepOutput:
        """Return one PDF page as both parsed text and a rendered image."""
        started = time.monotonic()
        if len(parts) != 3:
            return StepOutput(
                "Usage: view_page <document> <PDF-page-number>\n"
                "Example: view_page treasury_bulletin_1954_01 13"
            )

        document = Path(parts[1]).name
        stem = Path(document).stem
        if not stem or not re.fullmatch(r"[A-Za-z0-9_.-]+", stem):
            return StepOutput(f"Invalid document name: {parts[1]!r}")
        try:
            page_number = int(parts[2])
        except ValueError:
            return StepOutput(f"PDF page number must be an integer: {parts[2]!r}")
        if page_number < 1:
            return StepOutput("PDF page number must be at least 1.")

        text_path = self.page_text_dir / f"{stem}.txt"
        pdf_path = (self.pdf_dir / f"{stem}.pdf") if self.pdf_dir is not None else None
        if not text_path.is_file():
            return StepOutput(f"Page-indexed text document not found: {stem}.txt")
        if pdf_path is None or not pdf_path.is_file():
            return StepOutput(f"PDF document not found: {stem}.pdf")

        try:
            page_text = self._extract_marked_page_text(text_path, page_number)
        except (OSError, UnicodeError) as exc:
            return StepOutput(f"Could not read page text for {stem}: {exc}")
        if page_text is None:
            return StepOutput(
                f"Page {page_number} is not present in {text_path.name}. Search "
                "the `--- PAGE N ---` markers in officeqa_pages/ and try again."
            )

        try:
            image = self._render_pdf_page(
                pdf_path,
                page_number,
                self.config.pdf_render_dpi,
            )
        except ModuleNotFoundError:
            return StepOutput(
                "Visual PDF rendering requires Pillow and pypdfium2; install the "
                "'officeqa-rl[inference]' optional dependencies."
            )
        except (IndexError, ValueError) as exc:
            return StepOutput(f"Could not render PDF page {page_number}: {exc}")
        except Exception as exc:
            return StepOutput(
                f"Could not render {pdf_path.name} page {page_number} "
                f"({type(exc).__name__}): {exc}"
            )

        output = self._truncate_output(
            f"Document: {stem}\nPDF page: {page_number}\n\n"
            f"Parsed page text:\n{page_text}"
        ).rstrip()
        self.command_history.append(
            {
                "command": f"view_page {stem} {page_number}",
                "runtime": "host",
                "returncode": 0,
                "duration": time.monotonic() - started,
                "timed_out": False,
                "document": stem,
                "page_number": page_number,
            }
        )
        return StepOutput(output=output, image=image)

    @staticmethod
    def _extract_marked_page_text(path: Path, page_number: int) -> str | None:
        for marked_page_number, page_text in FullCorpusEnvironment._read_marked_pages(
            path
        ):
            if marked_page_number == page_number:
                return page_text
        return None

    @staticmethod
    def _render_pdf_page(pdf_path: Path, page_number: int, dpi: int) -> Any:
        pdfium = importlib.import_module("pypdfium2")
        document = pdfium.PdfDocument(str(pdf_path))
        try:
            if page_number > len(document):
                raise IndexError(
                    f"page {page_number} exceeds the PDF's {len(document)} pages"
                )
            page = document[page_number - 1]
            try:
                bitmap = page.render(scale=dpi / 72.0)
                try:
                    # Detach the Pillow image from PDFium-owned bitmap memory
                    # before closing the page and document.
                    return bitmap.to_pil().convert("RGB").copy()
                finally:
                    bitmap.close()
            finally:
                page.close()
        finally:
            document.close()

    async def _execute_command(self, command: str) -> StepOutput:
        started = time.monotonic()
        try:
            returncode, output, timed_out = await self._run_command(command)
            elapsed = time.monotonic() - started
            if timed_out:
                observation = (
                    f"Command timed out after {self.config.command_timeout_sec}s | "
                    f"Duration: {elapsed:.2f}s\n{self._truncate_output(output)}"
                )
                self._record_command(
                    command=command,
                    returncode=None,
                    duration=elapsed,
                    timed_out=True,
                )
                return StepOutput(observation.rstrip())

            header = f"Exit code: {returncode} | Duration: {elapsed:.2f}s\n"
            observation = header + self._truncate_output(output)
            self._record_command(
                command=command,
                returncode=returncode,
                duration=elapsed,
                timed_out=False,
            )
            observation = observation.rstrip() or header.rstrip()
            if returncode == 0 and self.config.enable_visual_pdf_tool:
                return self._attach_pdf_to_search_observation(command, observation)
            return StepOutput(observation)
        except Exception as exc:  # command failures become agent observations
            elapsed = time.monotonic() - started
            output = self._exception_output(exc)
            observation = (
                f"Command execution error ({type(exc).__name__}): {exc} | "
                f"Duration: {elapsed:.2f}s\n{self._truncate_output(output)}"
            )
            self._record_command(
                command=command,
                returncode=-1,
                duration=elapsed,
                timed_out=False,
                exception_type=type(exc).__name__,
            )
            return StepOutput(observation.rstrip())

    async def _run_command(self, command: str) -> tuple[int | None, str, bool]:
        if self.config.sandbox_runtime == "docker":
            if self.command_container_id is None:
                raise RuntimeError("Docker command container was not started")
            cmd = [*self._docker_command(), "exec", "-w", "/workspace"]
            for key, value in self._command_env(docker=True).items():
                cmd.extend(["-e", f"{key}={value}"])
            cmd.extend([self.command_container_id, "bash", "-lc", command])
            return await self._run_process(
                cmd,
                timeout=self.config.command_timeout_sec,
            )
        return await self._run_process(
            command,
            timeout=self.config.command_timeout_sec,
            cwd=self.workspace,
            env={**os.environ, **self._command_env(docker=False)},
            shell=True,
        )

    async def _start_command_container(self) -> None:
        if not self.config.sandbox_image:
            raise ValueError("sandbox_image is required for the Docker sandbox runtime")
        cmd = [
            *self._docker_command(),
            "run",
            "-d",
            "--name",
            self._container_name,
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev",
            "-v",
            f"{self.workspace}:/workspace",
            "-v",
            f"{self.corpus_dir}:/workspace/officeqa_corpus:ro",
        ]
        if self.pdf_dir is not None:
            cmd.extend(["-v", f"{self.pdf_dir}:/workspace/officeqa_pdfs:ro"])
        if self.config.enable_visual_pdf_tool:
            cmd.extend(
                ["-v", f"{self.page_text_dir}:/workspace/officeqa_pages:ro"]
            )
        cmd.extend(
            [
                "-w",
                "/workspace",
                self.config.sandbox_image,
                "sleep",
                "infinity",
            ]
        )
        returncode, output, timed_out = await self._run_process(
            cmd,
            timeout=max(120, self.config.command_timeout_sec),
        )
        if timed_out:
            raise RuntimeError("Timed out starting Docker command container")
        output = output.strip()
        if returncode != 0:
            raise RuntimeError(
                f"Failed to start Docker command container: {output}"
            )
        self.command_container_id = output

    @staticmethod
    def _docker_command() -> list[str]:
        """Return the Docker CLI prefix for this evaluator process.

        Container-Optimized OS nodes expose Docker only through passwordless
        sudo.  Keep that exception scoped to the evaluator's fixed run/exec/rm
        commands rather than installing a general root-equivalent PATH shim.
        """
        if os.environ.get("OFFICEQA_DOCKER_USE_SUDO") == "1":
            return ["sudo", "-n", "docker"]
        return ["docker"]

    async def _run_process(
        self,
        command: str | list[str],
        *,
        timeout: float,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        shell: bool = False,
    ) -> tuple[int | None, str, bool]:
        """Run a process without blocking on its completion.

        Output is file-backed so a verbose child cannot fill a pipe while this
        coroutine polls. Polling also avoids relying on event-loop SIGCHLD
        delivery, which is unavailable in some rollout sandboxes.
        """
        with tempfile.TemporaryFile() as output_file:
            process = subprocess.Popen(
                command,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                shell=shell,
                executable="/bin/bash" if shell else None,
                stdout=output_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            try:
                deadline = asyncio.get_running_loop().time() + timeout
                while process.poll() is None:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        timed_out = True
                        await self._terminate_process(process)
                        break
                    await asyncio.sleep(min(0.05, remaining))
            except BaseException:
                await self._terminate_process(process)
                raise

            returncode = process.poll()
            output_file.seek(0)
            output = output_file.read().decode("utf-8", errors="replace")
            return returncode, output, timed_out

    @staticmethod
    async def _terminate_process(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                process.kill()
            except ProcessLookupError:
                return
        for _ in range(100):
            if process.poll() is not None:
                return
            await asyncio.sleep(0.01)

    def _command_env(self, *, docker: bool) -> dict[str, str]:
        workspace = "/workspace" if docker else str(self.workspace)
        corpus = (
            "/workspace/officeqa_corpus"
            if docker
            else str(self.workspace / "officeqa_corpus")
        )
        pdf_dir = (
            "/workspace/officeqa_pdfs"
            if docker and self.pdf_dir is not None
            else str(self.pdf_dir or "")
        )
        command_env = {
            "OFFICEQA_WORKSPACE": workspace,
            "OFFICEQA_CORPUS_DIR": corpus,
            "OFFICEQA_PDF_DIR": pdf_dir,
            "OFFICEQA_UID": self.task.uid,
            "OFFICEQA_RUN_ID": self.run_id,
        }
        if self.config.enable_visual_pdf_tool:
            command_env["OFFICEQA_PAGE_TEXT_DIR"] = (
                "/workspace/officeqa_pages"
                if docker
                else str(self.page_text_dir)
            )
        return command_env

    def _record_command(
        self,
        *,
        command: str,
        returncode: int | None,
        duration: float,
        timed_out: bool,
        exception_type: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "command": command,
            "runtime": self.config.sandbox_runtime,
            "returncode": returncode,
            "duration": duration,
            "timed_out": timed_out,
        }
        if self.command_container_id is not None:
            record["container_id"] = self.command_container_id
        if exception_type is not None:
            record["exception_type"] = exception_type
        self.command_history.append(record)

    @staticmethod
    def _exception_output(exc: BaseException) -> str:
        output = getattr(exc, "stdout", None) or getattr(exc, "output", None) or ""
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return str(output)

    def _truncate_output(self, output: str) -> str:
        keep = self.config.max_observation_chars
        if len(output) <= keep:
            return output
        omitted = len(output) - keep
        left = keep // 2
        right = keep - left
        return (
            output[:left]
            + f"\n\n[... omitted {omitted} characters ...]\n\n"
            + output[-right:]
        )

    def get_info(self) -> dict[str, Any]:
        """Return diagnostics safe to serialize with a trajectory."""
        info = {
            "uid": self.task.uid,
            "run_id": self.run_id,
            "workspace": str(self.workspace),
            "corpus_dir": str(self.corpus_dir),
            "pdf_dir": str(self.pdf_dir) if self.pdf_dir is not None else None,
            "source_files": self.task.source_files,
            "sandbox_runtime": self.config.sandbox_runtime,
            "sandbox_image": self.config.sandbox_image,
            "command_container_id": self.command_container_id,
            "command_history": list(self.command_history),
        }
        if self.config.enable_visual_pdf_tool:
            info.update(
                {
                    "page_text_dir": str(self.page_text_dir),
                    "visual_pdf_tool_enabled": True,
                    "automatic_page_retrievals": list(
                        self.automatic_page_retrievals
                    ),
                }
            )
        return info

    async def cleanup(self) -> None:
        """Stop the command container if present; safe to call repeatedly."""
        target = self.command_container_id
        if target is None and self.config.sandbox_runtime == "docker":
            target = self._container_name
        if target is not None:
            await self._remove_container(target)
        self.command_container_id = None
        self._active = False

    async def _remove_container(self, target: str) -> None:
        try:
            await self._run_process(
                [*self._docker_command(), "rm", "-f", target],
                timeout=60,
            )
        except Exception:
            pass

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("Environment must be used inside 'async with'")


__all__ = ["FullCorpusEnvironment"]
