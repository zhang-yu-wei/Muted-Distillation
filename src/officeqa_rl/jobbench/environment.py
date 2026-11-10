"""Sandbox implementing mini-swe-agent's ``Environment`` protocol for JobBench.

The workspace mirrors what the official JobBench runner builds for a CLI agent::

    <workspace>/task_folder/   copy of the task's task_folder (instructions +
                               local reference files); writable scratch space
    <workspace>/output/        empty; only the final deliverables belong here,
                               and only this directory is graded

``RUBRICS.json`` and ``task_card.md`` are never staged — they carry the gold
answers.

Commands run through ``docker exec`` against a per-episode container with no
network and a memory cap, the same Docker-outside-of-Docker arrangement the
OfficeQA and SpreadsheetBench environments use. ``sandbox_runtime="local"``
runs them as plain subprocesses in the workspace instead, for tests and local
debugging; it offers no isolation, so never point it at an untrusted policy.

Submission follows the upstream mini-swe-agent contract: the agent runs
``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` and the environment raises
``Submitted``. Unlike upstream, a submission with an empty ``output/`` is
refused with an observation instead (``require_output_files``), because an
episode that produced no deliverable scores zero on every rubric and is worth a
retry rather than a terminated episode.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from officeqa_rl.jobbench._msa import Submitted, recursive_merge
from officeqa_rl.jobbench.dataset import (
    INSTRUCTIONS_NAME,
    JobBenchTask,
    SearchFilePolicy,
    TASK_FOLDER_NAME,
)

logger = logging.getLogger("officeqa_rl.jobbench.environment")

#: Upstream mini-swe-agent submission sentinel. Keep in sync with the prompt.
SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

#: Mount point of the episode workspace inside the container.
CONTAINER_WORKSPACE = "/workspace"

OUTPUT_DIR_NAME = "output"


@dataclass(frozen=True)
class JobBenchEnvironmentConfig:
    """Settings for one JobBench episode sandbox."""

    workspace_root: Path
    sandbox_image: str | None = None
    sandbox_runtime: str = "docker"
    command_timeout_sec: int = 120
    max_observation_chars: int = 12000
    container_memory_limit: str = "8g"
    container_lifetime: str = "2h"
    network: bool = False
    # Hardening for a container that runs policy-written code. The workspace
    # bind mount stays writable under --read-only, and HOME points at the
    # tmpfs, so ordinary python and shell work is unaffected. A training host
    # needs these: one unbounded allocation or fork bomb in a rollout can trip
    # the node memory monitor and take the whole job down.
    read_only_rootfs: bool = True
    pids_limit: int = 512
    tmpfs_tmp_size: str = "1g"
    # Run container commands as the host user. Without this the agent's files
    # land in the bind-mounted workspace owned by root, and any directory it
    # creates cannot be removed afterwards by the (non-root) host process that
    # grades and cleans up. HOME already points at the tmpfs, so nothing needs
    # a passwd entry.
    run_as_host_user: bool = True
    search_file_policy: SearchFilePolicy = SearchFilePolicy.OMIT
    require_output_files: bool = True
    docker_executable: str = field(
        default_factory=lambda: os.getenv("OFFICEQA_DOCKER_EXECUTABLE", "docker")
    )
    interpreter: tuple[str, ...] = ("bash", "-lc")


class JobBenchEnvironment:
    """A per-episode workspace plus a command runner for one JobBench task."""

    def __init__(
        self,
        config: JobBenchEnvironmentConfig,
        task: JobBenchTask,
        *,
        run_id: str,
    ) -> None:
        self.config = config
        self.task = task
        self.run_id = run_id
        # Absolute, because the docker -v source must be a path: a relative
        # source is interpreted as a *named volume* and the run fails with
        # "includes invalid characters for a local volume name". os.path.abspath
        # rather than Path.resolve so a symlinked workspace root keeps the name
        # the caller gave it — under DooD that path has to match on the host.
        self.workspace = Path(
            os.path.abspath(config.workspace_root / _workspace_name(task.uid, run_id))
        )
        self.container_id: str | None = None
        self.container_name: str | None = None
        self.command_history: list[dict[str, Any]] = []
        self._active = False
        self._staged_search_files: tuple[str, ...] = ()

    # ---------------------------------------------------------------- lifecycle

    def __enter__(self) -> JobBenchEnvironment:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.cleanup()
        return False

    def start(self) -> None:
        """Stage the workspace and, for the docker runtime, boot the container."""
        if self._active:
            return
        self._prepare_workspace()
        if self.config.sandbox_runtime == "docker":
            self._start_container()
        self._active = True

    def cleanup(self) -> None:
        """Remove the container. The workspace is left for the caller to grade."""
        self._active = False
        target = self.container_id or self.container_name
        if target is None:
            return
        self.container_id = None
        self.container_name = None
        subprocess.Popen(
            f"(timeout 60 {self.config.docker_executable} rm -f {target}) "
            ">/dev/null 2>&1 &",
            shell=True,
        )

    # ---------------------------------------------------------------- workspace

    @property
    def host_task_folder(self) -> Path:
        return self.workspace / TASK_FOLDER_NAME

    @property
    def host_output_dir(self) -> Path:
        return self.workspace / OUTPUT_DIR_NAME

    @property
    def container_task_folder(self) -> str:
        return f"{CONTAINER_WORKSPACE}/{TASK_FOLDER_NAME}"

    @property
    def container_output_dir(self) -> str:
        return f"{CONTAINER_WORKSPACE}/{OUTPUT_DIR_NAME}"

    def _prepare_workspace(self) -> None:
        self.config.workspace_root.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=False, exist_ok=False)
        shutil.copytree(self.task.task_folder, self.host_task_folder)
        self.host_output_dir.mkdir()
        if self.config.search_file_policy is SearchFilePolicy.STAGE:
            self._staged_search_files = self._stage_search_files()

    def _stage_search_files(self) -> tuple[str, ...]:
        """Copy ``files_required_to_search/`` in beside the local references.

        The instructions name these documents as obtained elsewhere (the
        official runner expects the agent to find them on the web). Staging
        them makes the task solvable without network access and makes the score
        incomparable with published JobBench numbers.
        """
        source = self.task.search_files_dir
        if not source.is_dir():
            return ()
        staged: list[str] = []
        for entry in sorted(source.rglob("*")):
            if not entry.is_file():
                continue
            relative = entry.relative_to(source)
            destination = self.host_task_folder / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, destination)
            staged.append(relative.as_posix())
        return tuple(staged)

    def output_files(self) -> list[Path]:
        """Return the deliverables the judge will read, in a stable order."""
        if not self.host_output_dir.is_dir():
            return []
        return sorted(p for p in self.host_output_dir.rglob("*") if p.is_file())

    # ------------------------------------------------------------------ docker

    def _start_container(self) -> None:
        if not self.config.sandbox_image:
            raise ValueError("sandbox_image is required for the docker runtime")
        self.container_name = _container_name(self.task.uid, self.run_id)
        command = [
            self.config.docker_executable,
            "run",
            "-d",
            "--name",
            self.container_name,
            "-w",
            CONTAINER_WORKSPACE,
            "-v",
            f"{self.workspace}:{CONTAINER_WORKSPACE}",
            "--memory",
            self.config.container_memory_limit,
        ]
        if not self.config.network:
            command += ["--network", "none"]
        if self.config.container_memory_limit:
            command += ["--memory-swap", self.config.container_memory_limit]
        if self.config.read_only_rootfs:
            command += ["--read-only"]
        if self.config.pids_limit > 0:
            command += ["--pids-limit", str(self.config.pids_limit)]
        if self.config.tmpfs_tmp_size:
            command += ["--tmpfs", f"/tmp:rw,nosuid,nodev,size={self.config.tmpfs_tmp_size}"]
        if self.config.run_as_host_user:
            command += ["--user", f"{os.getuid()}:{os.getgid()}"]
        command += [self.config.sandbox_image, "sleep", self.config.container_lifetime]
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start JobBench sandbox container: {result.stderr.strip()}"
            )
        self.container_id = result.stdout.strip()

    # ------------------------------------------------------- Environment protocol

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Run one command and return mini-swe-agent's output dict."""
        if not self._active:
            raise RuntimeError("JobBenchEnvironment.start() was not called")
        command = action.get("command", "") if isinstance(action, dict) else str(action)
        started = time.monotonic()
        output = self._run(command, cwd=cwd or self.container_task_folder, timeout=timeout)
        raw_output = output["output"]
        output["output"] = self._truncate(raw_output)
        self.command_history.append(
            {
                "command": command,
                "returncode": output["returncode"],
                "duration_sec": round(time.monotonic() - started, 3),
                "output_chars": len(raw_output),
            }
        )
        self._check_finished(output, raw_output)
        return output

    def _run(self, command: str, *, cwd: str, timeout: int | None) -> dict[str, Any]:
        if self.config.sandbox_runtime == "docker":
            argv = [
                self.config.docker_executable,
                "exec",
                "-w",
                cwd,
                *self._env_flags(),
                str(self.container_id),
                *self.config.interpreter,
                command,
            ]
            run_cwd = None
        else:
            argv = [*self.config.interpreter, command]
            run_cwd = self._local_path(cwd)
        try:
            result = subprocess.run(
                argv,
                cwd=run_cwd,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout or self.config.command_timeout_sec,
                env=self._local_env() if run_cwd is not None else None,
            )
            return {
                "output": result.stdout or "",
                "returncode": result.returncode,
                "exception_info": "",
            }
        except subprocess.TimeoutExpired as error:
            partial = error.output or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            return {
                "output": partial,
                "returncode": -1,
                "exception_info": (
                    f"Command timed out after "
                    f"{timeout or self.config.command_timeout_sec}s and was killed."
                ),
                "extra": {"exception_type": "TimeoutExpired"},
            }
        except Exception as error:  # pragma: no cover - defensive
            return {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {error}",
                "extra": {"exception_type": type(error).__name__},
            }

    def _env_flags(self) -> list[str]:
        flags: list[str] = []
        for key, value in self._command_env().items():
            flags += ["-e", f"{key}={value}"]
        return flags

    def _command_env(self) -> dict[str, str]:
        return {
            "HOME": "/tmp",
            "PAGER": "cat",
            "MANPAGER": "cat",
            "PIP_PROGRESS_BAR": "off",
            "TQDM_DISABLE": "1",
            "JB_TASK_FOLDER": self.container_task_folder,
            "JB_OUTPUT_DIR": self.container_output_dir,
            "JB_UID": self.task.uid,
        }

    def _local_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self._command_env())
        env["HOME"] = str(self.workspace)
        env["JB_TASK_FOLDER"] = str(self.host_task_folder)
        env["JB_OUTPUT_DIR"] = str(self.host_output_dir)
        return env

    def _local_path(self, container_path: str) -> str:
        """Map a container path back onto the host workspace for local runs."""
        if container_path.startswith(CONTAINER_WORKSPACE):
            suffix = container_path[len(CONTAINER_WORKSPACE) :].lstrip("/")
            return str(self.workspace / suffix) if suffix else str(self.workspace)
        return container_path

    def _truncate(self, output: str) -> str:
        limit = self.config.max_observation_chars
        if limit <= 0 or len(output) <= limit:
            return output
        head = output[: limit // 2]
        tail = output[-(limit // 2) :]
        elided = len(output) - len(head) - len(tail)
        return f"{head}\n... [{elided} characters elided] ...\n{tail}"

    def _check_finished(self, output: dict[str, Any], raw_output: str) -> None:
        """Raise ``Submitted`` on the sentinel, matching upstream's contract."""
        lines = raw_output.lstrip().splitlines(keepends=True)
        if not lines or lines[0].strip() != SUBMIT_SENTINEL:
            return
        if output["returncode"] != 0:
            return
        if self.config.require_output_files and not self.output_files():
            output["output"] = (
                f"Cannot submit yet: {self.container_output_dir} is empty. "
                "Write the final deliverables there, then submit again."
            )
            output["returncode"] = 1
            return
        submission = "".join(lines[1:])
        raise Submitted(
            {
                "role": "exit",
                "content": submission,
                "extra": {"exit_status": "Submitted", "submission": submission},
            }
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        """Variables exposed to the prompt templates."""
        task = self.task
        base = {
            "uid": task.uid,
            "split": task.split,
            "occupation": task.occupation,
            "task_id": task.task_id,
            "instructions": task.instructions,
            "task_folder": self.container_task_folder,
            "output_dir": self.container_output_dir,
            "workspace": CONTAINER_WORKSPACE,
            "local_reference_files": list(task.local_reference_files),
            "staged_search_files": list(self._staged_search_files),
            "instructions_file": f"{self.container_task_folder}/{INSTRUCTIONS_NAME}",
            "submit_sentinel": SUBMIT_SENTINEL,
        }
        return recursive_merge(base, platform.uname()._asdict(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.get_info(),
                    "environment_type": (
                        f"{self.__class__.__module__}.{self.__class__.__name__}"
                    ),
                }
            }
        }

    def get_info(self) -> dict[str, Any]:
        return {
            "uid": self.task.uid,
            "benchmark": "jobbench",
            "split": self.task.split,
            "occupation": self.task.occupation,
            "task_id": self.task.task_id,
            "run_id": self.run_id,
            "workspace": str(self.workspace),
            "sandbox_runtime": self.config.sandbox_runtime,
            "sandbox_image": self.config.sandbox_image,
            "container_id": self.container_id,
            "search_file_policy": self.config.search_file_policy.value,
            "staged_search_files": list(self._staged_search_files),
            "command_timeout_sec": self.config.command_timeout_sec,
            "command_history": list(self.command_history),
        }


_CONTAINER_NAME_MAX = 60
_WORKSPACE_NAME_MAX = 120


def _workspace_name(uid: str, run_id: str) -> str:
    # Preserve the unique tail just as _container_name does. Some JobBench
    # occupation slugs are long enough that truncating the combined string
    # discarded run_id, making concurrent pass@k attempts race for one
    # retained workspace.
    #
    # run_id alone is NOT enough to make this unique. Its shape is
    # ``idx<i>-grp<g>-rid<r>-<hash>``, so ``run_id[:12]`` on a short index
    # yields just ``idx34-grp34-`` and drops the hash entirely -- two episodes
    # that reuse an index (successive rounds of the online-SFT batch driver,
    # or a retried episode) then build the same path and the second dies with
    # FileExistsError, because _prepare_workspace uses exist_ok=False.
    # Measured 2026-09-04: every episode of two consecutive SFT rounds failed
    # this way. A fresh uuid suffix makes the name unique regardless of how
    # much of run_id survives truncation, exactly as _container_name does.
    suffix = uuid.uuid4().hex[:6]
    tail = f"-{run_id[:12]}-{suffix}"
    return uid[: _WORKSPACE_NAME_MAX - len(tail)] + tail


def _container_name(uid: str, run_id: str) -> str:
    # The run/random tail is what makes the name unique, so it must survive the
    # length cap: truncating the whole string dropped it for long occupation
    # names (e.g. training_and_development_specialists), and every concurrent
    # sample of that task then raced for one container name and lost.
    suffix = uuid.uuid4().hex[:6]
    tail = f"-{run_id[:8]}-{suffix}"
    head = f"jobbench-{uid.lower().replace('__', '-')}"
    return head[: _CONTAINER_NAME_MAX - len(tail)] + tail


__all__ = [
    "CONTAINER_WORKSPACE",
    "JobBenchEnvironment",
    "JobBenchEnvironmentConfig",
    "OUTPUT_DIR_NAME",
    "SUBMIT_SENTINEL",
]
