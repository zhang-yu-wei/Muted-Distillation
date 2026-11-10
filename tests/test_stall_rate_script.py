"""scripts/stall_rate.py — the readout for officeqa_opd_stall_penalty."""
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stall_rate.py"


def _turn(uid, response, exit_status):
    return {
        "response": response,
        "metadata": {"uid": uid, "exit_status": exit_status},
    }


@pytest.fixture
def run_dir(tmp_path):
    dumps = tmp_path / "debug_rollout"
    dumps.mkdir()
    samples = [
        # Stalled: think never closes, so no command either.
        _turn("a", "<think>Let me reconsider", "limits_exceeded"),
        # Closed think but no fenced command — a symptom, not a stall.
        _turn("a", "<think>Hmm.</think>\n\nI am lost.", "limits_exceeded"),
        # Healthy: closed think and a command.
        _turn("b", "<think>Done.</think>\n\n```bash\nsubmit 42\n```", "submitted"),
        _turn("b", "<think>Again.</think>\n\n```bash\nls\n```", "submitted"),
    ]
    torch.save({"rollout_id": 0, "samples": samples}, dumps / "rollout_eval_7.pt")
    return tmp_path


def _run(*args):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def test_counts_stalls_fenceless_turns_and_capped_episodes(run_dir):
    row = _run(str(run_dir)).splitlines()[1].split()
    step, turns, stall, nocmd, episodes, limits = row
    assert step == "7" and turns == "4"
    assert stall == "25.0%", "1 of 4 turns never closes </think>"
    assert nocmd == "50.0%", "2 of 4 turns carry no fenced command"
    assert episodes == "2"
    assert limits == "50.0%", "1 of 2 episodes ended limits_exceeded"


def test_accepts_the_debug_rollout_directory_itself(run_dir):
    assert _run(str(run_dir / "debug_rollout")) == _run(str(run_dir))


def test_missing_dumps_is_an_error_not_an_empty_table(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "no debug_rollout directory" in result.stderr
