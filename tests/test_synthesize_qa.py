"""Tests for the author-only synthesis orchestration."""

from types import SimpleNamespace

from officeqa_rl import synthesis
from scripts.synthesize_qa import CandidateJob, Synthesizer, parse_args


class FakeAuthor:
    def __init__(self, reply: str):
        self.reply = reply

    def generate(self, prompt, **kwargs):
        del prompt, kwargs
        return self.reply


def test_author_only_is_the_default_and_only_mode():
    args = parse_args([])
    assert not hasattr(args, "author_only")


def test_author_only_accepts_schema_valid_row_without_old_gates():
    # The raw value is absent from the excerpt and the computation disagrees
    # with the answer. Author-only mode intentionally keeps the row; Kimi
    # result agreement validates the answer later in Step 6.
    reply = """{
      "question": "What was the synthetic amount in millions of dollars?",
      "answer": "123",
      "difficulty_tier": "T1",
      "cells_used": [{"row_label": "missing", "col_label": "value", "raw_value": "9,999"}],
      "computation": "9999",
      "units_and_rounding": "millions of dollars"
    }"""
    orchestrator = Synthesizer.__new__(Synthesizer)
    orchestrator.args = SimpleNamespace(author_temperature=0.3)
    orchestrator.author = FakeAuthor(reply)
    region = synthesis.TableRegion(
        source_file="source.txt",
        start_line=1,
        end_line=6,
        context="Synthetic table",
        table_text="| year | value | note |\n|---|---|---|\n| 1950 | 1 | x |\n" * 4,
        n_rows=4,
        n_cols=3,
        numeric_density=0.5,
    )

    result = orchestrator.run_candidate(
        CandidateJob(candidate_id=1, tier="T1", regions=[region]),
        exemplars=[],
    )

    assert result.accepted is True
    assert result.rejection is None
    assert result.author_status == {"parsed": True, "accepted": True}
