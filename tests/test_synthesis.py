"""Unit tests for the synthetic-QA generation logic (no network)."""

import random

import pytest

from officeqa_rl import synthesis


TOY_TABLE = """Section 1 - Budget Receipts

Table 2.- Monthly Receipts, Calendar Year 1953
(In millions of dollars)

| Period | Receipts | Expenditures | Surplus |
| --- | --- | --- | --- |
| January | 5,061 | 5,737 | (676) |
| February | 5,479 | 5,595 | -116.0 |
| March | 10,502 | 6,187 | 4,315 |
| April | 2,849 | 6,362 | -3,513 |
| May | 4,380 | 6,241 | −1,862 |

Some trailing prose after the table.
"""


def toy_region() -> synthesis.TableRegion:
    regions = synthesis.extract_table_regions(TOY_TABLE, "treasury_bulletin_1954_02.txt")
    assert len(regions) == 1
    return regions[0]


class TestTableExtraction:
    def test_extracts_qualifying_table_with_context(self):
        region = toy_region()
        assert region.n_rows == 5
        assert region.n_cols == 4
        assert region.numeric_density > 0.5
        assert "Monthly Receipts" in region.context
        assert "| January | 5,061" in region.table_text

    def test_small_or_texty_tables_are_filtered(self):
        small = "| a | b |\n| --- | --- |\n| 1 | 2 |\n"
        assert synthesis.extract_table_regions(small, "f.txt") == []
        texty = "\n".join(
            ["| name | role | city |", "| --- | --- | --- |"]
            + ["| alice | admin | boston |"] * 6
        )
        assert synthesis.extract_table_regions(texty, "f.txt") == []

    def test_window_around_caps_size(self):
        text = "\n".join(f"line {i}" for i in range(2000))
        region = synthesis.TableRegion(
            source_file="f.txt",
            start_line=1000,
            end_line=1010,
            context="",
            table_text="",
            n_rows=5,
            n_cols=3,
            numeric_density=1.0,
        )
        window = synthesis.window_around(text, region, max_chars=600)
        assert len(window) <= 700
        assert "line 1005" in window


class TestCellParsing:
    @pytest.mark.parametrize(
        ("raw", "value"),
        [
            ("2,602", 2602.0),
            ("(1,613)", -1613.0),
            ("3,293r", 3293.0),
            ("-21490.0", -21490.0),
            ("57%", 57.0),
            ("−1,862", -1862.0),
        ],
    )
    def test_cell_numeric_value(self, raw, value):
        assert synthesis.cell_numeric_value(raw) == value

    def test_nan_is_not_numeric_cell(self):
        assert synthesis.extract_table_regions(
            "| a | b | c |\n| --- | --- | --- |\n" + "| nan | nan | nan |\n" * 5,
            "f.txt",
        ) == []


class TestGrounding:
    def test_verbatim_value_found_after_normalization(self):
        sources = {"f.txt": synthesis.normalize_for_grounding("total:  5,061 \n end −1,862")}
        ok, failures = synthesis.check_grounding(
            [{"raw_value": "5,061"}, {"raw_value": "-1,862"}], sources
        )
        assert ok, failures

    def test_invented_value_fails(self):
        sources = {"f.txt": "receipts 5,061"}
        ok, failures = synthesis.check_grounding([{"raw_value": "9,999"}], sources)
        assert not ok
        assert failures == ["9,999"]


class TestComputation:
    def test_reexecution_matches_authored_answer(self):
        cells = [{"raw_value": "5,061"}, {"raw_value": "5,479"}]
        ok, detail = synthesis.check_computation("10,540", "5061 + 5479", cells)
        assert ok, detail

    def test_mismatched_answer_rejected(self):
        cells = [{"raw_value": "5,061"}, {"raw_value": "5,479"}]
        ok, detail = synthesis.check_computation("10,000", "5061 + 5479", cells)
        assert not ok

    def test_invented_literal_rejected(self):
        cells = [{"raw_value": "5,061"}]
        ok, detail = synthesis.check_computation("15,061", "5061 + 10000", cells)
        assert not ok
        assert "literals" in detail

    def test_small_constants_allowed(self):
        cells = [{"raw_value": "5,061"}, {"raw_value": "5,479"}]
        ok, _ = synthesis.check_computation("5,270", "round((5061 + 5479) / 2)", cells)
        assert ok

    @pytest.mark.parametrize(
        "expression",
        ["__import__('os')", "x + 1", "(1).__class__", "[1 for _ in range(2)]"],
    )
    def test_safe_eval_rejects_non_arithmetic(self, expression):
        with pytest.raises((ValueError, SyntaxError)):
            synthesis.safe_eval_expression(expression)

    def test_percent_change_expression(self):
        ok, detail = synthesis.check_computation(
            "-11.65",
            "round((5595 - 6334) / 6334 * 100, 2)",
            [{"raw_value": "5,595"}, {"raw_value": "6,334"}],
        )
        assert ok, detail


class TestDedup:
    def test_near_duplicate_question_detected(self):
        question = "What were the total expenditures (in millions of nominal dollars) for U.S national defense in the calendar year of 1940?"
        near = "What were the total expenditures (in millions of nominal dollars) for US national defense in calendar year 1940?"
        score, _ = synthesis.max_question_similarity(near, [question])
        assert score >= synthesis.JACCARD_THRESHOLD

    def test_different_question_passes(self):
        question = "What were the total expenditures for national defense in 1940?"
        other = "What was the mean of monthly internal revenue collections during FY 1953?"
        score, _ = synthesis.max_question_similarity(other, [question])
        assert score < synthesis.JACCARD_THRESHOLD

    def test_answer_collision_same_file(self):
        rows = [
            {"source_files": "treasury_bulletin_1941_01.txt", "answer": "2,602"},
        ]
        assert synthesis.answer_collision(
            ["treasury_bulletin_1941_01.txt"], "2602", rows
        )
        assert not synthesis.answer_collision(
            ["treasury_bulletin_1950_01.txt"], "2602", rows
        )


class TestAuthoredParsing:
    def test_parses_fenced_json(self):
        reply = """```json
{"question": "What were total receipts in millions of dollars in January 1953?",
 "answer": "5,061", "difficulty_tier": "T1",
 "cells_used": [{"row_label": "January", "col_label": "Receipts", "raw_value": "5,061"}],
 "computation": "5061", "units_and_rounding": "millions"}
```"""
        authored = synthesis.parse_authored_json(reply)
        assert authored["answer"] == "5,061"

    def test_missing_field_raises(self):
        with pytest.raises(ValueError, match="missing fields"):
            synthesis.parse_authored_json('{"question": "long enough question text here?"}')

    def test_skip_is_passed_through(self):
        assert synthesis.parse_authored_json('{"skip": true}') == {"skip": True}


class TestSampling:
    def test_round_robin_spreads_across_buckets(self):
        buckets = {"1940s": list(range(10)), "1950s": list(range(10, 20)), "1960s": [20]}
        picked = synthesis.round_robin_sample(buckets, 7, random.Random(0))
        assert len(picked) == 7
        assert 20 in picked  # small bucket still represented

    def test_find_cross_year_pair_matches_heading(self):
        def region(source_file, heading):
            return synthesis.TableRegion(
                source_file=source_file,
                start_line=1,
                end_line=10,
                context=heading,
                table_text="",
                n_rows=5,
                n_cols=3,
                numeric_density=1.0,
            )

        base = region("treasury_bulletin_1954_02.txt", "Summary of Federal fiscal operations")
        same_year = region("treasury_bulletin_1954_05.txt", "Summary of Federal fiscal operations")
        other_year = region("treasury_bulletin_1960_02.txt", "Summary of Federal fiscal operations")
        unrelated = region("treasury_bulletin_1960_03.txt", "Market quotations on securities")
        pair = synthesis.find_cross_year_pair(base, [same_year, unrelated, other_year])
        assert pair is other_year
