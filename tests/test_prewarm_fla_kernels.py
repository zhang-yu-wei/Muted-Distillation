import argparse

import pytest

from scripts.prewarm_fla_kernels import (
    balanced_cu_seqlens,
    build_case_plan,
    norm_bucket,
    parse_int_ranges,
    token_lengths_for_norm_buckets,
)


def test_parse_int_ranges() -> None:
    assert parse_int_ranges("1-3, 2, 32, 34-35") == [1, 2, 3, 32, 34, 35]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_int_ranges("3-1")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_int_ranges("0")


def test_norm_bucket_lengths_cover_training_and_padding() -> None:
    lengths = token_lengths_for_norm_buckets(
        min_tokens=32768,
        max_tokens=65536,
        num_v_heads=32,
        padding_buckets=1,
    )
    assert [norm_bucket(length, 32) for length in lengths] == list(range(16, 34))
    assert lengths[0] == 32768
    assert lengths[-1] == 65537


def test_case_plan_covers_both_key_spaces_without_cross_product() -> None:
    lengths = token_lengths_for_norm_buckets(32768, 65536, 32, 1)
    counts = parse_int_ranges("1-3,32-65")
    cases = build_case_plan(lengths, counts, num_v_heads=32)
    assert len(cases) == len(counts)
    assert {case.packed_sequences for case in cases} == set(counts)
    assert {case.norm_bucket for case in cases} == set(range(16, 34))


def test_balanced_cu_seqlens() -> None:
    boundaries = balanced_cu_seqlens(total_tokens=17, packed_sequences=4)
    assert boundaries == [0, 5, 9, 13, 17]
    assert all(right > left for left, right in zip(boundaries, boundaries[1:]))
