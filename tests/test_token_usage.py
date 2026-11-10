from officeqa_rl.token_usage import add_usage, episode_usage, per_task_means, turn_usage
from officeqa_rl.types import GeneratedTurn


def test_openai_style_usage_splits_cached_from_fresh_input():
    turn = GeneratedTurn(
        text="",
        usage={
            "prompt_tokens": 1_000,
            "completion_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 900},
            "completion_tokens_details": {"reasoning_tokens": 80},
        },
    )
    counted = turn_usage(turn)
    assert counted["prompt_tokens"] == 1_000
    assert counted["cached_input_tokens"] == 900
    assert counted["input_tokens"] == 100
    assert counted["output_tokens"] == 120
    assert counted["reasoning_tokens"] == 80
    assert counted["source"] == "provider"


def test_gemini_style_usage_is_normalized():
    turn = GeneratedTurn(
        text="",
        usage={
            "prompt_token_count": 5_000,
            "candidates_token_count": 300,
            "cached_content_token_count": 4_096,
            "thoughts_token_count": 250,
            "total_token_count": 5_300,
        },
    )
    counted = turn_usage(turn)
    assert (counted["prompt_tokens"], counted["cached_input_tokens"]) == (5_000, 4_096)
    assert counted["input_tokens"] == 904
    assert counted["output_tokens"] == 300
    assert counted["reasoning_tokens"] == 250


def test_vllm_prefix_cache_hits_count_as_cached_input():
    turn = GeneratedTurn(
        text="",
        usage={"prompt_tokens": 2_048, "completion_tokens": 64, "cached_tokens": 1_920},
    )
    counted = turn_usage(turn)
    assert counted["cached_input_tokens"] == 1_920
    assert counted["input_tokens"] == 128


def test_token_ids_are_the_fallback_when_no_usage_is_reported():
    turn = GeneratedTurn(text="", prompt_ids=[1] * 700, output_ids=[2] * 30)
    counted = turn_usage(turn)
    assert counted["prompt_tokens"] == 700
    assert counted["output_tokens"] == 30
    assert counted["cached_input_tokens"] == 0
    assert counted["source"] == "token_ids"


def test_cached_tokens_never_exceed_the_prompt():
    turn = GeneratedTurn(text="", usage={"prompt_tokens": 10, "cached_tokens": 99})
    assert turn_usage(turn)["cached_input_tokens"] == 10
    assert turn_usage(turn)["input_tokens"] == 0


def test_episode_totals_track_the_resent_transcript():
    """Each turn resends the transcript, so prompt tokens accumulate."""
    turns = [
        GeneratedTurn(text="", usage={"prompt_tokens": p, "completion_tokens": 100})
        for p in (1_000, 2_000, 3_000)
    ]
    episode = episode_usage(turns)
    assert episode["prompt_tokens"] == 6_000
    assert episode["output_tokens"] == 300
    assert episode["model_calls"] == 3
    assert episode["final_prompt_tokens"] == 3_000


def test_run_totals_and_per_task_means():
    episode = {
        "prompt_tokens": 100,
        "cached_input_tokens": 40,
        "input_tokens": 60,
        "output_tokens": 10,
        "reasoning_tokens": 0,
        "model_calls": 2,
        "final_prompt_tokens": 70,
        "source": "provider",
    }
    total = add_usage(add_usage(None, episode), episode)
    assert total["prompt_tokens"] == 200
    assert total["episodes"] == 2
    assert total["max_final_prompt_tokens"] == 70
    assert "final_prompt_tokens" not in total

    # Two attempts of one task: the task costs both of them.
    means = per_task_means(total, tasks=1)
    assert means["prompt_tokens_per_task"] == 200.0
    assert means["output_tokens_per_task"] == 20.0
    assert "episodes_per_task" not in means


def test_episode_usage_is_empty_for_a_run_with_no_turns():
    episode = episode_usage([])
    assert episode["prompt_tokens"] == 0
    assert episode["model_calls"] == 0
    assert episode["source"] == "none"
