"""Tests for the Vertex rawPredict / SGLang generation backend."""

import pytest

from officeqa_rl.backends.vertex import (
    VertexAPIError,
    VertexSGLangGenerationBackend,
    rawpredict_url,
)


class ChatTemplateTokenizer:
    """Deterministic fake: renders roles/contents and the generation opener."""

    def apply_chat_template(
        self, messages, *, add_generation_prompt, tokenize, **template_kwargs
    ):
        del template_kwargs
        assert add_generation_prompt and not tokenize
        rendered = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return rendered + "<assistant><think>\n"


def sglang_response(text, *, finish_reason="stop", meta_extra=None):
    meta = {
        "finish_reason": {"type": finish_reason},
        "prompt_tokens": 25,
        "completion_tokens": 135,
        "cached_tokens": 10,
    }
    meta.update(meta_extra or {})
    return {"text": text, "meta_info": meta}


def make_backend(**kwargs):
    kwargs.setdefault("tokenizer", ChatTemplateTokenizer())
    kwargs.setdefault("base_url", "http://10.0.0.1:30000/generate")
    return VertexSGLangGenerationBackend("teacher-9b", **kwargs)


def test_rawpredict_url_regional_and_global():
    assert rawpredict_url("proj", "us-east1", "17") == (
        "https://us-east1-aiplatform.googleapis.com/v1/projects/proj"
        "/locations/us-east1/endpoints/17:rawPredict"
    )
    assert rawpredict_url("proj", "global", "17").startswith(
        "https://aiplatform.googleapis.com/v1/"
    )


def test_renders_chat_template_and_maps_sampling_params(monkeypatch):
    backend = make_backend(max_tokens=4096)
    seen = {}

    def fake_post(payload):
        seen.update(payload)
        return sglang_response("</think>done")

    monkeypatch.setattr(backend, "_post_json", fake_post)
    turn = backend._generate_sync(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "exit", "content": "ignored"},
        ],
        {"temperature": 0.7, "max_tokens": 128},
    )

    assert seen["text"] == "<system>sys<user>go<assistant><think>\n"
    assert seen["sampling_params"] == {"max_new_tokens": 128, "temperature": 0.7}
    assert turn.text == "done"
    assert turn.finish_reason == "stop"
    assert turn.usage == {"prompt_tokens": 25, "completion_tokens": 135, "cached_tokens": 10}


def test_enable_thinking_is_forwarded_to_chat_template():
    seen = {}

    class GemmaTokenizer:
        def apply_chat_template(
            self, messages, *, add_generation_prompt, tokenize, enable_thinking
        ):
            seen["enable_thinking"] = enable_thinking
            suffix = "<|turn>model\n" if enable_thinking else (
                "<|turn>model\n<|channel>thought\n<channel|>"
            )
            return suffix

    backend = make_backend(tokenizer=GemmaTokenizer(), enable_thinking=True)
    assert backend._render_prompt([{"role": "user", "content": "go"}]) == (
        "<|turn>model\n"
    )
    assert seen == {"enable_thinking": True}


def test_enable_thinking_preserves_sglang_channel_markers(monkeypatch):
    backend = make_backend(enable_thinking=True, max_tokens=4096)
    seen = {}

    def fake_post(payload):
        seen.update(payload)
        return sglang_response(
            "<|channel>thought\nPlan.<channel|>\n```bash\nls\n```"
        )

    monkeypatch.setattr(backend, "_post_json", fake_post)
    turn = backend._generate_sync(
        [{"role": "user", "content": "go"}], {"temperature": 1.0}
    )

    assert seen["sampling_params"] == {
        "max_new_tokens": 4096,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "temperature": 1.0,
    }
    assert turn.reasoning_content == "Plan."
    assert turn.text == "```bash\nls\n```"


def test_splits_gemma_thought_channel(monkeypatch):
    backend = make_backend()
    generated = (
        "<|channel>thought\nInspect the workbook first.<channel|>\n"
        "```bash\npython /workspace/edit.py\n```"
    )
    monkeypatch.setattr(backend, "_post_json", lambda p: sglang_response(generated))

    turn = backend._generate_sync([{"role": "user", "content": "go"}], {})

    assert turn.reasoning_content == "Inspect the workbook first."
    assert turn.text == "```bash\npython /workspace/edit.py\n```"


def test_unclosed_gemma_thought_channel_is_not_executable(monkeypatch):
    backend = make_backend()
    generated = "<|channel>thought\nMaybe run\n```bash\nrm -rf /\n```"
    monkeypatch.setattr(backend, "_post_json", lambda p: sglang_response(generated))

    turn = backend._generate_sync([{"role": "user", "content": "go"}], {})

    assert turn.reasoning_content.startswith("Maybe run")
    assert turn.text == ""


def test_gemma_repeated_thoughts_are_all_removed_and_usage_is_unchanged(monkeypatch):
    backend = make_backend()
    generated = (
        "<|channel>thought\nfirst<channel|>"
        "<|channel>thought\n<channel|>"
        "<|channel>thought\nsecond ```bash\necho private\n```<channel|>"
        "```bash\nls\n```"
    )
    monkeypatch.setattr(backend, "_post_json", lambda p: sglang_response(generated))
    turn = backend._generate_sync([{"role": "user", "content": "go"}], {})
    assert turn.text == "```bash\nls\n```"
    assert turn.reasoning_content == "first\nsecond ```bash\necho private\n```"
    assert turn.usage["completion_tokens"] == 135


def test_gemma_unclosed_later_thought_keeps_only_preceding_visible_text(monkeypatch):
    backend = make_backend()
    generated = "<|channel>thought\na<channel|>visible<|channel>thought\nprivate"
    monkeypatch.setattr(backend, "_post_json", lambda p: sglang_response(generated))
    turn = backend._generate_sync([{"role": "user", "content": "go"}], {})
    assert turn.text == "visible"
    assert turn.reasoning_content == "a\nprivate"


@pytest.mark.parametrize("protocol", ["sglang", "vllm"])
def test_gemma_extra_close_does_not_expose_candidate_commands(protocol):
    backend = make_backend(protocol=protocol)
    generated = (
        "<|channel>thought\ninitial plan<channel|>"
        "```bash\necho candidate\n```\nWait, use the final command.<channel|>"
        "```bash\necho final\n```"
    )
    response = (sglang_response(generated) if protocol == "sglang" else
                {"choices": [{"text": generated, "finish_reason": "stop"}],
                 "usage": {"completion_tokens": 135}})
    turn = backend._parse_response(response)
    assert turn.text == "```bash\necho final\n```"
    assert "echo candidate" in turn.reasoning_content
    assert "initial plan" in turn.reasoning_content
    assert "<channel|>" not in turn.reasoning_content
    assert turn.usage["completion_tokens"] == 135


def test_gemma_multiple_closes_without_opener_are_all_parsed():
    backend = make_backend()
    turn = backend._parse_response(sglang_response(
        "first plan<channel|>second plan<channel|>```bash\nls\n```"))
    assert turn.reasoning_content == "first plan\nsecond plan"
    assert turn.text == "```bash\nls\n```"


def test_gemma_nested_opener_and_visible_prefix_are_preserved_correctly():
    backend = make_backend()
    turn = backend._parse_response(sglang_response(
        "visible prefix\n<|channel>thought\nfirst<|channel>thought\nsecond"
        "<channel|>candidate<channel|>visible suffix"))
    assert turn.reasoning_content == "first\nsecond\ncandidate"
    assert turn.text == "visible prefix\nvisible suffix"


def test_gemma_orphan_close_is_removed_from_prior_history():
    backend = VertexSGLangGenerationBackend(
        "google/gemma-4-31B-it", base_url="http://localhost/generate",
        tokenizer=ChatTemplateTokenizer(), enable_thinking=True,
    )
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content":
         "<|channel>thought\nPRIVATE_A<channel|>PRIVATE_B<channel|>```bash\nls\n```"},
        {"role": "user", "content": "next"},
    ]
    rendered = backend._render_prompt(messages)
    assert "PRIVATE" not in rendered and "<channel|>" not in rendered
    assert "ls" in rendered
    assert "PRIVATE_B" in messages[1]["content"]


def test_gemma_empty_think_keeps_markers_and_visible_only_history():
    backend = VertexSGLangGenerationBackend(
        "google/gemma-4-31B-it", base_url="http://localhost/generate",
        tokenizer=ChatTemplateTokenizer(), enable_thinking=False,
    )
    params = backend._sglang_sampling_params({"temperature": 1.0})
    assert params["skip_special_tokens"] is False
    assert params["spaces_between_special_tokens"] is False
    messages = [
        {"role": "user", "content": "list"},
        {"role": "assistant", "content":
         "<|channel>thought\nPRIVATE_A<channel|>```bash\nls\n```"
         "<|channel>thought\nPRIVATE_B<channel|>",
         "reasoning_content": "PRIVATE_C"},
        {"role": "user", "content": "notes.txt"},
    ]
    prompt = backend._render_prompt(messages)
    assert "PRIVATE" not in prompt
    assert "ls" in prompt and "notes.txt" in prompt
    assert "PRIVATE_A" in messages[1]["content"]  # Do not mutate saved traces.


def test_system_instruction_prepends_own_system_block(monkeypatch):
    backend = make_backend(system_instruction="Be quick.")
    seen = {}

    def fake_post(payload):
        seen.update(payload)
        return sglang_response("</think>done")

    monkeypatch.setattr(backend, "_post_json", fake_post)
    backend._generate_sync(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}],
        {"temperature": 0.0},
    )

    # The instruction becomes its own leading system block before the agent's
    # rendered conversation — the OPD teacher's training-time context layout.
    assert seen["text"] == (
        "<|im_start|>system\nBe quick.<|im_end|>\n<system>sys<user>go<assistant><think>\n"
    )


def test_system_instruction_already_rendered_is_used_verbatim(monkeypatch):
    backend = make_backend(
        system_instruction="<|im_start|>user\ncustom<|im_end|>\n"
    )
    seen = {}

    def fake_post(payload):
        seen.update(payload)
        return sglang_response("</think>done")

    monkeypatch.setattr(backend, "_post_json", fake_post)
    backend._generate_sync([{"role": "user", "content": "go"}], {"temperature": 0.0})
    assert seen["text"] == "<|im_start|>user\ncustom<|im_end|>\n<user>go<assistant><think>\n"


def test_system_instruction_applies_to_vllm_protocol_too(monkeypatch):
    backend = make_backend(protocol="vllm", system_instruction="Be quick.")
    seen = {}

    def fake_post(payload):
        seen.update(payload)
        return {"choices": [{"index": 0, "text": "</think>ok", "finish_reason": "stop"}]}

    monkeypatch.setattr(backend, "_post_json", fake_post)
    backend._generate_sync([{"role": "user", "content": "go"}], {"temperature": 0.0})
    assert seen["prompt"] == (
        "<|im_start|>system\nBe quick.<|im_end|>\n<user>go<assistant><think>\n"
    )


def test_instruction_block_matches_teacher_client():
    pytest.importorskip("aiohttp")
    from officeqa_rl.backends import vertex
    from officeqa_rl.training import vertex_teacher_client

    assert vertex._INSTRUCTION_BLOCK == vertex_teacher_client._INSTRUCTION_BLOCK


def test_vllm_completion_protocol_reuses_local_chat_template(monkeypatch):
    backend = make_backend(protocol="vllm", max_tokens=4096)
    seen = {}

    def fake_post(payload):
        seen.update(payload)
        return {
            "model": "Qwen/Qwen3.5-122B-A10B",
            "choices": [
                {
                    "index": 0,
                    "text": "plan</think>done",
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 3,
                "total_tokens": 28,
            },
        }

    monkeypatch.setattr(backend, "_post_json", fake_post)
    turn = backend._generate_sync(
        [{"role": "user", "content": "go"}],
        {"temperature": 0.0, "max_tokens": 128},
    )

    assert seen == {
        "model": "teacher-9b",
        "prompt": "<user>go<assistant><think>\n",
        "max_tokens": 128,
        "temperature": 0.0,
    }
    assert turn.text == "done"
    assert turn.reasoning_content == "plan"
    assert turn.finish_reason == "stop"
    assert turn.usage == {
        "prompt_tokens": 25,
        "completion_tokens": 3,
        "total_tokens": 28,
    }


def test_splits_reasoning_without_opening_tag(monkeypatch):
    # Qwen templates place <think> in the prompt, so generations start
    # mid-reasoning and only the close tag appears in the output.
    backend = make_backend()
    inline = "Plan: run\n```bash\nls /data\n```\n</think>Listing.\n```bash\nls /w\n```"
    monkeypatch.setattr(backend, "_post_json", lambda p: sglang_response(inline))

    turn = backend._generate_sync([{"role": "user", "content": "go"}], {})

    assert turn.text == "Listing.\n```bash\nls /w\n```"
    assert turn.reasoning_content.startswith("Plan: run")
    assert "</think>" not in turn.text


def test_unclosed_think_is_all_reasoning(monkeypatch):
    backend = make_backend()
    truncated = "<think>maybe\n```bash\nrm -rf /\n```\nno wait"
    monkeypatch.setattr(backend, "_post_json", lambda p: sglang_response(truncated))

    turn = backend._generate_sync([{"role": "user", "content": "go"}], {})

    assert turn.text == ""
    assert turn.reasoning_content.startswith("maybe")


def test_capacity_errors_use_the_larger_retry_budget(monkeypatch):
    backend = make_backend(max_retries=0, quota_max_retries=3)
    monkeypatch.setattr(backend, "_retry_delay", lambda attempt: 0.0)
    calls = {"n": 0}

    def flaky(payload):
        calls["n"] += 1
        if calls["n"] < 3:
            raise VertexAPIError(503, "Model server is not available")
        return sglang_response("</think>ok")

    monkeypatch.setattr(backend, "_post_json", flaky)
    turn = backend._generate_sync([{"role": "user", "content": "go"}], {})
    assert turn.text == "ok"
    assert calls["n"] == 3


def test_non_capacity_error_respects_max_retries(monkeypatch):
    backend = make_backend(max_retries=1, quota_max_retries=10)
    monkeypatch.setattr(backend, "_retry_delay", lambda attempt: 0.0)

    def always_400(payload):
        raise VertexAPIError(400, "bad request")

    monkeypatch.setattr(backend, "_post_json", always_400)
    with pytest.raises(RuntimeError, match="failed after retries"):
        backend._generate_sync([{"role": "user", "content": "go"}], {})


def test_context_length_error_is_not_retried(monkeypatch):
    backend = make_backend(max_retries=3, quota_max_retries=10)
    calls = {"n": 0}

    def context_overflow(payload):
        calls["n"] += 1
        raise VertexAPIError(
            400,
            "Input length (243493 tokens) exceeds the maximum allowed length "
            "(243450 tokens).",
        )

    monkeypatch.setattr(backend, "_post_json", context_overflow)
    monkeypatch.setattr(
        backend,
        "_retry_delay",
        lambda attempt: pytest.fail("a context overflow must not be retried"),
    )

    with pytest.raises(RuntimeError, match="exceeds context length"):
        backend._generate_sync([{"role": "user", "content": "go"}], {})

    assert calls["n"] == 1


def test_multimodal_content_is_rejected():
    backend = make_backend()
    with pytest.raises(TypeError, match="text-only"):
        backend._render_prompt(
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        )


def test_no_auth_header_for_plain_http():
    backend = make_backend()  # http:// URL
    assert backend._access_token() is None


def test_stats_aggregate_usage(monkeypatch):
    backend = make_backend()
    monkeypatch.setattr(backend, "_post_json", lambda p: sglang_response("</think>x"))
    backend._generate_sync([{"role": "user", "content": "a"}], {})
    backend._generate_sync([{"role": "user", "content": "b"}], {})
    stats = backend.get_stats()
    assert stats["n_calls"] == 2
    assert stats["usage_totals"]["completion_tokens"] == 270


def test_retry_delay_uses_the_configured_base_and_caps_at_60s():
    backend = make_backend(backoff_base_sec=20.0)
    # Jitter is 0.5x-1.5x, so bound each attempt rather than assert equality.
    for attempt, unjittered in ((1, 20.0), (2, 40.0), (3, 60.0), (9, 60.0)):
        delays = [backend._retry_delay(attempt) for _ in range(200)]
        assert min(delays) >= 0.5 * unjittered
        assert max(delays) <= 1.5 * unjittered
        assert len(set(delays)) > 1, "backoff must be jittered, not fixed"


def test_retry_delay_default_base_matches_the_previous_behaviour():
    backend = make_backend()
    assert backend.backoff_base_sec == 2.0
    delays = [backend._retry_delay(3) for _ in range(200)]
    assert min(delays) >= 4.0 and max(delays) <= 12.0


def test_capacity_and_transient_budgets_are_counted_apart(monkeypatch):
    """A model-server restart (503) must not eat the transient budget."""
    backend = make_backend(max_retries=1, quota_max_retries=4)
    monkeypatch.setattr(backend, "_retry_delay", lambda attempt: 0.0)
    calls = {"n": 0}

    def always_capacity(payload):
        calls["n"] += 1
        raise VertexAPIError(503, "model server not available")

    monkeypatch.setattr(backend, "_post_json", always_capacity)
    with pytest.raises(RuntimeError):
        backend._generate_sync([{"role": "user", "content": "go"}], {})
    assert calls["n"] == 5, "1 initial call + quota_max_retries=4 retries"
