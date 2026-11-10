"""Tests for the Vertex OPD teacher client and its rollout-hook wiring.

The client is exercised against local aiohttp fakes speaking the SGLang
``/generate`` and vLLM ``/v1/completions`` wire formats over plain http, so
the ADC auth path stays untouched and no Google credentials are needed. The
rollout integration reuses the fakes and helpers from ``test_slime_rollout``.
"""

import json

import pytest
from aiohttp import web

pytest.importorskip("slime", reason="requires the vendored slime on PYTHONPATH")

from slime.utils.types import Sample

from officeqa_rl.training import metrics, rollout
from officeqa_rl.training.vertex_teacher_client import (
    ScoredToken,
    TeacherScoringError,
    VertexTeacherClient,
    get_teacher_client,
    rawpredict_url,
)
from tests.sglang_fakes import FakeSGLangServer
from tests.test_slime_rollout import (  # noqa: F401  (fixtures register on import)
    SUBMIT_TURN,
    corpus_dir,
    fake_tokenizer,
    make_args,
    make_base_sample,
    scripted_responder,
)


def teacher_scores(payload):
    """SGLang /generate scoring response: one input_token_logprobs entry per
    token of input_ids[logprob_start_len:], the first entry of the window
    unscored, then -0.25 * global position for every later token."""
    ids = payload["input_ids"]
    start = payload.get("logprob_start_len", 0)
    window = ids[start:]
    items = [[None, window[0], None]] + [
        [-0.25 * (start + offset), token, None]
        for offset, token in enumerate(window[1:], start=1)
    ]
    return {"text": "", "meta_info": {"input_token_logprobs": items}}


async def teacher_responder(payload):
    return teacher_scores(payload)


def fast_client(url, **overrides):
    kwargs = dict(
        teacher_url=url,
        max_concurrent=4,
        timeout_sec=5.0,
        max_attempts=3,
        backoff_base_sec=0.01,
    )
    kwargs.update(overrides)
    return VertexTeacherClient(**kwargs)


class FakeNativeTeacherServer:
    """Record requests to any predict route and delegate the response."""

    def __init__(self, responder):
        self.responder = responder
        self.payloads = []

    async def __aenter__(self):
        app = web.Application()
        app.router.add_post("/{route:.*}", self._predict)
        self._runner = web.AppRunner(app, shutdown_timeout=0.5)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = self._runner.addresses[0][1]
        self.base_url = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self._runner.cleanup()
        return False

    async def _predict(self, request):
        payload = await request.json()
        self.payloads.append(payload)
        response = await self.responder(payload)
        if isinstance(response, web.Response):
            return response
        return web.json_response(response)


def vllm_scores(payload):
    """vLLM completion response with echoed prompt-token log-probabilities."""
    ids = payload["prompt"]
    return {
        "choices": [
            {
                "text": "",
                "index": 0,
                "logprobs": {
                    "tokens": [str(token) for token in ids],
                    "token_logprobs": [None]
                    + [-0.25 * position for position in range(1, len(ids))],
                    "top_logprobs": [None] * len(ids),
                    "text_offset": [0] * len(ids),
                },
                # Recent vLLM versions can return this extension. The client
                # validates it when present but does not require it.
                "prompt_token_ids": ids,
                "finish_reason": "length",
            }
        ]
    }


async def vllm_responder(payload):
    return vllm_scores(payload)


def test_rawpredict_url_regional_and_global():
    assert rawpredict_url("proj", "us-central1", "123") == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/proj"
        "/locations/us-central1/endpoints/123:rawPredict"
    )
    assert rawpredict_url("proj", "global", "123").startswith(
        "https://aiplatform.googleapis.com/v1/"
    )


async def test_score_tokens_speaks_the_sglang_opd_convention():
    async with FakeSGLangServer(teacher_responder) as server:
        client = fast_client(f"{server.base_url}/generate")
        try:
            log_probs = await client.score_tokens([10, 11, 12, 13])
        finally:
            await client.close()

    assert log_probs == pytest.approx([-0.25, -0.5, -0.75])
    payload = server.generate_payloads[0]
    assert payload["input_ids"] == [10, 11, 12, 13]
    assert payload["return_logprob"] is True
    assert payload["logprob_start_len"] == 0
    assert payload["sampling_params"]["max_new_tokens"] == 0
    assert client.resolved_backend == "sglang"
    # Plain-http teacher: no bearer token, so no google-auth involvement.
    assert "Authorization" not in server.generate_headers[0]


async def test_score_tokens_speaks_the_vllm_completion_convention():
    async with FakeNativeTeacherServer(vllm_responder) as server:
        client = fast_client(
            f"{server.base_url}/v1/completions", model="teacher-model"
        )
        try:
            log_probs = await client.score_tokens(
                [10, 11, 12, 13, 14, 15], logprob_start_len=2
            )
        finally:
            await client.close()

    assert log_probs == pytest.approx([-0.75, -1.0, -1.25])
    assert client.resolved_backend == "vllm"
    assert server.payloads == [
        {
            "prompt": [10, 11, 12, 13, 14, 15],
            "temperature": 0,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
            "model": "teacher-model",
        }
    ]


async def test_top_logprobs_sglang_returns_scored_tokens():
    async def responder(payload):
        response = teacher_scores(payload)
        items = response["meta_info"]["input_token_logprobs"]
        # Position-aligned top-k: None for the unscored first window token,
        # then a descending [log_prob, token_id] list per scored position.
        response["meta_info"]["input_top_logprobs"] = [None] + [
            [[item[0], item[1]], [item[0] - 1.0, 777]] for item in items[1:]
        ]
        return response

    async with FakeSGLangServer(responder) as server:
        client = fast_client(f"{server.base_url}/generate")
        try:
            scored = await client.score_tokens([10, 11, 12, 13], top_logprobs=2)
        finally:
            await client.close()

    assert server.generate_payloads[0]["top_logprobs_num"] == 2
    assert [token.log_prob for token in scored] == pytest.approx([-0.25, -0.5, -0.75])
    assert scored[0].top_log_probs == pytest.approx([-0.25, -1.25])
    assert scored[0].top_token_ids == [11, 777]


async def test_top_logprobs_sglang_missing_tops_degrade_to_none():
    async with FakeSGLangServer(teacher_responder) as server:
        client = fast_client(f"{server.base_url}/generate")
        try:
            scored = await client.score_tokens([10, 11, 12, 13], top_logprobs=4)
        finally:
            await client.close()

    assert [token.log_prob for token in scored] == pytest.approx([-0.25, -0.5, -0.75])
    assert all(token.top_log_probs is None for token in scored)


async def test_top_logprobs_vllm_returns_sorted_values_without_ids():
    async def responder(payload):
        response = vllm_scores(payload)
        logprobs = response["choices"][0]["logprobs"]
        # The completion echo keys top entries by token text; deliberately
        # unsorted values to prove the client re-sorts descending.
        logprobs["top_logprobs"] = [None] + [
            {"a": lp - 2.0, "b": lp, "c": lp - 1.0}
            for lp in logprobs["token_logprobs"][1:]
        ]
        return response

    async with FakeNativeTeacherServer(responder) as server:
        client = fast_client(f"{server.base_url}/v1/completions")
        try:
            scored = await client.score_tokens(
                [10, 11, 12, 13, 14, 15], logprob_start_len=2, top_logprobs=3
            )
        finally:
            await client.close()

    assert server.payloads[0]["logprobs"] == 3
    assert [token.log_prob for token in scored] == pytest.approx([-0.75, -1.0, -1.25])
    assert scored[0].top_log_probs == pytest.approx([-0.75, -1.75, -2.75])
    assert scored[0].top_token_ids is None
    assert isinstance(scored[0], ScoredToken)


async def test_generate_tokens_returns_generated_ids():
    async def responder(payload):
        if payload["sampling_params"]["max_new_tokens"] == 0:
            return teacher_scores(payload)  # the protocol-resolution probe
        assert payload["sampling_params"]["max_new_tokens"] == 64
        assert payload["sampling_params"]["temperature"] == 0.7
        return {
            "text": "",
            "meta_info": {
                "output_token_logprobs": [[-0.1, 500], [-0.2, 501], [-0.3, 502]]
            },
        }

    async with FakeSGLangServer(responder) as server:
        client = fast_client(f"{server.base_url}/generate")
        try:
            ids = await client.generate_tokens(
                [10, 11, 12], max_new_tokens=64, temperature=0.7
            )
        finally:
            await client.close()

    assert ids == [500, 501, 502]
    payload = server.generate_payloads[-1]
    assert payload["input_ids"] == [10, 11, 12]
    assert payload["sampling_params"]["no_stop_trim"] is True


async def test_generate_tokens_puts_bos_before_instruction():
    async def responder(payload):
        assert payload["input_ids"] == [2, 100, 101, 10, 11, 12]
        return {
            "text": "",
            "meta_info": {"output_token_logprobs": [[-0.1, 500]]},
        }

    async with FakeSGLangServer(responder) as server:
        client = fast_client(
            f"{server.base_url}/generate",
            instruction_token_ids=[100, 101],
            bos_token_id=2,
        )
        try:
            ids = await client.generate_tokens([10, 11, 12], max_new_tokens=8)
        finally:
            await client.close()

    assert ids == [500]


async def test_generate_tokens_rejected_for_vllm_backend():
    client = fast_client("http://127.0.0.1:1/v1/completions")
    try:
        with pytest.raises(TeacherScoringError, match="SGLang"):
            await client.generate_tokens([1, 2, 3], max_new_tokens=8)
    finally:
        await client.close()


async def test_top_logprobs_rejected_in_contrast_mode():
    client = fast_client(
        "http://127.0.0.1:1/generate",
        instruction_token_ids=[1],
        heavy_instruction_token_ids=[2],
    )
    try:
        with pytest.raises(ValueError, match="contrast"):
            await client.score_tokens([10, 11, 12], top_logprobs=2)
    finally:
        await client.close()


async def test_rawpredict_auto_detects_vllm_once_and_caches_it():
    async def reject_sglang(payload):
        if "input_ids" in payload:
            return web.Response(status=422, text="prompt field required")
        return vllm_scores(payload)

    async with FakeNativeTeacherServer(reject_sglang) as server:
        client = fast_client(f"{server.base_url}/endpoint:rawPredict")
        try:
            first = await client.score_tokens([1, 2, 3])
            second = await client.score_tokens([4, 5, 6])
        finally:
            await client.close()

    assert first == pytest.approx([-0.25, -0.5])
    assert second == pytest.approx([-0.25, -0.5])
    assert client.resolved_backend == "vllm"
    # Only the first call probes the historical SGLang request shape.
    assert "input_ids" in server.payloads[0]
    assert [payload["prompt"] for payload in server.payloads[1:]] == [
        [1, 2, 3],
        [4, 5, 6],
    ]


async def test_rawpredict_keeps_existing_sglang_request_shape():
    async with FakeNativeTeacherServer(teacher_responder) as server:
        client = fast_client(f"{server.base_url}/endpoint:rawPredict")
        try:
            first = await client.score_tokens([1, 2, 3])
            second = await client.score_tokens([4, 5, 6])
        finally:
            await client.close()

    assert first == pytest.approx([-0.25, -0.5])
    assert second == pytest.approx([-0.25, -0.5])
    assert client.resolved_backend == "sglang"
    assert [payload["input_ids"] for payload in server.payloads] == [
        [1, 2, 3],
        [4, 5, 6],
    ]
    assert all("prompt" not in payload for payload in server.payloads)


async def test_score_tokens_with_prompt_skip():
    """logprob_start_len skips prompt positions; the window's first entry is
    unscored, so start = prompt_length - 1 yields exactly the response
    log-probs at their global positions."""
    async with FakeSGLangServer(teacher_responder) as server:
        client = fast_client(f"{server.base_url}/generate")
        try:
            log_probs = await client.score_tokens(
                [10, 11, 12, 13, 14, 15], logprob_start_len=2
            )
        finally:
            await client.close()

    # Positions 3, 4, 5 (everything after the unscored window head at 2).
    assert log_probs == pytest.approx([-0.75, -1.0, -1.25])
    assert server.generate_payloads[0]["logprob_start_len"] == 2


async def test_transient_429_is_retried():
    calls = {"n": 0}

    async def flaky(payload):
        calls["n"] += 1
        if calls["n"] == 1:
            return web.Response(status=429, text="resource contention")
        return teacher_scores(payload)

    async with FakeSGLangServer(flaky) as server:
        client = fast_client(f"{server.base_url}/generate")
        try:
            log_probs = await client.score_tokens([1, 2, 3])
        finally:
            await client.close()

    assert calls["n"] == 2
    assert log_probs == pytest.approx([-0.25, -0.5])


async def test_non_retryable_status_fails_immediately():
    async def not_found(payload):
        return web.Response(status=404, text="no such endpoint")

    async with FakeSGLangServer(not_found) as server:
        client = fast_client(f"{server.base_url}/generate")
        try:
            with pytest.raises(TeacherScoringError, match="HTTP 404"):
                await client.score_tokens([1, 2, 3])
        finally:
            await client.close()

    assert len(server.generate_payloads) == 1


async def test_exhausted_retries_raise_teacher_scoring_error():
    async def always_500(payload):
        return web.Response(status=500, text="teacher down")

    async with FakeSGLangServer(always_500) as server:
        client = fast_client(f"{server.base_url}/generate", max_attempts=2)
        try:
            with pytest.raises(TeacherScoringError, match="after 2 attempts"):
                await client.score_tokens([1, 2, 3])
        finally:
            await client.close()

    assert len(server.generate_payloads) == 2


async def test_token_count_mismatch_is_fatal():
    async def short_score(payload):
        return {"meta_info": {"input_token_logprobs": [[None, 1], [-0.1, 2]]}}

    async with FakeSGLangServer(short_score) as server:
        client = fast_client(f"{server.base_url}/generate")
        try:
            with pytest.raises(TeacherScoringError, match="scored 1 tokens"):
                await client.score_tokens([1, 2, 3])
        finally:
            await client.close()


async def test_low_resource_instruction_is_prepended_and_shifted():
    """COPD low-resource conditioning: the instruction tokens are prepended
    and logprob_start_len shifts by the same amount, so exactly the caller's
    window is scored — under the instructed context — and the result length
    matches the un-instructed contract."""
    async with FakeSGLangServer(teacher_responder) as server:
        client = fast_client(
            f"{server.base_url}/generate", instruction_token_ids=[100, 101, 102]
        )
        try:
            log_probs = await client.score_tokens([10, 11, 12, 13], logprob_start_len=1)
        finally:
            await client.close()

    payload = server.generate_payloads[0]
    assert payload["input_ids"] == [100, 101, 102, 10, 11, 12, 13]
    assert payload["logprob_start_len"] == 4  # 1 + 3 instruction tokens
    # Same result shape as without the instruction: scores for tokens 12, 13.
    assert log_probs == pytest.approx([-1.25, -1.5])


async def test_bos_precedes_instruction_and_preserves_scoring_window():
    """A model-required BOS leads the request, including with COPD context."""
    async with FakeSGLangServer(teacher_responder) as server:
        client = fast_client(
            f"{server.base_url}/generate",
            instruction_token_ids=[100, 101, 102],
            bos_token_id=2,
        )
        try:
            log_probs = await client.score_tokens(
                [10, 11, 12, 13], logprob_start_len=1
            )
        finally:
            await client.close()

    payload = server.generate_payloads[0]
    assert payload["input_ids"] == [2, 100, 101, 102, 10, 11, 12, 13]
    assert payload["logprob_start_len"] == 5  # 1 + BOS + 3 instruction tokens
    assert log_probs == pytest.approx([-1.5, -1.75])


async def test_existing_bos_is_moved_before_instruction_without_duplication():
    async with FakeSGLangServer(teacher_responder) as server:
        client = fast_client(
            f"{server.base_url}/generate",
            instruction_token_ids=[100, 101],
            bos_token_id=2,
        )
        try:
            log_probs = await client.score_tokens([2, 10, 11, 12], logprob_start_len=1)
        finally:
            await client.close()

    payload = server.generate_payloads[0]
    assert payload["input_ids"] == [2, 100, 101, 10, 11, 12]
    assert payload["logprob_start_len"] == 3  # caller BOS reused; only prefix shifts
    assert log_probs == pytest.approx([-1.0, -1.25])


def test_render_instruction_wraps_plain_text_as_system_block():
    from officeqa_rl.training import vertex_teacher_client as vtc

    assert vtc._render_instruction("be brief") == (
        "<|im_start|>system\nbe brief<|im_end|>\n"
    )
    # Explicit chat markers pass through verbatim.
    marked = "<|im_start|>user\ncustom<|im_end|>\n"
    assert vtc._render_instruction(marked) == marked


def test_get_teacher_client_tokenizes_instruction(monkeypatch):
    import types

    from officeqa_rl.training import vertex_teacher_client as vtc

    seen = {}

    def fake_tokenize(args, text):
        seen["text"] = text
        return [7, 8, 9]

    monkeypatch.setattr(vtc, "_instruction_token_ids", fake_tokenize)
    base = dict(
        officeqa_opd_teacher_url="http://instruction-cache.test/generate",
        hf_checkpoint="/fake/ckpt",
    )
    instructed = get_teacher_client(
        types.SimpleNamespace(**base, officeqa_opd_instruction="be brief")
    )
    plain = get_teacher_client(types.SimpleNamespace(**base))
    with_bos = get_teacher_client(
        types.SimpleNamespace(**base, officeqa_opd_bos_token_id=2)
    )

    assert seen["text"] == "be brief"
    assert instructed.instruction_prefix_len == 3
    # The instruction is part of the cache key: same URL, distinct clients.
    assert plain is not instructed
    assert plain.instruction_prefix_len == 0
    assert plain.bos_token_id is None
    assert with_bos.bos_token_id == 2
    assert with_bos is not plain


async def test_contrast_scores_twice_and_returns_clipped_diff():
    """COPD paired scoring: one call = two teacher passes (light and heavy
    prefixes), returning per-token light-minus-heavy contrasts for exactly
    the same window as single-context scoring."""
    async with FakeSGLangServer(teacher_responder) as server:
        client = fast_client(
            f"{server.base_url}/generate",
            instruction_token_ids=[100, 101, 102],
            heavy_instruction_token_ids=[200, 201, 202, 203, 204],
        )
        try:
            contrast = await client.score_tokens([10, 11, 12, 13], logprob_start_len=1)
        finally:
            await client.close()

    payloads = {p["input_ids"][0]: p for p in server.generate_payloads}
    assert set(payloads) == {100, 200}
    assert payloads[100]["input_ids"] == [100, 101, 102, 10, 11, 12, 13]
    assert payloads[100]["logprob_start_len"] == 4  # 1 + 3 light tokens
    assert payloads[200]["input_ids"] == [200, 201, 202, 203, 204, 10, 11, 12, 13]
    assert payloads[200]["logprob_start_len"] == 6  # 1 + 5 heavy tokens
    # Fake scores by global sent position: light scores tokens 12,13 at
    # positions 5,6 (-1.25,-1.5); heavy at 7,8 (-1.75,-2.0) -> diff 0.5 each.
    assert contrast == pytest.approx([0.5, 0.5])


async def test_contrast_clip_bounds_the_signal():
    async with FakeSGLangServer(teacher_responder) as server:
        client = fast_client(
            f"{server.base_url}/generate",
            instruction_token_ids=[100],
            heavy_instruction_token_ids=[200, 201, 202, 203, 204],
            contrast_clip=0.3,
        )
        try:
            contrast = await client.score_tokens([10, 11, 12, 13], logprob_start_len=1)
        finally:
            await client.close()
    # Unclipped diff would be (len(heavy)-len(light))*0.25 = 1.0 per token.
    assert contrast == pytest.approx([0.3, 0.3])


def test_get_teacher_client_contrast_config(monkeypatch):
    import types

    from officeqa_rl.training import vertex_teacher_client as vtc

    monkeypatch.setattr(
        vtc, "_instruction_token_ids", lambda args, text: [1] * (10 if text == "ht" else 4)
    )
    base = dict(
        officeqa_opd_teacher_url="http://contrast-cache.test/generate",
        hf_checkpoint="/fake/ckpt",
        officeqa_opd_instruction="lt",
    )
    contrast = get_teacher_client(
        types.SimpleNamespace(
            **base, officeqa_opd_instruction_high="ht", officeqa_opd_contrast_clip=5.0
        )
    )
    light_only = get_teacher_client(types.SimpleNamespace(**base))

    assert contrast.contrast_enabled
    assert contrast.instruction_prefix_len == 4
    assert contrast.heavy_prefix_len == 10
    assert contrast._contrast_clip == 5.0
    assert not light_only.contrast_enabled
    assert light_only is not contrast


def test_loss_contrast_branch_adds_clipped_contrast_directly():
    """In contrast mode the slime loss adds opd_kl_coef * contrast (no
    student term); teacher_log_probs carries the client-computed contrast."""
    pytest.importorskip("megatron", reason="requires Megatron (training container)")
    import types

    import torch
    from slime.backends.megatron_utils.loss import apply_opd_kl_to_advantages

    adv = [torch.tensor([1.0, -1.0, 0.0])]
    contrast = [torch.tensor([0.5, -0.25, 2.0])]
    student = [torch.tensor([9.0, 9.0, 9.0])]  # must be ignored in contrast mode
    rollout_data = {"teacher_log_probs": contrast}
    args = types.SimpleNamespace(
        use_opd=True, opd_type="sglang", opd_kl_coef=2.0, officeqa_opd_contrast=True
    )
    apply_opd_kl_to_advantages(args, rollout_data, adv, student)
    assert torch.allclose(adv[0], torch.tensor([2.0, -1.5, 4.0]))
    assert torch.allclose(rollout_data["opd_reverse_kl"][0], contrast[0])


def test_loss_pure_opd_discards_reward_advantage():
    """With officeqa_opd_pure the base (reward) advantage is zeroed and the
    OPD tilt is the whole gradient, at any group size."""
    pytest.importorskip("megatron", reason="requires Megatron (training container)")
    import types

    import torch
    from slime.backends.megatron_utils.loss import apply_opd_kl_to_advantages

    adv = [torch.tensor([5.0, -3.0, 1.0])]  # reward advantage: must vanish
    student = [torch.tensor([0.5, 0.5, 0.5])]
    teacher = [torch.tensor([0.0, 1.0, 0.5])]
    rollout_data = {"teacher_log_probs": teacher}
    args = types.SimpleNamespace(
        use_opd=True, opd_type="sglang", opd_kl_coef=2.0, officeqa_opd_pure=True
    )
    apply_opd_kl_to_advantages(args, rollout_data, adv, student)
    # -coef * (student - teacher) with no reward term underneath
    assert torch.allclose(adv[0], torch.tensor([-1.0, 1.0, 0.0]))


def test_loss_pure_opd_composes_with_contrast():
    """Pure + contrast: advantage is exactly opd_kl_coef * contrast."""
    pytest.importorskip("megatron", reason="requires Megatron (training container)")
    import types

    import torch
    from slime.backends.megatron_utils.loss import apply_opd_kl_to_advantages

    adv = [torch.tensor([5.0, -3.0])]
    contrast = [torch.tensor([0.5, -0.25])]
    student = [torch.tensor([9.0, 9.0])]
    rollout_data = {"teacher_log_probs": contrast}
    args = types.SimpleNamespace(
        use_opd=True,
        opd_type="sglang",
        opd_kl_coef=2.0,
        officeqa_opd_contrast=True,
        officeqa_opd_pure=True,
    )
    apply_opd_kl_to_advantages(args, rollout_data, adv, student)
    assert torch.allclose(adv[0], torch.tensor([1.0, -0.5]))


async def test_round_robin_assigns_episodes_with_sticky_affinity():
    """Multi-endpoint pool: new episodes rotate across endpoints; every call
    of one episode sticks to its endpoint (prefix-cache affinity)."""
    async with (
        FakeSGLangServer(teacher_responder) as s0,
        FakeSGLangServer(teacher_responder) as s1,
    ):
        client = fast_client(f"{s0.base_url}/generate,{s1.base_url}/generate")
        try:
            await client.score_tokens([1, 2, 3], affinity_key="ep-A")
            await client.score_tokens([4, 5, 6], affinity_key="ep-B")
            await client.score_tokens([7, 8, 9], affinity_key="ep-A")
            await client.score_tokens([10, 11, 12], affinity_key="ep-C")
        finally:
            await client.close()

    assert client.endpoint_urls == (
        f"{s0.base_url}/generate",
        f"{s1.base_url}/generate",
    )
    # A -> endpoint 0 (first seen), B -> 1, A again -> 0 (sticky), C -> 0.
    assert [p["input_ids"] for p in s0.generate_payloads] == [
        [1, 2, 3],
        [7, 8, 9],
        [10, 11, 12],
    ]
    assert [p["input_ids"] for p in s1.generate_payloads] == [[4, 5, 6]]


async def test_contrast_pair_shares_the_episode_endpoint():
    async with (
        FakeSGLangServer(teacher_responder) as s0,
        FakeSGLangServer(teacher_responder) as s1,
    ):
        client = fast_client(
            f"{s0.base_url}/generate,{s1.base_url}/generate",
            instruction_token_ids=[100],
            heavy_instruction_token_ids=[200, 201],
        )
        try:
            await client.score_tokens([1, 2, 3], affinity_key="ep-A")  # -> endpoint 0
            await client.score_tokens([4, 5, 6], affinity_key="ep-B")  # -> endpoint 1
        finally:
            await client.close()

    # Each episode's light AND heavy passes hit the same endpoint.
    assert len(s0.generate_payloads) == 2
    assert {tuple(p["input_ids"]) for p in s0.generate_payloads} == {
        (100, 1, 2, 3),
        (200, 201, 1, 2, 3),
    }
    assert len(s1.generate_payloads) == 2
    assert {tuple(p["input_ids"]) for p in s1.generate_payloads} == {
        (100, 4, 5, 6),
        (200, 201, 4, 5, 6),
    }


async def test_endpoint_index_override_probes_each_endpoint():
    async with (
        FakeSGLangServer(teacher_responder) as s0,
        FakeSGLangServer(teacher_responder) as s1,
    ):
        client = fast_client(f"{s0.base_url}/generate,{s1.base_url}/generate")
        try:
            for index in range(2):
                await client.score_tokens([1, 2, 3], endpoint_index=index)
        finally:
            await client.close()
    assert len(s0.generate_payloads) == 1
    assert len(s1.generate_payloads) == 1


def test_teacher_url_from_args_expands_endpoint_id_list():
    import types

    from officeqa_rl.training.vertex_teacher_client import _teacher_url_from_args

    urls = _teacher_url_from_args(
        types.SimpleNamespace(
            officeqa_opd_teacher_url=None,
            officeqa_opd_project="proj",
            officeqa_opd_location="us-east1",
            officeqa_opd_endpoint_id="111, 222",
        )
    )
    assert urls.split(",") == [
        rawpredict_url("proj", "us-east1", "111"),
        rawpredict_url("proj", "us-east1", "222"),
    ]


def opd_args(actor_server, teacher_server, corpus, tmp_path, **overrides):
    values = dict(
        use_opd=True,
        opd_type="sglang",
        officeqa_opd_teacher_url=f"{teacher_server.base_url}/generate",
        officeqa_opd_backoff_sec=0.01,
    )
    values.update(overrides)
    return make_args(actor_server, corpus, tmp_path, **values)


async def test_generate_attaches_teacher_log_probs(corpus_dir, tmp_path):
    async with (
        FakeSGLangServer(scripted_responder([SUBMIT_TURN])) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        args = opd_args(actor, teacher, corpus_dir, tmp_path)
        sample = await rollout.generate(args, make_base_sample(), {"max_new_tokens": 512})
        await get_teacher_client(args).close()

    assert isinstance(sample, Sample)
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == 1.0

    # The teacher scored the exact training token sequence with the prompt
    # region skipped, and the sample keeps exactly the response log-probs.
    prompt_length = len(sample.tokens) - sample.response_length
    payload = teacher.generate_payloads[0]
    assert payload["input_ids"] == sample.tokens
    assert payload["logprob_start_len"] == prompt_length - 1
    assert len(sample.teacher_log_probs) == sample.response_length
    assert sample.teacher_log_probs == pytest.approx(
        [-0.25 * position for position in range(prompt_length, len(sample.tokens))]
    )


async def test_generate_raises_when_teacher_scoring_fails(corpus_dir, tmp_path):
    """Teacher-scoring failures are deliberately fatal (not aborts): the run
    must stop loudly rather than continue without teacher signal — but the
    failure log is still written first."""

    async def broken(payload):
        return web.Response(status=500, text="teacher down")

    failure_log = tmp_path / "failures" / "opd.jsonl"
    async with (
        FakeSGLangServer(scripted_responder([SUBMIT_TURN])) as actor,
        FakeSGLangServer(broken) as teacher,
    ):
        args = opd_args(
            actor,
            teacher,
            corpus_dir,
            tmp_path,
            officeqa_opd_attempts=2,
            officeqa_opd_failure_log=str(failure_log),
        )
        with pytest.raises(TeacherScoringError, match="after 2 attempts"):
            await rollout.generate(args, make_base_sample(), {"max_new_tokens": 512})
        await get_teacher_client(args).close()

    # The terminal failure reason is durably recorded before the raise.
    records = [json.loads(line) for line in failure_log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["uid"] == "UID0001"
    assert "idx3-grp1-rid7-" in records[0]["run_id"]
    assert "TeacherScoringError" in records[0]["error"]
    assert "HTTP 500" in records[0]["error"]
    assert "teacher down" in records[0]["error"]


def test_episode_reward_mean_is_episode_weighted():
    """A long failing episode contributes many fragments but one episode
    reward; the fragment-weighted mean understates the per-question mean."""
    fail = [
        Sample(prompt="p", status=Sample.Status.TRUNCATED, reward=0.0,
               metadata={"run_id": "ep-fail", "exit_status": "limits_exceeded"})
        for _ in range(9)
    ]
    ok = [Sample(prompt="p", status=Sample.Status.COMPLETED, reward=1.0,
                 metadata={"run_id": "ep-ok", "exit_status": "submitted"})]
    log_dict = metrics._truncation_metrics(
        fail + ok, max_response_len=None, max_context_len=None
    )
    # Fragment-weighted mean would be 0.1; the episode-weighted mean is 0.5.
    assert log_dict["episode/reward_mean"] == pytest.approx(0.5)
    assert log_dict["episode/count"] == 2.0


def test_abort_metrics_split_teacher_failures_by_category():
    ok = Sample(prompt="p", status=Sample.Status.COMPLETED)
    teacher_fail = Sample(
        prompt="p",
        status=Sample.Status.ABORTED,
        metadata={"abort_reason": "opd_teacher_failure: TeacherScoringError: HTTP 500"},
    )
    env_fail = Sample(
        prompt="p",
        status=Sample.Status.ABORTED,
        metadata={"abort_reason": "environment_failure: OSError: boom"},
    )
    log_dict = metrics._abort_metrics([ok, ok, teacher_fail, env_fail])
    assert log_dict["abort/ratio"] == pytest.approx(0.5)
    assert log_dict["abort/opd_teacher_failure"] == 1.0
    assert log_dict["abort/environment_failure"] == 1.0

    # A clean batch still logs the (stable) ratio line, and no category keys.
    assert metrics._abort_metrics([ok]) == {"abort/ratio": 0.0}


async def test_no_teacher_calls_when_disabled_or_evaluating(corpus_dir, tmp_path):
    async with (
        FakeSGLangServer(scripted_responder([SUBMIT_TURN, SUBMIT_TURN])) as actor,
        FakeSGLangServer(teacher_responder) as teacher,
    ):
        # OPD flags absent: the hook never builds a teacher client.
        args = make_args(actor, corpus_dir, tmp_path)
        sample = await rollout.generate(args, make_base_sample(), {"max_new_tokens": 512})
        assert sample.teacher_log_probs is None

        # OPD on, but eval rollouts never train, so no teacher round-trip.
        args = opd_args(actor, teacher, corpus_dir, tmp_path)
        sample = await rollout.generate(
            args, make_base_sample(), {"max_new_tokens": 512}, evaluation=True
        )
        assert sample.teacher_log_probs is None

    assert teacher.generate_payloads == []
