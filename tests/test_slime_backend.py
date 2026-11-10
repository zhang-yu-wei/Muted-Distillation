import asyncio

import pytest
from aiohttp import web

from officeqa_rl.training.slime_sglang_backend import SlimeSGLangBackend
from tests.sglang_fakes import ByteTokenizer, FakeSGLangServer, sglang_response


def make_backend(server: FakeSGLangServer, **overrides) -> SlimeSGLangBackend:
    values = {
        "base_url": server.base_url,
        "tokenizer": ByteTokenizer(),
        "max_context_len": 4096,
    }
    values.update(overrides)
    return SlimeSGLangBackend(**values)


MESSAGES = [
    {"role": "system", "content": "Answer the question."},
    {"role": "user", "content": "What is the revenue?", "extra": {"secret": "x"}},
]


async def test_payload_has_exact_prompt_ids_and_clamped_budget():
    async def responder(payload):
        return sglang_response("ok")

    async with FakeSGLangServer(responder) as server:
        prompt_ids = ByteTokenizer().apply_chat_template(
            [{"role": m["role"], "content": m["content"]} for m in MESSAGES],
            add_generation_prompt=True,
            tokenize=True,
        )
        backend = make_backend(server, max_context_len=len(prompt_ids) + 10)
        caller_params = {"temperature": 0.8, "max_new_tokens": 100}

        await backend.generate(MESSAGES, caller_params)

    payload = server.generate_payloads[0]
    assert payload["input_ids"] == prompt_ids
    assert payload["return_logprob"] is True
    assert len(payload["rid"]) == 32
    sent = payload["sampling_params"]
    assert sent["max_new_tokens"] == 10  # clamped from 100 to the remaining budget
    assert sent["temperature"] == 0.8
    assert sent["skip_special_tokens"] is False
    assert sent["no_stop_trim"] is True
    # The caller's dict must not be mutated by clamping or defaults.
    assert caller_params == {"temperature": 0.8, "max_new_tokens": 100}
    # Message "extra" payloads never reach the server.
    assert "secret" not in str(payload)


async def test_dict_returning_chat_template_still_sends_flat_ids():
    """Newer transformers apply_chat_template returns a BatchEncoding dict."""

    class DictTokenizer(ByteTokenizer):
        def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
            ids = super().apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
            )
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    async def responder(payload):
        return sglang_response("ok")

    async with FakeSGLangServer(responder) as server:
        backend = make_backend(server, tokenizer=DictTokenizer())
        turn = await backend.generate(MESSAGES, {"max_new_tokens": 50})

    expected_ids = ByteTokenizer().apply_chat_template(
        [{"role": m["role"], "content": m["content"]} for m in MESSAGES],
        add_generation_prompt=True,
        tokenize=True,
    )
    assert server.generate_payloads[0]["input_ids"] == expected_ids
    assert turn.prompt_ids == expected_ids


async def test_output_tokens_and_log_probs_stay_aligned():
    async def responder(payload):
        return sglang_response("AB")

    async with FakeSGLangServer(responder) as server:
        backend = make_backend(server)
        turn = await backend.generate(MESSAGES, {"max_new_tokens": 50})

    assert turn.output_ids == [ord("A"), ord("B")]
    assert turn.output_log_probs == [-0.5, -1.0]
    assert len(turn.output_ids) == len(turn.output_log_probs)
    assert turn.text == "AB"  # decoded from token ids, not copied from response text
    assert turn.finish_reason == "stop"
    assert turn.prompt_ids == server.generate_payloads[0]["input_ids"]
    assert len(backend.session_turns) == 1
    assert backend.session_turns[0].turn is turn


async def test_exhausted_context_returns_length_turn_without_request():
    async def responder(payload):
        raise AssertionError("no request should be sent")

    async with FakeSGLangServer(responder) as server:
        backend = make_backend(server, max_context_len=5)
        turn = await backend.generate(MESSAGES, {"max_new_tokens": 50})

    assert turn.finish_reason == "length"
    assert turn.output_ids == []
    assert turn.output_log_probs == []
    assert turn.text == ""
    assert len(turn.prompt_ids) > 5
    assert server.generate_payloads == []
    assert len(backend.session_turns) == 1  # recorded so truncation is exportable


async def test_engine_context_length_400_returns_length_turn():
    # The engine's admissible input is slightly smaller than max_context_len,
    # so this rejection can arrive for a prompt the local guard let through.
    async def responder(payload):
        return web.Response(
            status=400,
            text=(
                "Input length (64084 tokens) exceeds the maximum allowed "
                "length (64069 tokens). Use a shorter input or enable "
                "--allow-auto-truncate."
            ),
        )

    async with FakeSGLangServer(responder) as server:
        backend = make_backend(server)
        turn = await backend.generate(MESSAGES, {"max_new_tokens": 50})

    assert turn.finish_reason == "length"
    assert turn.output_ids == []
    assert turn.text == ""
    assert len(backend.session_turns) == 1  # recorded so truncation is exportable
    assert server.abort_payloads == []


async def test_other_400_still_raises():
    async def responder(payload):
        return web.Response(status=400, text="malformed sampling params")

    async with FakeSGLangServer(responder) as server:
        backend = make_backend(server)
        with pytest.raises(RuntimeError, match="400.*malformed sampling params"):
            await backend.generate(MESSAGES, {"max_new_tokens": 50})

    assert backend.session_turns == []


async def test_non_200_response_raises_with_status():
    async def responder(payload):
        return web.Response(status=503, text="router overloaded")

    async with FakeSGLangServer(responder) as server:
        backend = make_backend(server)
        with pytest.raises(RuntimeError, match="503.*router overloaded"):
            await backend.generate(MESSAGES, {"max_new_tokens": 50})

    assert backend.session_turns == []
    assert server.abort_payloads == []  # server answered; nothing to abort


async def test_cancellation_aborts_the_inflight_request():
    request_started = asyncio.Event()

    async def responder(payload):
        request_started.set()
        await asyncio.sleep(5)
        return sglang_response("late")

    async with FakeSGLangServer(responder) as server:
        backend = make_backend(server)
        task = asyncio.create_task(backend.generate(MESSAGES, {"max_new_tokens": 50}))
        await asyncio.wait_for(request_started.wait(), timeout=10)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        for _ in range(100):
            if server.abort_payloads:
                break
            await asyncio.sleep(0.01)

    assert len(server.abort_payloads) == 1
    assert server.abort_payloads[0]["rid"] == server.generate_payloads[0]["rid"]
    assert backend.session_turns == []


async def test_multi_turn_session_shrinks_budget_and_records_turns():
    async def responder(payload):
        return sglang_response("go")

    async with FakeSGLangServer(responder) as server:
        first_prompt_ids = ByteTokenizer().apply_chat_template(
            [{"role": m["role"], "content": m["content"]} for m in MESSAGES],
            add_generation_prompt=True,
            tokenize=True,
        )
        backend = make_backend(server, max_context_len=len(first_prompt_ids) + 60)

        first = await backend.generate(MESSAGES, {"max_new_tokens": 1000})
        longer = [
            *MESSAGES,
            {"role": "assistant", "content": first.text},
            {"role": "user", "content": "Exit code: 0 | output"},
        ]
        await backend.generate(longer, {"max_new_tokens": 1000})

    first_budget = server.generate_payloads[0]["sampling_params"]["max_new_tokens"]
    second_budget = server.generate_payloads[1]["sampling_params"]["max_new_tokens"]
    assert first_budget == 60
    # Growth: "<assistant>go" (13) + "<user>Exit code: 0 | output" (27).
    assert second_budget == 20
    assert [len(t.turn.output_ids) for t in backend.session_turns] == [2, 2]
    assert backend.session_turns[0].prompt_messages[0]["role"] == "system"
    assert backend.session_turns[1].prompt_messages[-1]["role"] == "user"
