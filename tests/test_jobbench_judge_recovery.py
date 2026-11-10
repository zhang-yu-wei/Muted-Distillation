import json
from types import SimpleNamespace

import pytest

from officeqa_rl.jobbench.judge import JudgeConfig, judge_rubric, parse_judge_json


def test_literal_backslash_in_explanation_preserves_verdict():
    text = r'{"rubric_passed": false, "overall_reasoning": "Uses \\valid and \Delta, \$5."}'
    result = parse_judge_json(text)
    assert result["rubric_passed"] is False
    assert result["overall_reasoning"] == r"Uses \valid and \Delta, \$5."


def test_valid_json_escapes_are_unchanged():
    result = {"rubric_passed": True, "reason": 'newline\nquote"tab\tpath\\file λ'}
    assert parse_judge_json(json.dumps(result)) == result


@pytest.mark.parametrize("text", [
    '{"rubric_passed": tru}',
    '{"rubric_passed": true, "reason": "unfinished',
    r'{"rubric_passed": true, "reason": "\uZZZZ"}',
])
def test_structurally_invalid_judgments_still_fail(text):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_judge_json(text)


@pytest.mark.asyncio
async def test_vertex_refreshes_adc_before_each_rubric_request(monkeypatch):
    import officeqa_rl.jobbench.judge as module
    tokens = iter(["fresh-1", "fresh-2"])
    monkeypatch.setattr(module, "_adc_token", lambda: next(tokens))
    client = SimpleNamespace(api_key="expired")
    observed = []

    async def create(**kwargs):
        observed.append(client.api_key)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"rubric_passed": true, "criteria_results": []}'
        ))])

    client.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    config = JudgeConfig(api_key="", api_base="https://aiplatform.googleapis.com/v1")
    for _ in range(2):
        result = await judge_rubric(client, 0, {"weight": 1}, "files", config)
        assert result.passed
    assert observed == ["fresh-1", "fresh-2"]


@pytest.mark.asyncio
async def test_truncated_judge_response_is_not_retried_at_same_budget():
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(content="Still counting the CSV rows..."),
            )],
            usage=SimpleNamespace(completion_tokens_details=SimpleNamespace(
                reasoning_tokens=15727,
            )),
        )

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create),
    ))
    config = JudgeConfig(api_key="test", max_retries=8, max_completion_tokens=16384)
    result = await judge_rubric(client, 0, {"weight": 1}, "files", config)
    assert len(calls) == 1
    assert "judge_response_truncated" in result.error
    assert "max_completion_tokens=16384" in result.error
    assert "reasoning_tokens=15727" in result.error
    assert not result.passed


@pytest.mark.asyncio
async def test_missing_document_reader_is_ungraded_and_never_calls_judge(tmp_path, monkeypatch):
    from officeqa_rl.jobbench import judge as module
    from officeqa_rl.jobbench.extraction import DocumentReaderUnavailableError
    (tmp_path / "deliverable.docx").write_bytes(b"saved artifact")
    task = SimpleNamespace(uid="test", rubric_dicts=lambda: [
        {"rubric": "Correct answer", "weight": 10, "criterion": ["Correct"]}])

    def missing(*args):
        raise DocumentReaderUnavailableError("mammoth/python-docx unavailable")

    monkeypatch.setattr(module, "extract_all_file_contents", missing)
    monkeypatch.setattr(module, "_client", lambda *a: pytest.fail("must not call judge"))
    result = await module.judge_episode(task, tmp_path, JudgeConfig(api_key="test"))
    assert not result.graded
    assert result.judge_failures == 1
    assert "document_extraction_unavailable" in result.rubrics[0].error


@pytest.mark.asyncio
async def test_truncated_judge_response_retries_once_with_fallback_effort():
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if "reasoning_effort" not in kwargs:
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content="Still counting the CSV rows..."),
                )],
                usage=SimpleNamespace(completion_tokens_details=SimpleNamespace(
                    reasoning_tokens=31455,
                )),
            )
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content='{"rubric_passed": true, "criteria_results": []}'),
        )])

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create),
    ))
    config = JudgeConfig(api_key="test", max_retries=8, max_completion_tokens=32768,
                         truncation_fallback_reasoning_effort="low")
    result = await judge_rubric(client, 0, {"weight": 1}, "files", config)
    assert [c.get("reasoning_effort") for c in calls] == [None, "low"]
    assert result.passed and not result.error


@pytest.mark.asyncio
async def test_truncation_under_fallback_effort_is_final():
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(content="..."),
            )],
            usage=SimpleNamespace(completion_tokens_details=SimpleNamespace(
                reasoning_tokens=31000,
            )),
        )

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create),
    ))
    config = JudgeConfig(api_key="test", max_retries=8, max_completion_tokens=32768,
                         truncation_fallback_reasoning_effort="low")
    result = await judge_rubric(client, 0, {"weight": 1}, "files", config)
    assert len(calls) == 2 and calls[1]["reasoning_effort"] == "low"
    assert "judge_response_truncated" in result.error and not result.passed


@pytest.mark.asyncio
async def test_empty_message_object_is_retried_not_permanent():
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=None)])
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content='{"rubric_passed": true, "criteria_results": []}'),
        )])

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create),
    ))
    config = JudgeConfig(api_key="test", max_retries=8, retry_delay_sec=0.0, retry_max_delay_sec=0.0)
    result = await judge_rubric(client, 0, {"weight": 1}, "files", config)
    assert len(calls) == 2
    assert result.passed and not result.error


@pytest.mark.asyncio
async def test_invalid_argument_400_is_retried_a_bounded_number_of_times(monkeypatch):
    import officeqa_rl.jobbench.judge as module

    monkeypatch.setattr(module, "INVALID_ARGUMENT_RETRY_DELAY_SEC", 0.0)
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if len(calls) <= 2:
            raise RuntimeError(
                "Error code: 400 - [{'error': {'code': 400, 'message': "
                "'Request contains an invalid argument.', 'status': 'INVALID_ARGUMENT'}}]"
            )
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content='{"rubric_passed": true, "criteria_results": []}'),
        )])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    config = JudgeConfig(api_key="test", max_retries=8, retry_delay_sec=0.0, retry_max_delay_sec=0.0)
    result = await judge_rubric(client, 0, {"weight": 1}, "files", config)
    assert len(calls) == 3  # original + INVALID_ARGUMENT_RETRIES
    assert result.passed and not result.error


@pytest.mark.asyncio
async def test_persistent_invalid_argument_400_gives_up_after_the_bounded_retries(monkeypatch):
    import officeqa_rl.jobbench.judge as module

    monkeypatch.setattr(module, "INVALID_ARGUMENT_RETRY_DELAY_SEC", 0.0)
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("Error code: 400 - status INVALID_ARGUMENT")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    config = JudgeConfig(api_key="test", max_retries=8, retry_delay_sec=0.0, retry_max_delay_sec=0.0)
    result = await judge_rubric(client, 0, {"weight": 1}, "files", config)
    assert len(calls) == 1 + module.INVALID_ARGUMENT_RETRIES
    assert not result.passed and "judge_error" in (result.error or "")


@pytest.mark.asyncio
async def test_other_400s_are_still_permanent():
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("Error code: 400 - model not found for this project")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    config = JudgeConfig(api_key="test", max_retries=8, retry_delay_sec=0.0, retry_max_delay_sec=0.0)
    result = await judge_rubric(client, 0, {"weight": 1}, "files", config)
    assert len(calls) == 1
    assert not result.passed
