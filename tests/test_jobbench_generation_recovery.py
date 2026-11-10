import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
from types import SimpleNamespace

import pytest

from officeqa_rl.jobbench.models import GenerationTimeoutError, LoopBridge


def test_bridge_cancels_hung_generation(monkeypatch):
    cancelled = []
    def wait(timeout):
        raise FutureTimeoutError()
    future = SimpleNamespace(result=wait, cancel=lambda: cancelled.append(True))
    monkeypatch.setattr(asyncio, 'run_coroutine_threadsafe', lambda *a: future)
    with pytest.raises(GenerationTimeoutError):
        LoopBridge(object()).run(None, timeout=1800)
    assert cancelled == [True]


def test_standalone_bridge_also_honors_deadline():
    async def hangs():
        await asyncio.sleep(60)
    with pytest.raises(GenerationTimeoutError):
        LoopBridge().run(hangs(), timeout=.01)


def test_generation_timeout_is_preserved_but_not_scored(tmp_path, monkeypatch):
    import run_jobbench_eval as evaluator
    episode = SimpleNamespace(exit_status='GenerationTimeoutError',
                              workspace=tmp_path / 'workspace',
                              messages=[{'content': 'partial trajectory'}], metadata={})
    async def run_episode(*args, **kwargs):
        return episode
    async def must_not_grade(*args, **kwargs):
        pytest.fail('An infrastructure timeout must not be graded')
    monkeypatch.setattr(evaluator, 'run_jobbench_episode', run_episode)
    monkeypatch.setattr(evaluator, 'judge_episode', must_not_grade)
    args = SimpleNamespace(model='same-checkpoint', temperature=1., max_tokens=16384,
                           generation_timeout_sec=1800, keep_workspaces=True)
    async def scenario():
        await evaluator.run_attempt(args, object(), SimpleNamespace(uid='JB__easy__task'), 0,
                                    env_config=None, agent_config=None, judge_config=None,
                                    episode_semaphore=asyncio.Semaphore(1),
                                    boot_semaphore=asyncio.Semaphore(1), output_dir=tmp_path)
    with pytest.raises(RuntimeError, match='partial trajectory retained'):
        asyncio.run(scenario())
    assert list((tmp_path / '.infra-failures').glob('*.json'))
    assert not list(tmp_path.glob('JB__*.json'))
