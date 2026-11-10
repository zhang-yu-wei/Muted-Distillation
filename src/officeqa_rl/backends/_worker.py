"""Async bridge for synchronous generation SDKs."""

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import TypeVar


T = TypeVar("T")


async def run_sync_in_daemon_thread(
    function: Callable[..., T],
    *args: object,
    thread_name: str,
) -> T:
    """Run synchronous SDK work without blocking or retaining an executor.

    Cancelling the awaiting coroutine stops the wait immediately, but a call
    that already started cannot be interrupted: the daemon thread keeps
    running detached and its result is discarded.
    """
    future: Future[T] = Future()

    def run() -> None:
        # Claim the running state so a cancelled awaiter cannot race the
        # result; if the future was cancelled before the call started, skip it.
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(function(*args))
        except BaseException as error:
            future.set_exception(error)

    threading.Thread(target=run, name=thread_name, daemon=True).start()
    return await asyncio.wrap_future(future)


__all__ = ["run_sync_in_daemon_thread"]
