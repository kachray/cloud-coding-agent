"""Unit tests for the functional suite's suspend-poll helper.

The helper's one non-trivial job is keeping an infrastructure failure (rate
limit, API error) distinct from model non-response. Conflating the two once
sent an entire investigation down the wrong path, so the branches are pinned
here. No API, no Groq mock — synthetic asyncio tasks only.
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.functional.test_agent_loop import _await_pending_question  # noqa: E402


def _agent(question=None):
    return SimpleNamespace(
        user_handler=SimpleNamespace(pending_question=question)
    )


async def _settle(task):
    """Await a task that may raise, for cleanup."""
    try:
        await task
    except BaseException:
        pass


async def test_reports_failure_when_loop_task_raises():
    async def boom():
        await asyncio.sleep(0.01)
        raise RuntimeError("Error code: 429 - rate_limit_exceeded")

    task = asyncio.create_task(boom())
    elapsed, failure = await _await_pending_question(_agent(), task, timeout=5.0)

    assert failure is not None, (
        "a task that raised must be reported as a failure, not as silence"
    )
    assert "RuntimeError" in failure and "429" in failure, (
        f"failure must carry the exception, got {failure!r}"
    )
    assert elapsed < 5.0, (
        f"must break as soon as the task ends; waited {elapsed:.2f}s"
    )


async def test_clean_finish_without_asking_is_not_a_failure():
    async def quiet():
        return "answered without asking"

    task = asyncio.create_task(quiet())
    elapsed, failure = await _await_pending_question(_agent(), task, timeout=5.0)

    assert failure is None, (
        "a clean finish with no question is model behavior — the caller's "
        "pending_question assertion is the right thing to fail there"
    )
    assert elapsed < 5.0


async def test_cancelled_task_does_not_raise_out_of_the_helper():
    async def never():
        await asyncio.sleep(30)

    task = asyncio.create_task(never())
    task.cancel()
    await _settle(task)

    elapsed, failure = await _await_pending_question(_agent(), task, timeout=5.0)

    assert failure is None, (
        "task.exception() raises on a cancelled task; the helper must guard it"
    )
    assert elapsed < 5.0


async def test_breaks_immediately_when_question_is_pending():
    async def still_working():
        await asyncio.sleep(30)

    task = asyncio.create_task(still_working())
    elapsed, failure = await _await_pending_question(
        _agent("what is the access code?"), task, timeout=5.0
    )

    assert failure is None
    assert elapsed < 1.0, (
        f"a pending question must end the poll at once; waited {elapsed:.2f}s"
    )

    task.cancel()
    await _settle(task)
