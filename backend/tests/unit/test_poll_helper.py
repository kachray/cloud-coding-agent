"""Unit tests for the functional suite's suspend-poll helper.

The helper's one non-trivial job is keeping infrastructure conditions (rate
limit, API error, cancellation, a still-running loop) distinct from model
non-response. Conflating the two once sent an entire investigation down the
wrong path, so every branch is pinned here. No API, no Groq mock — synthetic
asyncio tasks only.
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.functional.test_agent_loop import (  # noqa: E402
    _await_pending_question,
    _suspend_gap,
)


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
    status, elapsed, detail = await _await_pending_question(
        _agent(), task, timeout=5.0
    )

    assert status == "raised", (
        f"a task that raised must be reported as such, not as silence; "
        f"got {status!r}"
    )
    assert "RuntimeError" in detail and "429" in detail, (
        f"detail must carry the exception, got {detail!r}"
    )
    assert elapsed < 5.0, (
        f"must break as soon as the task ends; waited {elapsed:.2f}s"
    )


async def test_clean_finish_without_asking_is_model_behavior():
    async def quiet():
        return "answered without asking"

    task = asyncio.create_task(quiet())
    status, elapsed, detail = await _await_pending_question(
        _agent(), task, timeout=5.0
    )

    assert status == "finished", (
        f"a run that returned without asking is the model choosing not to "
        f"use the tool; got {status!r}"
    )
    assert detail == ""
    assert elapsed < 5.0


async def test_still_running_at_deadline_is_inconclusive_not_silence():
    async def never_returns():
        await asyncio.sleep(30)

    task = asyncio.create_task(never_returns())
    status, elapsed, _ = await _await_pending_question(
        _agent(), task, timeout=0.3
    )

    assert status == "timeout", (
        f"a task still running at the deadline must NOT be reported as a "
        f"clean finish or as model non-response; got {status!r}"
    )
    assert elapsed >= 0.3, f"must respect the deadline; waited {elapsed:.2f}s"

    task.cancel()
    await _settle(task)


async def test_cancelled_task_is_reported_as_cancelled():
    async def never():
        await asyncio.sleep(30)

    task = asyncio.create_task(never())
    task.cancel()
    await _settle(task)

    status, elapsed, detail = await _await_pending_question(
        _agent(), task, timeout=5.0
    )

    assert status == "cancelled", (
        "task.exception() raises on a cancelled task, so the helper must "
        f"branch on cancellation first; got {status!r}"
    )
    assert elapsed < 5.0
    assert detail


async def test_breaks_immediately_when_question_is_pending():
    async def still_working():
        await asyncio.sleep(30)

    task = asyncio.create_task(still_working())
    status, elapsed, _ = await _await_pending_question(
        _agent("what is the access code?"), task, timeout=5.0
    )

    assert status == "suspended"
    assert elapsed < 1.0, (
        f"a pending question must end the poll at once; waited {elapsed:.2f}s"
    )

    task.cancel()
    await _settle(task)


async def test_only_the_finished_gap_blames_the_model():
    """The whole point of the helper: infrastructure must not read as model silence."""
    infra = {
        "raised": _suspend_gap("raised", 1.0, "RuntimeError: 429", "user_question"),
        "cancelled": _suspend_gap("cancelled", 1.0, "cancelled", "user_question"),
        "timeout": _suspend_gap("timeout", 60.0, "", "user_question"),
    }
    for status, text in infra.items():
        assert "NOT" in text, (
            f"an infrastructure status ({status}) must explicitly disclaim "
            f"model non-response rather than leave the claim standing: {text!r}"
        )
    assert "inconclusive" in infra["timeout"].lower(), (
        "a still-running loop is not evidence either way; say so: "
        f"{infra['timeout']!r}"
    )

    finished = _suspend_gap("finished", 0.1, "", "user_question")
    assert "didn't call the tool" in finished, (
        f"a clean finish IS model non-response and should say so: {finished!r}"
    )
