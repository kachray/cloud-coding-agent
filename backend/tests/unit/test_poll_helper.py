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

    text = _suspend_gap(status, elapsed, detail, "user_question")
    assert "infrastructure" in text.lower(), (
        f"a rate-limited loop must be named as infrastructure: {text!r}"
    )
    assert "429" in text, (
        f"the gap report must carry the underlying error, not just a label: "
        f"{text!r}"
    )
    assert "NOT" in text, (
        f"the infrastructure report must explicitly disclaim model "
        f"non-response rather than leaving it as the standing explanation: "
        f"{text!r}"
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

    text = _suspend_gap(status, elapsed, detail, "user_question")
    assert "infrastructure" not in text.lower(), (
        f"a clean finish is model behavior, NOT an infrastructure failure — "
        f"the inverse misattribution, and just as wrong: {text!r}"
    )
    assert "didn't call the tool" in text, (
        f"a clean finish is the one case that IS model non-response and "
        f"should say so plainly: {text!r}"
    )


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


async def test_stale_pending_flag_does_not_hide_a_raised_task():
    """ask() raises without clearing _pending_question; task state must win."""

    async def boom():
        raise RuntimeError("user_question timed out waiting for response.")

    task = asyncio.create_task(boom())
    await _settle(task)

    status, _, detail = await _await_pending_question(
        _agent("what is the access code?"), task, timeout=5.0
    )

    assert status == "raised", (
        f"a dead loop with a stale pending_question must not read as "
        f"suspended — that is infra reported as the model calling the tool; "
        f"got {status!r}"
    )
    assert "timed out" in detail


async def test_stale_pending_flag_on_a_clean_finish_is_flagged():
    async def quiet():
        return "done"

    task = asyncio.create_task(quiet())
    await _settle(task)

    status, elapsed, detail = await _await_pending_question(
        _agent("what is the access code?"), task, timeout=5.0
    )

    assert status == "finished"
    assert detail, (
        "a finished loop that still has pending_question set must say so; "
        "silently reporting a plain clean finish hides a handler bug"
    )

    text = _suspend_gap(status, elapsed, detail, "user_question")
    assert "didn't call the tool" not in text, (
        f"a stale flag is not evidence about the model, and must not be "
        f"reported as if it were: {text!r}"
    )


def test_unknown_status_raises_instead_of_blaming_the_model():
    """The formatter's catch-all must not default to 'the model didn't ask'."""
    for bogus in ("", "suspend", "suspendedd", "error", "errored"):
        try:
            text = _suspend_gap(bogus, 1.0, "", "user_question")
        except ValueError:
            continue
        raise AssertionError(
            f"_suspend_gap accepted unknown status {bogus!r} and rendered it "
            f"as {text!r} — a silent default here is how the misattribution "
            f"comes back"
        )


def test_only_the_finished_gap_blames_the_model():
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
