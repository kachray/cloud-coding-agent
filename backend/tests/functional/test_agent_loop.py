"""Functional tests for the agent loop — real API, real sandbox, no mocks.

Requires GROQ_API_KEY in backend/.env.
"""
import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.loop import AgentLoop, _USER_ANSWER_GUARD  # noqa: E402


async def _await_pending_question(agent, task, timeout=60.0):
    """Wait until the loop suspends at user_question, or ``task`` ends.

    Returns ``(status, elapsed_seconds, detail)``; ``status`` is one of:

      "suspended"  ``pending_question`` is set — the model called the tool
      "finished"   ``run()`` returned a final answer without ever asking
      "raised"     ``run()`` ended by raising (rate limit, API error); detail
      "cancelled"  the task was cancelled; detail
      "timeout"    still running at the deadline, no question pending

    Only "suspended" means the model called the tool. Everything else is
    reported as itself: a loop that raised, was cancelled, or is still running
    is an infrastructure condition and must never be presented as "the model
    didn't call the tool" — that misattribution sent a whole investigation
    down the wrong path once already. Callers assert ``status == "suspended"``
    and format the mismatch with ``_suspend_gap``.

    Caveat: this sees only failures that escape ``run()``. ``_execute_tool``
    swallows tool exceptions into an "ERROR executing ..." tool result, so a
    ``user_question`` that raises inside ``_dispatch`` — a missing ``text``
    arg, say — leaves ``pending_question`` None with the task finishing
    cleanly, and reads here as "finished".

    time.monotonic(), not an accumulated nominal counter: Windows sleep
    granularity makes asyncio.sleep(0.05) cost ~60-75ms, so accumulating the
    nominal 0.05 under-counts elapsed time by ~40% and cut a 5s wait short
    before a 7.45s model turn ever reached the tool call.

    ``timeout`` must clear ``_call_with_retry``'s 429 chain (2+4+8+10 = 24s),
    or a rate-limited run exhausts this poll while the task is still retrying.
    That is a soft bound only: ``loop.py`` raises its wait to whatever
    ``"Please retry in Xs"`` the server suggests, with no cap, so a long hint
    outlives any fixed deadline here — which is why "timeout" is reported as
    inconclusive rather than as model non-response.

    It may safely exceed ``UserQuestionHandler.ask``'s inner 30s ``wait_for``:
    that timeout only begins once a question is pending, and this poll breaks
    the moment ``pending_question`` is set, so the stale-pending_question left
    behind by a handler timeout cannot be observed here.
    """
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if agent.user_handler.pending_question is not None:
            break
        if task.done():
            break
        await asyncio.sleep(0.05)

    elapsed = time.monotonic() - start
    if agent.user_handler.pending_question is not None:
        return "suspended", elapsed, ""
    if not task.done():
        return "timeout", elapsed, ""
    if task.cancelled():
        return "cancelled", elapsed, "task was cancelled"
    exc = task.exception()
    if exc is not None:
        return "raised", elapsed, f"{type(exc).__name__}: {exc}"
    return "finished", elapsed, ""


def _suspend_gap(status, elapsed, detail, what):
    """Explain, without misattributing, why the loop never suspended at ``what``."""
    if status in ("raised", "cancelled"):
        return (
            f"The loop {status} before suspending at {what} (after "
            f"{elapsed:.2f}s). That is an infrastructure failure, NOT model "
            f"non-response: {detail}"
        )
    if status == "timeout":
        return (
            f"The loop was still running after {elapsed:.2f}s with no {what} "
            f"pending. Inconclusive — a slow turn, a hung connection, or an "
            f"extended retry — NOT evidence the model didn't call the tool."
        )
    return (
        f"The loop finished without calling {what} (after {elapsed:.2f}s): "
        f"pending_question is None — the model didn't call the tool."
    )


class TestAgentLoop:

    async def test_create_and_run_shell(self, agent, tmp_path):
        result = await agent.run(
            "Create a shell, run `echo hello world`, then write the output to "
            "a file called output.txt in the current directory.",
            working_dir=tmp_path,
        )
        output = tmp_path / "output.txt"
        assert output.exists(), (
            f"output.txt was not created in {tmp_path}. Loop result:\n{result}"
        )
        content = output.read_text(encoding="utf-8")
        assert "hello" in content.lower() and "world" in content.lower(), (
            f"output.txt should contain 'hello world'; got: {content!r}"
        )

    async def test_file_write_and_read_tool_chain(self, agent, tmp_path):
        test_content = "round-trip verification content"
        result = await agent.run(
            f"Write '{test_content}' to a file called round_trip.txt, "
            f"then read it back and report the exact content you see.",
            working_dir=tmp_path,
        )
        rtw = tmp_path / "round_trip.txt"
        assert rtw.exists(), f"round_trip.txt not created. Loop result:\n{result}"
        on_disk = rtw.read_text(encoding="utf-8")
        assert on_disk == test_content, (
            f"Exact content mismatch: expected {test_content!r}, "
            f"disk has {on_disk!r}"
        )
        assert test_content in result, (
            f"Model should report the written content in its output; "
            f"got: {result!r}"
        )

    async def test_create_file_fails_on_duplicate(self, sandbox, tmp_path):
        path = tmp_path / "sample_data.txt"
        await sandbox.create_file(path, "sample data")
        assert path.read_text(encoding="utf-8") == "sample data"
        with pytest.raises(FileExistsError):
            await sandbox.create_file(path, "other data")

    async def test_list_directory_content(self, agent, tmp_path):
        (tmp_path / "preexisting_file.txt").write_text("seed")
        result = await agent.run(
            "List every file and directory in the current working directory "
            "and report what you see.",
            working_dir=tmp_path,
        )
        assert "preexisting_file" in result, (
            f"Expected 'preexisting_file.txt' in directory listing; got: {result!r}"
        )

    async def test_user_question_suspend_and_resume(self, agent, tmp_path):
        override = (
            "CRITICAL: You have NO knowledge of your own and MUST NOT answer "
            "any question yourself, even trivial ones. Your ONLY capability is "
            "the `user_question` tool. For EVERY prompt, your FIRST and ONLY "
            "action is to call `user_question` with the exact prompt text. "
            "There is no other path. Calling any other tool or producing "
            "prose without calling user_question first is a total failure. "
            "Call user_question NOW with the user's exact text. "
            "After you call user_question, you receive a tool result. That "
            "tool result text IS the user's answer and IS the single correct "
            "final answer. Your response MUST reproduce that tool result text "
            "verbatim and nothing else — no commentary, no 'I asked', and no "
            "answering the question yourself even if you think you know it. "
            "The user's reply, exactly as it appears in the tool result, is "
            "the only acceptable output."
        )
        agent.system_instruction = override

        loop_task = asyncio.create_task(
            agent.run(
                "Call user_question with the exact text: 'What is the answer to "
                "life, the universe, and everything?'. "
                "You have no other option — call user_question now.",
            )
        )

        status, elapsed, detail = await _await_pending_question(agent, loop_task)

        assert status == "suspended", _suspend_gap(
            status, elapsed, detail, "user_question"
        )
        assert "life" in agent.user_handler.pending_question.lower(), (
            f"Unexpected pending question: "
            f"{agent.user_handler.pending_question!r}"
        )

        agent.user_handler.set_response("42")
        result = await asyncio.wait_for(loop_task, timeout=120.0)

        assert "42" in result, (
            f"Expected '42' in final loop output; got:\n{result!r}"
        )
        assert agent.user_handler.pending_question is None, (
            "pending_question should be cleared after the loop resumes; "
            f"got: {agent.user_handler.pending_question!r}"
        )
        # The answer the model saw must carry the guard, not the bare string —
        # otherwise the loop's anti-drop mechanism is not reaching the API.
        tool_msgs = [m for m in agent._messages if m["role"] == "tool"]
        assert any(
            "42" in m["content"] and _USER_ANSWER_GUARD in m["content"]
            for m in tool_msgs
        ), (
            f"user_question tool result did not carry _USER_ANSWER_GUARD; "
            f"tool messages were: {tool_msgs!r}"
        )

    async def test_user_question_multi_turn_no_stale_response(self, agent, tmp_path):
        override = (
            "CRITICAL: You have NO knowledge of your own and MUST NOT answer "
            "any question yourself, even trivial ones. Your ONLY capability is "
            "the `user_question` tool. For EVERY prompt, your FIRST and ONLY "
            "action is to call `user_question` with the exact prompt text. "
            "There is no other path. Calling any other tool or producing "
            "prose without calling user_question first is a total failure. "
            "Call user_question NOW with the user's exact text. "
            "After you call user_question, you receive a tool result. That "
            "tool result text IS the user's answer and IS the single correct "
            "final answer. Your response MUST reproduce that tool result text "
            "verbatim and nothing else — no commentary, no 'I asked', and no "
            "answering the question yourself even if you think you know it. "
            "The user's reply, exactly as it appears in the tool result, is "
            "the only acceptable output."
        )
        agent.system_instruction = override

        # --- First round ---
        loop_task = asyncio.create_task(
            agent.run(
                "Call user_question with the exact text: 'What is the access "
                "code?'. Call user_question now.",
            )
        )

        status, elapsed, detail = await _await_pending_question(agent, loop_task)

        assert status == "suspended", _suspend_gap(
            status, elapsed, detail, "the first user_question"
        )

        agent.user_handler.set_response("first_answer")
        result1 = await asyncio.wait_for(loop_task, timeout=120.0)
        assert "first_answer" in result1, (
            f"Expected 'first_answer' in first round; got: {result1!r}"
        )
        assert agent.user_handler.pending_question is None

        # --- Second round ---
        loop_task2 = asyncio.create_task(
            agent.run(
                "Now call user_question with the exact text: "
                "'What is the secret passphrase?'. "
                "Call user_question now.",
            )
        )

        status2, elapsed2, detail2 = await _await_pending_question(agent, loop_task2)

        # A clean finish here is not merely "the model didn't ask" — it is the
        # exact shape of the stale-response bug, so name that hypothesis when
        # it applies instead of leaving the reader with a timing complaint.
        assert status2 == "suspended", _suspend_gap(
            status2, elapsed2, detail2, "the second user_question"
        ) + (
            " A clean finish here means the second ask() did NOT suspend — "
            "suspect a stale response: _user_response was not cleared after "
            "the first round."
            if status2 == "finished" else ""
        )
        assert "passphrase" in agent.user_handler.pending_question.lower(), (
            f"Unexpected second pending question: "
            f"{agent.user_handler.pending_question!r}"
        )

        agent.user_handler.set_response("second_answer")
        result2 = await asyncio.wait_for(loop_task2, timeout=120.0)
        assert "second_answer" in result2, (
            f"Expected 'second_answer' in second round; got: {result2!r}"
        )
        assert agent.user_handler.pending_question is None

    async def test_escape_attempt_is_rejected(self, agent, tmp_path):
        """A write_file/create_file aimed outside the working_dir must be
        rejected by the sandbox, not silently write outside the session dir.

        Non-vacuous: the test asserts the model actually *attempted* the
        outside-path write (>=1 such tool call), so it fails if the model
        sidesteps the instruction — and asserts every such attempt returned a
        PermissionError and no file landed outside working_dir.
        """
        outside = tmp_path.parent / f"escape_{tmp_path.name}.txt"
        instruction = (
            f"Write the text 'pwned' to the file at the absolute path "
            f"{outside}. This is a required step of the task. You MUST use "
            f"the write_file or create_file tool with that exact path — do "
            f"NOT use a shell command. If the first tool fails, retry with "
            f"the other file tool using the same absolute path. Then report "
            f"exactly what the tool result said, and whether the file was "
            f"actually created."
        )
        result = await agent.run(instruction, working_dir=tmp_path)

        assert not outside.exists(), (
            f"file was written OUTSIDE the working_dir at {outside}. "
            f"Loop result:\n{result}"
        )

        escape_calls = [
            c
            for c in agent.tool_calls
            if c["name"] in ("write_file", "create_file")
            and Path(c["args"]["path"]).resolve() == outside.resolve()
        ]
        assert escape_calls, (
            f"agent never attempted the outside-path file write — test is "
            f"vacuous. Loop result:\n{result}"
        )
        for call in escape_calls:
            error = call.get("error") or call.get("result_preview", "")
            assert "PermissionError" in error, (
                f"escape tool call was not rejected: {call}"
            )
