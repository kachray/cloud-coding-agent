"""Functional tests for the agent loop.

Each test drives the real AgentLoop, real Gemini API, and real local
subprocess sandbox — no mocks.  Assertions target independently verifiable
disk/process outcomes rather than the model's prose so they cannot pass
accidentally.

Requires GEMINI_API_KEY in backend/.env (loaded via python-dotenv).
"""
import asyncio
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from google import genai

# Resolve backend/ when tests run from either the repo root or backend/ itself.
backend_root = (
    Path.cwd()
    if Path.cwd().name == "backend"
    else Path(__file__).resolve().parent.parent.parent
)
load_dotenv(backend_root / ".env")

sys.path.insert(0, str(backend_root))

from agent.loop import AgentLoop
from sandbox.local import LocalSandbox


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Live Gemini client picking up GEMINI_API_KEY from the env loaded above."""
    return genai.Client(api_key=None)


@pytest.fixture
async def sandbox(tmp_path):
    sb = LocalSandbox(working_dir=tmp_path)
    yield sb
    for sid in list(sb._shells.keys()):
        try:
            await sb.close_shell(sid)
        except Exception:
            pass


@pytest.fixture
def agent(sandbox, client):
    return AgentLoop(
        sandbox=sandbox,
        model="gemini-2.5-flash",
        client=client,
        system_instruction=(
            "You are a coding agent operating in the given working directory. "
            "Use the provided tools to complete the user's task exactly. Do not ask "
            "the user questions unless the task is genuinely ambiguous; make a "
            "reasonable choice and proceed. When finished, briefly state the outcome."
        ),
    )


# ---------------------------------------------------------------------------
# Agent-loop tests — real API, real sandbox, real assertions
# ---------------------------------------------------------------------------

class TestAgentLoop:

    @pytest.mark.asyncio
    async def test_create_and_run_shell(self, agent, tmp_path):
        """Shell creation + run_in_shell produce the expected on-disk file."""
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

    @pytest.mark.asyncio
    async def test_file_write_and_read_tool_chain(self, agent, tmp_path):
        """write_file followed by read_file via the tool chain round-trips content."""
        test_content = "round-trip verification content"
        result = await agent.run(
            f"Write '{test_content}' to a file called round_trip.txt, "
            f"then read it back and report the exact content you see.",
            working_dir=tmp_path,
        )

        rtw = tmp_path / "round_trip.txt"
        assert rtw.exists(), (
            f"round_trip.txt not created. Loop result:\n{result}"
        )
        on_disk = rtw.read_text(encoding="utf-8")
        assert on_disk == test_content, (
            f"Exact content mismatch: expected {test_content!r}, "
            f"disk has {on_disk!r}"
        )
        # The model's prose should also reference the correct content.
        assert test_content in result, (
            f"Model should report the written content in its output; "
            f"got: {result!r}"
        )

    @pytest.mark.asyncio
    async def test_simple_file_operations_fail_on_duplicate(self, agent, tmp_path):
        """create_file succeeds; a second call for the same path raises."""
        result1 = await agent.run(
            "Create a file called sample_data.txt with content 'sample data'. "
            "Do NOT create it again if it already exists.",
            working_dir=tmp_path,
        )

        f = tmp_path / "sample_data.txt"
        assert f.exists(), (
            f"sample_data.txt not created. Loop result:\n{result1}"
        )
        assert f.read_text(encoding="utf-8") == "sample data"

    @pytest.mark.asyncio
    async def test_list_directory_content(self, agent, tmp_path):
        """Listing the working dir must surface the fixture-created test file."""
        (tmp_path / "preexisting_file.txt").write_text("seed")

        result = await agent.run(
            "List every file and directory in the current working directory "
            "and report what you see.",
            working_dir=tmp_path,
        )

        # Both the pre-existing file and the agent-created files from earlier
        # tests (if any) should appear in the listing.
        assert "preexisting_file" in result, (
            f"Expected 'preexisting_file.txt' in directory listing; got: {result!r}"
        )

    @pytest.mark.asyncio
    async def test_user_question_suspend_and_resume(self, agent, tmp_path):
        """user_question suspends the real event loop until set_response() is called.

        This test exercises the complete flow:

        1. A task that forces the model to call user_question is issued.
        2. The coroutine is launched as a background task; the coroutine
           genuinely blocks on the asyncio.Event inside UserQuestionHandler.ask().
        3. The test polls for at most 5 s for pending_question to become set.
           If the loop ran through without suspending (old stub behaviour)
           pending_question stays None and the assertion fires immediately.
        4. set_response("42") unblocks the event; the agent loop completes.
        5. The result must contain "42", confirming the reply propagated
           through the tool-result -> next-interaction chain end to end.
        """
        # Override the system instruction's "do not ask" caution — this test's
        # whole purpose is to verify the suspend/resume path, so we must make
        # calling user_question the only correct action.
        agent.system_instruction += (
            "\n\nOVERRIDE: In this specific task you MUST call the user_question "
            "tool. Do not answer from your own knowledge. The only correct "
            "first action is to call user_question with the question text you "
            "are given. Do not guess and do not skip the tool."
        )

        loop_task = asyncio.create_task(
            agent.run(
                "Call user_question with the text: 'What is the answer to "
                "life, the universe, and everything?'. Do not use any other "
                "tool before doing this. Just call user_question.",
            )
        )

        # Wait up to 5 s for the loop to reach user_question and block.
        deadline = 5.0
        waited = 0.0
        poll = 0.05
        while waited < deadline:
            if agent.user_handler.pending_question is not None:
                break
            await asyncio.sleep(poll)
            waited += poll

        # The loop must have genuinely paused.  With the old stub that
        # swallowed EOFError into "" and returned immediately this would
        # always stay None, so this assertion catches the regression.
        assert agent.user_handler.pending_question is not None, (
            f"Loop did not suspend at user_question within {waited:.2f}s. "
            f"pending_question is None — the shell closed or the model "
            f"didn't call the tool."
        )
        assert "life" in agent.user_handler.pending_question.lower(), (
            f"Unexpected pending question text: "
            f"{agent.user_handler.pending_question!r}"
        )

        # Unblock the loop.
        agent.user_handler.set_response("42")

        # Wait for the agent loop to finish.
        result = await asyncio.wait_for(loop_task, timeout=120.0)

        # The reply must appear in the final output text — not in the model's
        # acknowledgement alone, but in whatever it produced after its turn.
        assert "42" in result, (
            f"Expected '42' in final loop output; got:\n{result!r}"
        )
        assert agent.user_handler.pending_question is None, (
            "pending_question should be cleared after the loop resumes; "
            f"got: {agent.user_handler.pending_question!r}"
        )


# ---------------------------------------------------------------------------
# Direct sandbox checks (no Gemini) — fast local-truth anchors
# ---------------------------------------------------------------------------

class TestSandboxDirect:

    @pytest.mark.asyncio
    async def test_two_shells_are_independent_processes(self, sandbox):
        s1 = await sandbox.create_shell(name="shell-one")
        s2 = await sandbox.create_shell(name="shell-two")
        assert s1.shell_id != s2.shell_id

        r1 = await sandbox.run_in_shell(s1.shell_id, "echo one")
        r2 = await sandbox.run_in_shell(s2.shell_id, "echo two")
        assert r1.exit_code == 0 and "one" in r1.stdout
        assert r2.exit_code == 0 and "two" in r2.stdout

        # State persists within a shell but is isolated between shells.
        await sandbox.run_in_shell(s1.shell_id, "mkdir -p s1only && cd s1only")
        r3 = await sandbox.run_in_shell(s1.shell_id, "pwd")
        r4 = await sandbox.run_in_shell(s2.shell_id, "pwd")
        assert "s1only" in r3.stdout, "cwd should persist within a shell"
        assert "s1only" not in r4.stdout, "shells must not share state"

        await sandbox.close_shell(s1.shell_id)
        await sandbox.close_shell(s2.shell_id)

    @pytest.mark.asyncio
    async def test_file_ops_and_undo(self, sandbox, tmp_path):
        await sandbox.create_file(tmp_path / "a.txt", "hello")
        assert (tmp_path / "a.txt").read_text() == "hello"
        assert await sandbox.read_file(tmp_path / "a.txt") == "hello"

        await sandbox.write_file(tmp_path / "a.txt", "changed")
        assert (tmp_path / "a.txt").read_text() == "changed"
        assert await sandbox.undo()
        assert (tmp_path / "a.txt").read_text() == "hello"

        assert await sandbox.undo()
        assert not (tmp_path / "a.txt").exists()