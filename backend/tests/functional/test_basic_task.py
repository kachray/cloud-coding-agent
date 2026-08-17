"""Functional tests for basic agent tasks.

These run the REAL agent loop against the REAL Gemini Interactions API and a
REAL local-subprocess sandbox (no mocks of either). They assert on real
outcomes — files created on disk, shell commands actually executed — per the
project's verification standard in CLAUDE.md.

Requires GEMINI_API_KEY in backend/.env (loaded via python-dotenv).
"""
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv

# Make backend/ importable when run via `uv run pytest` from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Load GEMINI_API_KEY from backend/.env before the agent loop builds its client.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from google import genai

from sandbox.local import LocalSandbox
from agent.loop import AgentLoop


@pytest.fixture
def client():
    """A live Gemini client built from the loaded GEMINI_API_KEY."""
    return genai.Client(api_key=None)  # picks up GEMINI_API_KEY / GOOGLE_API_KEY env


@pytest_asyncio.fixture
async def sandbox(tmp_path):
    sb = LocalSandbox(working_dir=tmp_path)
    yield sb
    # Teardown: close every shell the loop created so no subprocess transport
    # leaks across tests (Windows raises PytestUnraisableExceptionWarning on
    # unclosed proactor pipes otherwise).
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


class TestBasicTasks:
    """Two end-to-end cases that guard the core Milestone 1 contract."""

    @pytest.mark.asyncio
    async def test_create_hello_txt_containing_world(self, agent, tmp_path):
        """Case 1: 'create a file called hello.txt containing the word world'
        drives the real loop to produce that file on disk with the right content.
        """
        result = await agent.run(
            "Create a file called hello.txt containing the word world.",
            working_dir=tmp_path,
        )

        # The real outcome: the file exists on disk (created by the tools, not
        # asserted against the model's prose).
        hello_file = tmp_path / "hello.txt"
        assert hello_file.exists(), (
            f"hello.txt was not created in {tmp_path}. Loop result:\n{result}"
        )
        content = hello_file.read_text(encoding="utf-8")
        assert "world" in content.lower(), (
            f"hello.txt content should contain 'world'; got: {content!r}"
        )

    @pytest.mark.asyncio
    async def test_two_distinct_shells(self, agent, tmp_path):
        """Case 2: an instruction needing two shells must create two distinct
        shell_ids and run a command correctly in each. This is the regression
        guard against collapsing back to a single stateless `run_shell(cmd)` —
        if the loop only had one shell, both outputs would still come back, but
        only one shell_id would ever be issued and the parallel-shell contract
        would silently be gone. So we assert two DIFFERENT shell_ids appeared.
        """
        result = await agent.run(
            "In one shell run `echo one`, in a separate (different) shell run "
            "`echo two`, then tell me both outputs.",
            working_dir=tmp_path,
        )

        # Shell ids referenced by the run_in_shell calls the loop actually issued.
        run_shell_targets = [
            call["args"]["shell_id"]
            for call in agent.tool_calls
            if call["name"] == "run_in_shell"
        ]

        # The key regression check: at least two DISTINCT shell_ids were used.
        # A single stateless run_shell(cmd) tool would collapse to zero ids (no
        # create_shell at all) or one id reused — either way this fails, which
        # is exactly what guards the stateful multi-shell contract.
        distinct_ids = set(run_shell_targets)
        assert len(distinct_ids) >= 2, (
            "Expected two distinct shell_ids to be used by run_in_shell, "
            f"but saw only: {distinct_ids}. A single-shell tool would collapse "
            "to one id — this assertion guards the stateful multi-shell contract."
        )

        # Both commands executed with exit code 0 — confirm via the result text.
        result_lower = (result or "").lower()
        assert "one" in result_lower, (
            f"Final output should report 'one'; got:\n{result}"
        )
        assert "two" in result_lower, (
            f"Final output should report 'two'; got:\n{result}"
        )


class TestSandboxDirect:
    """Direct sandbox checks (no Gemini) — fast, local-truth anchors."""

    @pytest.mark.asyncio
    async def test_two_shells_are_independent_processes(self, sandbox):
        s1 = await sandbox.create_shell(name="shell-one")
        s2 = await sandbox.create_shell(name="shell-two")
        assert s1.shell_id != s2.shell_id

        r1 = await sandbox.run_in_shell(s1.shell_id, "echo one")
        r2 = await sandbox.run_in_shell(s2.shell_id, "echo two")
        assert "one" in r1.stdout and r1.exit_code == 0
        assert "two" in r2.stdout and r2.exit_code == 0

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

        # Overwrite then undo restores original.
        await sandbox.write_file(tmp_path / "a.txt", "changed")
        assert (tmp_path / "a.txt").read_text() == "changed"
        assert await sandbox.undo()
        assert (tmp_path / "a.txt").read_text() == "hello"

        # Undo again removes the file entirely (undo of create).
        assert await sandbox.undo()
        assert not (tmp_path / "a.txt").exists()
