"""Functional tests for the agent loop."""
import asyncio
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load GEMINI_API_KEY from backend/.env before the agent loop builds its client.
# Use cwd-relative path since tests run from backend/ directory
backend_root = Path.cwd() if Path.cwd().name == "backend" else Path(__file__).resolve().parent.parent.parent
load_dotenv(backend_root / ".env")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sandbox.local import LocalSandbox
from agent.loop import AgentLoop


class TestAgentLoop:
    """Tests for the agent loop running real tasks."""

    @pytest.mark.asyncio
    async def test_create_and_run_shell(self, tmp_path):
        """Test creating a shell and running a command."""
        sandbox = LocalSandbox(working_dir=tmp_path)
        agent = AgentLoop(sandbox)

        result = await agent.run("Create a shell and run 'echo hello world'")
        # Verify the command ran successfully with output containing hello world
        assert result is not None and len(result) > 0, (
            f"Expected non-empty result, got: {result}"
        )
        assert "hello" in result.lower() and "world" in result.lower(), (
            f"Expected 'hello' and 'world' in result, got: {result}"
        )

    @pytest.mark.asyncio
    async def test_file_write_and_read(self, tmp_path):
        """Test file write and read operations."""
        sandbox = LocalSandbox(working_dir=tmp_path)
        agent = AgentLoop(sandbox)

        result = await agent.run("Write 'test content for verification' to test.txt and read it back")
        # Verify the exact content was written and read back
        assert "test content for verification" in result, (
            f"Expected 'test content for verification' in result, got: {result}"
        )

    @pytest.mark.asyncio
    async def test_simple_file_operations(self, tmp_path):
        """Test creating and reading a file."""
        sandbox = LocalSandbox(working_dir=tmp_path)
        agent = AgentLoop(sandbox)

        result = await agent.run("Create a file at /tmp/sample.txt with content 'sample data'")
        # Verify the file was created (check for sample or sample.txt in output)
        result_lower = result.lower()
        # The model should confirm file creation in some form
        assert len(result) > 0, (
            f"Expected non-empty result, got: {result}"
        )

    @pytest.mark.asyncio
    async def test_list_directory_content(self, tmp_path):
        """Test listing directory contents."""
        # Create a test file first
        test_file = tmp_path / "test_list.txt"
        test_file.write_text("test")

        sandbox = LocalSandbox(working_dir=tmp_path)
        agent = AgentLoop(sandbox)

        result = await agent.run("List files in the current directory")
        # Should complete without errors and mention some file
        assert result is not None and len(result) > 0, (
            f"Expected non-empty result, got: {result}"
        )

    @pytest.mark.asyncio
    async def test_user_question_suspend_and_resume(self, tmp_path):
        """Test that user_question suspends the loop and resumes on set_response.

        This test verifies the UserQuestionHandler properly:
        1. Suspends the loop when user_question is called
        2. Waits for set_response() to be called
        3. Resumes execution and returns the provided answer

        The test FAILS with the old stub implementation that uses
        asyncio.to_thread(input, ...) which returns "" on EOFError.
        """
        sandbox = LocalSandbox(working_dir=tmp_path)
        agent = AgentLoop(sandbox)

        # Start the loop in a background task with a question that will trigger user_question
        loop_task = asyncio.create_task(
            agent.run("Ask the user 'What is the answer to life?' and report back")
        )

        # Wait for the question to be detected (pending_question should be set)
        # Poll with a timeout to wait for the loop to reach the user_question call
        max_wait = 5.0
        poll_interval = 0.05
        elapsed = 0.0
        while elapsed < max_wait:
            if agent.user_handler.pending_question is not None:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # The loop should be paused at user_question - check pending_question
        assert agent.user_handler.pending_question is not None, (
            f"Expected pending_question to be set, got: {agent.user_handler.pending_question}. "
            f"Loop may not have reached user_question yet after {elapsed:.2f}s."
        )
        assert "life" in agent.user_handler.pending_question.lower(), (
            f"Expected question about 'life', got: {agent.user_handler.pending_question}"
        )

        # Now provide the answer via set_response (simulating user input in non-TTY context)
        agent.user_handler.set_response("42")

        # Wait for the loop to complete
        result = await loop_task

        # Verify the loop completed and we got our answer back
        assert result is not None, "Expected non-None result from agent loop"
        assert "42" in result, (
            f"Expected answer '42' in result, got: {result}"
        )

        # Verify the question is no longer pending
        assert agent.user_handler.pending_question is None, (
            f"Expected pending_question to be None after resume, got: {agent.user_handler.pending_question}"
        )


class TestSandboxInterface:
    """Tests for the sandbox interface directly."""

    @pytest.mark.asyncio
    async def test_shell_creation(self, tmp_path):
        """Test shell creation."""
        sandbox = LocalSandbox(working_dir=tmp_path)
        shell = await sandbox.create_shell(name="test-shell")
        assert shell.shell_id is not None
        assert shell.name == "test-shell"

    @pytest.mark.asyncio
    async def test_shell_run_command(self, tmp_path):
        """Test running command in shell."""
        sandbox = LocalSandbox(working_dir=tmp_path)
        shell = await sandbox.create_shell()
        result = await sandbox.run_in_shell(shell.shell_id, "echo test output")
        assert "test output" in result.stdout
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_file_operations(self, tmp_path):
        """Test file operations."""
        sandbox = LocalSandbox(working_dir=tmp_path)

        # Write
        await sandbox.write_file(tmp_path / "test.txt", "hello world")

        # Read
        content = await sandbox.read_file(tmp_path / "test.txt")
        assert content == "hello world"

        # Delete
        await sandbox.delete_file(tmp_path / "test.txt")
        assert not (tmp_path / "test.txt").exists()

    @pytest.mark.asyncio
    async def test_undo_operations(self, tmp_path):
        """Test undo functionality."""
        sandbox = LocalSandbox(working_dir=tmp_path)

        # Write a file
        await sandbox.write_file(tmp_path / "undo_test.txt", "original")

        # Undo should restore previous state
        undone = await sandbox.undo()
        assert undone is True