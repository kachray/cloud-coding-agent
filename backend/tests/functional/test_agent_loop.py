"""Functional tests for the agent loop."""
import asyncio
import sys
from pathlib import Path

import pytest

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
        assert "shell_id" in result.lower() or "shell" in result.lower()
        assert "hello" in result.lower()

    @pytest.mark.asyncio
    async def test_file_write_and_read(self, tmp_path):
        """Test file write and read operations."""
        sandbox = LocalSandbox(working_dir=tmp_path)
        agent = AgentLoop(sandbox)

        result = await agent.run("Write 'test content for verification' to test.txt and read it back")
        assert "test content for verification" in result

    @pytest.mark.asyncio
    async def test_simple_file_operations(self, tmp_path):
        """Test creating and reading a file."""
        sandbox = LocalSandbox(working_dir=tmp_path)
        agent = AgentLoop(sandbox)

        result = await agent.run("Create a file at /tmp/sample.txt with content 'sample data'")
        assert "created" in result.lower() or "temp" in result.lower()

    @pytest.mark.asyncio
    async def test_list_directory_content(self, tmp_path):
        """Test listing directory contents."""
        # Create a test file first
        test_file = tmp_path / "test_list.txt"
        test_file.write_text("test")

        sandbox = LocalSandbox(working_dir=tmp_path)
        agent = AgentLoop(sandbox)

        result = await agent.run("List files in the current directory")
        # Should complete without errors
        assert result is not None


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