"""Sandbox-level tests for Milestone 2: working-dir path containment and
command visibility logging.

Token-free (no LLM calls) and mock-free: real filesystem, real subprocess
shells. These test the concrete ``LocalSandbox`` directly, mirroring how
``test_docker_sandbox.py`` tests ``DockerSandbox`` directly.
"""
import datetime
import os
from pathlib import Path

import pytest

from sandbox.local import LocalSandbox


@pytest.fixture
async def make_sandbox(tmp_path):
    """Factory for LocalSandbox instances; closes their shells on teardown."""
    sandboxes = []

    async def _make(command_log=None):
        sb = LocalSandbox(working_dir=tmp_path, command_log=command_log)
        sandboxes.append(sb)
        return sb

    yield _make

    for sb in sandboxes:
        for sid in list(getattr(sb, "_shells", {}).keys()):
            try:
                await sb.close_shell(sid)
            except Exception:
                pass


def _try_symlink(target: Path, link: Path) -> None:
    """Create a file symlink or skip the test where the platform forbids it."""
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")


# --- working-dir containment --------------------------------------------


async def test_relative_dotdot_escape_rejected(tmp_path):
    sb = LocalSandbox(working_dir=tmp_path)
    with pytest.raises(PermissionError):
        await sb.write_file(Path("../evil.txt"), "nope")
    assert not (tmp_path.parent / "evil.txt").exists()


async def test_absolute_outside_rejected(tmp_path):
    sb = LocalSandbox(working_dir=tmp_path)
    outside = tmp_path.parent / "evil.txt"
    with pytest.raises(PermissionError):
        await sb.write_file(outside, "nope")
    with pytest.raises(PermissionError):
        await sb.read_file(outside)
    assert not outside.exists()


async def test_symlink_escape_rejected(tmp_path):
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret")
    _try_symlink(outside, tmp_path / "link.txt")
    sb = LocalSandbox(working_dir=tmp_path)
    with pytest.raises(PermissionError):
        await sb.read_file(Path("link.txt"))


async def test_symlink_outside_to_inside_allowed(tmp_path):
    # A symlink (even one located outside working_dir) that resolves to a
    # target *inside* working_dir must be allowed: the containment check runs
    # on the resolved path, not the literal argument.
    target = tmp_path / "real.txt"
    target.write_text("payload")
    link = tmp_path.parent / "inlink.txt"
    _try_symlink(target, link)
    sb = LocalSandbox(working_dir=tmp_path)
    assert await sb.read_file(link) == "payload"


async def test_dotdot_staying_inside_allowed(tmp_path):
    sb = LocalSandbox(working_dir=tmp_path)
    await sb.write_file(Path("subdir/../a.txt"), "ok")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "ok"


async def test_inside_operations_still_work(tmp_path):
    sb = LocalSandbox(working_dir=tmp_path)
    await sb.write_file(Path("b.txt"), "data")
    assert await sb.read_file(Path("b.txt")) == "data"
    await sb.create_file(Path("c.txt"))
    with pytest.raises(FileExistsError):
        await sb.create_file(Path("c.txt"))
    await sb.delete_file(Path("b.txt"))
    with pytest.raises(FileNotFoundError):
        await sb.read_file(Path("b.txt"))


async def test_undo_restores_after_containment(tmp_path):
    sb = LocalSandbox(working_dir=tmp_path)
    await sb.write_file(Path("a.txt"), "v1")
    await sb.write_file(Path("a.txt"), "v2")
    assert await sb.undo() is True          # write v2 -> v1
    assert await sb.read_file(Path("a.txt")) == "v1"
    await sb.delete_file(Path("a.txt"))
    assert await sb.undo() is True          # delete -> restored
    assert await sb.read_file(Path("a.txt")) == "v1"
    assert await sb.undo() is True          # original create -> file removed
    with pytest.raises(FileNotFoundError):
        await sb.read_file(Path("a.txt"))
    assert await sb.undo() is False         # history exhausted


async def test_rejected_escape_not_recorded_in_history(tmp_path):
    sb = LocalSandbox(working_dir=tmp_path)
    with pytest.raises(PermissionError):
        await sb.write_file(Path("../evil.txt"), "nope")
    assert await sb.undo() is False  # the failed call recorded nothing


# --- command visibility --------------------------------------------------


async def test_run_in_shell_logs_commands(make_sandbox, tmp_path):
    log = tmp_path / "cmd.log"
    sb = await make_sandbox(command_log=log)
    shell = await sb.create_shell()
    r1 = await sb.run_in_shell(shell.shell_id, "echo hello")
    r2 = await sb.run_in_shell(shell.shell_id, "echo world")
    assert r1.exit_code == 0, r1
    assert r2.exit_code == 0, r2

    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # one line per command
    for line in lines:
        ts, sid, _cmd = line.split("\t", 2)
        datetime.datetime.fromisoformat(ts)  # timestamp is parseable ISO
        assert sid == shell.shell_id
    assert "echo hello" in lines[0]
    assert "echo world" in lines[1]


async def test_multiline_command_is_one_log_line(make_sandbox, tmp_path):
    log = tmp_path / "cmd.log"
    sb = await make_sandbox(command_log=log)
    shell = await sb.create_shell()
    multi = "echo one\necho two"
    r = await sb.run_in_shell(shell.shell_id, multi)
    assert r.exit_code == 0, r

    text = log.read_text(encoding="utf-8").strip()
    assert len(text.splitlines()) == 1  # newline escaped, still one line
    assert "echo one\\necho two" in text


async def test_default_log_path_outside_working_dir(make_sandbox, tmp_path):
    sb = await make_sandbox()  # no override: default path derived from working_dir
    shell = await sb.create_shell()
    await sb.run_in_shell(shell.shell_id, "echo hi")

    default = tmp_path.parent / f"{tmp_path.name}.commands.log"
    assert default.exists()
    assert "echo hi" in default.read_text(encoding="utf-8")


async def test_timed_out_command_is_logged(make_sandbox, tmp_path):
    log = tmp_path / "cmd.log"
    sb = await make_sandbox(command_log=log)
    shell = await sb.create_shell()
    result = await sb.run_in_shell(shell.shell_id, "sleep 10", timeout=1)
    assert result.timeout is True
    # Logged *before* execution, so a command that never completes is recorded.
    assert "sleep 10" in log.read_text(encoding="utf-8")
