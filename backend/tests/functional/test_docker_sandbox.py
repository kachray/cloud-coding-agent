"""Docker sandbox tests — real Docker, real behavior, no mocks.

The sandbox-level tests here are cheap and token-free; they auto-run whenever a
Docker daemon is reachable. The single real-agent test at the bottom spends LLM
tokens, so it is gated behind CCA_SANDBOX=docker (explicit opt-in, same switch
that routes the existing suite through Docker).
"""
import os
from pathlib import Path

import pytest

try:
    import docker as docker_py
except Exception:  # pragma: no cover - docker import env varies
    docker_py = None

from sandbox.docker_sandbox import DockerSandbox  # noqa: E402


def _docker_up() -> bool:
    if docker_py is None:
        return False
    try:
        docker_py.from_env().ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_up(), reason="Docker daemon not reachable"
)


@pytest.fixture
async def docker_sandbox(tmp_path):
    """DockerSandbox directly (not queued) so tests can reach ``_container``.

    Function-scoped: each test gets a fresh container bound to its own tmp_path,
    matching how the local functional tests get a fresh LocalSandbox per test.
    """
    sb = DockerSandbox(working_dir=tmp_path)
    yield sb
    await sb.cleanup()


# --- stateful shells -------------------------------------------------------


async def test_cwd_persists_across_commands(docker_sandbox):
    shell = await docker_sandbox.create_shell()
    r1 = await docker_sandbox.run_in_shell(shell.shell_id, "cd /tmp")
    assert r1.exit_code == 0, r1
    r2 = await docker_sandbox.run_in_shell(shell.shell_id, "pwd")
    assert r2.exit_code == 0, r2
    assert r2.stdout.strip() == "/tmp"


async def test_two_shells_are_independent(docker_sandbox):
    a = await docker_sandbox.create_shell()
    b = await docker_sandbox.create_shell()
    ra = await docker_sandbox.run_in_shell(a.shell_id, "cd /tmp")
    assert ra.exit_code == 0, ra
    rb = await docker_sandbox.run_in_shell(b.shell_id, "pwd")
    assert rb.exit_code == 0, rb
    # b was never cd'd — still at the default /workspace.
    assert rb.stdout.strip() == "/workspace"


# --- file operations through the container archive API ---------------------


async def test_file_roundtrip_and_bind_mount(docker_sandbox, tmp_path):
    await docker_sandbox.write_file("hello.txt", "hello docker")
    assert await docker_sandbox.read_file("hello.txt") == "hello docker"
    # working_dir is bind-mounted at /workspace, so the write is visible on host.
    host_file = tmp_path / "hello.txt"
    assert host_file.exists()
    assert host_file.read_text(encoding="utf-8") == "hello docker"


async def test_host_absolute_path_maps_into_container(docker_sandbox, tmp_path):
    # A host path under working_dir (= the /workspace bind mount) must resolve
    # to the same file inside the container — interface parity with local.
    p = tmp_path / "dup.txt"
    await docker_sandbox.create_file(p, "sample data")
    assert p.read_text(encoding="utf-8") == "sample data"
    with pytest.raises(FileExistsError):
        await docker_sandbox.create_file(p, "other data")
    with pytest.raises(FileNotFoundError):
        await docker_sandbox.delete_file(tmp_path / "missing.txt")


async def test_undo_restores_previous_content(docker_sandbox):
    await docker_sandbox.write_file("app.py", "v1")
    await docker_sandbox.write_file("app.py", "v2")
    assert await docker_sandbox.read_file("app.py") == "v2"
    assert await docker_sandbox.undo() is True  # undo the write -> back to v1
    assert await docker_sandbox.read_file("app.py") == "v1"
    assert await docker_sandbox.undo() is True  # undo the create -> file removed
    with pytest.raises(FileNotFoundError):
        await docker_sandbox.read_file("app.py")
    assert await docker_sandbox.undo() is False  # history exhausted


async def test_dotdot_and_absolute_escape_rejected(docker_sandbox, tmp_path):
    with pytest.raises(PermissionError):
        await docker_sandbox.write_file(Path("../evil.txt"), "nope")
    with pytest.raises(PermissionError):
        await docker_sandbox.write_file(tmp_path.parent / "evil.txt", "nope")
    assert not (tmp_path.parent / "evil.txt").exists()


async def test_run_in_shell_logs_commands(docker_sandbox, tmp_path):
    shell = await docker_sandbox.create_shell()
    r = await docker_sandbox.run_in_shell(shell.shell_id, "echo hello")
    assert r.exit_code == 0, r
    default = tmp_path.parent / f"{tmp_path.name}.commands.log"
    assert default.exists()
    assert "echo hello" in default.read_text(encoding="utf-8")


# --- timeouts / lifecycle / resource caps ---------------------------------


async def test_timeout_returns_timeout_result_and_drops_shell(docker_sandbox):
    shell = await docker_sandbox.create_shell()
    result = await docker_sandbox.run_in_shell(shell.shell_id, "sleep 60", timeout=2)
    assert result.timeout is True
    # a timed-out shell is dead — subsequent commands on that shell_id are gone
    with pytest.raises(ValueError):
        await docker_sandbox.run_in_shell(shell.shell_id, "echo hi")


async def test_cleanup_removes_container(docker_sandbox):
    cid = docker_sandbox._container.id
    await docker_sandbox.cleanup()
    assert docker_sandbox._container is None
    remaining = {c.id for c in docker_sandbox.client.containers.list(all=True)}
    assert cid not in remaining, "container leaked after cleanup"


async def test_resource_limits_applied(docker_sandbox):
    host_config = docker_sandbox._container.attrs["HostConfig"]
    assert host_config["Memory"] == 2 * 1024 ** 3  # "2g"
    assert host_config["NanoCpus"] == 2_000_000_000  # 2 CPUs
    assert host_config["PidsLimit"] == 512


# --- real agent run through queue + Docker (opt-in, spends tokens) ---------


@pytest.mark.skipif(
    os.environ.get("CCA_SANDBOX") != "docker",
    reason="LLM test: opt in via CCA_SANDBOX=docker",
)
async def test_agent_run_end_to_end_against_docker(client, tmp_path):
    from agent.loop import AgentLoop
    from sandbox.queued import QueuedSandbox

    docker_sb = DockerSandbox(working_dir=tmp_path)
    queued = QueuedSandbox(docker_sb)
    agent = AgentLoop(
        sandbox=queued,
        model="openai/gpt-oss-120b",
        client=client,
        system_instruction=(
            "You are a coding agent operating in the given working directory. "
            "Use the provided tools to complete the user's task exactly. Do not ask "
            "the user questions unless the task is genuinely ambiguous; make a "
            "reasonable choice and proceed. When finished, briefly state the outcome."
        ),
    )
    try:
        await agent.run(
            "Create a shell, run `echo hello world`, then write the output to "
            "a file called output.txt in the current directory.",
            working_dir=tmp_path,
        )
    finally:
        await queued.cleanup()
    output = tmp_path / "output.txt"
    assert output.exists(), "output.txt not created by the agent"
    content = output.read_text(encoding="utf-8")
    assert "hello" in content.lower() and "world" in content.lower(), content
