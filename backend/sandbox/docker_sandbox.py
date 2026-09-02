"""Docker-backed implementation of the sandbox interface.

Milestone 2: one fresh container per DockerSandbox instance (per agent
session, never shared or reused). The agent works inside ``/workspace``,
which is a bind mount of the host working_dir — so host-side assertions
(tests) see the same files the model writes.

Shells are persistent non-tty ``docker exec`` sessions (one exec per
shell_id) driven with the same ``__CCA_CMD_DONE__`` marker protocol as the
Milestone 1 local sandbox. A one-shot exec per command would spawn a fresh
process each time and lose ``cd``/env state, violating the stateful-shell
requirement. tty=False means no input echo and no 80-col output wrap; the
stream is docker's 8-byte multiplexed frames (stream type + payload size),
which the reader parses and folds into one output — same trade-off as the
local session (stderr folded into stdout).

``agent/`` never imports this module's Docker dependency; it talks to
``SandboxInterface`` only.
"""
import asyncio
import io
import tarfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import docker
from docker.errors import ImageNotFound, NotFound

from .local import (
    _CMD_MARKER,
    _DEFAULT_CMD_TIMEOUT,
    SandboxInterface,
    Shell,
    ShellResult,
)

IMAGE_TAG = "cloud-agent-sandbox:dev"
_CONTAINER_LABEL = {"app": "cloud-agent-sandbox"}
# Container-internal path the host working_dir is bind-mounted at.
_WORKSPACE = "/workspace"


class DockerSandbox(SandboxInterface):
    """Sandbox backed by one Docker container per instance.

    All docker-py calls are blocking HTTP calls, so every one goes through
    ``asyncio.to_thread``. The container is created lazily on the first
    sandbox op (``AgentLoop.run`` sets ``working_dir`` before any tool runs,
    so lazy creation always sees the final dir; a later change tears down and
    recreates on the next op). Call ``cleanup()`` when the session ends so no
    container leaks.
    """

    def __init__(
        self,
        working_dir: Optional[Path] = None,
        image: str = IMAGE_TAG,
        dockerfile_dir: Optional[Path] = None,
        mem_limit: str = "2g",
        nano_cpus: int = 2_000_000_000,
        pids_limit: int = 512,
        client: Optional[Any] = None,
    ) -> None:
        self.working_dir = (working_dir or Path.cwd()).resolve()
        self.image = image
        self.dockerfile_dir = dockerfile_dir
        self.mem_limit = mem_limit
        self.nano_cpus = nano_cpus
        self.pids_limit = pids_limit
        self._client = client
        self._container = None
        self._mounted_dir: Optional[Path] = None
        self._shells: Dict[str, "_DockerShellSession"] = {}
        # Same undo-history shape as LocalSandbox: (op, path, old_content).
        self._file_history: List[Tuple[str, str, Optional[str]]] = []

    # --- container lifecycle --------------------------------------------

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    async def _ensure_image(self) -> None:
        try:
            await asyncio.to_thread(self.client.images.get, self.image)
        except ImageNotFound:
            # Build on first use so tests are self-sufficient.
            build_dir = str(self.dockerfile_dir or Path(__file__).parent.parent / "docker")
            await asyncio.to_thread(
                self.client.images.build, path=build_dir, tag=self.image
            )

    async def _ensure_container(self) -> None:
        current = self.working_dir
        if self._container is not None:
            if self._mounted_dir == current:
                return
            # working_dir changed after the container was created — a fresh
            # container is a fresh session, so tear down and recreate.
            await self.cleanup()
        await self._ensure_image()
        name = f"cca-sandbox-{uuid.uuid4().hex[:8]}"
        # auto_remove: a crashed run leaves at most a stopped container that
        # Docker removes itself; the label makes any stray findable via
        # `docker ps -a --filter label=app=cloud-agent-sandbox`.
        # ponytail: no sweeper for strays — add one if strays are ever observed.
        container = await asyncio.to_thread(
            self.client.containers.run,
            self.image,
            command=["sleep", "infinity"],
            name=name,
            labels=_CONTAINER_LABEL,
            # ponytail: no disk quota (storage-opt is driver-dependent) — add
            # if workspace-filling becomes a real problem.
            mem_limit=self.mem_limit,
            memswap_limit=self.mem_limit,  # == mem_limit: no swap
            nano_cpus=self.nano_cpus,
            pids_limit=self.pids_limit,
            volumes={str(current): {"bind": _WORKSPACE, "mode": "rw"}},
            working_dir=_WORKSPACE,
            detach=True,
            auto_remove=True,
        )
        self._container = container
        self._mounted_dir = current

    async def cleanup(self) -> None:
        """Tear down the container and all shell sessions. Idempotent."""
        for sid in list(self._shells.keys()):
            session = self._shells.pop(sid, None)
            if session is not None:
                await session.close()
        if self._container is not None:
            container, self._container, self._mounted_dir = self._container, None, None
            try:
                # auto_remove containers vanish on stop; force covers a
                # still-running container too. NotFound if auto-remove won
                # the race — that's success.
                await asyncio.to_thread(container.remove, v=True, force=True)
            except NotFound:
                pass

    # --- shells ----------------------------------------------------------

    async def create_shell(self, name: Optional[str] = None, cwd: Optional[Path] = None) -> Shell:
        await self._ensure_container()
        shell_id = str(uuid.uuid4())
        container_dir = self._full(cwd) if cwd is not None else _WORKSPACE
        session = await _DockerShellSession.start(
            self.client, self._container, container_dir
        )
        self._shells[shell_id] = session
        return Shell(shell_id=shell_id, name=name, cwd=Path(container_dir))

    async def run_in_shell(self, shell_id: str, cmd: str, timeout: Optional[int] = None) -> ShellResult:
        session = self._shells.get(shell_id)
        if session is None:
            raise ValueError(f"Unknown shell_id: {shell_id}")
        try:
            exit_code, stdout, timed_out = await session.run(cmd, timeout=timeout)
        except Exception as exc:  # exec'd bash died -> surface as an error result
            self._shells.pop(shell_id, None)
            return ShellResult(shell_id, exit_code=-1, stdout="", stderr=f"shell error: {exc}")
        if timed_out:
            # Matches local semantics: a timed-out shell is dead. Closing the
            # socket doesn't reliably kill the bash inside, but the whole
            # container is removed at cleanup, so nothing leaks past the
            # session. (ponytail ceiling: no per-command process kill.)
            self._shells.pop(shell_id, None)
            await session.close()
        return ShellResult(
            shell_id=shell_id, exit_code=exit_code, stdout=stdout, stderr="", timeout=timed_out
        )

    async def close_shell(self, shell_id: str) -> None:
        session = self._shells.pop(shell_id, None)
        if session is not None:
            await session.close()

    # --- file operations ---------------------------------------------------
    # All file content moves through the container's archive API (tar in/out),
    # never host-side file access — this is what keeps the remote-sandbox swap
    # possible later.

    def _full(self, path: Optional[Path]) -> str:
        """Map a host/relative path to a container path.

        The host working_dir *is* the /workspace bind mount, so the three cases
        coincide: relative paths and host-absolute paths under working_dir both
        become /workspace/<rel>; any other absolute path (a real container path
        like /tmp/...) is passed through unchanged.
        """
        if path is None:
            return _WORKSPACE
        p = Path(path)
        if not p.is_absolute():
            return f"{_WORKSPACE}/{p.as_posix().lstrip('/')}"
        try:
            rel = p.relative_to(self.working_dir)
        except ValueError:
            return str(p)
        return f"{_WORKSPACE}/{rel.as_posix()}"

    async def read_file(self, path: Path) -> str:
        await self._ensure_container()
        full = self._full(path)
        try:
            bits, _ = await asyncio.to_thread(self._container.get_archive, full)
        except NotFound:
            raise FileNotFoundError(f"File not found: {full}")
        with tarfile.open(fileobj=io.BytesIO(b"".join(bits))) as tar:
            member = tar.getmember(_tar_member_name(tar, full))
            return tar.extractfile(member).read().decode("utf-8")

    async def write_file(self, path: Path, content: str) -> None:
        await self._ensure_container()
        full = self._full(path)
        if await self._exists(full):
            old = await self.read_file(path)
            self._file_history.append(("write", full, old))
        else:
            self._file_history.append(("create", full, None))
        await self._put(full, content)

    async def create_file(self, path: Path, content: str = "") -> None:
        await self._ensure_container()
        full = self._full(path)
        if await self._exists(full):
            raise FileExistsError(f"File already exists: {full}")
        await self.write_file(path, content)

    async def delete_file(self, path: Path) -> None:
        await self._ensure_container()
        full = self._full(path)
        if not await self._exists(full):
            raise FileNotFoundError(f"File not found: {full}")
        old = await self.read_file(path)
        self._file_history.append(("delete", full, old))
        # One-shot exec with an argv list (no shell) — no escaping risk.
        await asyncio.to_thread(self._container.exec_run, ["rm", "--", full])

    async def undo(self) -> bool:
        if not self._file_history:
            return False
        await self._ensure_container()
        op_type, full, content = self._file_history.pop()
        if op_type == "create":
            if await self._exists(full):
                await asyncio.to_thread(self._container.exec_run, ["rm", "--", full])
        elif op_type == "delete":
            await self._put(full, content or "")
        elif op_type == "write":
            if content is None:
                if await self._exists(full):
                    await asyncio.to_thread(self._container.exec_run, ["rm", "--", full])
            else:
                await self._put(full, content)
        return True

    async def _exists(self, full: str) -> bool:
        result = await asyncio.to_thread(
            self._container.exec_run, ["test", "-e", full]
        )
        return result.exit_code == 0

    async def _put(self, full: str, content: str) -> None:
        """Write *content* to container path *full* via an in-memory tar."""
        full_path = Path(full)
        parent = str(full_path.parent)
        await asyncio.to_thread(
            self._container.exec_run, ["mkdir", "-p", "--", parent]
        )
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=full_path.name)
            data = content.encode("utf-8")
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
        await asyncio.to_thread(self._container.put_archive, parent, buf.getvalue())


def _tar_member_name(tar: tarfile.TarFile, full: str) -> str:
    """get_archive tars the path itself, so the member is the basename."""
    name = Path(full).name
    if name in tar.getnames():
        return name
    return tar.getnames()[0]  # e.g. "workspace" for /workspace itself


class _DockerShellSession:
    """A long-lived non-tty ``docker exec`` bash session.

    Same marker protocol as the local sandbox: write the command, then
    ``echo "__CCA_CMD_DONE__$?"``, read output lines until the marker.
    Non-interactive bash on a pipe prints no prompt and echoes nothing, so the
    stream contains only command output.

    Output arrives as docker 8-byte multiplexed frames (1-byte stream type,
    3 padding, 4-byte big-endian payload size); ``_readline`` consumes headers +
    payload and splits on newlines, folding stderr into stdout. The exec socket
    is blocking, so reads run in a worker thread; the per-command deadline is
    ``sock.settimeout`` where the transport honours it plus a wait backstop on
    transports where it's a no-op (Windows npipe).
    """

    def __init__(self, sock: Any):
        self._sock = sock
        self._lock = asyncio.Lock()  # one command at a time per shell
        self._raw = b""   # undecoded frame bytes
        self._text = ""   # decoded, newline-unterminated text

    @classmethod
    async def start(cls, client: Any, container: Any, cwd: str) -> "_DockerShellSession":
        def _open() -> Any:
            exec_id = client.api.exec_create(
                container.id,
                ["bash"],
                stdin=True,
                stdout=True,
                stderr=True,
                tty=False,
                workdir=cwd,
            )["Id"]
            return client.api.exec_start(exec_id, socket=True)

        sock = await asyncio.to_thread(_open)
        return cls(sock)

    async def _send(self, data: str) -> None:
        await asyncio.to_thread(self._sock.sendall, data.encode())

    # --- blocking frame/line reading (runs in a worker thread) ------------

    def _recv_exact(self, n: int) -> bytes:
        while len(self._raw) < n:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("exec socket closed")
            self._raw += chunk
        out, self._raw = self._raw[:n], self._raw[n:]
        return out

    def _readline(self, timeout: float) -> str:
        """Blocking readline over docker's 8-byte multiplexed frames.

        Runs in a worker thread. Raises ``TimeoutError`` if the socket-level
        deadline fires (settimeout works) or ``ConnectionError`` if the exec'd
        bash exits (recv returns empty).
        """
        try:
            self._sock.settimeout(timeout)
        except (AttributeError, OSError):
            pass  # no-op settimeout transports: the wait backstop in _next_line
        while "\n" not in self._text:
            header = self._recv_exact(8)
            size = int.from_bytes(header[4:8], "big")
            # header[0] is the stream type (1=stdout, 2=stderr); we fold both,
            # same trade-off as the local session.
            self._text += self._recv_exact(size).decode(errors="replace")
        line, self._text = self._text.split("\n", 1)
        return line

    async def _next_line(self, deadline: float) -> Optional[str]:
        """Read one output line; ``None`` means the command hit the deadline."""
        task = asyncio.create_task(asyncio.to_thread(self._readline, deadline))
        done, _ = await asyncio.wait({task}, timeout=deadline + 2.0)
        if not done:
            # Reader blocked past the deadline (no-op settimeout transport).
            # Release it by closing the socket, reap the thread, report timeout.
            await self._close_sock()
            await asyncio.wait({task})
            try:
                task.result()
            except BaseException:
                pass
            return None
        try:
            return task.result()
        except TimeoutError:
            # In-thread socket timeout (settimeout worked); drop the session.
            await self._close_sock()
            return None
        except (ConnectionError, OSError) as exc:
            raise ConnectionError(f"exec socket closed: {exc}") from None

    async def run(self, cmd: str, timeout: Optional[int] = None) -> Tuple[int, str, bool]:
        async with self._lock:
            deadline = timeout if timeout is not None else _DEFAULT_CMD_TIMEOUT
            await self._send(f'{cmd}\necho "{_CMD_MARKER}$?"\n')

            stdout_lines: List[str] = []
            exit_code = -1
            while True:
                line = await self._next_line(deadline)  # None -> timed out
                if line is None:
                    return exit_code, "".join(f"{l}\n" for l in stdout_lines), True
                idx = line.find(_CMD_MARKER)
                if idx != -1:
                    if idx > 0:
                        stdout_lines.append(line[:idx])
                    after = line[idx + len(_CMD_MARKER):].strip().strip('"').strip()
                    try:
                        exit_code = int(after)
                    except ValueError:
                        exit_code = 0
                    return exit_code, "".join(f"{l}\n" for l in stdout_lines), False
                stdout_lines.append(line)

    async def close(self) -> None:
        def _close() -> None:
            try:
                self._sock.sendall(b"exit 0\n")
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass

        await asyncio.to_thread(_close)

    async def _close_sock(self) -> None:
        def _close() -> None:
            try:
                self._sock.close()
            except OSError:
                pass

        await asyncio.to_thread(_close)
