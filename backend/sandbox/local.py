"""Local subprocess implementation of the sandbox interface.

Milestone 1: shells are backed by a long-lived local subprocess (bash on
Unix, or Git-bash when available on Windows) whose environment and working
directory therefore persist across `run_in_shell` calls within the same
shell_id. Each call to `create_shell` produces a fresh, distinct process with
its own shell_id, so two shells are genuinely independent — this is the
property the functional regression test for two-shells checks.

Communication with the shell process uses a sentinel marker so the wrapper
can detect when a command has finished and capture its exit status and
output reliably (a naive `communicate()` would close the pipe after one
read and is not usable for a stateful session).
"""
import asyncio
import logging
import os
import platform
import shutil
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiofiles

logger = logging.getLogger(__name__)

# Opaque marker printed after every command so the wrapper can detect the end
# of a command's output and capture its exit status reliably. A naive
# `communicate()` closes the pipe after one read and can't drive a stateful
# session, so we delimit each command with this token instead. The shell prints
# the marker verbatim followed by the exit code, e.g. `__CCA_CMD_DONE__0`; the
# bytes after the prefix on that line are the integer exit code.
_CMD_MARKER = "__CCA_CMD_DONE__"
_CMD_MARKER_BYTES = _CMD_MARKER.encode()
_DEFAULT_CMD_TIMEOUT = 30.0


@dataclass
class Shell:
    """Represents a shell environment (one process per shell_id)."""
    shell_id: str
    name: Optional[str] = None
    cwd: Optional[Path] = None


@dataclass
class ShellResult:
    """Result from running a command in a shell."""
    shell_id: str
    exit_code: int
    stdout: str
    stderr: str
    timeout: bool = False


def _resolve_inside(path: Path, root: Path) -> Path:
    """Resolve *path* relative to *root* (normalizes `..`, resolves symlinks,
    non-strict so a not-yet-existing create target works) and require the
    result stays inside *root*; raise PermissionError otherwise."""
    full = (path if path.is_absolute() else root / path).resolve()
    try:
        full.relative_to(root)
    except ValueError:
        raise PermissionError(
            f"Path {str(path)!r} resolves to {full}, outside working_dir {root}"
        ) from None
    return full


async def _append_command_log(log_path: Path, shell_id: str, cmd: str) -> None:
    """Append one line per executed command to the session log file.

    Best-effort: a log write failure must never break the command execution.
    """
    line = (
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t"
        f"{shell_id}\t{cmd.replace(chr(10), '\\n')}\n"
    )
    async with aiofiles.open(log_path, mode="a", encoding="utf-8") as f:
        await f.write(line)
        await f.flush()


class SandboxInterface(ABC):
    """Abstract interface for sandbox operations.

    `agent/` talks only to this interface. The Milestone 2 Docker-backed
    sandbox will implement the same methods, so the agent loop never imports
    Docker directly.
    """

    @abstractmethod
    async def create_shell(self, name: Optional[str] = None, cwd: Optional[Path] = None) -> Shell:
        """Create a new, independent shell environment; returns its shell_id."""
        pass

    @abstractmethod
    async def run_in_shell(self, shell_id: str, cmd: str, timeout: Optional[int] = None) -> ShellResult:
        """Run a command in an existing shell, preserving that shell's state."""
        pass

    @abstractmethod
    async def read_file(self, path: Path) -> str:
        pass

    @abstractmethod
    async def write_file(self, path: Path, content: str) -> None:
        pass

    @abstractmethod
    async def create_file(self, path: Path, content: str = "") -> None:
        pass

    @abstractmethod
    async def delete_file(self, path: Path) -> None:
        pass

    @abstractmethod
    async def undo(self) -> bool:
        pass

    @abstractmethod
    async def close_shell(self, shell_id: str) -> None:
        pass


def _resolve_shell_bin() -> str:
    """Pick a shell binary, preferring bash if available (works on Windows too)."""
    bash = shutil.which("bash")
    if bash:
        return bash
    if platform.system() == "Windows":
        # Fallback to cmd on Windows when no bash is on PATH
        return shutil.which("cmd.exe") or os.environ.get("COMSPEC", "cmd.exe")
    return "/bin/sh"


class LocalSandbox(SandboxInterface):
    """Local subprocess implementation for Milestone 1.

    Each shell is a long-lived process reading commands from stdin. We write a
    command, then a statement that prints `_STATUS_MARKER` formatted with the
    last command's exit code, then read stdout until that marker appears. This
    reliably delimits one command's output from the next within the same
    stateful shell session.
    """

    def __init__(
        self,
        working_dir: Optional[Path] = None,
        command_log: Optional[Path] = None,
    ):
        self.working_dir = (working_dir or Path.cwd()).resolve()
        self._command_log = Path(command_log).resolve() if command_log else None
        self._shells: Dict[str, "_ShellSession"] = {}
        self._file_history: List[Tuple[str, str, Optional[str]]] = []
        self._shell_bin = _resolve_shell_bin()
        self._is_cmd = self._shell_bin.endswith("cmd.exe") or self._shell_bin.endswith("cmd")

    @property
    def command_log_path(self) -> Path:
        """Per-session command log file.

        Lazily derived from the current working_dir (``AgentLoop.run`` can
        reassign ``working_dir`` after construction) and placed *outside* the
        workspace so the model cannot see or edit its own audit trail.
        """
        if self._command_log is not None:
            return self._command_log
        return self.working_dir.parent / f"{self.working_dir.name}.commands.log"

    # --- shells ---------------------------------------------------------

    async def create_shell(self, name: Optional[str] = None, cwd: Optional[Path] = None) -> Shell:
        """Create a new, independent shell session."""
        shell_id = str(uuid.uuid4())
        working_dir = (cwd or self.working_dir).resolve()
        session = await _ShellSession.start(self._shell_bin, working_dir, self._is_cmd)
        self._shells[shell_id] = session
        return Shell(shell_id=shell_id, name=name, cwd=working_dir)

    async def run_in_shell(self, shell_id: str, cmd: str, timeout: Optional[int] = None) -> ShellResult:
        if shell_id not in self._shells:
            raise ValueError(f"Unknown shell_id: {shell_id}")
        try:
            await _append_command_log(self.command_log_path, shell_id, cmd)
        except Exception as exc:  # logging is visibility, never block execution
            logger.warning("Failed to append command log for shell %s: %s", shell_id, exc)
        session = self._shells[shell_id]
        try:
            exit_code, stdout, stderr, timed_out = await session.run(cmd, timeout=timeout)
        except Exception as exc:  # shell died -> surface as an error result
            return ShellResult(shell_id, exit_code=-1, stdout="", stderr=f"shell error: {exc}")
        return ShellResult(
            shell_id=shell_id,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timeout=timed_out,
        )

    async def close_shell(self, shell_id: str) -> None:
        session = self._shells.pop(shell_id, None)
        if session is not None:
            await session.close()

    # --- file operations -------------------------------------------------

    def _full(self, path: Path) -> Path:
        return _resolve_inside(path, self.working_dir)

    async def read_file(self, path: Path) -> str:
        full = self._full(path)
        async with aiofiles.open(full, mode="r", encoding="utf-8") as f:
            return await f.read()

    async def write_file(self, path: Path, content: str) -> None:
        full = self._full(path)
        # Snapshot for undo
        if full.exists():
            old = await self.read_file(path)
            self._file_history.append(("write", str(full), old))
        else:
            self._file_history.append(("create", str(full), None))
        full.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(full, mode="w", encoding="utf-8") as f:
            await f.write(content)

    async def create_file(self, path: Path, content: str = "") -> None:
        full = self._full(path)
        if full.exists():
            raise FileExistsError(f"File already exists: {full}")
        await self.write_file(path, content)

    async def delete_file(self, path: Path) -> None:
        full = self._full(path)
        if not full.exists():
            raise FileNotFoundError(f"File not found: {full}")
        old = await self.read_file(path)
        self._file_history.append(("delete", str(full), old))
        full.unlink()

    async def undo(self) -> bool:
        if not self._file_history:
            return False
        op_type, path_str, content = self._file_history.pop()
        path = Path(path_str)
        if op_type == "create":
            if path.exists():
                path.unlink()
        elif op_type == "delete":
            async with aiofiles.open(path, mode="w", encoding="utf-8") as f:
                await f.write(content or "")
        elif op_type == "write":
            if content is None:
                if path.exists():
                    path.unlink()
            else:
                async with aiofiles.open(path, mode="w", encoding="utf-8") as f:
                    await f.write(content)
        return True


class _ShellSession:
    """A long-lived shell subprocess with marker-delimited command execution.

    stderr is redirected to stdout (`stderr=subprocess.STDOUT`) so the shell
    has a single output stream to read until the marker; this avoids needing
    to concurrently drain a second pipe that never EOFs on a persistent shell.
    The trade-off is command stderr is folded into `stdout` (acceptable for a
    coding agent — the model sees command output either way).
    """

    def __init__(self, process: asyncio.subprocess.Process, is_cmd: bool):
        self._proc = process
        self._is_cmd = is_cmd
        # Serialize access per shell so commands run one at a time.
        self._lock = asyncio.Lock()

    @classmethod
    async def start(cls, shell_bin: str, cwd: Path, is_cmd: bool) -> "_ShellSession":
        proc = await asyncio.create_subprocess_exec(
            shell_bin,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd),
        )
        return cls(proc, is_cmd=is_cmd)

    def _build_input(self, cmd: str) -> str:
        """Run the command, then print MARKER+exit_code on its own line.

        The marker string is concatenated *directly* (not wrapped in ${}), so
        only the `$?` / `%errorlevel%` trailer is interpreted by the shell —
        the literal marker text passes through untouched. (wrapping it as
        ``${_CMD_MARKER}`` would make bash treat it as variable expansion and
        silently expand it to the empty string, defeating the sentinel.)
        """
        if self._is_cmd:
            return f"{cmd}\r\necho {_CMD_MARKER}%errorlevel%\r\n"
        return f'{cmd}\necho "{_CMD_MARKER}$?"\n'

    async def run(self, cmd: str, timeout: Optional[int] = None) -> Tuple[int, str, str, bool]:
        async with self._lock:
            stdin = self._proc.stdin
            if stdin is None:
                raise RuntimeError("shell stdin closed")
            stdin.write(self._build_input(cmd).encode())
            await stdin.drain()

            stdout_lines: List[str] = []
            exit_code = -1
            timed_out = False
            deadline = (timeout or _DEFAULT_CMD_TIMEOUT)
            try:
                while True:
                    line_bytes = await asyncio.wait_for(
                        self._proc.stdout.readline(), timeout=deadline
                    )
                    if not line_bytes:
                        # Process closed stdout unexpectedly — shell is dead.
                        break
                    line = line_bytes.decode(errors="replace")
                    idx = line.find(_CMD_MARKER)
                    if idx != -1:
                        # Anything before the marker on this line is real output.
                        prefix = line[:idx]
                        if prefix:
                            stdout_lines.append(prefix)
                        # Bytes after the marker prefix are the integer exit code.
                        after = line[idx + len(_CMD_MARKER):].strip().strip('"').strip()
                        try:
                            exit_code = int(after)
                        except ValueError:
                            exit_code = 0
                        break
                    stdout_lines.append(line)
            except asyncio.TimeoutError:
                timed_out = True
                try:
                    self._proc.kill()
                except Exception:
                    pass
            return exit_code, "".join(stdout_lines), "", timed_out

    async def close(self) -> None:
        # Ask the shell to exit gracefully, then force-kill as a fallback so no
        # subprocess transport leaks (Windows proactor pipes warn if left open).
        try:
            if self._proc.stdin and not self._proc.stdin.is_closing():
                self._proc.stdin.write(b"exit 0\n")
                await self._proc.stdin.drain()
        except Exception:
            pass
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=2)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=2)
            except Exception:
                pass
        # Close the stdin write transport explicitly: Windows' proactor loop
        # emits a "transport was not closed" warning at shutdown when a write
        # pipe is left dangling after its process exits.
        if self._proc.stdin is not None and not self._proc.stdin.is_closing():
            try:
                self._proc.stdin.close()
            except Exception:
                pass
