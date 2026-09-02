"""SandboxInterface decorator that routes every op through a queue.

QueuedSandbox implements SandboxInterface by putting each op on an
``asyncio.Queue`` and awaiting a Future that a single worker task resolves by
calling the wrapped sandbox.

Single host: ``asyncio.Queue``. Multi-host later: the same request/response
shape (op name, args, one in-flight future per op) over Redis pub/sub — swap
the transport, not the interface or anything in ``agent/``. This is the layer
that lets Docker be swapped for a remote sandbox service without touching the
agent loop.
"""
import asyncio
from pathlib import Path
from typing import Any, Callable, Optional

from .local import SandboxInterface, Shell, ShellResult


class QueuedSandbox(SandboxInterface):
    """Route all sandbox ops through a queue to a single worker task.

    The worker serializes ops (one at a time) — matches how a single remote
    sandbox session serves requests today. ``working_dir`` is forwarded to the
    target so ``AgentLoop.run``'s direct assignment keeps working.
    """

    def __init__(self, target: SandboxInterface) -> None:
        if not isinstance(target, SandboxInterface):
            raise TypeError(
                f"QueuedSandbox target must be a SandboxInterface; "
                f"got {type(target).__name__}"
            )
        self.target = target
        self._queue: "asyncio.Queue" = asyncio.Queue()
        self._worker: Optional[asyncio.Task] = None

    # --- plumbing --------------------------------------------------------

    @property
    def working_dir(self) -> Path:
        return Path(self.target.working_dir)

    @working_dir.setter
    def working_dir(self, value: Path) -> None:
        self.target.working_dir = Path(value).resolve()

    async def _submit(self, op: str, **kwargs: Any) -> Any:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())
        future = asyncio.get_running_loop().create_future()
        await self._queue.put((op, kwargs, future))
        return await future

    async def _run(self) -> None:
        while True:
            op, kwargs, future = await self._queue.get()
            if op is None:  # sentinel: stop
                future.set_result(None)
                return
            try:
                method: Callable = getattr(self.target, op)
                future.set_result(await method(**kwargs))
            except Exception as exc:  # surface the error to the awaiting op
                future.set_exception(exc)

    async def cleanup(self) -> None:
        """Stop the worker, then forward cleanup to the target if it has one.

        Only DockerSandbox defines ``cleanup`` (removes the container);
        LocalSandbox's shells are closed via ``close_shell`` by its caller.
        """
        if self._worker is not None:
            worker, self._worker = self._worker, None
            stop = asyncio.get_running_loop().create_future()
            await self._queue.put((None, {}, stop))
            await stop
            if not worker.done():
                worker.cancel()
        target_cleanup = getattr(self.target, "cleanup", None)
        if target_cleanup is not None:
            await target_cleanup()

    # --- SandboxInterface --------------------------------------------------

    async def create_shell(self, name: Optional[str] = None, cwd: Optional[Path] = None) -> Shell:
        return await self._submit("create_shell", name=name, cwd=cwd)

    async def run_in_shell(self, shell_id: str, cmd: str, timeout: Optional[int] = None) -> ShellResult:
        return await self._submit("run_in_shell", shell_id=shell_id, cmd=cmd, timeout=timeout)

    async def read_file(self, path: Path) -> str:
        return await self._submit("read_file", path=path)

    async def write_file(self, path: Path, content: str) -> None:
        return await self._submit("write_file", path=path, content=content)

    async def create_file(self, path: Path, content: str = "") -> None:
        return await self._submit("create_file", path=path, content=content)

    async def delete_file(self, path: Path) -> None:
        return await self._submit("delete_file", path=path)

    async def undo(self) -> bool:
        return await self._submit("undo")

    async def close_shell(self, shell_id: str) -> None:
        return await self._submit("close_shell", shell_id=shell_id)
