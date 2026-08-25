"""Functional-test fixtures shared across tests/functional/."""
import asyncio
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from google import genai

backend_root = (
    Path.cwd()
    if Path.cwd().name == "backend"
    else Path(__file__).resolve().parent.parent.parent
)
load_dotenv(backend_root / ".env")
sys.path.insert(0, str(backend_root))

from sandbox.local import LocalSandbox  # noqa: E402
from agent.loop import AgentLoop        # noqa: E402


# ---- Test-body serialization ---------------------------------------------
# Uses a per-session `asyncio.Semaphore(1)` as an async barrier.
# Only one async test body holds the semaphore at a time; every other
# test body blocks at `await _serial_cm()` until the holder releases.
# Because the holder keeps it for the entire body (async with … yield),
# the ordering is strict and the API is never hit concurrently.
#
# Tests opt in by adding ``_serial_cm`` to their signature::
#
#   async def test_something(self, agent, _serial_cm, tmp_path):
#       async with _serial_cm():
#           result = await agent.run(...)

class _SerialCM:
    """Async context manager: holds an asyncio.Semaphore(1) for its body."""

    def __init__(self):
        self._sem = asyncio.Semaphore(1)

    def __call__(self):
        return self

    async def __aenter__(self):
        await self._sem.acquire()
        return self

    async def __aexit__(self, *exc):
        self._sem.release()
        return False


@pytest.fixture(scope="session")
def _serial_cm():
    """One shared barrier for the whole test session."""
    return _SerialCM()


# ---- Shared fixtures -----------------------------------------------------

@pytest.fixture
def client():
    """Live Gemini client — no rate-limit call here.

    Slot acquisition happens inside ``_serial_cm`` in each test body,
    immediately before ``agent.run()``, so there is no concurrent
    client-setup window that could burst the rate limit.
    """
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