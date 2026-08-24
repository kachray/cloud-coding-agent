"""Functional-test fixtures shared across tests/functional/."""
import sys
import threading
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


# ---- Test-body serialisation ---------------------------------------------
# An asyncio context manager that holds a threading.Lock for the *entire*
# duration of the test body, preventing pytest-asyncio from running two
# async test bodies concurrently (which would still share the event loop
# and therefore blast the API simultaneously).
_serial = threading.Lock()

class _SerialCM:
    """Callable async context manager for serialising test bodies.

    Usage::

        async with _serial_cm():
            # ... test body ...
    """

    async def __aenter__(self):
        while True:
            acquired = _serial.acquire(blocking=False)
            if acquired:
                return self
            import time
            time.sleep(0.05)

    async def __aexit__(self, *exc):
        _serial.release()

    # Make the instance itself callable so ``async with _serial_cm():``
    # works (the () triggers __call__ which returns self, an async CtxMgr).
    def __call__(self):
        return self


@pytest.fixture(scope="session")
def _serial_cm():
    """Session-scoped serialisation gate — call with ``async with _serial_cm():``."""
    return _SerialCM()


@pytest.fixture
def client(_acquire_gemini_slot):
    """Live Gemini client with session-level rate gating."""
    _acquire_gemini_slot()
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