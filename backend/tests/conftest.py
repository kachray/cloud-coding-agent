"""Shared pytest fixtures for the entire test suite.

Serialises functional test execution so only one test uses Gemini at a
time, and spaces API calls within each test so the suite stays inside the
20 req/min free-tier ceiling.
"""
import sys
import time
from pathlib import Path

import pytest

backend_root = (
    Path.cwd()
    if Path.cwd().name == "backend"
    else Path(__file__).resolve().parent.parent
)
sys.path.insert(0, str(backend_root))

# --- Shared mutable state in a dict so nested closures can mutate it
#     without running into Python's nonlocal / cell-variable scoping quirks.
_state = {
    "lock":        None,   # set to threading.Lock() at first use
    "running":     False,  # True while a test body is executing
    "last_api":    0.0,    # monotonic timestamp of the most recent API call
}

# Mutable container avoids nonlocal scoping issues in deeply nested closures.
_calls = {"n": 0}


def _get_lock():
    if _state["lock"] is None:
        import threading
        _state["lock"] = threading.Lock()
    return _state["lock"]


@pytest.fixture(scope="session")
def _acquire_gemini_slot():
    """Global gate for Gemini API calls.

    Returns a synchronous callable — call it right before each API call.
    The first call in a new test enforces a 14 s spacing (to account for
    the burst of calls that test will make); subsequent calls inside the
    same test use a tighter 5 s spacing.  Thread-safe across the many
    event loops pytest-asyncio creates.
    """
    _test_started = [0.0]   # monotonic; 0 == not yet initialised

    def _acquire() -> None:
        with _get_lock():
            now = time.monotonic()
            elapsed = now - _test_started[0]
            is_first_call = (
                _test_started[0] == 0.0
                or elapsed > 14.0   # >14 s since first call => new test
            )
            gap = 14.0 if is_first_call else 5.0
            wait = _state["last_api"] + gap - now
            if wait > 0:
                time.sleep(wait)
            _state["last_api"] = time.monotonic()
            if is_first_call:
                _test_started[0] = now

    return _acquire