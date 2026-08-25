"""Shared pytest fixtures for the entire test suite.

Provides _acquire_gemini_slot: a session-scoped rate limiter that spaces
Gemini API calls so the suite stays within the free-tier 20 req/min ceiling.
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

_lock = __import__("threading").Lock()
_last_call = [0.0]


@pytest.fixture(scope="session")
def _acquire_gemini_slot():
    """Rate-limits Gemini API calls across the whole session.

    Uses a threading.Lock (survives event-loop churn) plus time.sleep to
    enforce a minimum gap between calls.  Back-to-back calls within a test
    use a 5 s gap; the first call after a >30 s gap (i.e., a new test)
    is treated as a fresh start.

    Returns a **synchronous** callable to invoke right before each API call::

        _acquire_gemini_slot()
        # ... make Gemini API call ...
    """
    _test_start = [0.0]

    def _acquire():
        with _lock:
            now = time.monotonic()
            is_first = not _test_start[0] or (now - _test_start[0]) > 30.0
            gap = 30.0 if is_first else 5.0
            wait = _last_call[0] + gap - now
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.monotonic()
            if is_first:
                _test_start[0] = now

    return _acquire