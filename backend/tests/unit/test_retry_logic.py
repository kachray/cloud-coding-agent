"""Unit tests for `_call_with_retry` — synthetic exceptions, no real API.

These mock (synthetically raise) the exact exception types the openai SDK
produces, which is the case CLAUDE.md's clarified rule allows mocking for:
the retry/error-handling branch is what's under test, not the sandbox or the
agent's real behavior.
"""
import sys
from pathlib import Path

import httpx2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import openai  # noqa: E402

from agent.loop import _call_with_retry  # noqa: E402


def _status_error(cls, status_code: int, message: str) -> openai.APIStatusError:
    """Build a real openai SDK exception the way the client raises it."""
    request = httpx2.Request("POST", "https://api.groq.com/openai/v1")
    response = httpx2.Response(status_code, request=request)
    return cls(message, response=response, body={"error": {"message": message}})


async def test_retries_on_rate_limit_429():
    calls = []

    async def create_fn():
        calls.append(1)
        if len(calls) < 3:
            raise _status_error(
                openai.RateLimitError, 429, "Error code: 429 - rate limited"
            )
        return "ok"

    result = await _call_with_retry(create_fn, base_backoff=0.0)
    assert result == "ok"
    assert len(calls) == 3


async def test_retries_on_internal_server_error_500():
    calls = []

    async def create_fn():
        calls.append(1)
        if len(calls) < 3:
            raise _status_error(
                openai.InternalServerError, 500, "Error code: 500"
            )
        return "ok"

    result = await _call_with_retry(create_fn, base_backoff=0.0)
    assert result == "ok"
    assert len(calls) == 3


async def test_does_not_retry_on_400():
    calls = []

    async def create_fn():
        calls.append(1)
        # Body mentions "rate limit" — must still be treated as a 400 and
        # raised immediately, not mistaken for a retryable rate-limit error.
        raise _status_error(
            openai.BadRequestError, 400, "Error code: 400 - rate limit misuse"
        )

    with pytest.raises(openai.BadRequestError):
        await _call_with_retry(create_fn, base_backoff=0.0)
    assert len(calls) == 1