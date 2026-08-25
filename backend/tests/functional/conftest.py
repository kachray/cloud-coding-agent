"""Functional-test fixtures shared across tests/functional/."""
import asyncio
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from openai import AsyncOpenAI

backend_root = (
    Path.cwd()
    if Path.cwd().name == "backend"
    else Path(__file__).resolve().parent.parent.parent
)
load_dotenv(backend_root / ".env")
sys.path.insert(0, str(backend_root))

from sandbox.local import LocalSandbox  # noqa: E402
from agent.loop import AgentLoop        # noqa: E402


@pytest.fixture
def client():
    """Live OpenAI-compatible client pointed at Groq.

    Rate limiting is handled inside ``agent.loop._call_with_retry``, so no
    test-side throttle fixture is needed.
    """
    return AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    )


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
        model="openai/gpt-oss-120b",
        client=client,
        system_instruction=(
            "You are a coding agent operating in the given working directory. "
            "Use the provided tools to complete the user's task exactly. Do not ask "
            "the user questions unless the task is genuinely ambiguous; make a "
            "reasonable choice and proceed. When finished, briefly state the outcome."
        ),
    )