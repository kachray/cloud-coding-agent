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
from sandbox.queued import QueuedSandbox  # noqa: E402
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
    """Sandbox behind the queue. Default: LocalSandbox; set CCA_SANDBOX=docker
    to run the whole suite against a Docker container instead.

    Every op the agent issues crosses the QueuedSandbox boundary, so the
    existing tests exercise the queue path for free in both modes.
    """
    if os.environ.get("CCA_SANDBOX") == "docker":
        from sandbox.docker_sandbox import DockerSandbox

        concrete = DockerSandbox(working_dir=tmp_path)
    else:
        concrete = LocalSandbox(working_dir=tmp_path)
    sb = QueuedSandbox(concrete)
    yield sb
    await sb.cleanup()
    # LocalSandbox has no cleanup(); close its shells so no subprocess leaks
    # across tests (DockerSandbox.cleanup already closed + removed everything).
    for sid in list(getattr(concrete, "_shells", {}).keys()):
        try:
            await concrete.close_shell(sid)
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
