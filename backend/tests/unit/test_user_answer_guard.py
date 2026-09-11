"""Unit tests for the `user_question` answer guard — no real API, no real user.

Both tests drive `AgentLoop.run()` end to end against a stub client, because
that is the layer the bug lived at: the guard must be attached to the
assembled `role: tool` message on success, and must never reach it on failure.
Testing `_dispatch` or `_execute_tool` directly would pass against the
pre-fix code, where the guard was appended in `run()`'s tool-result loop and
so leaked onto the "ERROR executing user_question: ..." string.

Stub client is allowed here per CLAUDE.md: tests/unit/ may mock, tests/functional/ may not.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sandbox.local import LocalSandbox  # noqa: E402
from agent.loop import AgentLoop, _USER_ANSWER_GUARD  # noqa: E402


class _Msg:
    """Stands in for the SDK's assistant message (needs .model_dump)."""

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant"}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return d


def _fake_client(messages):
    """Client that replays `messages` one create() call at a time."""
    queue = list(messages)
    sent = []

    async def create(**kwargs):
        sent.append(kwargs["messages"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=queue.pop(0))]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    # Exposed so tests can assert on what the API actually received, not just
    # on _messages — those are the same object today, but a future change that
    # passes create() a filtered copy would otherwise slip past.
    client.sent = sent
    return client, sent


def _user_question_turn():
    return _Msg(tool_calls=[
        SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(
                name="user_question",
                arguments=json.dumps({"text": "meaning?"}),
            ),
        )
    ])


def _tool_messages(messages):
    return [m for m in messages if m["role"] == "tool"]


@pytest.fixture
def loop_factory(tmp_path):
    def build(turns):
        client, _ = _fake_client(turns)
        return AgentLoop(sandbox=LocalSandbox(working_dir=tmp_path), client=client)
    return build


async def test_successful_answer_reaches_tool_message_with_guard(loop_factory):
    loop = loop_factory([_user_question_turn(), _Msg(content="done")])
    loop.user_handler.set_response("42")

    await loop.run("ask the user")

    (tool_msg,) = _tool_messages(loop._messages)
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"].startswith("42"), (
        f"the answer must head the content so substring checks still work; "
        f"got {tool_msg['content']!r}"
    )
    assert _USER_ANSWER_GUARD in tool_msg["content"], (
        "a successful answer must carry the guard into the API messages"
    )

    # ...and the turn-2 API call must actually have received it.
    received = loop._client.sent[-1]
    assert any(
        m["role"] == "tool" and _USER_ANSWER_GUARD in m["content"]
        for m in received
    ), (
        f"the API call after the answer did not receive the guarded tool "
        f"message; it saw: {received!r}"
    )


async def test_handler_error_reaches_tool_message_without_guard(loop_factory):
    loop = loop_factory([_user_question_turn(), _Msg(content="done")])

    async def boom(question):
        raise RuntimeError("user_question timed out waiting for response.")

    loop.user_handler.ask = boom

    await loop.run("ask the user")

    (tool_msg,) = _tool_messages(loop._messages)
    assert tool_msg["content"].startswith("ERROR executing user_question"), (
        f"expected the error routed through _execute_tool; got "
        f"{tool_msg['content']!r}"
    )
    assert _USER_ANSWER_GUARD not in tool_msg["content"], (
        "an error must never be labelled as the user's authoritative answer"
    )
