"""Main agent loop orchestration.

Implements the OpenAI-compatible chat.completions pattern (Groq in production):

  - Call ``client.chat.completions.create(model, messages=messages, tools=tools)``
  - Read tool calls from ``response.choices[0].message.tool_calls``
  - Execute tools and append both the assistant turn and per-tool ``role: tool``
    results back into ``messages``
  - Repeat until ``tool_calls`` is empty — the assistant's text reply is final

The agent loop and sandbox are separate concerns: ``agent/`` imports only
``SandboxInterface``, never a concrete sandbox implementation.
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from sandbox import SandboxInterface

from .tools import (
    create_file_declaration,
    create_shell_declaration,
    delete_file_declaration,
    read_file_declaration,
    run_in_shell_declaration,
    undo_declaration,
    user_question_declaration,
    write_file_declaration,
)

logger = logging.getLogger(__name__)

# --------------- retry helper (covers all API calls) ------------------------

_MAX_RETRIES = 5
_BASE_BACKOFF = 2.0   # seconds; doubled after each 429, capped at _BACKOFF_CAP
_BACKOFF_CAP = 10.0


async def _call_with_retry(create_fn, **kwargs):
    """Call the coroutine *create_fn* with exponential-backoff on rate-limit errors.

    Parses the ``"Please retry in Xs"`` hint the provider returns and honours
    it (preferring the server-side hint over the local formula when the hint
    is longer).  Propagates the last exception once *max_retries* is
    exhausted so real errors surface immediately rather than being swallowed.

    ``kwargs`` override ``_MAX_RETRIES`` / ``_BASE_BACKOFF`` / ``_BACKOFF_CAP``
    defaults for one-off callers.
    """
    import re

    max_retries = kwargs.pop("max_retries", _MAX_RETRIES)
    base_backoff = kwargs.pop("base_backoff", _BASE_BACKOFF)
    backoff_cap = kwargs.pop("backoff_cap", _BACKOFF_CAP)
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return await create_fn()
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            is_rate_limit = (
                "429" in msg
                or "rate_limit" in msg
                or "rate limit" in msg.lower()
                or "too_many_requests" in msg
                or "QUOTA_EXCEEDED" in msg
                or "Please retry in" in msg
                or ("5" in msg[:3] and "server_error" in msg.lower())
            )
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            m = re.search(r"Please retry in (\d+\.?\d*)s", msg)
            suggested = float(m.group(1)) if m else 0.0
            wait = min(base_backoff * (2 ** attempt), backoff_cap)
            wait = max(wait, suggested)
            logger.warning(
                "API rate-limited (attempt %d/%d); sleeping %.1fs then retrying.",
                attempt + 1, max_retries, wait,
            )
            await asyncio.sleep(wait)
    raise last_exc  # type: ignore[misc]


# --------------- tool declarations (OpenAI/Groq envelope) -------------------

# Raw declarations carry the function schema; the wrapper adds the
# {"type": "function", "function": ...} envelope required by the
# OpenAI-compatible tool-calling format (Groq, OpenRouter, LM Studio, …).
_RAW_TOOL_DECLARATIONS: List[Dict[str, Any]] = [
    create_shell_declaration,
    run_in_shell_declaration,
    read_file_declaration,
    write_file_declaration,
    create_file_declaration,
    delete_file_declaration,
    undo_declaration,
    user_question_declaration,
]


_TOOL_DECLARATIONS: List[Dict[str, Any]] = [
    {"type": "function", "function": d} for d in _RAW_TOOL_DECLARATIONS
]


# --------------- user-question suspend/resume -------------------------------

class UserQuestionHandler:
    """Suspends the agent loop until the user answers.

    The loop genuinely pauses on ``await ask(...)``.  Tests and the future
    Milestone 4 WebSocket frontend deliver a reply via ``set_response(...)``,
    which releases the ``_response_event`` the loop is awaiting.  For human
    use in a TTY, an optional ``input_provider`` fallback reads from stdin via
    ``asyncio.to_thread`` so a real CLI session still works; when stdin is not
    a TTY (e.g. under pytest) this fallback raises ``EOFError`` and the
    programmatic ``set_response`` path is the only way to unblock.
    """

    def __init__(
        self,
        input_provider: Optional[Any] = None,
    ) -> None:
        self._response_event = asyncio.Event()
        self._user_response: Optional[str] = None
        self._pending_question: Optional[str] = None
        self._input_provider = (
            input_provider if input_provider is not None else input
        )

    async def ask(self, question: str) -> str:
        # If set_response was called before this method started, we have a reply
        if self._user_response is not None:
            response = self._user_response
            self._user_response = None
            self._response_event.clear()
            self._pending_question = None
            return response

        self._response_event.clear()
        self._pending_question = question

        in_test_env = (
            "PYTEST_CURRENT_TEST" in os.environ
            or "PYTESTLAUNCH" in os.environ
            or not hasattr(sys, "stdin")  # type: ignore[attr-defined]
            or not hasattr(sys.stdin, "fileno")  # type: ignore[attr-defined]
            or sys.stdin.fileno() < 0  # type: ignore[attr-defined]
        )
        is_tty = sys.stdin.isatty() and not in_test_env  # type: ignore[attr-defined]

        if is_tty:
            prompt = f"\n[AGENT QUESTION]: {question}\nYour response: "
            stdin_task = asyncio.create_task(
                asyncio.to_thread(self._input_provider, prompt)
            )
            event_task = asyncio.create_task(self._response_event.wait())

            done, pending = await asyncio.wait(
                [stdin_task, event_task], return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            if self._user_response is not None:
                response = self._user_response
                self._user_response = None
                self._response_event.clear()
                self._pending_question = None
                return response

            if stdin_task.done() and not stdin_task.cancelled():
                try:
                    response = stdin_task.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to get input from stdin: {exc}"
                    ) from exc
                self._pending_question = None
                return response

            raise RuntimeError("User question wait failed unexpectedly")
        else:
            try:
                await asyncio.wait_for(
                    self._response_event.wait(), timeout=30.0
                )
            except asyncio.TimeoutError:
                if self._user_response is not None:
                    response = self._user_response
                    self._user_response = None
                    self._response_event.clear()
                    self._pending_question = None
                    return response
                raise RuntimeError(
                    "user_question timed out waiting for response. "
                    "In non-interactive contexts, call set_response() to "
                    "provide an answer."
                )

            response = self._user_response
            self._user_response = None
            self._response_event.clear()
            self._pending_question = None
            assert response is not None
            return response

    def set_response(self, response: str) -> None:
        if self._user_response is None:
            self._user_response = response
            self._response_event.set()
        else:
            self._user_response = response
            if not self._response_event.is_set():
                self._response_event.set()

    @property
    def pending_question(self) -> Optional[str]:
        return self._pending_question


# --------------- main agent loop -------------------------------------------

class AgentLoop:
    """Coordinates OpenAI/Groq tool-calling API with sandbox tool execution.

    Conversation state is fully client-managed: ``self._messages`` holds the
    entire ``messages`` list sent to the API on every turn.  Each round
    appends the assistant's tool-call turn and then one ``role: tool`` result
    per call before sending the next request.
    """

    def __init__(
        self,
        sandbox: "SandboxInterface",
        model: str = "openai/gpt-oss-120b",
        client: Optional[AsyncOpenAI] = None,
        system_instruction: str = (
            "You are a helpful coding agent operating in a working directory. "
            "Use the provided tools to accomplish the user's task. When the task "
            "is complete, summarize what you did."
        ),
        max_iterations: int = 25,
    ) -> None:
        if not isinstance(sandbox, SandboxInterface):
            raise TypeError(
                f"AgentLoop.sandbox must be a SandboxInterface; "
                f"got {type(sandbox).__name__}"
            )
        self.sandbox: SandboxInterface = sandbox
        self.model = model
        self._client = client
        self.system_instruction = system_instruction
        self.max_iterations = max_iterations
        self._messages: List[Dict[str, Any]] = []
        self.user_handler = UserQuestionHandler()
        self.tool_calls: List[Dict[str, Any]] = []

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY"),
            )
        return self._client

    async def run(self, instruction: str, working_dir: Optional[Path] = None) -> str:
        """Run the agent loop to completion and return the final output text."""
        if working_dir is not None:
            self.sandbox.working_dir = Path(working_dir).resolve()

        self._messages = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user",   "content": instruction},
        ]
        self.tool_calls = []

        for _ in range(self.max_iterations):
            response = await _call_with_retry(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=self._messages,
                    tools=_TOOL_DECLARATIONS,
                )
            )

            msg = response.choices[0].message
            tool_calls = msg.tool_calls or []

            if not tool_calls:
                # Final answer — no further tool calls.
                self._messages.append(
                    {"role": "assistant", "content": msg.content}
                )
                return msg.content or ""

            # Record the assistant turn (includes tool_calls).
            # model_dump strips None values; exclude_none keeps the dict clean.
            assistant_msg = msg.model_dump(exclude_none=True)
            self._messages.append(assistant_msg)

            # Execute every tool call concurrently — they are independent.
            results = await asyncio.gather(*[
                self._execute_tool(
                    tc.function.name,
                    json.loads(tc.function.arguments),
                )
                for tc in tool_calls
            ])

            # Record each tool result.
            for tc, result in zip(tool_calls, results):
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        raise RuntimeError(
            f"agent loop did not finish within {self.max_iterations} iterations"
        )

    async def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Dispatch one tool call to the sandbox and return a result string."""
        record = {"name": name, "args": dict(args)}
        try:
            result = await self._dispatch(name, args)
            record["result_preview"] = result[:200]
            return result
        except Exception as exc:
            err = f"ERROR executing {name}: {type(exc).__name__}: {exc}"
            record["error"] = err
            return err
        finally:
            self.tool_calls.append(record)

    async def _dispatch(self, name: str, args: Dict[str, Any]) -> str:
        if name == "create_shell":
            shell = await self.sandbox.create_shell(name=args.get("name"))
            return f"shell_id: {shell.shell_id}"

        if name == "run_in_shell":
            result = await self.sandbox.run_in_shell(
                args["shell_id"], args["cmd"]
            )
            out = (
                f"exit_code: {result.exit_code}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
            if result.timeout:
                out += "\n[timed out]"
            return out

        if name == "read_file":
            return await self.sandbox.read_file(Path(args["path"]))

        if name == "write_file":
            await self.sandbox.write_file(Path(args["path"]), args["content"])
            return f"Wrote {len(args['content'])} bytes to {args['path']}"

        if name == "create_file":
            await self.sandbox.create_file(
                Path(args["path"]), args.get("content", "")
            )
            return f"Created {args['path']}"

        if name == "delete_file":
            await self.sandbox.delete_file(Path(args["path"]))
            return f"Deleted {args['path']}"

        if name == "undo":
            success = await self.sandbox.undo()
            return "Undo successful" if success else "Nothing to undo"

        if name == "user_question":
            return await self.user_handler.ask(args["text"])

        return f"Unknown tool: {name}"