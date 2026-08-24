"""Main agent loop orchestration.

Implements the Gemini Interactions API pattern described in CLAUDE.md:

  - Call `client.aio.interactions.create(model, input, tools=...)`.
  - Iterate the returned `interaction.steps` for `function_call` entries
    (Gemini may emit several — parallel calls are native).
  - Execute each tool, then send the results back as `function_result`
    steps with `call_id` set to the call's `id`, chaining with
    `previous_interaction_id`.
  - Repeat until an interaction's `output_text` is the final answer (status
    `completed`) and no further function_call steps are present.

The loop never imports the sandbox implementation directly — it holds a
`SandboxInterface` handle — so the Milestone 2 Docker-backed sandbox can be
swapped in without touching `agent/`.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from google import genai
from google.genai._gaos.types.interactions.functionresultstep import (
    FunctionResultStep,
)

from sandbox import SandboxInterface

from .tools import (
    create_shell_declaration,
    run_in_shell_declaration,
    read_file_declaration,
    write_file_declaration,
    create_file_declaration,
    delete_file_declaration,
    undo_declaration,
    user_question_declaration,
)

# --------------- retry helper for Gemini calls -----------------------------

_MAX_RETRIES = 5
_BASE_BACKOFF = 1.0   # seconds; doubled after each 429, capped at _BACKOFF_CAP
_BACKOFF_CAP = 30.0


async def _call_with_retry(create_fn, **kwargs):
    """Call the coroutine *create_fn* with exponential-backoff on 429s.

    Parses the ``"Please retry in Xs"`` hint that the free-tier quota
    error returns and honours it (preferring the server-side hint over the
    local formula when the hint is the longer of the two).  Propagates the
    last exception once *max_retries* is exhausted so real errors surface
    immediately rather than being swallowed.

    ``kwargs`` override ``_MAX_RETRIES`` / ``_BASE_BACKOFF`` / ``_BACKOFF_CAP``
    defaults for one-off callers.
    """
    import re  # local import keeps the top-level namespace clean
    max_retries = kwargs.pop("max_retries", _MAX_RETRIES)
    base_backoff = kwargs.pop("base_backoff", _BASE_BACKOFF)
    backoff_cap   = kwargs.pop("backoff_cap",   _BACKOFF_CAP)
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return await create_fn()
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            is_rate_limit = (
                "429" in msg
                or "exceeded your current quota" in msg
                or "Please retry in" in msg
                or "too_many_requests" in msg
            )
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            m = re.search(r"Please retry in (\d+\.?\d*)s", msg)
            suggested = float(m.group(1)) if m else 0.0
            wait = min(base_backoff * (2 ** attempt), backoff_cap)
            wait = max(wait, suggested)
            logger.warning(
                "Gemini rate-limited (attempt %d/%d); sleeping %.1fs then retrying.",
                attempt + 1, max_retries, wait,
            )
            await asyncio.sleep(wait)
    raise last_exc  # type: ignore[misc]

# All tool declarations sent to Gemini on every interaction.
TOOL_DECLARATIONS: List[Dict[str, Any]] = [
    create_shell_declaration,
    run_in_shell_declaration,
    read_file_declaration,
    write_file_declaration,
    create_file_declaration,
    delete_file_declaration,
    undo_declaration,
    user_question_declaration,
]


class UserQuestionHandler:
    """Suspends the agent loop until the user answers.

    The loop genuinely pauses on `await ask(...)` — it does not return on
    EOFError, it does not fake a "no reply" path. Tests and the future
    Milestone 4 WebSocket frontend deliver a reply via `set_response(...)`,
    which releases the `_response_event` the loop is awaiting. For human use
    in a TTY, an optional `input_provider` fallback reads from stdin via
    `asyncio.to_thread` so a real CLI session still works; when stdin is not a
    TTY (e.g. under pytest without a tty) this fallback raises EOFError and the
    programmatic `set_response` path is the only way to unblock.
    """

    def __init__(
        self,
        input_provider: Optional[Any] = None,
    ) -> None:
        """Create the handler.

        * `input_provider` — a sync callable `(prompt: str) -> str` used in
          real TTY sessions as a fallback when `set_response()` hasn't arrived
          yet. Defaults to `builtins.input`. Ignored in non-TTY mode (tests,
          CI, the future WebSocket frontend).
        """
        self._response_event = asyncio.Event()
        self._user_response: Optional[str] = None
        self._pending_question: Optional[str] = None
        # Defaults: use `builtins.input` for human CLI use (async-wrapped).
        self._input_provider = input_provider if input_provider is not None else input

    async def ask(self, question: str) -> str:
        """Ask a question and suspend until a reply arrives via set_response().

        The loop blocks on `_response_event.wait()`; the caller MUST call
        `set_response(...)` to release it. The stdin fallback only runs if no
        programmatic reply has arrived yet — and it does not silently swallow
        EOFError into "" anymore: in a non-interactive test run there is no
        human, so falling back to "" would be a lie. We let the exception
        propagate so the test/frontend path is the only reliable input.
        """
        # If set_response was called before this method started, we already have a reply
        if self._user_response is not None:
            response = self._user_response
            self._user_response = None
            self._response_event.clear()
            self._pending_question = None
            return response

        self._response_event.clear()
        self._pending_question = question

        # Check if we should use stdin fallback or event-only mode.
        # Use stdin only in real TTY environments where stdin has an actual terminal.
        # Check for test environment markers and also verify stdin has a valid fileno.
        in_test_env = (
            "PYTEST_CURRENT_TEST" in os.environ or
            "PYTESTLAUNCH" in os.environ or
            not hasattr(sys.stdin, 'fileno') or
            sys.stdin.fileno() < 0
        )
        is_tty = sys.stdin.isatty() and not in_test_env

        if is_tty:
            prompt = f"\n[AGENT QUESTION]: {question}\nYour response: "
            # Race between set_response() and stdin input.
            stdin_task = asyncio.create_task(
                asyncio.to_thread(self._input_provider, prompt)
            )
            event_task = asyncio.create_task(self._response_event.wait())

            # Wait for whichever completes first
            done, pending = await asyncio.wait(
                [stdin_task, event_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel the pending task
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Check if we got a response via set_response
            if self._user_response is not None:
                response = self._user_response
                self._user_response = None
                self._response_event.clear()
                self._pending_question = None
                return response

            # Fall back to stdin input result (only if stdin task finished)
            if stdin_task.done() and not stdin_task.cancelled():
                try:
                    response = stdin_task.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed to get input from stdin: {exc}") from exc
                self._pending_question = None
                return response

            # Neither completed - this shouldn't happen, but raise an error
            raise RuntimeError("User question wait failed unexpectedly")
        else:
            # Non-TTY context (tests, CI) - just wait for set_response()
            # with a timeout to provide a helpful error if test forgets
            try:
                await asyncio.wait_for(self._response_event.wait(), timeout=300.0)
            except asyncio.TimeoutError:
                # set_response may have released the event while wait_for was
                # cancelling the inner task (the race documented in Python's
                # asyncio.wait_for notes).  Check before raising so a response
                # that arrived during cancellation is not silently dropped.
                if self._user_response is not None:
                    response = self._user_response
                    self._user_response = None
                    self._pending_question = None
                    return response
                raise RuntimeError(
                    "user_question timed out waiting for response. "
                    "In non-interactive contexts, call set_response() to provide an answer."
                )

            response = self._user_response
            self._user_response = None
            self._response_event.clear()
            self._pending_question = None
            assert response is not None  # set_response guarantees this
            return response

    def set_response(self, response: str) -> None:
        """Deliver a reply (used by future WebSocket frontend, and tests).

        Releases the loop's `await self._response_event.wait()` so the agent
        loop resumes. If called before `ask(...)` is awaited (rare — e.g.
        tests racing the loop), the response is queued and the next `ask()`
        call observes it on entry.
        """
        # Store response even if event isn't set yet (for pre-call race condition)
        if self._user_response is None:
            self._user_response = response
            self._response_event.set()
        else:
            # Already have a response - update it and set event (idempotent)
            self._user_response = response
            if not self._response_event.is_set():
                self._response_event.set()

    @property
    def pending_question(self) -> Optional[str]:
        """The question the loop is currently blocked on, or None if not waiting."""
        return self._pending_question


class AgentLoop:
    """Coordinates Gemini Interactions API calls with sandbox tool execution."""

    def __init__(
        self,
        sandbox: "SandboxInterface",
        model: str = "gemini-2.5-flash",
        client: Optional[genai.Client] = None,
        system_instruction: str = (
            "You are a helpful coding agent operating in a working directory. "
            "Use the provided tools to accomplish the user's task. When the task "
            "is complete, summarize what you did."
        ),
        max_iterations: int = 25,
    ) -> None:
        # Constrain the param to SandboxInterface so the agent/→sandbox/
        # boundary is explicit, not duck-typed. `isinstance` would also be
        # fine; using a type annotation makes mypy/type-checkers happy and
        # makes the architectural invariant in CLAUDE.md enforceable.
        # Enforce the Milestone 1 architectural invariant: `agent/` must never
        # import a concrete sandbox implementation. A TypeError here catches
        # accidental coupling at construction time rather than letting it
        # surface as an AttributeError deep inside a tool call.
        if not isinstance(sandbox, SandboxInterface):
            raise TypeError(
                f"AgentLoop.sandbox must be a SandboxInterface; "
                f"got {type(sandbox).__name__}"
            )
        self.sandbox: SandboxInterface = sandbox
        self.model = model
        # Allow injecting a client (tests pass `client=...` for live use; in
        # normal use we lazily build one from the GEMINI_API_KEY env var).
        self._client = client
        self.system_instruction = system_instruction
        self.max_iterations = max_iterations
        self.previous_interaction_id: Optional[str] = None
        self.user_handler = UserQuestionHandler()
        # Observability hooks for tests: every tool invocation is recorded.
        self.tool_calls: List[Dict[str, Any]] = []

    @property
    def client(self):
        if self._client is None:
            self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        return self._client

    async def run(self, instruction: str, working_dir: Optional[Path] = None) -> str:
        """Run the agent loop to completion and return the final output text."""
        if working_dir is not None:
            self.sandbox.working_dir = Path(working_dir).resolve()
        self.previous_interaction_id = None
        self.tool_calls = []

        current_input: Any = instruction
        for _ in range(self.max_iterations):
            interaction = await _call_with_retry(
                lambda: self.client.aio.interactions.create(
                    model=self.model,
                    input=current_input,
                    tools=TOOL_DECLARATIONS,
                    system_instruction=self.system_instruction,
                    previous_interaction_id=self.previous_interaction_id,
                )
            )

            function_calls = [
                s for s in (interaction.steps or []) if s.type == "function_call"
            ]

            if not function_calls:
                # No tool calls means the model produced its final answer.
                self.previous_interaction_id = interaction.id
                return interaction.output_text or ""

            # Execute every function call, then send the results back as
            # FunctionResultStep SDK objects — plain dicts are rejected by the
            # API with "Invalid input received." on turn 2+ because pydantic
            # validation/serialization must run on the result field.
            result_steps: List[FunctionResultStep] = []
            for fc in function_calls:
                result_text = await self._execute_tool(fc.name, dict(fc.arguments))
                result_steps.append(
                    FunctionResultStep(
                        call_id=fc.id,
                        name=fc.name,
                        result=result_text,
                    )
                )

            self.previous_interaction_id = interaction.id
            current_input = result_steps

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
            shell_id = args["shell_id"]
            cmd = args["cmd"]
            result = await self.sandbox.run_in_shell(shell_id, cmd)
            out = (
                f"exit_code: {result.exit_code}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
            if result.timeout:
                out += f"\n[timed out]"
            return out

        if name == "read_file":
            return await self.sandbox.read_file(Path(args["path"]))

        if name == "write_file":
            await self.sandbox.write_file(Path(args["path"]), args["content"])
            return f"Wrote {len(args['content'])} bytes to {args['path']}"

        if name == "create_file":
            await self.sandbox.create_file(Path(args["path"]), args.get("content", ""))
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
