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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai

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

    Milestone 1 stand-in: blocks on CLI stdin via `asyncio.to_thread` so the
    event loop isn't frozen while waiting. The same `asyncio.Event`-based
    `set_response` hook is what the Milestone 4 WebSocket frontend will call
    to deliver a reply; only the *input source* changes later, not this
    object's role in the loop.
    """

    def __init__(self) -> None:
        self._response_event = asyncio.Event()
        self._user_response: Optional[str] = None

    async def ask(self, question: str) -> str:
        """Ask a question and block until a reply is available."""
        self._response_event.clear()
        self._user_response = None
        # Milestone 1: read from stdin. Milestone 4: this is replaced by
        # awaiting the WebSocket-delivered reply that calls set_response().
        prompt = f"\n[AGENT QUESTION]: {question}\nYour response: "
        try:
            response = await asyncio.to_thread(input, prompt)
        except EOFError:
            response = ""
        return response

    def set_response(self, response: str) -> None:
        """Deliver a reply (used by future WebSocket frontend, and tests)."""
        self._user_response = response
        self._response_event.set()


class AgentLoop:
    """Coordinates Gemini Interactions API calls with sandbox tool execution."""

    def __init__(
        self,
        sandbox,
        model: str = "gemini-2.5-flash",
        client: Optional[genai.Client] = None,
        system_instruction: str = (
            "You are a helpful coding agent operating in a working directory. "
            "Use the provided tools to accomplish the user's task. When the task "
            "is complete, summarize what you did."
        ),
        max_iterations: int = 25,
    ) -> None:
        self.sandbox = sandbox
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
            interaction = await self.client.aio.interactions.create(
                model=self.model,
                input=current_input,
                tools=TOOL_DECLARATIONS,
                system_instruction=self.system_instruction,
                previous_interaction_id=self.previous_interaction_id,
            )

            function_calls = [
                s for s in (interaction.steps or []) if s.type == "function_call"
            ]

            if not function_calls:
                # No tool calls means the model produced its final answer.
                self.previous_interaction_id = interaction.id
                return interaction.output_text or ""

            # Execute every function call, then send the results back.
            result_steps: List[Dict[str, Any]] = []
            for fc in function_calls:
                result_text = await self._execute_tool(fc.name, dict(fc.arguments))
                result_steps.append(
                    {
                        "type": "function_result",
                        "call_id": fc.id,
                        "name": fc.name,
                        "result": result_text,
                    }
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
