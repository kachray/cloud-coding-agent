# CLAUDE.md — Cloud Coding Agent

## Stack
Backend: Python 3.11+, managed with `uv` (not pip/venv directly — use `uv add`,
`uv run`). FastAPI. LLM: Groq via the `openai` SDK pointed at Groq's
OpenAI-compatible endpoint (`base_url="https://api.groq.com/openai/v1"`,
`api_key=GROQ_API_KEY`) — NOT the `groq` SDK; the OpenAI-compatible path is
what CLAUDE.md standardizes on since it's the better-documented, more
battle-tested shape. Model: openai/gpt-oss-120b (real tool-calling
reliability; openai/gpt-oss-120b). Docker SDK for Python (docker-py). Frontend: Vite +
React + Tailwind, WebSocket client.

## Groq / OpenAI-compatible tool-calling — the pattern this project uses
- Standard `chat.completions.create(model=..., messages=[...], tools=[...])`.
  `tools` is a list of `{"type": "function", "function": {"name", "description",
  "parameters"}}` dicts — write these explicitly, same discipline as before.
- Conversation state is client-managed, not server-managed (no
  `previous_interaction_id` equivalent) — agent/loop.py owns the full
  `messages` list and appends to it every turn: the assistant's message
  (including any `tool_calls`), then one `{"role": "tool", "tool_call_id":
  <id from the tool_call>, "content": <result>}` message per tool call executed.
- Read tool calls from `response.choices[0].message.tool_calls` (a list —
  parallel calls are native here too). Final text is
  `response.choices[0].message.content` once `tool_calls` is empty/None.
- **Rate limiting lives inside the API-call wrapper itself** (e.g. a
  `_call_with_retry` function in agent/loop.py), not bolted onto individual
  test call sites. Every call through AgentLoop — tests, and later the real
  multi-sandbox production path — is automatically serialized/throttled this
  way. Do not add per-test rate-limit boilerplate; if a test needs one, the
  wrapper is broken, fix it there.



## Architecture invariant
The agent loop (calls Claude, decides what to do) and the sandbox (where code
actually executes) are separate concerns, kept in separate modules
(agent/ vs sandbox/) from Milestone 1 onward, even before Docker exists in
Milestone 2. Never let agent/ import Docker directly — it should call an
interface (e.g. `sandbox.create_shell()`, `sandbox.run_in_shell(id, cmd)`)
that Milestone 1 implements against a local subprocess and Milestone 2
re-implements against Docker, without agent/ changing. Communication between
agent/ and sandbox/ goes through a queue, not a direct call, from Milestone 2
onward — this is what lets Docker be swapped for a remote sandbox service
later without touching the agent loop.

## Tool shapes (fixed from Milestone 1 — don't redesign later)
- `create_shell()` -> shell_id ; `run_in_shell(shell_id, cmd)` — stateful,
  supports parallel shells. NOT a single stateless run_shell(cmd) call.
- File editor: read_file, write_file, create_file, delete_file, undo.
- `user_question(text)` — suspends the agent loop until a reply arrives.
  Build the suspend/resume mechanism now even though it's only visibly
  useful once Milestone 4 wires up the frontend.
- Hard tools (LSP, browser, deploy) come later, in that difficulty order —
  don't attempt them before Milestone 5.

## Verification standard
No milestone is done until backend/tests/functional/ passes against it AND a
separate session or read-only subagent — not the one that implemented the
milestone — has independently confirmed the tests actually assert real
behavior and the implementation matches the plan. These tests run a real task
through the real agent loop and assert on real outcomes (files created,
commands run, commits made) — not mocks of the Groq API or the sandbox,
which would defeat the purpose. An agent checking its own work in the same
context that produced it is not a verification pass; treat it as one anyway
and you will eventually ship a milestone whose "passing" tests don't
actually test anything.

## Codebase search
Use grep/glob to find things, not semantic/fuzzy search tooling — this
project's own creator's team found exact literal matching outperforms vector
search for codebases, and there's no reason this project is the exception.

## Security-relevant code
Anything touching GitHub tokens, sandbox isolation, or credential handling:
be explicit and conservative, don't optimize for cleverness. Short-lived
installation tokens only (per the GitHub App pattern) — never persist a
long-lived PAT or refresh token unless the feature genuinely needs user
metadata (it doesn't, for repo access alone).

## Session hygiene
/clear between milestones. One milestone, fully working, harness-passing, AND
independently verified, before starting the next — don't parallelize across
milestones. For the two or three genuinely hard milestones (Docker+queue
swap, browser control, deployment), consider "ultrathink" in the planning
prompt for extra reasoning depth — not needed for the straightforward ones.