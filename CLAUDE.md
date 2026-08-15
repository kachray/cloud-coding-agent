# CLAUDE.md — Cloud Coding Agent

## Stack
Backend: Python 3.11+, FastAPI, Anthropic SDK (tool use / function calling),
Docker SDK for Python (docker-py). Frontend: Vite + React + Tailwind, WebSocket
client. Package managers: uv for Python, npm for frontend.

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
commands run, commits made) — not mocks of the Anthropic API or the sandbox,
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