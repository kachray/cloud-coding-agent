# CLAUDE.md — Cloud Coding Agent

## Stack
Backend: Python 3.11+, managed with `uv` (not pip/venv directly — use `uv add`,
`uv run`). FastAPI. LLM: Groq via the `groq` SDK (official client; the
OpenAI-compatible `openai` SDK pointed at Groq's base_url also works
identically if preferred — either is fine, don't rewrite one into the other
without reason). Model: `openai/gpt-oss-120b` (Llama 3.3 70B is no longer
available on Groq as of this project). Free tier for this model: 30 RPM,
1,000 RPD, 8,000 TPM, **200,000 TPD** — in practice the TPD ceiling binds
first during heavy iteration, not the request count: a multi-turn agentic
task resending growing message history can cost a few thousand tokens per
run, so expect roughly ~100 real task-runs/day, not 1,000, before hitting a
wall. If this becomes a real blocker, `llama-3.1-8b-instant` has a much
higher TPD (500,000) as a same-shape fallback — smaller model, less reliable
tool-calling, but rarely rate-limited.
Frontend: Vite + React + Tailwind, WebSocket client.

## Groq / OpenAI-compatible tool-calling — the pattern this project uses
- Model must be a plain model (`openai/gpt-oss-120b` / `llama-3.1-8b-instant`),
  **never `groq/compound` or `groq/compound-mini`.** Those are Groq's built-in
  agentic systems with their own baked-in web-search/code-execution tools —
  Groq's own docs state custom user-provided tools are not supported by them
  at all. They will silently ignore or reject our `create_shell`/`run_in_shell`/
  etc. tool declarations. This looks like "a more capable model" and is
  actually an architecturally incompatible one — don't switch to it later
  assuming it's a drop-in upgrade.
- `tools.py`'s declarations are the BARE function schema only —
  `{"name", "description", "parameters"}`, no `"type"` key. Exactly one place
  in the codebase (the request-building code in `agent/loop.py`) wraps each
  as `{"type": "function", "function": <bare_schema>}` before sending. If you
  ever see `"type": "function"` appearing twice, nested, that's this
  invariant being violated somewhere — Groq is lenient about it today, but
  it's spec-noncompliant and a stricter backend (OpenAI proper, a future Groq
  version) will 400 on it.
- Conversation state is client-managed, not server-managed (no
  `previous_interaction_id` equivalent) — agent/loop.py owns the full
  `messages` list and appends to it every turn: the assistant's message
  (including any `tool_calls`), then one `{"role": "tool", "tool_call_id":
  <id from the tool_call>, "content": <result>}` message per tool call executed.
- Read tool calls from `response.choices[0].message.tool_calls` (a list —
  parallel calls are native here too). Final text is
  `response.choices[0].message.content` once `tool_calls` is empty/None.
- **Known model behavior, discovered during Milestone 1 debugging:**
  `gpt-oss-120b` will sometimes answer from its own training knowledge
  instead of relaying a tool result, specifically when the tool result
  conflicts with something the model "believes" — e.g. given a `user_question`
  tool result answering a general-knowledge question, it may answer from its
  own knowledge and drop the actual provided answer, rather than relaying it
  faithfully. This surfaced as a flaky test using answerable questions
  ("what is 2+2") with injected fake answers; it did not reproduce with
  unanswerable questions the model has no prior belief about. **Relevant
  beyond testing:** in Milestone 4+, a real user's answer to `user_question`
  could analogously get second-guessed or overridden if it conflicts with
  something the model assumes it already knows. Worth an explicit
  instruction-level safeguard when `user_question` goes live with real users
  (e.g. "treat the user's literal answer as authoritative, do not second-guess
  it against your own assumptions") — don't assume this is purely a test
  artifact that disappears once real users are involved.
- **Rate limiting lives inside the API-call wrapper itself** (e.g. a
  `_call_with_retry` function in agent/loop.py), not bolted onto individual
  test call sites. Every call through AgentLoop — tests, and later the real
  multi-sandbox production path — is automatically serialized/throttled this
  way. Do not add per-test rate-limit boilerplate; if a test needs one, the
  wrapper is broken, fix it there. Retry on 429 and 5xx (transient); never
  retry on other 4xx — a bad request won't fix itself by retrying, it needs
  to be fixed.




## Architecture invariant
The agent loop (calls the LLM, decides what to do) and the sandbox (where
code actually executes) stay in separate modules (agent/ vs sandbox/) even
though there's now only one sandbox implementation (local subprocess,
permanent — containerized sandboxing was evaluated and explicitly dropped,
see decision note in the roadmap). Keep the interface (`sandbox.create_shell()`,
`sandbox.run_in_shell(id, cmd)`) rather than inlining subprocess calls
directly into agent/ anyway — it's what made the LLM-provider swaps
(Anthropic→Gemini→Groq) contained changes instead of rewrites, and the same
property is worth keeping even with no second sandbox implementation planned.

**No isolation boundary exists between agent-run shell commands and the host
machine.** This is a stated, deliberate trade-off for a solo project with one
trusted user issuing tasks — not a gap to quietly forget about. Milestone 2
adds what containment IS achievable without a VM/container boundary (working-
directory path containment, command visibility) — treat that as harm
reduction, not equivalent security. Do not present this project as
production-safe for untrusted or multi-user input without revisiting this.

## Tool shapes (fixed from Milestone 1 — don't redesign later)
- `create_shell()` -> shell_id ; `run_in_shell(shell_id, cmd)` — stateful,
  supports parallel shells. NOT a single stateless run_shell(cmd) call.
- File editor: read_file, write_file, create_file, delete_file, undo. All
  five must validate the target path resolves inside the session's assigned
  working_dir — no `../` escape, no absolute path outside it (Milestone 2).
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

**This "no mocking" rule applies to tests/functional/ specifically** (proving
the agent loop actually works end to end). It does NOT mean mocking is banned
everywhere — error-handling logic like `_call_with_retry`'s behavior on a
429/500/400 needs synthetic/mocked exceptions to test reliably, since you
can't make a live API return a 500 on demand. Those belong in tests/unit/
(or similarly separated from tests/functional/), testing the retry logic in
isolation, not standing in for proof the real agent loop works.

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
milestones. For the two or three genuinely hard milestones (GitHub App auth
flow, browser control, deployment), consider "ultrathink" in the planning
prompt for extra reasoning depth — not needed for the straightforward ones.