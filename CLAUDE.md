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


## Known open model-behavior risks (Milestone 2 investigation)
Two separate risks, different mechanisms, different blast radius. Neither is
fully mitigated. Both are documented here so Milestone 3+ doesn't rediscover
them as "flaky tests" and paper over them.

### Risk 1 — `user_question` tool-result drop ("Mode B")
After a `user_question` ask resolves, `gpt-oss-120b` sometimes fails to attend
to the `role: tool` result at all: `finish_reason='stop'`, `tool_calls=[]`,
re-emitting the prior user turn's text instead of the answer. Same underlying
behavior as the Milestone 1 note above, but the failure is total rather than a
substitution — the answer never enters the response.

**Mitigation in place:** `_USER_ANSWER_GUARD` in `agent/loop.py` is appended
to the answer at the success point in `_dispatch`, so it rides inside the tool
content. No extra API call, no mid-conversation role. It is applied there and
not in `run()`'s tool-result append specifically so it can never land on
`_execute_tool`'s `"ERROR executing user_question: ..."` string.

**What the evidence does and does not support. At least 32** post-guard
observations of `test_user_question_multi_turn_no_stale_response`, 0 failures —
13 isolated runs and 4 full-suite runs before the final 15-run isolated batch.
That count is a **floor, not a tally**: the record preserves those runs but not
the repeated-run shell loops in between, so the true number is higher and
unknown. At n=32 it rules out a ~20% rate (P(0/32 | p=0.2) ≈ 0.08%) but cannot
rule out something in the 2–9% range (95% upper bound ≈ 8.9%). "Post-guard"
rests on the session record, not on git history — and the reason is stronger
than commit timing: the guard was wired in before the first run, but its
*placement* changed mid-batch, from `run()`'s tool-result loop to `_dispatch`'s
success point, which is what the committed code has. Git cannot vouch for what
any pre-batch run executed. **Every one of those observations was
under the test's adversarial `system_instruction` override**, which orders the
model to reproduce the tool result verbatim — conditions specifically designed
to suppress this exact failure. **The production rate under the default system
instruction has not been measured.** An attempt to measure it was lost to Groq
TPD exhaustion and a probe bug, not to a clean result. Treat this as open, not
fixed.

**If it manifests in Milestone 4:** build the bounded detect-and-retry that was
scoped but deliberately not built — a single re-prompt when the final content
following a `user_question` tool result does not reference the answer, attached
to the real WebSocket-facing `user_question` flow. Do not build it
speculatively now, and do not treat 32 green observations as proof it's
unnecessary.

### Risk 2 — transcription corruption on exact-value tokens
**Separate mechanism, and not scoped to `user_question`.** Observed cleanly: the
model was given `ZXQ-4471-KESTREL` in a tool result, attended to it, used it,
and emitted `ZXT-4471-KESTREL` — one character substituted, the same
substitution in all three runs whose turns were captured. That same run also
wrote to a file on disk and the file did **not** contain the correct value; the
probe's file check is "the exact answer is in the file", which is also false
when no file was written, so that is an inference from the `write_file` call
having happened, not an observation. The file's text was not captured either
way, so the exact character it wrote is inferred too. Nothing in the run
flagged a write error, and nothing in the loop could have: the bad value
would reach disk silently. The surrounding digits (`4471-KESTREL`) survived
intact; only the third character flipped.

Working hypothesis: an unusual low-probability token gets sampled to a more
likely near-neighbour. In that batch the probe's own counters scored runs 1–11
as 5 exact / 6 not-exact (its SUMMARY read 5 pass / 7 fail / 3 no-ask over 15
runs, FAILING RUNS `[2, 5, 6, 7, 9, 10, 12]`). Run 12 is excluded here as
rate-limit-contaminated — its raw final is a 429 TPD `RateLimitError`, and its
28.86s runtime shows the retries — as are runs 13–15, which the probe scored
NO_ASK. Those exclusions are mine, not the script's. Of the 6 not-exact runs,
three (7, 9, 10) still had their raw turns and all three are the
single-character substitution above. The other three (2, 5, 6) were scored
not-exact by the same counters, but their per-turn detail was lost to output
truncation, so they are **not** confirmed as this mechanism. Runs 1–6 lost
their turns entirely, so the Risk 1 drop signature could not be checked in
them at all; no run whose turns survived showed it. That batch also had a probe
bug — the model asks `user_question` twice and only the first ask was answered,
so an error string entered the history — and the corruption is not *directly*
explained by it: in every affected run whose turns were captured, the model saw
the correct value in turn 1's tool result, and the wrong value it emitted is not
the error string copied. Whether the interrupted second ask contributed to the
substitution is not established. The sample is small and the answer token was
deliberately exotic, so the *rate* is unknown; the *mechanism* is real.

**Why this matters more than it looks.** Wherever this agent transcribes an
exact string that a user or a tool supplied, a silent single-character change is
worse than a crash — it produces plausible, wrong output that nothing
downstream can flag. Concretely: **Milestone 3's GitHub integration** (commit
messages, branch names, file contents, anything a SHA, token, or path is echoed
into), and any future deploy or credential-echoing path. No mitigation is
built. Watch for it. If it shows up where exactness matters, the fix belongs at
the transcription boundary — verify a supplied exact value round-trips, or have
the model copy from a verbatim-constrained channel — not in a retry.




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