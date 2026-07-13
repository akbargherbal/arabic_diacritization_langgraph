# Phased Plan v4 — Diacritizer Refactor

**Purpose of this document:** long-term memory across stateless sessions.
This file tells the next session *what the whole project is trying to
achieve and what's already been done*. It is paired with a
`Session_N_Handover.md` (short-term memory: what happened *last* session).
At the start of every session, both documents should be attached/pasted.

**Flexibility note:** phases are not required to map 1:1 to sessions. One
phase may span several sessions (obstinate bugs, unclear design) or several
phases may complete in one session, depending on how the work goes. Session
boundaries are recorded in the handover files, not in this plan.

---

## Background — why this refactor

Original objection (see project brief, session 1): the existing
DeepAgents/LangGraph pipeline over-engineered the diacritization step by
having the diacritizer subagent reason about meter feet mechanically
(bit-pattern scansion, zihaf tables) and by dispatching **one verse per
model call** instead of a full batch. A single-line Arabic prompt against a
plain WebUI model, given a whole batch as one JSON payload, outperformed
this pipeline on both speed and correctness. The fix is not "throw out
verification" (pyarud's deciding-gate role, lock-on-pass, max-3-passes, and
the advisory-only grammar/naturalness checks in `AGENTS.md` are sound and
being kept) — it's specifically: (a) make the diacritizer's job as close to
"just diacritize, in one batch call" as possible, (b) stop teaching it
meter arithmetic it doesn't need to do its job, (c) add a dedicated
formatter/salvage agent instead of overloading the diacritizer's retry path
with format-recovery duties, and (d) resolve the DeepAgents/LangGraph
documentation confusion so the next session isn't re-litigating "which
framework is this."

## Codebase facts established during planning (Session 1)

- The runtime pipeline (`main.py` → `langgraph_pipeline.py`) is **pure
  `langgraph.graph.StateGraph`**. DeepAgents is explicitly retired — the
  docstrings say so. Confusion came from stale prose in `README.md`,
  `docs/TRACING.md`, and `docs/CONFIGURATION.md`, not from the running code.
- `AGENTS.md` is human/agent-onboarding memory; nothing in the code loads
  it at runtime.
- `docs/CONFIGURATION.md`'s "Sandbox Filesystem & Security" section
  (`CompositeBackend`, per-path permission gateways) describes a
  DeepAgents-era mechanism that **no longer exists in code**. The current
  protection model is simpler: subagents are only ever given the tools
  `bind_tools([...])` explicitly grants them — the diacritizer gets
  `meter_schema_tool` (read-only) and `read_workspace_file` (hard-restricted
  to paths under `workspace/`); `irab_checker`/`naturalness_critic` get *no*
  tools at all. Dataset writes go exclusively through `commit_verse_tool`.
  No file in the current code implements a `CompositeBackend` or
  path-permission gateway.
- `docs/TRACING.md` describes trace attribution via watching a `task` tool
  call and reading its `subagent_type` argument
  (`tools/tracing.py:DISPATCH_TOOL_NAME = "task"`). The current
  `langgraph_pipeline.py` does **not** dispatch subagents through a `task`
  tool at all — it calls Python functions directly from graph nodes
  (`make_dispatch_diacritizer`, `run_irab_checker_batch`, etc.). This means
  the per-agent attribution in `tools/trace_report.py`'s output (the
  `agent` column showing `diacritizer`, `irab_checker_batch`, etc.) is
  likely **not actually working** under the current architecture and would
  silently attribute every LLM call to `orchestrator`. This is a **real
  functional bug**, not just stale docs — flagged for its own phase
  (see Phase 8), not fixed as part of Phase 0's docs-only cleanup.
- The diacritizer's system prompt (`subagents/diacritizer.py`) taught the
  model to read `correction_report` text as raw `'1'`/`'0'` bit patterns,
  but `verification/arabic_prosody_feedback.py`'s `binary_to_ux()` actually
  renders reports in `U`/`_` notation before they ever reach the model.
  That block was describing an encoding the model would never actually see
  in a report — fixed in Phase 1.
- `dispatch_diacritizer` in `langgraph_pipeline.py` calls the diacritizer
  **once per verse** via `ThreadPoolExecutor`, not once per batch — the
  direct cause of the "not one verse at a time!" objection. Fixed in Phase 1.
- Batches in `dataset/inputs/*.jsonl` are small (mostly 12 verses, a few
  3-verse smoke-test files) — batch-mode dispatch is well within normal
  context/output limits for a single call.

---

## Phase status legend
`[ ]` not started · `[~]` in progress · `[x]` complete & checkpoint verified

---

### Phase 0 — Documentation & terminology cleanup (no logic changes) `[x]`
- Fix `README.md`'s `cd arabic_diacritization_deepagent` → correct repo dir name.
- Fix `README.md`'s doc-index description of `docs/CONFIGURATION.md`.
- Rewrite `docs/CONFIGURATION.md` Section 2 to describe the *actual* current
  protection model (tool-grant-based, not a runtime sandbox), with the old
  `CompositeBackend` design kept as clearly-labeled historical context (it's
  still useful — it's the rationale behind `AGENTS.md` rule 3).
- Fix stale function names in `docs/CONFIGURATION.md` (`build_agent` →
  `build_langgraph_pipeline`, `build_studio_batch_agent` → `build_studio_graph`).
- Fix `docs/TRACING.md`'s `LANGCHAIN_PROJECT` example name, and add a flagged
  note that Section 2's `task`/`subagent_type` attribution mechanism doesn't
  match the current dispatch shape (see Phase 8) — noted, not fixed here.
- Add a one-line clarifying note to `AGENTS.md` about its own role (persistent
  memory, not runtime-loaded).
- **Checkpoint**: `grep -ri deepagent .` (excluding `.git` and this plan file)
  returns only clearly-labeled historical/rationale mentions. Human review of
  the corrected docs for accuracy.

### Phase 1 — Redefine the diacritizer's contract: batch-in, batch-out `[x]`
**Live-run checkpoint CLOSED in Session 2 (continued).** First-ever successful
end-to-end run: `python main.py 3VERSES_1919_batch_00.jsonl` completed
without crashing, diacritizer batch call → verify → advisory → commit all
worked. 2/3 verses locked at 100% and committed to `dataset/verses.jsonl`;
1/3 (`1919-5`) rejected as `structurally_incompatible` at 93% (see Phase 2
notes below — open question, not a blocker). Diacritizer token usage for
this batch: 15,818 total tokens (2,978 in + 12,840 out) — comfortably under
the 65,536 cap, no truncation. This required one real bug fix, logged
below — the checkpoint is closed *with* that fix applied, not against the
original Phase 1 code.
- Replace `dispatch_diacritizer`'s per-verse `ThreadPoolExecutor` fan-out with
  a single model call carrying the *entire* pass's target-verse array as one
  JSON payload; parse back a JSON array, not N separate objects.
- Rewrite `DIACRITIZER_SYSTEM_PROMPT`: drop the exhaustive foot/zihaf tables
  (available programmatically via `meter_schema_tool`) and the "trial-and-error
  match your bit pattern" scaffolding; keep only the output-shape contract,
  the hard letter-preservation constraint, and a corrected (U/_, not 1/0)
  explanation of how to read a correction_report.
- Raise the per-call token budget to accommodate a full batch response instead
  of a single verse (was tuned for one verse's ~20.5k observed reasoning
  tokens; a 12-verse batch call needs headroom — treated as a starting
  estimate to validate against `tools/trace_report.py` telemetry, not a
  guaranteed-correct number).
- **Checkpoint**: Run the existing test batches
  (`dataset/inputs/*_batch_00.jsonl`) end-to-end via `python main.py`; compare
  pass-1 pyarud pass rate, latency, and token usage (via
  `python -m tools.trace_report`) against the pre-refactor baseline. Confirm
  no truncated/malformed batch responses at the new token budget. Human
  judges whether output quality holds.

### Phase 2 — Simplify the verify → correct loop `[~]`
- Confirm `verify_batch_tool`'s correction_report still gives the (now
  batch-mode) diacritizer enough signal per broken verse without
  over-specifying mechanics.
- **Session 2 (continued) finding — open question, not yet investigated:**
  during the first successful live run, verse `1919-5` scored 93% and was
  auto-routed to `structurally_incompatible` (permanent rejection, no
  pass-2 retry) rather than `broken` (gets a retry), because
  `verify_batch_tool` treats any `extra_bits is not None` as structural/
  letter-level regardless of how small. pyarud's own report on this verse
  described the fix as a foot reweight + removing one trailing mora —
  language that reads as diacritic-fixable, not letter-level. Unclear
  whether the classifier is correctly conservative here or too aggressive
  on near-misses. User deferred investigation — pick this up before
  finalizing Phase 2's verify→correct loop simplification, since it's the
  same code path (`verify_batch_tool`'s locked/broken/structurally_incompatible
  triage in `tools/prosody_tools.py`).
- **Session 2 finding (not yet acted on):** `generate_poem_correction_report`
  duplicates every fix prescription verbatim — once inline per-verse under
  "CORRECTION PRESCRIPTION", again in the poem-level "CONSOLIDATED FIX LIST"
  at the end — and renders box-drawing expected/actual grids that repeat
  what's already stated in plain prose in "DETAILED DIAGNOSIS". Measured on
  a real 3-verse worst-case batch (`3VERSES_1919_batch_00.jsonl`, fully
  undiacritized input): report was 14,291 chars (~3.6k tokens). Candidate
  trim: drop the CONSOLIDATED FIX LIST section (redundant with per-verse
  prescriptions) and/or the box-drawing grids (redundant with DETAILED
  DIAGNOSIS prose). Not implemented — needs a live before/after comparison
  to confirm no signal loss, and Phase 1's own live baseline isn't
  confirmed yet either (see Known Issues).
- **Checkpoint**: `tests/test_alignment_guards.py`, `test_prosody_tools.py`,
  `test_advisory_ledger.py` still pass; no regression in unresolved-verse rate
  across a fixed sample of 5 batches vs. the Phase 1 baseline.

### Phase 3 — Introduce the formatter/salvage agent `[ ]`
- New agent, invoked only when the diacritizer's batch JSON fails
  parsing/structural checks (missing verse IDs, wrong count, broken JSON) —
  distinct from the diacritizer itself.
- Job: salvage what's fixable (JSON syntax, ID reattachment) and use judgment
  to decide "formatting problem" vs. "diacritization problem." On the latter,
  return actionable feedback (e.g. "input 12 verses, got 10, verses 2 and 8
  missing") rather than dumping raw logs back at the diacritizer.
- **Checkpoint**: Feed it deliberately malformed synthetic outputs (missing
  verse, broken braces, truncated array) and confirm correct classification
  of each as salvageable-format vs. needs-diacritizer-retry, with no verse
  silently dropped.

### Phase 4 — Tighten the irab/grammar feedback contract `[ ]`
- Verify `subagents/irab_checker_agent.py`'s feedback matches the "tell them
  what's wrong, not how to fix it" style (بحر البسيط تفعيلاته / الزحافات
  الجائزة framing) rather than prescriptive corrections.
- **Checkpoint**: Sample 10 flagged verses; confirm feedback text names the
  problem/meter foot without stating the corrected diacritic.

### Phase 5 — Confirm batch-level grammar review behavior `[ ]`
- Verify the final grammar-plausibility pass (intermediate/middle-school-level
  errors, e.g. ḍamma-instead-of-kasra on a genitive) approves the whole batch
  when clean, and — when not — annotates only the questionable verse(s)
  without rejecting or blocking the batch.
- **Checkpoint**: Run one batch with an intentionally injected case-ending
  error; confirm only that verse is flagged in `logs/disagreements/`, the
  rest of the batch still commits.

### Phase 6 — Arabic-prompt experiment (diacritizer only, to start) `[ ]`
- Fork the Phase 1 batch-mode diacritizer prompt into an Arabic-only variant
  modeled on the working one-liner from session 1's brief.
- Run both English and Arabic variants against the same 5-10 batches; compare
  pass-1 pyarud pass rate, needs_review rate, and latency.
- **Checkpoint**: Side-by-side result table; human decides English, Arabic,
  or both behind a config flag.

### Phase 7 — Cleanup pass `[ ]`
- Remove dead code paths made obsolete by Phases 1-3 (old per-verse dispatch
  helpers if any remain, superseded comments about DeepAgents-era
  constraints), update `docs/ARCHITECTURE.md` and `README.md` to match the
  final shape.
- **Checkpoint**: Fresh read-through of README + ARCHITECTURE.md; no
  references to retired components.

### Phase 8 — Fix trace attribution for the LangGraph dispatch shape `[ ]`
*(Newly identified during Session 1 planning — inserted after Phase 0's
discovery, not part of the original brief.)*
- `tools/tracing.py`'s per-agent attribution logic watches for a `task` tool
  call and reads `subagent_type` from it — a DeepAgents-era pattern that the
  current LangGraph node-function dispatch never produces. Diagnose whether
  `python -m tools.trace_report` is in fact currently mis-attributing every
  LLM call to `orchestrator`, and if so, adapt the callback to attribute by
  LangGraph node/run context instead of the retired `task`-tool convention.
- **Checkpoint**: A batch run with mixed diacritizer/irab/naturalness calls
  produces a `trace_report` breakdown with correct per-agent rows (matching
  the example table in `docs/TRACING.md`), verified against manual reasoning
  about which calls happened.

---

## Log of completed sessions
*(Append one line per session here as a running index; full detail lives in
that session's `Session_N_Handover.md`.)*

- Session 1: Repo inspected, objections mapped to code, this plan written and
  approved. Phase 0 and Phase 1 executed and checkpointed.
- Session 2: No live model access available (no API key, restricted network
  egress) — Phase 1's live-run checkpoint still open. Re-verified Phase 0/1
  via fresh clone (grep + pytest + all 3 contract scripts, all pass, no
  drift). Did non-live Phase 2 prep: read `verify_batch_tool` +
  `generate_poem_correction_report`, ran it against a real batch via pyarud
  (deterministic, no LLM needed), found real duplication/verbosity in the
  report (see Phase 2 notes above).
  **Continued same session, user ran live locally:** first-ever successful
  end-to-end live run after fixing a real bug in `_extract_json` (model
  responses with conversational preamble before/around a JSON fence weren't
  being parsed — old regex only matched if the fence was the entire
  string). Fix verified against the real captured failing content plus 4
  other cases, no regressions in pytest/contract scripts. Phase 1's
  live-run checkpoint is now closed. Surfaced a new open question during
  that run (see Phase 2 notes: `1919-5` near-miss auto-routed to
  `structurally_incompatible`) — logged, not investigated yet per user's
  call. Fix given to user to commit/push; push not confirmed in-session.
