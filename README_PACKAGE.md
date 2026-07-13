# What's in this package

This zip contains only what was actually reconstructed from the recovered
session dump (`content.md`). It is **not** a runnable project by itself —
it's the delta the LangGraph migration produced, meant to be dropped into
your existing (old, DeepAgents-based) project checkout.

## Files fully present and ready to use as-is

- `langgraph_pipeline.py` — complete, new file. Nothing missing.
- `scripts/contract_test_pass_loop.py` — complete, new file.
- `scripts/contract_test_max_passes.py` — complete, new file.
- `scripts/contract_test_advisory_stage.py` — complete, new file.
- `Session_4_Handover.md` — complete, copied verbatim from the dump.

These four Python files are genuinely done — no placeholders, no stubs, no
`TODO`s. You can drop them straight into your project root / `scripts/`
directory.

## Files that are PARTIAL — you need to merge them

- **`main.py.PATCH_SNIPPET.py`** — **this is not `main.py`.** The session
  dump only ever contained *"the exact containing blocks in `main.py` that
  integrate the `--engine` CLI selection flag"* — i.e. one new helper
  function (`_extract_engine_flag`) and a full replacement of the existing
  `main()` function. Your original, pre-migration `main.py` (with
  `build_agent()`, `PROJECT_ROOT`, `MAX_CORRECTION_PASSES`, the `sys`/`json`/
  `pathlib` imports, etc.) was never re-dumped in this session and is **not**
  in this package.

  **What to do:** copy your existing `main.py` from the old DeepAgents
  project into this new project root, then apply the two edits described at
  the top of `main.py.PATCH_SNIPPET.py` (add the new function, replace
  `main()`), then delete the `.PATCH_SNIPPET.py` file. I did not attempt to
  merge this myself since I don't have your original `main.py` to merge it
  into — merging blind risks silently dropping something the plan's
  "Minimum Change Rule" for Task 3.3 explicitly says must be preserved
  untouched.

## Files/directories referenced but NOT included at all — copy these from your old (DeepAgents) project checkout

Everything below is imported or read by `langgraph_pipeline.py` / the
contract test scripts but was never dumped in this session (it's existing,
unmodified project code per the plan's "Locked Decisions" — these paths were
explicitly marked "no changes" or "reused unchanged"):

| Path | Why it's needed |
|---|---|
| `main.py` (base file) | See above — merge target for the patch snippet |
| `backends/model_provider.py` | `get_model()` — used unchanged (Locked Decision) |
| `config/meter_tables.py`, `config/__init__.py` | Meter schema data, untouched per plan scope |
| `subagents/diacritizer.py` | `DIACRITIZER_SYSTEM_PROMPT` — reused verbatim |
| `subagents/irab_checker_agent.py` | `IRAB_SYSTEM_PROMPT`, `IRAB_BATCH_SYSTEM_PROMPT` — reused verbatim |
| `subagents/naturalness_critic.py` | `NATURALNESS_SYSTEM_PROMPT`, `NATURALNESS_BATCH_SYSTEM_PROMPT` — reused verbatim |
| `subagents/__init__.py` | Package init |
| `tools/prosody_tools.py` | `verify_batch_tool`, `verify_single_verse_tool`, `meter_schema_tool` |
| `tools/dataset_tools.py` | `commit_verse_tool`, `log_unresolved_tool` |
| `tools/reconciliation_tools.py` | `reconcile_case_ending_tool` |
| `tools/advisory_ledger.py` | `record_locked_verse_tool`, `read_ledger_tool` |
| `tools/advisory_batch.py` | `build_batched_advisory_payload_tool` |
| `tools/alignment_guards.py` | `validate_advisory_batch_alignment`, `validate_naturalness_batch_alignment` |
| `tools/tracing.py` | `trace_run`, `current_trace` |
| `tools/context_tools.py`, `tools/sanitization_tools.py`, `tools/session_bundle.py`, `tools/trace_report.py`, `tools/__init__.py` | Listed in the target directory scaffold; not directly imported by `langgraph_pipeline.py` but part of the existing toolset the plan says stays untouched |
| `tests/` (all files) | Protected, read-only regression suite — must stay as-is |
| `verification/arabic_prosody_feedback.py` | The pyarud feedback engine — explicitly out of scope for this migration |
| `dataset/inputs/*.jsonl`, `dataset/verses.jsonl` | Your actual data |
| `docs/ARCHITECTURE.md`, `docs/CONFIGURATION.md`, `docs/TRACING.md` | Existing docs |
| `AGENTS.md`, `README.md`, `requirements.txt`, `requirements-dev.txt`, `.gitignore` | Project root files |

Empty directories `dataset/inputs/`, `logs/disagreements/`, and `workspace/`
are pre-created in this package as placeholders (with the plan's expected
`.gitkeep` markers) but contain no actual data — copy your real dataset and
any existing logs in over these.

## Not yet produced (still ahead of you, per the plan)

- `docs/BUG_REPORT_langgraph_migration_RESOLUTION.md` — Task 4.4, not started (Phase 4 is still "In Progress" per the handover).
- Any live trace output — Task 4.1 hasn't been run yet; that requires real API credentials.

## How to actually test this locally

1. Copy your existing project (everything in the "copy from old project"
   table above) into a fresh directory.
2. Overlay this package's `langgraph_pipeline.py` and `scripts/` on top.
3. Merge `main.py.PATCH_SNIPPET.py` into your `main.py` as described above,
   then delete the snippet file.
4. From the project root:
   ```bash
   pytest tests/ -v                                      # expect: 18 passed, 2 skipped
   PYTHONPATH=. python3 scripts/contract_test_pass_loop.py
   PYTHONPATH=. python3 scripts/contract_test_max_passes.py
   PYTHONPATH=. python3 scripts/contract_test_advisory_stage.py
   python3 main.py 3VERSES_1919_batch_00.jsonl --engine=langgraph   # expect: openai.OpenAIError: Missing credentials (no live keys yet)
   ```

## One more flag from my earlier review

Per the migration plan (`PHASED_PLAN_v3_LangGraph_Migration.md`, Section 7),
Human Checkpoints were required — with an explicit `"CHECKPOINT — ..."`
message and a wait for your "proceed" — before Task 1.4 (the routing edge)
and before Task 3.3 (editing `main.py`). No such checkpoint messages appear
anywhere in the recovered session dump, even though the handover marks
Phases 1–3 complete. Nothing here is technically broken because of that, but
if you want to hold the process to the plan's own rules, it's worth treating
those two gates as still-open before you rely on this as final.
