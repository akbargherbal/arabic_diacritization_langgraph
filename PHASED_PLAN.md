## Phased Plan

### Phase 0 — Build the safety net (do this before touching anything)

- Port `scripts/contract_test_*.py` into real `tests/test_pipeline_*.py` pytest modules (same fakes, just under pytest so they run in CI and with coverage).
- Add characterization tests around `build_graph()` / `run_one_batch()` that snapshot output for a few representative verses (locked, unresolved, needs_review paths) so later refactors have something to diff against.
- Run `graphify` again after this phase purely as a baseline snapshot to compare against later phases.

### Phase 1 — Kill the confirmed duplication (cheap, low-risk)

- Move `AR_CHARS`/`normalize_text` into a single shared module (or make `sanitization_tools.py` expose them publicly) and delete the duplicate in `fidelity_tools.py`. Remove the TODO.
- Formally document the Rule 6 tanwīn fatḥ / alif maqṣūra bug: turn the loose reference into a tracked test case (even an expected-fail one) so it stops being an isolated, invisible node.

### Phase 2 — Break up `langgraph_pipeline.py`

This file currently mixes: state schema, JSON-cleanup helpers, the diacritize-batch stage, advisory-stage construction, graph assembly, and the CLI-style entrypoint. Split along those seams into something like:

```
pipeline/
  state.py          # BatchState, merge/append helpers
  json_utils.py      # _cleanup_json_text, _extract_json
  diacritize_stage.py # _diacritize_batch, make_dispatch_diacritizer
  advisory_stage.py  # make_advisory_stage, resolve_and_commit
  graph.py           # build_graph, build_langgraph_pipeline, build_studio_graph
  runner.py          # run_one_batch (the actual entrypoint)
```

Each module gets its own focused tests, reusing the Phase 0 fakes.

### Phase 3 — Decompose `verification/arabic_prosody_feedback.py`

Same problem, same fix: 1392 lines and cohesion 0.06 means it's several loosely-related things filed together — meter/foot analysis, health scoring, and UX formatting (`binary_to_ux`, `_enrich_foot`, `_enrich_hemistich`) look like three separate concerns. Split into `analysis.py`, `scoring.py`, `reporting.py` (or similar), each independently testable.

### Phase 4 — Put a face on the cross-cutting infra

Rather than every stage calling `trace_run()` / `TokenTracingCallback` / `record_locked_verse_tool()` / `read_ledger_tool()` directly, introduce two thin facades:

- A `TracingContext` (wrapping `TraceStore`/`TokenTracingCallback`) so pipeline code depends on one small interface, not the tracing internals.
- A `LedgerClient` wrapping the lock/read/commit ledger calls, so dataset-writing code stops reaching into `dataset_tools.py` internals directly.

This is the phase that should visibly reduce the god-node edge counts on the next graph run.

### Phase 5 — Consolidate the batch/fallback duplication

The batched-vs-sequential fallback logic (`irab_checker_batch`→`irab_checker`, `naturalness_critic_batch`→`naturalness_critic`) is implemented twice in the pipeline with the same alignment-guard-then-fallback shape. Extract one parametrized `run_with_batch_fallback(batch_fn, single_fn, alignment_guard)` helper and use it for both, per the architecture doc's own description of the pattern.

### Phase 6 — Verify and lock in the gains

- Re-run `graphify` and compare: cohesion scores on the split communities should rise meaningfully above 0.06–0.09, and the top god nodes should show fewer edges.
- Wire up the 14 isolated nodes (mostly `langgraph.json`/config keys and the Rule 6 bug) into docs or tests so they stop showing up as "possible documentation gaps."
- Update `docs/ARCHITECTURE.md` to reference the new module boundaries so the doc and the graph stay in sync.

---

**Suggested order of execution:** 0 → 1 → 2 → 4 → 5 → 3 → 6. Phase 3 is independent of the pipeline work and lower risk, so it can slide later or run in parallel if you have two people on it; everything else has real sequencing dependencies (you need the safety net before splitting files, and you need `langgraph_pipeline.py` split before the tracing/ledger facades have a clean place to be injected).
