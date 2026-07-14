# Codebase Refactoring Plan

This document tracks the phased refactoring of the Arabic Diacritization and Verification pipeline. The goals are to increase cohesion, eliminate duplication, decompose monolithic modules, and establish clean architectural boundaries.

---

## Phase Status Summary

| Phase | Description | Status | Target Modules |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Safety Net | Completed ✅ | `tests/test_pipeline_*.py` |
| **Phase 1** | Kill Confirmed Duplication | Completed ✅ | `fidelity_tools.py`, `tests/` |
| **Phase 2** | Decompose Orchestrator | Completed ✅ | `pipeline/` package, `langgraph_pipeline.py` |
| **Phase 3** | Decompose Prosody Module | Completed ✅ | `verification/prosody/` package |
| **Phase 4** | Cross-cutting Infrastructure Facades | Completed ✅ | `facades/` package |
| **Phase 5** | Consolidate Fallback Duplication | Completed ✅ | `pipeline/advisory_stage.py` |
| **Phase 6** | Verify & Lock in Gains (Wiring & Docs) | Completed ✅ | `docs/`, `tests/`, `langgraph.json` |

---

## Detailed Phase Explanations

### Phase 0 — Build the safety net
- Port `scripts/contract_test_*.py` into real `tests/test_pipeline_*.py` pytest modules.
- Add characterization tests around `build_graph()` / `run_one_batch()` that snapshot output for representative verses.
- Establish the baseline snapshot for the dependency graph.

### Phase 1 — Kill the confirmed duplication
- Move `AR_CHARS` and `normalize_text` into a single shared module, deleting the duplicate in `fidelity_tools.py`.
- Formally document and add regression tests for the `Rule 6` tanwīn fatḥ / alif maqṣūra bug.

### Phase 2 — Break up `langgraph_pipeline.py`
Decomposed the 900-line monolithic orchestrator into the `pipeline/` package structure:
- `pipeline/state.py` — BatchState and state reducers
- `pipeline/json_utils.py` — JSON cleaning and extraction helpers
- `pipeline/diacritize_stage.py` — Diacritizer dispatching node
- `pipeline/verify_stage.py` — Verification node and routing logic
- `pipeline/advisory_stage.py` — Advisory stage construction
- `pipeline/graph.py` — Graph assembly and studio initialization
- `pipeline/runner.py` — Batch runner entrypoint

*Note: langgraph_pipeline.py remains as a backward-compatible facade/re-export shim to avoid breaking tests/scripts that monkeypatch it directly.*

### Phase 3 — Decompose `verification/arabic_prosody_feedback.py`
Split the 1392-line low-cohesion monolith (cohesion 0.06) into the `verification/prosody/` package:
- `verification/prosody/scoring.py` — Pure U/_-pattern comparison, zihaf/health classification
- `verification/prosody/analysis.py` — Data models, meter names, and pyarud-driving analysis
- `verification/prosody/reporting.py` — LLM-facing correction report generation

*Note: verification/arabic_prosody_feedback.py remains as a thin facade/re-export shim for backward compatibility.*

### Phase 4 — Put a face on the cross-cutting infra
Introduced thin facades to reduce coupling and god-node centrality:
- `TracingContext` — Wraps `TraceStore`/`TokenTracingCallback`
- `LedgerClient` — Wraps advisory ledger lock, read, and commit operations

### Phase 5 — Consolidate the batch/fallback duplication
- Created the parameterized `run_with_batch_fallback(batch_fn, single_fn, alignment_guard)` helper inside `pipeline/advisory_stage.py`.
- Used this single helper to consolidate the duplicate batched-vs-sequential fallback logic across `irab_checker` and `naturalness_critic`.

### Phase 6 — Verify and lock in the gains
- Document the refactored package layout in `docs/ARCHITECTURE.md`.
- Wire up the configuration-level isolated nodes (`langgraph.json`, config keys, env) into unit tests and architectural documentation to close knowledge/documentation gaps.
- Reconstruct `docs/REFACTOR_PLAN.md` to establish in-repo documentation of the refactoring phases.
- (Local Run) Re-run `graphify` and verify cohesion improvement.
