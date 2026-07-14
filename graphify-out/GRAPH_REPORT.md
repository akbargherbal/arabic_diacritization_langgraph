# Graph Report - .  (2026-07-14)

## Corpus Check
- Corpus is ~34,951 words - fits in a single context window. You may not need a graph.

## Summary
- 381 nodes · 636 edges · 37 communities (26 shown, 11 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Prosody Meter Config & Ledger Tests
- Prosody & Arabic Metric Feedback
- LangGraph Orchestration Pipeline
- Dataset Management & Locking Rules
- Token and LLM Tracing
- Fidelity & Reconciliation Tests
- Model Provider & Retry Mechanics
- Session Bundle Execution
- Text Normalization & Formatting
- LangGraph Configuration & Environments
- Correction Pass Contract Tests
- Irab (Case Ending) Checker Subagent
- Naturalness Critic Subagent
- Pass Loop Verification Scripts
- Advisory Stage Contract Tests
- Context Summarization Tools
- Orchestration Rules & Capping
- Security Model & Verification Read-only
- Graphify Workflow Rules
- Prosody Reconciliation Loop
- Linguistic Advisor Subagents
- Refactoring Plan & Attribution Bug
- Tanwīn Fatḥ Syllable Workaround
- Diacritizer Generation Agent
- Multi-Axis Validation Core
- LangGraph Local Checkpoints
- Ephemeral Security Ledgers
- Tracing Attribution Callback
- Refactor Core Background

## God Nodes (most connected - your core abstractions)
1. `trace_run()` - 25 edges
2. `TokenTracingCallback` - 20 edges
3. `record_locked_verse_tool()` - 18 edges
4. `read_ledger_tool()` - 16 edges
5. `commit_verse_tool()` - 13 edges
6. `analyze_poem()` - 11 edges
7. `heuristic_salvage()` - 10 edges
8. `validate_advisory_batch_alignment()` - 10 edges
9. `reconcile_case_ending_tool()` - 10 edges
10. `TraceStore` - 10 edges

## Surprising Connections (you probably didn't know these)
- `build_langgraph_pipeline()` --calls--> `get_model()`  [EXTRACTED]
  langgraph_pipeline.py → backends/model_provider.py
- `_current_thread_id()` --calls--> `current_trace()`  [EXTRACTED]
  langgraph_pipeline.py → tools/tracing.py
- `_diacritize_batch()` --calls--> `heuristic_salvage()`  [EXTRACTED]
  langgraph_pipeline.py → subagents/formatter.py
- `verify_pass()` --calls--> `record_locked_verse_tool()`  [EXTRACTED]
  langgraph_pipeline.py → tools/advisory_ledger.py
- `verify_pass()` --calls--> `log_unresolved_tool()`  [EXTRACTED]
  langgraph_pipeline.py → tools/dataset_tools.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Arabic Diacritization and Verification Flow** — docs_architecture_diacritizer_agent, docs_architecture_irab_checker, docs_architecture_naturalness_critic [INFERRED 0.85]
- **Deciding Validation Rules** — agents_rule_1_pyarud_reconciliation, agents_rule_3_verification_code_read_only, agents_rule_4_dataset_writes [INFERRED 0.85]

## Communities (37 total, 11 thin omitted)

### Community 0 - "Prosody Meter Config & Ledger Tests"
Cohesion: 0.07
Nodes (52): config/meter_tables.py ======================= Ported verbatim from arabic_proso, make_advisory_stage(), clean_ledger_environment(), tests/test_advisory_ledger.py ============================== Unit tests for Lock, A test that calls read_ledger_tool(clear=True)., A test that records 2 verses, builds payload, and asserts JSON structure., Ensure ledger is clean before and after every test., A test that calls record_locked_verse_tool with a verse whose skeleton doesn't m (+44 more)

### Community 1 - "Prosody & Arabic Metric Feedback"
Cohesion: 0.06
Nodes (55): FootStatus, HealthLevel, analyze_and_report(), analyze_poem(), analyze_verse(), binary_to_ux(), _enrich_foot(), _enrich_hemistich() (+47 more)

### Community 2 - "LangGraph Orchestration Pipeline"
Cohesion: 0.06
Nodes (46): _append_unique(), BatchState, build_graph(), build_langgraph_pipeline(), build_studio_graph(), _call_advisory_model(), _cleanup_json_text(), _current_thread_id() (+38 more)

### Community 3 - "Dataset Management & Locking Rules"
Cohesion: 0.09
Nodes (34): Literal port of AGENTS.md rule 1 / ORCHESTRATOR_SYSTEM_PROMPT's     "Handling a, resolve_and_commit(), _append_locked(), _append_rejected(), commit_verse_tool(), _iter_valid_records(), _load_seen(), _lock_file() (+26 more)

### Community 4 - "Token and LLM Tracing"
Cohesion: 0.11
Nodes (7): BaseCallbackHandler, Any, Thin wrapper around a SQLite file. Safe to share across a process --     includi, Attributes every LLM call to the agent that made it by walking up the     parent, TokenTracingCallback, TraceStore, _utcnow_iso()

### Community 5 - "Fidelity & Reconciliation Tests"
Cohesion: 0.12
Nodes (20): tests/test_prosody_tools.py ============================= Regression tests for t, test_reconcile_rejects_out_of_range_word_index(), test_reconcile_rejects_unknown_harakah(), test_reconcile_reports_no_basic_mark_found(), test_reconcile_swaps_damma_to_kasra(), test_sanitizer_accepts_plain_arabic(), test_sanitizer_allows_common_punctuation(), test_sanitizer_rejects_control_char() (+12 more)

### Community 6 - "Model Provider & Retry Mechanics"
Cohesion: 0.14
Nodes (20): _awith_retry(), _build_nvidia_model(), get_model(), _is_retryable(), _patch_nvidia_error_masking(), _patch_retry(), Any, backends/model_provider.py ============================ Provider-agnostic model (+12 more)

### Community 7 - "Session Bundle Execution"
Cohesion: 0.19
Nodes (18): Connection, build_bundle(), _capture(), main(), Path, tools/session_bundle.py  Assembles a single Markdown "evidence bundle" for one t, Run a trace_report print-based function and capture its stdout., _section() (+10 more)

### Community 8 - "Text Normalization & Formatting"
Cohesion: 0.29
Nodes (10): clean_and_normalize(), heuristic_salvage(), Any, Strip Arabic diacritics (Tashkeel)., Normalize text for alignment comparisons., Attempt to deterministically align parsed items to expected_verses., strip_diacritics(), test_heuristic_salvage_exact_match() (+2 more)

### Community 9 - "LangGraph Configuration & Environments"
Cohesion: 0.33
Nodes (5): dependencies, env, graphs, batch_diacritization, .

### Community 10 - "Correction Pass Contract Tests"
Cohesion: 0.53
Nodes (5): fake_diacritize_batch(), fake_log_unresolved(), fake_verify_batch_tool(), main(), scripts/contract_test_max_passes.py ===================================== Proves

### Community 11 - "Irab (Case Ending) Checker Subagent"
Cohesion: 0.33
Nodes (5): build_irab_checker_batch_subagent(), build_irab_checker_subagent(), subagents/irab_checker_agent.py ================================= Advisory-only, Compile irab_checker as an isolated agent with no tools -- it never     needed f, Compile irab_checker_batch as an isolated agent with no tools.     Receives a JS

### Community 12 - "Naturalness Critic Subagent"
Cohesion: 0.33
Nodes (5): build_naturalness_critic_batch_subagent(), build_naturalness_critic_subagent(), subagents/naturalness_critic.py ================================= LLM-based advi, Compile naturalness_critic_batch as an isolated agent with no tools.     Receive, Compile naturalness_critic as an isolated agent with no tools -- it     never ne

### Community 14 - "Pass Loop Verification Scripts"
Cohesion: 0.60
Nodes (4): fake_diacritize_batch(), fake_verify_batch_tool(), main(), scripts/contract_test_pass_loop.py ==================================== Task 1.5

### Community 15 - "Advisory Stage Contract Tests"
Cohesion: 0.67
Nodes (3): fake_commit_verse_tool(), main(), scripts/contract_test_advisory_stage.py ========================================

### Community 16 - "Context Summarization Tools"
Cohesion: 0.50
Nodes (3): tools/context_tools.py ======================== New module (does not touch verif, Collapse a verify_batch_tool poem_result_json blob into a terse,     one-line-pe, summarize_correction_report_tool()

### Community 17 - "Orchestration Rules & Capping"
Cohesion: 0.67
Nodes (3): Rule 2: Locked Verses, Rule 5: Max 3 Correction Passes, Orchestrator Control Loop

### Community 18 - "Security Model & Verification Read-only"
Cohesion: 0.67
Nodes (3): Rule 3: Verification Code is Read-Only, Rule 4: Dataset Writes via commit_verse Only, Subagent Security Model

## Knowledge Gaps
- **14 isolated node(s):** `.`, `batch_diacritization`, `env`, `Rule 6: tanwīn fatḥ bug on alif maqṣūra`, `Diacritizer (Generative Subagent)` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `trace_run()` connect `Prosody Meter Config & Ledger Tests` to `LangGraph Orchestration Pipeline`, `Token and LLM Tracing`, `Advisory Stage Contract Tests`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Why does `TokenTracingCallback` connect `Token and LLM Tracing` to `Prosody Meter Config & Ledger Tests`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `heuristic_salvage()` connect `Text Normalization & Formatting` to `LangGraph Orchestration Pipeline`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **What connects `.`, `batch_diacritization`, `env` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Prosody Meter Config & Ledger Tests` be split into smaller, more focused modules?**
  _Cohesion score 0.06779661016949153 - nodes in this community are weakly interconnected._
- **Should `Prosody & Arabic Metric Feedback` be split into smaller, more focused modules?**
  _Cohesion score 0.05584415584415584 - nodes in this community are weakly interconnected._
- **Should `LangGraph Orchestration Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.0636734693877551 - nodes in this community are weakly interconnected._