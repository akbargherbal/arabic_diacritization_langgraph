"""
pipeline/
==========
Phase 2 of docs/REFACTOR_PLAN.md: extraction target for langgraph_pipeline.py.

langgraph_pipeline.py started as a single 890-line file mixing state schema,
JSON-cleanup helpers, the diacritize-batch stage, advisory-stage
construction, graph assembly, and the run_one_batch entrypoint (see
GRAPH_REPORT.md's "LangGraph Orchestration Pipeline" community, cohesion
0.06 -- the low score reflecting exactly this mix of unrelated concerns).

This package holds the pieces that are safe to extract without touching
langgraph_pipeline.py's public monkeypatch surface (`patch.object(lp, "...")`
in tests/ and scripts/): pure, stateless helpers with no test ever patching
them directly.

The stage-level functions (_diacritize_batch, make_advisory_stage,
resolve_and_commit, build_graph, run_one_batch, and the various
run_irab_checker_*/run_naturalness_critic_* wrappers) remain in
langgraph_pipeline.py for now -- moving those requires simultaneously
rewriting every test/script patch target that currently assumes they live
directly on the langgraph_pipeline module, which is a larger, separately
reviewed follow-up (Phase 2b in docs/REFACTOR_PLAN.md), not bundled in here.
"""
