"""
pipeline/graph.py
===================
Task 1.4/1.1's graph construction and Task 3.1's SqliteSaver checkpointing,
extracted from langgraph_pipeline.py (Phase 2b of docs/REFACTOR_PLAN.md).
"""

from __future__ import annotations

import sqlite3

from langgraph.graph import END, StateGraph

from backends.model_provider import get_model
from runtime import PROJECT_ROOT

from pipeline.advisory_stage import make_advisory_stage
from pipeline.diacritize_stage import make_dispatch_diacritizer
from pipeline.state import BatchState
from pipeline.verify_stage import route_after_verify, verify_pass

DIACRITIZER_MODEL_KWARGS = dict(
    max_completion_tokens=65536,
    max_tokens=65536,
    # Session 4 finding: model_provider.py's _REASONING_SUPPRESSION table
    # defaults deepseek to thinking DISABLED, which was Bug 2's fix (Session
    # 3) -- but for THIS callsite specifically, disabling the thinking
    # channel causes the model to relocate its scansion reasoning into
    # visible `content` instead of skipping it (Session 3's Finding 3).
    # Explicitly re-enabling thinking here routes that reasoning into
    # `additional_kwargs["reasoning_content"]` instead, confirmed via
    # _inspect_reasoning_content.py and _inspect_real_verses_thinking_enabled.py
    # against real verses. get_model()'s explicit-kwarg-wins-over-setdefault
    # contract (see model_provider.py docstring) means this overrides the
    # module default for this call only -- every other provider/callsite is
    # unaffected.
    #
    # Phase 1 (PHASED_PLAN_v4_Diacritizer_Refactor.md) note: this cap was
    # 24576, tuned for ONE verse per call (~20.5k observed reasoning tokens
    # for a single verse in the pre-refactor per-verse dispatch). Dispatch
    # is now one call per BATCH (up to ~12 verses per dataset/inputs/*.jsonl
    # file), so this is raised to 65536 as a starting estimate, not a
    # validated number -- Phase 1's checkpoint requires confirming via
    # `python -m tools.trace_report` that real batch calls aren't hitting
    # this ceiling (truncated JSON output is the symptom to watch for).
    # Adjust up (or down, if cost/latency is a concern and it's never come
    # close) once real telemetry exists.
    extra_body={"thinking": {"type": "enabled"}},
)


def build_graph(diacritizer_model, advisory_model, checkpointer=None):
    graph = StateGraph(BatchState)

    graph.add_node("dispatch_diacritizer", make_dispatch_diacritizer(diacritizer_model))
    graph.add_node("verify_pass", verify_pass)
    graph.add_node("advisory_stage", make_advisory_stage(advisory_model))

    graph.set_entry_point("dispatch_diacritizer")
    graph.add_edge("dispatch_diacritizer", "verify_pass")
    graph.add_conditional_edges(
        "verify_pass",
        route_after_verify,
        {
            "dispatch_diacritizer": "dispatch_diacritizer",
            "advisory_stage": "advisory_stage",
        },
    )
    graph.add_edge("advisory_stage", END)

    return graph.compile(checkpointer=checkpointer)


def build_langgraph_pipeline(use_checkpointer: bool = True):
    """Analogous to main.py::build_agent() -- returns (compiled_graph,
    checkpoint_conn, checkpoint_db_path) so main.py's CLI runner can treat
    both engines uniformly (Task 3.3).
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    model = get_model()
    diacritizer_model = get_model(**DIACRITIZER_MODEL_KWARGS)

    checkpointer = None
    checkpoint_conn = None
    checkpoint_db_path = None

    if use_checkpointer:
        checkpoint_db_path = PROJECT_ROOT / "checkpoints.sqlite"
        checkpoint_conn = sqlite3.connect(
            str(checkpoint_db_path), check_same_thread=False
        )
        checkpoint_conn.execute("PRAGMA busy_timeout = 30000")
        checkpoint_conn.execute("PRAGMA synchronous = NORMAL")
        try:
            cursor = checkpoint_conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            if integrity_result != "ok":
                print(
                    f"[!] Warning: Checkpoint database corrupted. PRAGMA integrity_check returned: {integrity_result}"
                )
        except Exception as exc:
            print(f"[-] Pre-run checkpoint diagnostic failed: {exc}")
        checkpointer = SqliteSaver(checkpoint_conn)

    graph = build_graph(diacritizer_model, model, checkpointer=checkpointer)
    return graph, checkpoint_conn, checkpoint_db_path


def build_studio_graph():
    """LangGraph Studio factory; Studio owns checkpoint persistence itself."""
    graph, _, _ = build_langgraph_pipeline(use_checkpointer=False)
    return graph
