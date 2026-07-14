"""
pipeline/runner.py
====================
Task 3.2's tracing integration + single-batch runner, extracted from
langgraph_pipeline.py (Phase 2b of docs/REFACTOR_PLAN.md).

Phase 4 of PHASED_PLAN.md: tracing goes through the TracingContext facade
(facades/tracing_context.py) instead of calling tools.tracing.trace_run()
and reaching into the raw Trace object directly. Patch
`pipeline.runner.TracingContext` when testing this module.
"""

from __future__ import annotations

from facades.tracing_context import TracingContext
from pipeline.state import BatchState


def run_one_batch(graph, verses_batch: list[dict], meter: str, thread_id: str) -> dict:
    """Invoke the graph for one (input file, meter) batch under a trace,
    with a caller-supplied LangGraph-namespaced thread id.
    """
    with TracingContext.start(label=meter, thread_id=thread_id) as tracing:
        run_config = tracing.run_config(thread_id)
        print(
            f"[*] trace_id='{tracing.trace_id}' (inspect with: python -m tools.trace_report --trace {tracing.trace_id})"
        )

        try:
            existing_state = graph.get_state(run_config)
        except Exception:
            existing_state = None

        initial_state: BatchState = {
            "verses": verses_batch,
            "meter_name": meter,
            "pass_number": 1,
            "locked": [],
            "broken": [],
            "structurally_incompatible": [],
            "drafts": {},
            "report_path": None,
            "thread_id": thread_id,
        }

        if existing_state and existing_state.next:
            print(
                f"[*] Resuming interrupted LangGraph thread '{thread_id}' (next step: {existing_state.next})...."
            )
            result = graph.invoke(None, config=run_config)
        else:
            result = graph.invoke(initial_state, config=run_config)

        print(
            f"[*] trace summary: python -m tools.trace_report --trace {tracing.trace_id}"
        )
        return result
