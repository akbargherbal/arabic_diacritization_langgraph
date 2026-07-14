"""
langgraph_pipeline.py
======================
Raw langgraph.graph.StateGraph replacement for the DeepAgents orchestrator
in main.py. See PHASED_PLAN_v3_LangGraph_Migration.md for the full spec
this file implements (Phases 1-3, Tasks 1.1-3.3's pipeline half).

Core idea (Locked Decision, plan Section 2): the dispatch -> verify -> route
pass-loop is enforced by the SHAPE of the graph's edges, not by an LLM's
willingness to follow a "check your work before dispatching again"
instruction. route_after_verify is a plain Python function; it cannot be
"not followed" the way a system-prompt rule could.

--- Phase 2b note (docs/REFACTOR_PLAN.md) ---
This module used to hold every stage's implementation directly (state
schema, JSON parsing, the diacritize/verify/advisory stage bodies, graph
assembly, and the run_one_batch entrypoint) in ~900 lines -- exactly the
low-cohesion "LangGraph Orchestration Pipeline" community GRAPH_REPORT.md
flagged (cohesion 0.06). That implementation now lives in the `pipeline/`
package, split by responsibility:

    pipeline/state.py            BatchState + reducers
    pipeline/json_utils.py       model-output JSON parsing
    pipeline/diacritize_stage.py dispatch_diacritizer node
    pipeline/verify_stage.py     verify_pass node + route_after_verify
    pipeline/advisory_stage.py   advisory subagent calls + resolve_and_commit
    pipeline/graph.py            build_graph / build_langgraph_pipeline / build_studio_graph
    pipeline/runner.py           run_one_batch

This file re-exports the same public names for backward compatibility --
`langgraph.json`'s `batch_diacritization` graph entry and main.py's imports
both still resolve unchanged. **If you're patching internals in a test,
patch them on the `pipeline.<module>` they now live in, not here** -- e.g.
`patch.object(pipeline.diacritize_stage, "_diacritize_batch", ...)`, not
`patch.object(langgraph_pipeline, "_diacritize_batch", ...)`. Each stage
call resolves bare names via its OWN module's globals at call time, so a
patch applied only to this facade's re-exported binding has no effect on
what the stage functions actually see.

--- Design note on Task 1.2's "Send" instruction ---
The pre-coding checklist confirms `from langgraph.types import Send` is
importable (langgraph 1.2.9). However Task 1.4's success criteria requires
`dispatch_diacritizer` to appear as an actual graph NODE with exactly one
outgoing edge, fixed, to `verify_pass` -- which is incompatible with using
Send as a conditional-edge router (Send-based fan-out routes to a
DIFFERENT node per invocation, not through a single node with one fixed
outgoing edge). Given that conflict, and that per-verse parallelism is
achievable equally well inside a single node via ThreadPoolExecutor
(each verse still gets its own model call, still all in parallel, still
zero conditional dispatch logic inside the node), this implementation
dispatches per-verse calls in parallel from INSIDE the `dispatch_diacritizer`
node body rather than via graph-level Send. This satisfies every stated
Task 1.2/1.4/1.5 success criterion (one call per verse per pass, no
pass_number/verify_batch_tool/looping construct in the node, exactly one
fixed edge dispatch_diacritizer -> verify_pass). Flagged explicitly here
and in the session handover rather than silently picked -- if a future
session wants true graph-level Send fan-out (splitting this into a router
+ a `diacritize_single_verse` node), the model-call logic in
pipeline/diacritize_stage.py (`_diacritize_batch`) already isolates cleanly
for that refactor.
"""

from __future__ import annotations

from tools.tracing import current_trace

# ===========================================================================
# Re-exports for backward compatibility -- see module docstring above for
# where each of these now actually lives, and the patch-target warning.
# ===========================================================================

from pipeline.state import BatchState, _merge_dicts, _append_unique  # noqa: F401
from pipeline.json_utils import _cleanup_json_text, _extract_json  # noqa: F401

from pipeline.diacritize_stage import (  # noqa: F401
    read_workspace_file,
    _diacritize_batch,
    make_dispatch_diacritizer,
)
from pipeline.verify_stage import verify_pass, route_after_verify  # noqa: F401
from pipeline.advisory_stage import (  # noqa: F401
    _call_advisory_model,
    run_irab_checker_batch,
    run_naturalness_critic_batch,
    run_irab_checker_single,
    run_naturalness_critic_single,
    resolve_and_commit,
    make_advisory_stage,
)
from pipeline.graph import (  # noqa: F401
    DIACRITIZER_MODEL_KWARGS,
    build_graph,
    build_langgraph_pipeline,
    build_studio_graph,
)
from pipeline.runner import run_one_batch  # noqa: F401

# Also re-export the tool functions each stage imports, so any external code
# (or existing test) that did `from langgraph_pipeline import verify_batch_tool`
# etc. keeps working. These are the exact names the old monolith imported at
# module scope; each is now imported directly by whichever pipeline/ module
# actually calls it (see that module's imports for the authoritative source).
from tools.prosody_tools import (  # noqa: F401
    verify_batch_tool,
    verify_single_verse_tool,
    meter_schema_tool,
)
from tools.dataset_tools import commit_verse_tool, log_unresolved_tool  # noqa: F401
from tools.reconciliation_tools import reconcile_case_ending_tool  # noqa: F401
from tools.advisory_ledger import (
    record_locked_verse_tool,
    read_ledger_tool,
)  # noqa: F401
from tools.advisory_batch import build_batched_advisory_payload_tool  # noqa: F401
from tools.alignment_guards import (  # noqa: F401
    validate_advisory_batch_alignment,
    validate_naturalness_batch_alignment,
)
from tools.tracing import trace_run  # noqa: F401


def _current_thread_id(state: BatchState) -> str:
    """NOTE: dead code -- grep across tests/, scripts/, and main.py found no
    caller of this function anywhere in the codebase. Left in place (Phase 2b
    of docs/REFACTOR_PLAN.md is a structural extraction, not a behavior
    change) but flagged here for a future cleanup pass: either wire it in
    where thread-id resolution is currently done ad hoc, or delete it.
    """
    trace = current_trace()
    if trace and trace.langgraph_thread_id:
        return trace.langgraph_thread_id
    return state.get("thread_id", "unknown")
