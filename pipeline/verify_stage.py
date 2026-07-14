"""
pipeline/verify_stage.py
==========================
Task 1.3/2.5's verify_pass node and Task 1.4's route_after_verify edge,
extracted from langgraph_pipeline.py (Phase 2b of docs/REFACTOR_PLAN.md).

Patch `verify_batch_tool` and `LedgerClient` HERE
(`pipeline.verify_stage.<name>`) when testing verify_pass -- they're
called as bare names resolved via this module's globals, not via
`langgraph_pipeline`'s re-exported bindings.

Phase 4 of PHASED_PLAN.md: recording locked verses and logging unresolved
ones now goes through the LedgerClient facade (facades/ledger_client.py)
instead of importing record_locked_verse_tool/log_unresolved_tool directly
from tools.advisory_ledger/tools.dataset_tools.
"""

from __future__ import annotations

from runtime import MAX_CORRECTION_PASSES, PROJECT_ROOT
from tools.prosody_tools import verify_batch_tool

from facades.ledger_client import LedgerClient
from pipeline.state import BatchState


def verify_pass(state: BatchState) -> dict:
    """Task 1.3: call verify_batch_tool unchanged; this is the ONLY place
    pass_number is ever incremented (see Task 1.3's dangerous-zone note and
    Task 3.5's hard-cap verification).

    Also folds in Task 2.5's immediate-logging requirement: any verse newly
    structurally_incompatible THIS pass is logged via log_unresolved_tool
    right away (it must never reach dispatch_diacritizer again -- excluding
    it here, rather than deferring to the terminal node, is what makes
    Task 3.4's exclusion guarantee airtight), and any newly-locked verse is
    recorded via record_locked_verse_tool immediately (matching the original
    ORCHESTRATOR_SYSTEM_PROMPT's "after each verify_batch_tool pass" cadence
    -- not deferred to the end of the batch).
    """
    prior_locked = set(state.get("locked", []))
    prior_incompatible = set(state.get("structurally_incompatible", []))
    drafts = state.get("drafts", {})
    pass_number = state.get("pass_number", 1)

    # Build the current text for every verse still in play (exclude verses
    # already excluded as structurally_incompatible in an earlier pass).
    verses_to_verify = []
    for v in state["verses"]:
        vid = v["verse_id"]
        if vid in prior_incompatible:
            continue
        draft = drafts.get(vid)
        text = (
            {"verse_id": vid, "sadr": draft["sadr"], "ajuz": draft.get("ajuz", "")}
            if draft
            else v
        )
        verses_to_verify.append(text)

    result = verify_batch_tool(verses_to_verify, state["meter_name"], pass_number)

    newly_locked = [vid for vid in result["locked"] if vid not in prior_locked]
    newly_incompatible = [
        vid
        for vid in result["structurally_incompatible"]
        if vid not in prior_incompatible
    ]

    report_text = None
    rp = result.get("report_path")
    if rp:
        p = PROJECT_ROOT / rp
        if p.exists():
            report_text = p.read_text(encoding="utf-8")

    for vid in newly_locked:
        draft = drafts.get(vid) or next(
            v for v in state["verses"] if v["verse_id"] == vid
        )
        LedgerClient.record_locked(
            verse_id=vid,
            sadr=draft["sadr"],
            ajuz=draft.get("ajuz", ""),
            meter=state["meter_name"],
        )

    for vid in newly_incompatible:
        draft = drafts.get(vid) or next(
            v for v in state["verses"] if v["verse_id"] == vid
        )
        LedgerClient.log_unresolved(
            verse_id=vid,
            sadr=draft["sadr"],
            ajuz=draft.get("ajuz", ""),
            meter=state["meter_name"],
            last_report=report_text or "",
            stage="structurally_incompatible",
            reason=(
                f"verify_batch_tool pass {pass_number}: mora/foot mismatch that "
                f"persists even in diacritized output -- see report_path={rp}"
            ),
        )

    return {
        "locked": newly_locked,  # merged via _append_unique reducer
        "broken": result["broken"],
        "structurally_incompatible": newly_incompatible,  # merged via reducer
        "report_path": rp,
        "pass_number": pass_number + 1,
    }


def route_after_verify(state: BatchState) -> str:
    """Task 1.4: the single edge this migration exists to get right. Pure
    Python, no LLM call, no branch that can fire dispatch_diacritizer twice
    without an intervening verify_pass -- guaranteed by the graph SHAPE
    (dispatch_diacritizer's only outgoing edge is the fixed one to
    verify_pass; this function is the only thing that can route back to
    dispatch_diacritizer, and it's only reachable AFTER verify_pass runs).
    """
    if state.get("broken") and state.get("pass_number", 1) <= MAX_CORRECTION_PASSES:
        return "dispatch_diacritizer"
    return "advisory_stage"
