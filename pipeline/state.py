"""
pipeline/state.py
===================
Task 1.1's state schema, extracted verbatim from langgraph_pipeline.py
(Phase 2 of docs/REFACTOR_PLAN.md). Pure data + reducer functions, no
model/tool calls, no test patches this file's names directly -- safe to
relocate without touching any patch.object(lp, ...) target.
"""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict


def _merge_dicts(a: Optional[dict], b: Optional[dict]) -> dict:
    """Reducer for BatchState["drafts"]: per-verse parallel writers each
    touch a disjoint verse_id key, so a plain last-writer-wins update per
    key is safe -- there is no cross-verse key collision by construction
    (each verse is dispatched exactly once per pass, see dispatch_diacritizer).
    """
    merged = dict(a) if a else {}
    if b:
        merged.update(b)
    return merged


def _append_unique(a: Optional[list], b: Optional[list]) -> list:
    """Reducer for cumulative id lists (locked / structurally_incompatible):
    append only ids not already present, preserving order."""
    out = list(a) if a else []
    seen = set(out)
    for item in b or []:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


class BatchState(TypedDict, total=False):
    verses: list[dict]  # [{verse_id, sadr, ajuz}, ...] -- original input, immutable
    meter_name: str
    pass_number: int  # starts at 1; incremented ONLY in verify_pass (Task 1.3)
    locked: Annotated[list[str], _append_unique]  # cumulative across passes
    broken: list[str]  # THIS pass's still-broken verse_ids (overwritten each pass)
    structurally_incompatible: Annotated[list[str], _append_unique]  # cumulative
    drafts: Annotated[dict, _merge_dicts]  # verse_id -> {"sadr", "ajuz"}
    report_path: Optional[str]  # most recent pass's correction-report file
    thread_id: str  # for workspace/ledger file scoping (mirrors trace context)
