"""
tools/context_tools.py
========================
New module (does not touch verification/ or config/, and does not change
the read-only pyarud report generator itself).

Addresses the context-window bloat identified across the orchestrator's
up-to-3 correction passes: verify_batch_tool's full-text correction_report
(complete per-foot diagnostics for every broken verse) re-enters the
orchestrator's own message history every pass, and largely repeats
diagnostics for verses that were already broken in a prior pass. The
orchestrator's OWN reasoning trace does not need that full text to persist
pass over pass -- it needs the CURRENT pass's full report (handed fresh to
the diacritizer subagent each time, via a scoped `task` call that does not
inherit the orchestrator's history) and only a terse per-verse summary for
its own bookkeeping across passes.

summarize_correction_report_tool is pure string/JSON processing over the
`poem_result_json` that verify_batch_tool already returns (see
tools/prosody_tools.py's verse_scores addition) -- it contains no prosodic
logic of its own and never touches verification/.
"""

from __future__ import annotations

import json


def summarize_correction_report_tool(poem_result_json: str) -> dict:
    """Collapse a verify_batch_tool poem_result_json blob into a terse,
    one-line-per-verse summary for the ORCHESTRATOR's OWN context --
    this is not a replacement for the full correction_report handed to the
    diacritizer subagent each pass, which still needs the complete
    per-foot diagnostics to act on.

    Args:
        poem_result_json: the "poem_result_json" string returned by
            verify_batch_tool (contains overall_score, is_metrically_sound,
            candidate_meters, and verse_scores).

    Returns:
        {"summary": "verse_003: 0.941; verse_007: 0.882; ..."}
        listing only verses below the pass threshold (0.99), since sound
        verses are locked and need no further mention across passes.
    """
    data = json.loads(poem_result_json)
    verse_scores = data.get("verse_scores", [])
    lines = [
        f"{v['verse_id']}: {v['combined_score']:.3f}"
        for v in verse_scores
        if v.get("combined_score", 1.0) < 0.99
    ]
    return {"summary": "; ".join(lines) if lines else "(no broken verses)"}
