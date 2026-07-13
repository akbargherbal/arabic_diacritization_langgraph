"""
tools/prosody_tools.py
========================
Agent-facing tools wrapping verification/arabic_prosody_feedback.py.

These are the ONLY sanctioned way an agent touches the pyarud gate. Note
that these are read-driven — none of them write into verification/, and the
orchestrator (not any subagent) is the one that calls verify_batch_tool,
per the design's separation between "the entity that decides pass/fail"
and "the entity trying to pass" (see main.py's system prompt).
"""

from __future__ import annotations
import json
from pathlib import Path

from verification import arabic_prosody_feedback as prosody
from config import meter_tables
from tools.tracing import current_trace
from runtime import MAX_CORRECTION_PASSES


def meter_schema_tool(meter_id: str) -> dict:
    """Return the canonical template and Arabic name for a meter id.

    Read-only lookup against config/meter_tables.py — never generated or
    guessed by the model.
    """
    resolved = meter_tables._METER_TABLE_TO_PYARUD.get(meter_id, meter_id)
    return {
        "meter_id": resolved,
        "template": meter_tables.METER_TEMPLATES.get(resolved),
        "arabic_name": meter_tables.METER_ARABIC_NAMES.get(resolved),
    }


def verify_batch_tool(verses: list[dict], meter_name: str, pass_number: int) -> dict:
    """Run the deterministic pyarud check over a batch of verses.

    Args:
        verses: list of {"verse_id": str, "sadr": str, "ajuz": str}
        meter_name: target meter (any supported alias)
        pass_number: current pass number, starting at 1

    Returns:
        {
          "locked": [verse_id, ...],       # passed, do not resubmit
          "broken": [verse_id, ...],       # failed, needs another pass
          "structurally_incompatible": [verse_id, ...], # failed due to structural mora deficits
          "report_path": str,              # path to the full correction report file
          "poem_result_json": str,         # full structured result, for logging
        }

    This is called by the ORCHESTRATOR directly, never delegated to the
    diacritizer subagent — the entity deciding pass/fail must not be the
    entity trying to pass.
    """
    pairs = [(v["sadr"], v.get("ajuz", "")) for v in verses]
    poem_result = prosody.analyze_poem(pairs, meter_name=meter_name)

    locked, broken = [], []
    structurally_incompatible = []
    # verse_scores is a terse (verse_id, combined_score) list, added purely
    # so the orchestrator's own bookkeeping across correction passes
    # (see tools/context_tools.py's summarize_correction_report_tool) can
    # carry a few tokens per verse instead of re-embedding the full
    # per-foot correction_report text every pass. This does NOT touch or
    # duplicate anything from verification/ itself -- it's assembled here,
    # in the agent-facing wrapper layer, from data verify_batch_tool
    # already computed.
    verse_scores = []
    for v, verse_result in zip(verses, poem_result.verses):
        if verse_result.combined_score >= 0.99:
            locked.append(v["verse_id"])
        else:
            # Check for structural incompatibility: expected vs actual mora counts do not match,
            # or there are missing feet / extra bits that cannot be resolved via diacritic-only tuning.
            is_incompatible = False
            
            # Check Sadr
            sadr_exp = sum(len(f.expected_pattern) for f in verse_result.sadr.feet)
            sadr_act = sum(len(f.actual_segment) for f in verse_result.sadr.feet)
            if sadr_exp != sadr_act:
                is_incompatible = True
            elif verse_result.sadr.missing_foot_count > 0 or verse_result.sadr.extra_bits is not None:
                is_incompatible = True
                
            # Check Ajuz if present
            if verse_result.ajuz:
                ajuz_exp = sum(len(f.expected_pattern) for f in verse_result.ajuz.feet)
                ajuz_act = sum(len(f.actual_segment) for f in verse_result.ajuz.feet)
                if ajuz_exp != ajuz_act:
                    is_incompatible = True
                elif verse_result.ajuz.missing_foot_count > 0 or verse_result.ajuz.extra_bits is not None:
                    is_incompatible = True
                    
            # Near-miss exception: if this is an early pass (pass_number < MAX_CORRECTION_PASSES)
            # and the combined_score is reasonably high (>= 0.80), we classify it as broken (retryable)
            # rather than structurally_incompatible to let the diacritizer attempt to tune the diacritics.
            if is_incompatible and pass_number < MAX_CORRECTION_PASSES and verse_result.combined_score >= 0.80:
                is_incompatible = False

            if is_incompatible:
                structurally_incompatible.append(v["verse_id"])
            else:
                broken.append(v["verse_id"])

        verse_scores.append(
            {"verse_id": v["verse_id"], "combined_score": verse_result.combined_score}
        )

    report = prosody.generate_poem_correction_report(poem_result, only_broken=True)

    # Obtain the thread id and sanitize it for file operations (subfolder isolation)
    trace = current_trace()
    thread_id = (
        trace.langgraph_thread_id if trace and trace.langgraph_thread_id else "unknown"
    )
    safe_thread = thread_id.replace(":", "_")

    report_path = f"workspace/{safe_thread}/pass_{pass_number}_report.json"

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")

    return {
        "locked": locked,
        "broken": broken,
        "structurally_incompatible": structurally_incompatible,
        "report_path": report_path,
        "poem_result_json": json.dumps(
            {
                "overall_score": poem_result.overall_score,
                "is_metrically_sound": poem_result.is_metrically_sound,
                "candidate_meters": poem_result.candidate_meters,
                "verse_scores": verse_scores,
            },
            ensure_ascii=False,
        ),
    }






def verify_single_verse_tool(sadr: str, ajuz: str, meter_name: str) -> dict:
    """Used by commit_verse_tool for the final re-check before a dataset write.
    Not intended for the diacritizer subagent to call directly during drafting —
    it drafts against the batch-level correction_report instead.
    """
    result = prosody.analyze_verse(sadr, ajuz, meter_name=meter_name)
    return {
        "combined_score": result.combined_score,
        "is_sound": result.combined_score >= 0.99,
        "issues": result.issues,
    }
