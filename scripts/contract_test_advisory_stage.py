"""
scripts/contract_test_advisory_stage.py
=========================================
Exercises advisory_stage / resolve_and_commit end-to-end using the REAL
tools.advisory_ledger / tools.alignment_guards / tools.reconciliation_tools
code, but with mock boundaries so no live dataset/verses.jsonl writes occur
(preserving the Task 2.4 locked-decision boundary).

Run: PYTHONPATH=. python3 scripts/contract_test_advisory_stage.py
"""

from unittest.mock import MagicMock, patch

import langgraph_pipeline as lp

COMMIT_CALLS = []


def fake_commit_verse_tool(**kwargs):
    COMMIT_CALLS.append(kwargs)
    return {"committed": True, "needs_review": kwargs.get("irab_flag", False) or kwargs.get("naturalness_flag", False)}


def main():
    # --- Scenario A: batched path, all clean (no flags) ---
    with patch.object(lp, "commit_verse_tool", side_effect=fake_commit_verse_tool), \
         patch("tools.advisory_ledger.verify_skeleton_fidelity_tool", return_value={"match": True}), \
         patch("tools.advisory_ledger.verify_single_verse_tool", return_value={"is_sound": True, "combined_score": 1.0}), \
         patch.object(lp, "run_irab_checker_batch", return_value=[
             {"verse_id": "V1", "flag": False, "fix_type": None, "note": "clean"},
             {"verse_id": "V2", "flag": False, "fix_type": None, "note": "clean"},
         ]), \
         patch.object(lp, "run_naturalness_critic_batch", return_value=[
             {"verse_id": "V1", "natural": True, "note": ""},
             {"verse_id": "V2", "natural": True, "note": ""},
         ]):

        from tools.tracing import trace_run
        with trace_run(langgraph_thread_id="contract_advisory_A"):
            lp.record_locked_verse_tool("V1", "سدر 1", "عجز 1", "ramal")
            lp.record_locked_verse_tool("V2", "سدر 2", "عجز 2", "ramal")

            state = {
                "verses": [
                    {"verse_id": "V1", "sadr": "سدر 1", "ajuz": "عجز 1"},
                    {"verse_id": "V2", "sadr": "سدر 2", "ajuz": "عجز 2"},
                ],
                "meter_name": "ramal",
                "pass_number": 2,
                "locked": ["V1", "V2"],
                "broken": [],
                "structurally_incompatible": [],
                "drafts": {},
                "report_path": None,
            }
            advisory_stage = lp.make_advisory_stage(MagicMock())
            advisory_stage(state)

    assert len(COMMIT_CALLS) == 2, f"FAIL: expected 2 commits (clean batch), got {len(COMMIT_CALLS)}"
    for c in COMMIT_CALLS:
        assert not c.get("irab_flag") and not c.get("naturalness_flag"), f"FAIL: expected clean commit, got {c}"
    print("Scenario A (clean batched path) PASSED:", [c["verse_id"] for c in COMMIT_CALLS])

    # --- Scenario B: case_ending_swap reconciles successfully (resolved, not a disagreement) ---
    COMMIT_CALLS.clear()
    with patch.object(lp, "reconcile_case_ending_tool", return_value={
            "success": True, "reconciled_text": "على الكتبِ", "reason": None}), \
         patch.object(lp, "verify_single_verse_tool", return_value={"is_sound": True, "combined_score": 1.0, "issues": []}), \
         patch.object(lp, "commit_verse_tool", side_effect=fake_commit_verse_tool):

        result = lp.resolve_and_commit(
            verse_id="V3", sadr="على الكتبُ", ajuz="عجز 3", meter="ramal",
            irab_verdict={"flag": True, "fix_type": "case_ending_swap", "hemistich": "sadr",
                          "word_index": 1, "target_harakah": "kasra", "note": "should be genitive"},
            naturalness_verdict={"natural": True, "note": ""},
        )
    assert len(COMMIT_CALLS) == 1
    c = COMMIT_CALLS[0]
    assert c["reconciled"] is True and c["sadr"] == "على الكتبِ" and c["original_sadr"] == "على الكتبُ"
    assert not c.get("irab_flag"), "FAIL: a resolved reconciliation must not be logged as irab_flag=True"
    print("Scenario B (reconciliation succeeds) PASSED:", c)

    # --- Scenario C: reconciliation fails re-verify -> falls back to precedence rule (pyarud wins, original text) ---
    COMMIT_CALLS.clear()
    with patch.object(lp, "reconcile_case_ending_tool", return_value={
            "success": True, "reconciled_text": "على الكتبِ", "reason": None}), \
         patch.object(lp, "verify_single_verse_tool", return_value={"is_sound": False, "combined_score": 0.5, "issues": ["broken"]}), \
         patch.object(lp, "commit_verse_tool", side_effect=fake_commit_verse_tool):

        lp.resolve_and_commit(
            verse_id="V4", sadr="على الكتبُ", ajuz="عجز 4", meter="ramal",
            irab_verdict={"flag": True, "fix_type": "case_ending_swap", "hemistich": "sadr",
                          "word_index": 1, "target_harakah": "kasra", "note": "should be genitive"},
            naturalness_verdict={"natural": True, "note": ""},
        )
    assert len(COMMIT_CALLS) == 1
    c = COMMIT_CALLS[0]
    assert c["sadr"] == "على الكتبُ" and c.get("irab_flag") is True and c.get("reconciled", False) is False
    print("Scenario C (reconciliation fails -> pyarud-wins precedence) PASSED:", c)

    # --- Scenario D: alignment guard failure triggers fallback to single-verse dispatch ---
    COMMIT_CALLS.clear()
    with patch.object(lp, "commit_verse_tool", side_effect=fake_commit_verse_tool), \
         patch("tools.advisory_ledger.verify_skeleton_fidelity_tool", return_value={"match": True}), \
         patch("tools.advisory_ledger.verify_single_verse_tool", return_value={"is_sound": True, "combined_score": 1.0}), \
         patch.object(lp, "run_irab_checker_batch", return_value=[
             {"verse_id": "V5", "flag": False, "fix_type": None, "note": "clean"},
             {"verse_id": "HALLUCINATED", "flag": False, "fix_type": None, "note": "bad"},
         ]), \
         patch.object(lp, "run_naturalness_critic_batch", return_value=[
             {"verse_id": "V5", "natural": True, "note": ""},
         ]), \
         patch.object(lp, "run_irab_checker_single", return_value={"flag": False, "fix_type": None, "note": ""}), \
         patch.object(lp, "run_naturalness_critic_single", return_value={"natural": True, "note": ""}):

        from tools.tracing import trace_run
        with trace_run(langgraph_thread_id="contract_advisory_D"):
            lp.record_locked_verse_tool("V5", "سدر 5", "عجز 5", "ramal")
            state = {
                "verses": [{"verse_id": "V5", "sadr": "سدر 5", "ajuz": "عجز 5"}],
                "meter_name": "ramal", "pass_number": 2,
                "locked": ["V5"], "broken": [], "structurally_incompatible": [],
                "drafts": {}, "report_path": None,
            }
            advisory_stage = lp.make_advisory_stage(MagicMock())
            advisory_stage(state)

    assert len(COMMIT_CALLS) == 1, f"FAIL: expected fallback path to still commit V5, got {COMMIT_CALLS}"
    print("Scenario D (alignment-guard failure -> per-verse fallback) PASSED:", COMMIT_CALLS[0]["verse_id"])

    print("\nALL ADVISORY_STAGE CONTRACT SCENARIOS PASSED.")


if __name__ == "__main__":
    main()
