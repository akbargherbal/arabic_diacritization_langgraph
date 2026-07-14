"""
tests/test_pipeline_advisory_stage.py
========================================
Pytest port of scripts/contract_test_advisory_stage.py (Phase 0 of
docs/REFACTOR_PLAN.md). Same fakes, same assertions, same four scenarios,
split into four independent test functions so a failure in one scenario
doesn't hide the others.

Exercises advisory_stage / resolve_and_commit end-to-end using the REAL
tools.advisory_ledger / tools.alignment_guards / tools.reconciliation_tools
code, but with mock boundaries so no live dataset/verses.jsonl writes occur.
"""

from unittest.mock import MagicMock, patch

import pipeline.advisory_stage as advisory_stage
from facades.ledger_client import LedgerClient
from tools.tracing import trace_run


def _fake_commit_verse_tool(commit_calls):
    def _commit(**kwargs):
        commit_calls.append(kwargs)
        return {
            "committed": True,
            "needs_review": kwargs.get("irab_flag", False)
            or kwargs.get("naturalness_flag", False),
        }

    return _commit


def test_scenario_a_clean_batched_path_commits_with_no_flags():
    commit_calls = []

    with patch.object(
        LedgerClient,
        "commit",
        side_effect=_fake_commit_verse_tool(commit_calls),
    ), patch(
        "tools.advisory_ledger.verify_skeleton_fidelity_tool",
        return_value={"match": True},
    ), patch(
        "tools.advisory_ledger.verify_single_verse_tool",
        return_value={"is_sound": True, "combined_score": 1.0},
    ), patch.object(
        advisory_stage,
        "run_irab_checker_batch",
        return_value=[
            {"verse_id": "V1", "flag": False, "fix_type": None, "note": "clean"},
            {"verse_id": "V2", "flag": False, "fix_type": None, "note": "clean"},
        ],
    ), patch.object(
        advisory_stage,
        "run_naturalness_critic_batch",
        return_value=[
            {"verse_id": "V1", "natural": True, "note": ""},
            {"verse_id": "V2", "natural": True, "note": ""},
        ],
    ):

        with trace_run(langgraph_thread_id="contract_advisory_A"):
            import tools.advisory_ledger as ledger

            ledger.record_locked_verse_tool("V1", "سدر 1", "عجز 1", "ramal")
            ledger.record_locked_verse_tool("V2", "سدر 2", "عجز 2", "ramal")

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
            node = advisory_stage.make_advisory_stage(MagicMock())
            node(state)

    assert len(commit_calls) == 2
    for c in commit_calls:
        assert not c.get("irab_flag") and not c.get("naturalness_flag")


def test_scenario_b_case_ending_swap_reconciles_successfully():
    commit_calls = []

    with patch.object(
        advisory_stage,
        "reconcile_case_ending_tool",
        return_value={"success": True, "reconciled_text": "على الكتبِ", "reason": None},
    ), patch.object(
        advisory_stage,
        "verify_single_verse_tool",
        return_value={"is_sound": True, "combined_score": 1.0, "issues": []},
    ), patch.object(
        LedgerClient,
        "commit",
        side_effect=_fake_commit_verse_tool(commit_calls),
    ):

        advisory_stage.resolve_and_commit(
            verse_id="V3",
            sadr="على الكتبُ",
            ajuz="عجز 3",
            meter="ramal",
            irab_verdict={
                "flag": True,
                "fix_type": "case_ending_swap",
                "hemistich": "sadr",
                "word_index": 1,
                "target_harakah": "kasra",
                "note": "should be genitive",
            },
            naturalness_verdict={"natural": True, "note": ""},
        )

    assert len(commit_calls) == 1
    c = commit_calls[0]
    assert c["reconciled"] is True
    assert c["sadr"] == "على الكتبِ"
    assert c["original_sadr"] == "على الكتبُ"
    assert not c.get(
        "irab_flag"
    ), "a resolved reconciliation must not be logged as irab_flag=True"


def test_scenario_c_failed_reconciliation_falls_back_to_pyarud_wins_precedence():
    commit_calls = []

    with patch.object(
        advisory_stage,
        "reconcile_case_ending_tool",
        return_value={"success": True, "reconciled_text": "على الكتبِ", "reason": None},
    ), patch.object(
        advisory_stage,
        "verify_single_verse_tool",
        return_value={"is_sound": False, "combined_score": 0.5, "issues": ["broken"]},
    ), patch.object(
        LedgerClient,
        "commit",
        side_effect=_fake_commit_verse_tool(commit_calls),
    ):

        advisory_stage.resolve_and_commit(
            verse_id="V4",
            sadr="على الكتبُ",
            ajuz="عجز 4",
            meter="ramal",
            irab_verdict={
                "flag": True,
                "fix_type": "case_ending_swap",
                "hemistich": "sadr",
                "word_index": 1,
                "target_harakah": "kasra",
                "note": "should be genitive",
            },
            naturalness_verdict={"natural": True, "note": ""},
        )

    assert len(commit_calls) == 1
    c = commit_calls[0]
    assert c["sadr"] == "على الكتبُ"
    assert c.get("irab_flag") is True
    assert c.get("reconciled", False) is False


def test_scenario_d_alignment_guard_failure_falls_back_to_per_verse_dispatch():
    commit_calls = []

    with patch.object(
        LedgerClient,
        "commit",
        side_effect=_fake_commit_verse_tool(commit_calls),
    ), patch(
        "tools.advisory_ledger.verify_skeleton_fidelity_tool",
        return_value={"match": True},
    ), patch(
        "tools.advisory_ledger.verify_single_verse_tool",
        return_value={"is_sound": True, "combined_score": 1.0},
    ), patch.object(
        advisory_stage,
        "run_irab_checker_batch",
        return_value=[
            {"verse_id": "V5", "flag": False, "fix_type": None, "note": "clean"},
            {
                "verse_id": "HALLUCINATED",
                "flag": False,
                "fix_type": None,
                "note": "bad",
            },
        ],
    ), patch.object(
        advisory_stage,
        "run_naturalness_critic_batch",
        return_value=[
            {"verse_id": "V5", "natural": True, "note": ""},
        ],
    ), patch.object(
        advisory_stage,
        "run_irab_checker_single",
        return_value={"flag": False, "fix_type": None, "note": ""},
    ), patch.object(
        advisory_stage,
        "run_naturalness_critic_single",
        return_value={"natural": True, "note": ""},
    ):

        with trace_run(langgraph_thread_id="contract_advisory_D"):
            import tools.advisory_ledger as ledger

            ledger.record_locked_verse_tool("V5", "سدر 5", "عجز 5", "ramal")
            state = {
                "verses": [{"verse_id": "V5", "sadr": "سدر 5", "ajuz": "عجز 5"}],
                "meter_name": "ramal",
                "pass_number": 2,
                "locked": ["V5"],
                "broken": [],
                "structurally_incompatible": [],
                "drafts": {},
                "report_path": None,
            }
            node = advisory_stage.make_advisory_stage(MagicMock())
            node(state)

    assert (
        len(commit_calls) == 1
    ), f"expected fallback path to still commit V5, got {commit_calls}"
    assert commit_calls[0]["verse_id"] == "V5"
