"""
tests/test_pipeline_max_passes.py
===================================
Pytest port of scripts/contract_test_max_passes.py (Phase 0 of
docs/REFACTOR_PLAN.md). Same fakes, same assertions, same graph.

Proves the max-3-passes hard cap and structural-incompatibility exclusion:
a verse that never passes must terminate at exactly pass 3, log
unresolved_max_passes, and never trigger a 4th dispatch wave. A
structurally_incompatible verse must be excluded from every subsequent
dispatch wave and logged immediately.
"""

from unittest.mock import MagicMock, patch

import pipeline.diacritize_stage as diacritize_stage
import pipeline.verify_stage as verify_stage
import pipeline.advisory_stage as advisory_stage
from facades.ledger_client import LedgerClient
from pipeline.graph import build_graph


def test_hard_cap_and_structural_incompatibility_exclusion():
    call_log = []
    unresolved_logs = []

    def fake_diacritize_batch(
        model, targets, meter_name, report_path, pass_number, config=None
    ):
        drafts = {}
        for verse in targets:
            call_log.append(("dispatch_verse", pass_number, verse["verse_id"]))
            drafts[verse["verse_id"]] = {
                "sadr": f"draft_p{pass_number}[{verse['sadr']}]",
                "ajuz": verse.get("ajuz", ""),
            }
        return drafts

    # NEVER_PASSES ("N") stays broken every pass, forever.
    # STRUCT_BAD ("S") becomes structurally_incompatible on pass 1 and must
    # never be dispatched or re-verified again.
    verify_sequence = [
        {
            "locked": [],
            "broken": ["N"],
            "structurally_incompatible": ["S"],
            "report_path": None,
            "poem_result_json": "{}",
        },
        {
            "locked": [],
            "broken": ["N"],
            "structurally_incompatible": [],
            "report_path": None,
            "poem_result_json": "{}",
        },
        {
            "locked": [],
            "broken": ["N"],
            "structurally_incompatible": [],
            "report_path": None,
            "poem_result_json": "{}",
        },
    ]
    idx = {"i": 0}

    def fake_verify_batch_tool(verses, meter_name, pass_number):
        call_log.append(("verify", pass_number))
        ids = {v["verse_id"] for v in verses}
        assert (
            "S" not in ids or pass_number == 1
        ), f"'S' was re-verified on pass {pass_number}, should have been excluded"
        result = verify_sequence[idx["i"]]
        idx["i"] += 1
        return result

    def fake_log_unresolved(
        verse_id,
        sadr,
        ajuz,
        meter,
        last_report,
        stage="unresolved_max_passes",
        reason=None,
    ):
        unresolved_logs.append((verse_id, stage))
        return {"logged": True}

    verses = [
        {"verse_id": "N", "sadr": "never passes", "ajuz": ""},
        {"verse_id": "S", "sadr": "structurally bad", "ajuz": ""},
    ]

    with patch.object(
        diacritize_stage, "_diacritize_batch", side_effect=fake_diacritize_batch
    ), patch.object(
        verify_stage, "verify_batch_tool", side_effect=fake_verify_batch_tool
    ), patch.object(
        LedgerClient, "record_locked", return_value={"recorded": True}
    ), patch.object(
        LedgerClient, "log_unresolved", side_effect=fake_log_unresolved
    ), patch.object(
        LedgerClient,
        "build_advisory_payload",
        return_value={"payload": None},
    ):

        graph = build_graph(MagicMock(), MagicMock())
        initial_state = {
            "verses": verses,
            "meter_name": "ramal",
            "pass_number": 1,
            "locked": [],
            "broken": [],
            "structurally_incompatible": [],
            "drafts": {},
            "report_path": None,
            "thread_id": "contract_test_2",
        }
        final_state = graph.invoke(
            initial_state, config={"configurable": {"thread_id": "contract_test_2"}}
        )

    dispatch_events = [e for e in call_log if e[0] == "dispatch_verse"]

    s_dispatches = [e for e in dispatch_events if e[2] == "S"]
    assert s_dispatches == [("dispatch_verse", 1, "S")]

    n_passes = sorted(e[1] for e in dispatch_events if e[2] == "N")
    assert n_passes == [1, 2, 3]

    verify_events = [e for e in call_log if e[0] == "verify"]
    assert len(verify_events) == 3

    assert final_state["pass_number"] == 4
    assert final_state["broken"] == ["N"]

    assert ("S", "structurally_incompatible") in unresolved_logs
    assert ("N", "unresolved_max_passes") in unresolved_logs
