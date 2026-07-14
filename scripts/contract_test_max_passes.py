"""
scripts/contract_test_max_passes.py
=====================================
Proves Task 3.5 (max-3-passes hard cap) and Task 3.4 (structural-incompatibility
exclusion). A verse that NEVER passes must terminate at exactly pass 3, log
unresolved_max_passes, and never trigger a 4th dispatch wave. A
structurally_incompatible verse must be excluded from every subsequent dispatch
wave and logged immediately.

Phase 1 update (PHASED_PLAN_v4_Diacritizer_Refactor.md): dispatch is now one
call per BATCH, not one call per verse -- this mocks `_diacritize_batch`
(taking the whole `targets` list) instead of the retired per-verse
`_diacritize_one_verse`. Per-verse CALL_LOG entries are still logged
individually so the existing assertions need no other changes.

Run: PYTHONPATH=. python3 scripts/contract_test_max_passes.py
"""

from unittest.mock import MagicMock, patch

import pipeline.diacritize_stage as diacritize_stage
import pipeline.verify_stage as verify_stage
import pipeline.advisory_stage as advisory_stage
from pipeline.graph import build_graph
from facades.ledger_client import LedgerClient

CALL_LOG = []
UNRESOLVED_LOGS = []


def fake_diacritize_batch(
    model, targets, meter_name, report_path, pass_number, config=None
):
    drafts = {}
    for verse in targets:
        CALL_LOG.append(("dispatch_verse", pass_number, verse["verse_id"]))
        drafts[verse["verse_id"]] = {
            "sadr": f"draft_p{pass_number}[{verse['sadr']}]",
            "ajuz": verse.get("ajuz", ""),
        }
    return drafts


# NEVER_PASSES ("N") stays broken every pass, forever.
# STRUCT_BAD ("S") becomes structurally_incompatible on pass 1 and must never
# be dispatched again.
VERIFY_SEQUENCE = [
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
_idx = {"i": 0}


def fake_verify_batch_tool(verses, meter_name, pass_number):
    CALL_LOG.append(("verify", pass_number))
    ids = {v["verse_id"] for v in verses}
    assert (
        "S" not in ids or pass_number == 1
    ), f"FAIL: 'S' was re-verified on pass {pass_number}, should have been excluded"
    result = VERIFY_SEQUENCE[_idx["i"]]
    _idx["i"] += 1
    return result


def fake_log_unresolved(
    verse_id, sadr, ajuz, meter, last_report, stage="unresolved_max_passes", reason=None
):
    UNRESOLVED_LOGS.append((verse_id, stage))
    return {"logged": True}


def main():
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

    print("=== CALL LOG ===")
    for e in CALL_LOG:
        print(" ", e)
    print("=== UNRESOLVED LOGS ===", UNRESOLVED_LOGS)
    print(
        "=== FINAL STATE broken/pass_number ===",
        final_state["broken"],
        final_state["pass_number"],
    )

    dispatch_events = [e for e in CALL_LOG if e[0] == "dispatch_verse"]
    s_dispatches = [e for e in dispatch_events if e[2] == "S"]
    assert s_dispatches == [
        ("dispatch_verse", 1, "S")
    ], f"FAIL: 'S' dispatched outside pass 1: {s_dispatches}"

    n_passes = sorted(e[1] for e in dispatch_events if e[2] == "N")
    assert n_passes == [
        1,
        2,
        3,
    ], f"FAIL: expected N dispatched on passes [1,2,3], got {n_passes}"

    verify_events = [e for e in CALL_LOG if e[0] == "verify"]
    assert (
        len(verify_events) == 3
    ), f"FAIL: expected 3 verify_batch_tool calls, got {len(verify_events)}"

    assert final_state["pass_number"] == 4
    assert final_state["broken"] == ["N"]

    assert ("S", "structurally_incompatible") in UNRESOLVED_LOGS
    assert ("N", "unresolved_max_passes") in UNRESOLVED_LOGS

    print(
        "\nCONTRACT TEST PASSED: max-3-passes hard cap and structural-incompatibility"
    )
    print("exclusion both hold -- no 4th dispatch wave, 'S' excluded after pass 1,")
    print("both verses logged with the correct, distinguishable stage values.")


if __name__ == "__main__":
    main()
