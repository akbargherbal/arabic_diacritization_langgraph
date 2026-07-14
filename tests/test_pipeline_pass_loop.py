"""
tests/test_pipeline_pass_loop.py
==================================
Pytest port of scripts/contract_test_pass_loop.py (Phase 0 of
docs/REFACTOR_PLAN.md). Same fakes, same assertions, same graph -- just
running under pytest instead of as a standalone `python3 scripts/...`
invocation, so it's collected in CI and covered by coverage tooling.

The original script is left in place (scripts/contract_test_pass_loop.py)
as a still-runnable, human-readable demonstration; this file is the
regression-tested version of the same contract.

Proves: exactly one dispatch wave per pass, in strict alternation with
verify_batch_tool, with no code path for a second same-pass wave.
"""

from unittest.mock import MagicMock, patch

import pipeline.diacritize_stage as diacritize_stage
import pipeline.verify_stage as verify_stage
import pipeline.advisory_stage as advisory_stage
from facades.ledger_client import LedgerClient
from pipeline.graph import build_graph


def test_pass_loop_alternates_dispatch_and_verify_until_all_locked():
    call_log = []  # ("dispatch_verse", pass_number, verse_id) / ("verify", pass_number)

    def fake_diacritize_batch(
        model, targets, meter_name, report_path, pass_number, config=None
    ):
        drafts = {}
        for verse in targets:
            call_log.append(("dispatch_verse", pass_number, verse["verse_id"]))
            drafts[verse["verse_id"]] = {
                "sadr": f"DIACRITIZED[{verse['sadr']}]",
                "ajuz": f"DIACRITIZED[{verse.get('ajuz', '')}]",
            }
        return drafts

    # pass 1 -> A locks, B/C stay broken; pass 2 -> B locks, C stays broken;
    # pass 3 -> C locks. Then advisory_stage should run.
    verify_sequence = [
        {
            "locked": ["A"],
            "broken": ["B", "C"],
            "structurally_incompatible": [],
            "report_path": None,
            "poem_result_json": "{}",
        },
        {
            "locked": ["B"],
            "broken": ["C"],
            "structurally_incompatible": [],
            "report_path": None,
            "poem_result_json": "{}",
        },
        {
            "locked": ["C"],
            "broken": [],
            "structurally_incompatible": [],
            "report_path": None,
            "poem_result_json": "{}",
        },
    ]
    verify_call_index = {"i": 0}

    def fake_verify_batch_tool(verses, meter_name, pass_number):
        call_log.append(("verify", pass_number))
        result = verify_sequence[verify_call_index["i"]]
        verify_call_index["i"] += 1
        return result

    verses = [
        {"verse_id": "A", "sadr": "seed a", "ajuz": "a2"},
        {"verse_id": "B", "sadr": "seed b", "ajuz": "b2"},
        {"verse_id": "C", "sadr": "seed c", "ajuz": "c2"},
    ]

    with patch.object(
        diacritize_stage, "_diacritize_batch", side_effect=fake_diacritize_batch
    ), patch.object(
        verify_stage, "verify_batch_tool", side_effect=fake_verify_batch_tool
    ), patch.object(
        LedgerClient, "record_locked", return_value={"recorded": True}
    ), patch.object(
        LedgerClient, "log_unresolved", return_value={"logged": True}
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
            "thread_id": "contract_test",
        }
        final_state = graph.invoke(
            initial_state, config={"configurable": {"thread_id": "contract_test"}}
        )

    dispatch_events = [e for e in call_log if e[0] == "dispatch_verse"]

    # 1. Strict D/V alternation, no back-to-back dispatch waves.
    collapsed = []
    for e in call_log:
        kind = "D" if e[0] == "dispatch_verse" else "V"
        if not collapsed or collapsed[-1] != kind:
            collapsed.append(kind)
    assert collapsed == ["D", "V"] * 3

    # 2. Correct broken-target filtering per pass.
    wave1 = {e[2] for e in dispatch_events if e[1] == 1}
    wave2 = {e[2] for e in dispatch_events if e[1] == 2}
    wave3 = {e[2] for e in dispatch_events if e[1] == 3}
    assert wave1 == {"A", "B", "C"}
    assert wave2 == {"B", "C"}
    assert wave3 == {"C"}

    # 3. No double dispatches within the same pass.
    for pass_num in (1, 2, 3):
        ids_this_pass = [e[2] for e in dispatch_events if e[1] == pass_num]
        assert len(ids_this_pass) == len(set(ids_this_pass))

    # 4. Correct resolution.
    assert set(final_state["locked"]) == {"A", "B", "C"}
    assert final_state["broken"] == []
    assert final_state["pass_number"] == 4
