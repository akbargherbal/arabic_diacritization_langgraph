"""
scripts/contract_test_pass_loop.py
====================================
Task 1.5's isolated contract-test run, adapted to not require a real model
provider API key. Proves the SAME thing Task 1.5 asks for: exactly one dispatch
wave per pass, in strict alternation with verify_batch_tool, with no code
path for a second same-pass wave.

Run: PYTHONPATH=. python3 scripts/contract_test_pass_loop.py
"""

import json
from unittest.mock import MagicMock, patch

import langgraph_pipeline as lp

CALL_LOG = []  # list of ("dispatch", pass_number, [verse_ids]) / ("verify", pass_number)


def fake_diacritize_one_verse(model, verse, report_path, pass_number):
    CALL_LOG.append(("dispatch_verse", pass_number, verse["verse_id"]))
    return {"sadr": f"DIACRITIZED[{verse['sadr']}]", "ajuz": f"DIACRITIZED[{verse.get('ajuz', '')}]"}


# Simulate: pass 1 -> verse A locks, verse B and C stay broken;
#           pass 2 -> verse B locks, verse C stays broken;
#           pass 3 -> verse C locks. Then advisory_stage should run.
VERIFY_SEQUENCE = [
    {"locked": ["A"], "broken": ["B", "C"], "structurally_incompatible": [], "report_path": None, "poem_result_json": "{}"},
    {"locked": ["B"], "broken": ["C"], "structurally_incompatible": [], "report_path": None, "poem_result_json": "{}"},
    {"locked": ["C"], "broken": [], "structurally_incompatible": [], "report_path": None, "poem_result_json": "{}"},
]
_verify_call_index = {"i": 0}


def fake_verify_batch_tool(verses, meter_name, pass_number):
    CALL_LOG.append(("verify", pass_number))
    result = VERIFY_SEQUENCE[_verify_call_index["i"]]
    _verify_call_index["i"] += 1
    return result


def main():
    verses = [
        {"verse_id": "A", "sadr": "seed a", "ajuz": "a2"},
        {"verse_id": "B", "sadr": "seed b", "ajuz": "b2"},
        {"verse_id": "C", "sadr": "seed c", "ajuz": "c2"},
    ]

    with patch.object(lp, "_diacritize_one_verse", side_effect=fake_diacritize_one_verse), \
         patch.object(lp, "verify_batch_tool", side_effect=fake_verify_batch_tool), \
         patch.object(lp, "record_locked_verse_tool", return_value={"recorded": True}), \
         patch.object(lp, "log_unresolved_tool", return_value={"logged": True}), \
         patch.object(lp, "build_batched_advisory_payload_tool", return_value={"payload": None}):

        graph = lp.build_graph(MagicMock(), MagicMock())

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
        final_state = graph.invoke(initial_state, config={"configurable": {"thread_id": "contract_test"}})

    print("=== CALL LOG ===")
    for entry in CALL_LOG:
        print(" ", entry)

    print("\n=== FINAL STATE ===")
    print(" locked:", final_state["locked"])
    print(" broken:", final_state["broken"])
    print(" pass_number:", final_state["pass_number"])

    # --- Exit-condition assertions (Task 1.5) ---
    dispatch_events = [e for e in CALL_LOG if e[0] == "dispatch_verse"]

    # 1. Alternation check
    collapsed = []
    for e in CALL_LOG:
        kind = "D" if e[0] == "dispatch_verse" else "V"
        if not collapsed or collapsed[-1] != kind:
            collapsed.append(kind)
    assert collapsed == ["D", "V"] * 3, f"FAIL: dispatch/verify did not alternate strictly: {collapsed}"

    # 2. Correct broken target filtering per pass
    wave1 = {e[2] for e in dispatch_events if e[1] == 1}
    wave2 = {e[2] for e in dispatch_events if e[1] == 2}
    wave3 = {e[2] for e in dispatch_events if e[1] == 3}
    assert wave1 == {"A", "B", "C"}, f"FAIL: pass 1 should dispatch ALL verses unconditionally, got {wave1}"
    assert wave2 == {"B", "C"}, f"FAIL: pass 2 should dispatch only still-broken verses, got {wave2}"
    assert wave3 == {"C"}, f"FAIL: pass 3 should dispatch only still-broken verses, got {wave3}"

    # 3. No double dispatches in the same pass
    for pass_num in (1, 2, 3):
        ids_this_pass = [e[2] for e in dispatch_events if e[1] == pass_num]
        assert len(ids_this_pass) == len(set(ids_this_pass)), f"FAIL: duplicate dispatch within pass {pass_num}"

    # 4. Correct resolution
    assert set(final_state["locked"]) == {"A", "B", "C"}
    assert final_state["broken"] == []
    assert final_state["pass_number"] == 4

    print("\nCONTRACT TEST PASSED: exactly one dispatch wave per pass, strict")
    print("alternation with verify_batch_tool, zero double-dispatch-before-verify")
    print("occurrences, loop terminated correctly with no runaway 4th pass.")


if __name__ == "__main__":
    main()
