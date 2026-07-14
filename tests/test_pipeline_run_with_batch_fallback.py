"""
tests/test_pipeline_run_with_batch_fallback.py
=================================================
Unit tests for pipeline.advisory_stage.run_with_batch_fallback (Phase 5 of
PHASED_PLAN.md), isolated from the full advisory_stage node so the shared
batch/alignment-guard/fallback shape is tested on its own.

Covers the joint-fallback rule from ARCHITECTURE.md: "If either alignment
guard fails ... falls back to calling standard single-verse subagents ...
sequentially" -- i.e. one checker's guard failure must also force the
OTHER checker (whose own guard passed) into per-verse fallback.
"""

from pipeline.advisory_stage import run_with_batch_fallback


def _ok_guard(verdicts):
    return True, None


def _failing_guard(verdicts):
    return False, "count mismatch"


def test_all_guards_pass_uses_batch_results_for_every_checker():
    calls = {"irab_single": 0, "nat_single": 0}

    def irab_batch(model, payload):
        return [{"verse_id": "V1", "flag": False}, {"verse_id": "V2", "flag": False}]

    def nat_batch(model, payload):
        return [{"verse_id": "V1", "natural": True}, {"verse_id": "V2", "natural": True}]

    def irab_single(model, verse):
        calls["irab_single"] += 1
        return {"flag": False}

    def nat_single(model, verse):
        calls["nat_single"] += 1
        return {"natural": True}

    ledger_verses = [{"verse_id": "V1"}, {"verse_id": "V2"}]

    results = run_with_batch_fallback(
        model=None,
        payload="{}",
        ledger_verses=ledger_verses,
        checkers=[
            (irab_batch, irab_single, _ok_guard, "irab_checker_batch"),
            (nat_batch, nat_single, _ok_guard, "naturalness_critic_batch"),
        ],
    )

    assert results["irab_checker_batch"]["V1"] == {"verse_id": "V1", "flag": False}
    assert results["naturalness_critic_batch"]["V2"] == {
        "verse_id": "V2",
        "natural": True,
    }
    assert calls == {"irab_single": 0, "nat_single": 0}


def test_one_guard_failing_forces_both_checkers_into_fallback():
    """The core joint-fallback assertion: naturalness_critic_batch's guard
    passes on its own, but because irab_checker_batch's guard fails, BOTH
    checkers must fall back to per-verse dispatch."""
    calls = {"irab_single": [], "nat_single": []}

    def irab_batch(model, payload):
        return [{"verse_id": "V1", "flag": False}]  # count mismatch vs ledger

    def nat_batch(model, payload):
        return [{"verse_id": "V1", "natural": True}, {"verse_id": "V2", "natural": True}]

    def irab_single(model, verse):
        calls["irab_single"].append(verse["verse_id"])
        return {"flag": False, "source": "single"}

    def nat_single(model, verse):
        calls["nat_single"].append(verse["verse_id"])
        return {"natural": True, "source": "single"}

    ledger_verses = [{"verse_id": "V1"}, {"verse_id": "V2"}]

    results = run_with_batch_fallback(
        model=None,
        payload="{}",
        ledger_verses=ledger_verses,
        checkers=[
            (irab_batch, irab_single, _failing_guard, "irab_checker_batch"),
            (nat_batch, nat_single, _ok_guard, "naturalness_critic_batch"),
        ],
    )

    # naturalness_critic_batch's own guard passed, but it must still have
    # been forced into per-verse fallback because irab's guard failed.
    assert sorted(calls["nat_single"]) == ["V1", "V2"]
    assert sorted(calls["irab_single"]) == ["V1", "V2"]
    assert results["irab_checker_batch"]["V1"]["source"] == "single"
    assert results["naturalness_critic_batch"]["V2"]["source"] == "single"
