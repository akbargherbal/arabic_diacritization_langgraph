"""
tests/test_facades_ledger_client.py
=====================================
Unit tests for the LedgerClient facade (Phase 4 of PHASED_PLAN.md).

These are thin pass-through tests: each LedgerClient method should forward
to exactly the underlying tool function it wraps, with the same kwargs,
and return that function's result unchanged. Pipeline-level behavior
(verify_stage / advisory_stage using LedgerClient correctly) is already
covered by tests/test_pipeline_*.py; this file only proves the facade
itself doesn't drop or mangle anything.
"""

from unittest.mock import patch

from facades.ledger_client import LedgerClient


def test_record_locked_forwards_to_record_locked_verse_tool():
    with patch(
        "facades.ledger_client.record_locked_verse_tool",
        return_value={"recorded": True},
    ) as mock_fn:
        result = LedgerClient.record_locked(
            verse_id="V1", sadr="سدر", ajuz="عجز", meter="ramal"
        )

    mock_fn.assert_called_once_with(
        verse_id="V1", sadr="سدر", ajuz="عجز", meter="ramal"
    )
    assert result == {"recorded": True}


def test_read_ledger_forwards_clear_flag():
    with patch(
        "facades.ledger_client.read_ledger_tool", return_value={"verses": []}
    ) as mock_fn:
        result = LedgerClient.read_ledger(clear=False)

    mock_fn.assert_called_once_with(clear=False)
    assert result == {"verses": []}


def test_reset_ledger_always_clears():
    with patch(
        "facades.ledger_client.read_ledger_tool", return_value={"verses": []}
    ) as mock_fn:
        LedgerClient.reset_ledger()

    mock_fn.assert_called_once_with(clear=True)


def test_build_advisory_payload_forwards_to_build_batched_advisory_payload_tool():
    with patch(
        "facades.ledger_client.build_batched_advisory_payload_tool",
        return_value={"payload": None, "reason": "ledger is empty"},
    ) as mock_fn:
        result = LedgerClient.build_advisory_payload()

    mock_fn.assert_called_once_with()
    assert result == {"payload": None, "reason": "ledger is empty"}


def test_commit_forwards_all_fields_to_commit_verse_tool():
    with patch(
        "facades.ledger_client.commit_verse_tool", return_value={"committed": True}
    ) as mock_fn:
        result = LedgerClient.commit(
            verse_id="V2",
            sadr="سدر 2",
            ajuz="عجز 2",
            meter="ramal",
            irab_flag=True,
            naturalness_flag=False,
            reconciled=True,
            original_sadr="orig sadr",
            original_ajuz="orig ajuz",
            notes="a note",
            fix_type="case_ending_swap",
            word_index=1,
            target_harakah="kasra",
        )

    mock_fn.assert_called_once_with(
        verse_id="V2",
        sadr="سدر 2",
        ajuz="عجز 2",
        meter="ramal",
        irab_flag=True,
        naturalness_flag=False,
        reconciled=True,
        original_sadr="orig sadr",
        original_ajuz="orig ajuz",
        notes="a note",
        fix_type="case_ending_swap",
        word_index=1,
        target_harakah="kasra",
    )
    assert result == {"committed": True}


def test_commit_defaults_match_commit_verse_tool_defaults():
    with patch(
        "facades.ledger_client.commit_verse_tool", return_value={"committed": True}
    ) as mock_fn:
        LedgerClient.commit(verse_id="V3", sadr="s", ajuz="a", meter="ramal")

    mock_fn.assert_called_once_with(
        verse_id="V3",
        sadr="s",
        ajuz="a",
        meter="ramal",
        irab_flag=False,
        naturalness_flag=False,
        reconciled=False,
        original_sadr=None,
        original_ajuz=None,
        notes="",
        fix_type=None,
        word_index=None,
        target_harakah=None,
    )


def test_log_unresolved_forwards_to_log_unresolved_tool():
    with patch(
        "facades.ledger_client.log_unresolved_tool", return_value={"logged": True}
    ) as mock_fn:
        result = LedgerClient.log_unresolved(
            verse_id="V4",
            sadr="s",
            ajuz="a",
            meter="ramal",
            last_report="report text",
            stage="structurally_incompatible",
            reason="mora mismatch",
        )

    mock_fn.assert_called_once_with(
        verse_id="V4",
        sadr="s",
        ajuz="a",
        meter="ramal",
        last_report="report text",
        stage="structurally_incompatible",
        reason="mora mismatch",
    )
    assert result == {"logged": True}


def test_log_unresolved_default_stage_is_unresolved_max_passes():
    with patch(
        "facades.ledger_client.log_unresolved_tool", return_value={"logged": True}
    ) as mock_fn:
        LedgerClient.log_unresolved(
            verse_id="V5", sadr="s", ajuz="a", meter="ramal", last_report=""
        )

    _, kwargs = mock_fn.call_args
    assert kwargs["stage"] == "unresolved_max_passes"
    assert kwargs["reason"] is None
