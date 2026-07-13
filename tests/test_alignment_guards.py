"""
tests/test_alignment_guards.py
===============================
Unit tests for alignment guards (Task 3).
"""

import pathlib
import pytest
from unittest.mock import patch

from tools.advisory_ledger import record_locked_verse_tool, read_ledger_tool
from tools.alignment_guards import (
    validate_advisory_batch_alignment,
    validate_naturalness_batch_alignment,
)
from tools.tracing import trace_run


@pytest.fixture(autouse=True)
def clean_ledger_environment():
    """Ensure ledger is clean before and after every test."""
    with trace_run(langgraph_thread_id="test_thread") as trace:
        ledger_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "workspace"
            / "test_thread"
            / "advisory_ledger.json"
        )
        if ledger_path.exists():
            try:
                ledger_path.unlink()
            except OSError:
                pass
        yield
        if ledger_path.exists():
            try:
                ledger_path.unlink()
            except OSError:
                pass


def test_validate_alignment_success():
    """Seeding 2 verses, case_ending_swap correctly matches target_word_skeleton at word_index."""
    with trace_run(langgraph_thread_id="test_thread"):
        with patch(
            "tools.advisory_ledger.verify_skeleton_fidelity_tool"
        ) as mock_fidelity, patch(
            "tools.advisory_ledger.verify_single_verse_tool"
        ) as mock_prosody:

            mock_fidelity.return_value = {"match": True}
            mock_prosody.return_value = {"is_sound": True}

            # Seed ledger with 2 verses
            # "الكتبِ" normalizes to "الكتب"
            record_locked_verse_tool("verse_1", "على الكتبِ", "عجز 1", "ramal")
            record_locked_verse_tool("verse_2", "سدر 2", "عجز 2", "ramal")

            # Word at word_index=1 in sadr is "الكتبِ", which has consonant skeleton "الكتب"
            batch_verdicts = [
                {
                    "verse_id": "verse_1",
                    "flag": True,
                    "fix_type": "case_ending_swap",
                    "hemistich": "sadr",
                    "word_index": 1,
                    "target_word_skeleton": "الكتب",
                    "target_harakah": "kasra",
                    "note": "correct",
                },
                {
                    "verse_id": "verse_2",
                    "flag": False,
                    "fix_type": None,
                    "note": "no issues",
                },
            ]

            success, reason = validate_advisory_batch_alignment(batch_verdicts)
            assert success is True
            assert reason is None


def test_validate_alignment_mismatched_skeleton():
    """A test with a mismatched skeleton -> asserts False with reason containing DATA CONTAMINATION SHIELD TRIGGERED."""
    with trace_run(langgraph_thread_id="test_thread"):
        with patch(
            "tools.advisory_ledger.verify_skeleton_fidelity_tool"
        ) as mock_fidelity, patch(
            "tools.advisory_ledger.verify_single_verse_tool"
        ) as mock_prosody:

            mock_fidelity.return_value = {"match": True}
            mock_prosody.return_value = {"is_sound": True}

            record_locked_verse_tool("verse_1", "على الكتبِ", "عجز 1", "ramal")
            record_locked_verse_tool("verse_2", "سدر 2", "عجز 2", "ramal")

            batch_verdicts = [
                {
                    "verse_id": "verse_1",
                    "flag": True,
                    "fix_type": "case_ending_swap",
                    "hemistich": "sadr",
                    "word_index": 1,
                    "target_word_skeleton": "مختلف",  # Mismatch!
                    "target_harakah": "kasra",
                    "note": "mismatch",
                },
                {
                    "verse_id": "verse_2",
                    "flag": False,
                    "fix_type": None,
                    "note": "no issues",
                },
            ]

            success, reason = validate_advisory_batch_alignment(batch_verdicts)
            assert success is False
            assert "DATA CONTAMINATION SHIELD TRIGGERED" in reason


def test_validate_alignment_hallucinated_id():
    """A test with a hallucinated verse_id not in the ledger."""
    with trace_run(langgraph_thread_id="test_thread"):
        with patch(
            "tools.advisory_ledger.verify_skeleton_fidelity_tool"
        ) as mock_fidelity, patch(
            "tools.advisory_ledger.verify_single_verse_tool"
        ) as mock_prosody:

            mock_fidelity.return_value = {"match": True}
            mock_prosody.return_value = {"is_sound": True}

            record_locked_verse_tool("verse_1", "على الكتبِ", "عجز 1", "ramal")
            record_locked_verse_tool("verse_2", "سدر 2", "عجز 2", "ramal")

            batch_verdicts = [
                {"verse_id": "verse_1", "flag": False, "fix_type": None, "note": "ok"},
                {
                    "verse_id": "hallucinated_id",  # Hallucinated!
                    "flag": False,
                    "fix_type": None,
                    "note": "ok",
                },
            ]

            success, reason = validate_advisory_batch_alignment(batch_verdicts)
            assert success is False
            assert "hallucinated" in reason


def test_validate_naturalness_alignment_duplicate_id():
    """A test for validate_naturalness_batch_alignment with a duplicate verse_id."""
    with trace_run(langgraph_thread_id="test_thread"):
        with patch(
            "tools.advisory_ledger.verify_skeleton_fidelity_tool"
        ) as mock_fidelity, patch(
            "tools.advisory_ledger.verify_single_verse_tool"
        ) as mock_prosody:

            mock_fidelity.return_value = {"match": True}
            mock_prosody.return_value = {"is_sound": True}

            record_locked_verse_tool("verse_1", "على الكتبِ", "عجز 1", "ramal")
            record_locked_verse_tool("verse_2", "سدر 2", "عجز 2", "ramal")

            batch_verdicts = [
                {"verse_id": "verse_1", "natural": True, "note": "ok"},
                {
                    "verse_id": "verse_1",  # Duplicate!
                    "natural": True,
                    "note": "duplicate",
                },
            ]

            success, reason = validate_naturalness_batch_alignment(batch_verdicts)
            assert success is False
            assert "duplicate" in reason
