"""
tests/test_advisory_ledger.py
==============================
Unit tests for Locked-Verse Ledger Tool (Task 1) and Batched Payload Builder (Task 2).
"""

import json
import pathlib
import pytest
from unittest.mock import patch

from tools.advisory_ledger import record_locked_verse_tool, read_ledger_tool
from tools.advisory_batch import build_batched_advisory_payload_tool
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


def test_record_locked_verse_fidelity_failure():
    """A test that calls record_locked_verse_tool with a verse whose skeleton doesn't match."""
    with trace_run(langgraph_thread_id="test_thread"):
        with patch(
            "tools.advisory_ledger.verify_skeleton_fidelity_tool"
        ) as mock_fidelity, patch(
            "tools.advisory_ledger.verify_single_verse_tool"
        ) as mock_prosody:

            mock_fidelity.return_value = {"match": False, "reason": "skeleton mismatch"}
            mock_prosody.return_value = {"is_sound": True}

            result = record_locked_verse_tool(
                verse_id="1919-0", sadr="مismatch", ajuz="مismatch", meter="ramal"
            )

            assert result["recorded"] is False
            assert "fidelity check failed" in result["reason"]

            # Assert no ledger file was created
            ledger = read_ledger_tool()
            assert len(ledger["verses"]) == 0


def test_record_and_read_multiple_valid_verses():
    """A test that records two distinct valid verses, then calls read_ledger_tool()."""
    with trace_run(langgraph_thread_id="test_thread"):
        with patch(
            "tools.advisory_ledger.verify_skeleton_fidelity_tool"
        ) as mock_fidelity, patch(
            "tools.advisory_ledger.verify_single_verse_tool"
        ) as mock_prosody:

            mock_fidelity.return_value = {"match": True}
            mock_prosody.return_value = {"is_sound": True}

            res1 = record_locked_verse_tool("1919-0", "سدر 1", "عجز 1", "ramal")
            res2 = record_locked_verse_tool("1919-1", "سدر 2", "عجز 2", "ramal")

            assert res1["recorded"] is True
            assert res2["recorded"] is True

            ledger = read_ledger_tool()
            assert len(ledger["verses"]) == 2
            assert ledger["verses"][0]["verse_id"] == "1919-0"
            assert ledger["verses"][0]["sadr"] == "سدر 1"
            assert ledger["verses"][1]["verse_id"] == "1919-1"
            assert "recorded_at" in ledger["verses"][0]


def test_duplicate_verse_id_prevention():
    """A test that records the same verse_id twice."""
    with trace_run(langgraph_thread_id="test_thread"):
        with patch(
            "tools.advisory_ledger.verify_skeleton_fidelity_tool"
        ) as mock_fidelity, patch(
            "tools.advisory_ledger.verify_single_verse_tool"
        ) as mock_prosody:

            mock_fidelity.return_value = {"match": True}
            mock_prosody.return_value = {"is_sound": True}

            res1 = record_locked_verse_tool("1919-0", "سدر 1", "عجز 1", "ramal")
            res2 = record_locked_verse_tool(
                "1919-0", "سدر 1 modified", "عجز 1 modified", "ramal"
            )

            assert res1["recorded"] is True
            assert res2["recorded"] is False
            assert res2.get("duplicate") is True

            ledger = read_ledger_tool()
            assert len(ledger["verses"]) == 1
            assert ledger["verses"][0]["sadr"] == "سدر 1"


def test_read_ledger_with_clear():
    """A test that calls read_ledger_tool(clear=True)."""
    with trace_run(langgraph_thread_id="test_thread"):
        with patch(
            "tools.advisory_ledger.verify_skeleton_fidelity_tool"
        ) as mock_fidelity, patch(
            "tools.advisory_ledger.verify_single_verse_tool"
        ) as mock_prosody:

            mock_fidelity.return_value = {"match": True}
            mock_prosody.return_value = {"is_sound": True}

            record_locked_verse_tool("1919-0", "سدر 1", "عجز 1", "ramal")

            # Read with clear=True
            ledger = read_ledger_tool(clear=True)
            assert len(ledger["verses"]) == 1

            # Subsequent read should be empty
            subsequent_ledger = read_ledger_tool()
            assert len(subsequent_ledger["verses"]) == 0


def test_build_batched_advisory_payload():
    """A test that records 2 verses, builds payload, and asserts JSON structure."""
    with trace_run(langgraph_thread_id="test_thread"):
        with patch(
            "tools.advisory_ledger.verify_skeleton_fidelity_tool"
        ) as mock_fidelity, patch(
            "tools.advisory_ledger.verify_single_verse_tool"
        ) as mock_prosody:

            mock_fidelity.return_value = {"match": True}
            mock_prosody.return_value = {"is_sound": True}

            record_locked_verse_tool("1919-0", "سدر 1", "عجز 1", "ramal")
            record_locked_verse_tool("1919-1", "سدر 2", "عجز 2", "ramal")

            result = build_batched_advisory_payload_tool()
            assert "payload" in result
            assert result["payload"] is not None

            parsed = json.loads(result["payload"])
            assert isinstance(parsed, list)
            assert len(parsed) == 2

            entry = parsed[0]
            assert set(entry.keys()) == {"verse_id", "sadr", "ajuz"}
            assert entry["verse_id"] == "1919-0"
            assert entry["sadr"] == "سدر 1"
            assert entry["ajuz"] == "عجز 1"
