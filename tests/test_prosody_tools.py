"""
tests/test_prosody_tools.py
=============================
Regression tests for the TOOL WRAPPERS (batch locking/broken split logic,
sanitizer, commit-path enforcement) — not for the pyarud gate itself, which
is out of scope for this project to test (it's your dependency's
correctness, not this scaffold's).

This file is permission-denied for agent write/edit (see main.py), same
rationale as verification/: a test suite the agent can edit is not a gate.

Run: pytest tests/
"""

import pytest

from tools.sanitization_tools import sanitize_output_tool
from tools.reconciliation_tools import reconcile_case_ending_tool


def test_reconcile_swaps_damma_to_kasra():
    # على الكتبُ -> على الكتبِ  (word_index=1, "الكتبُ" -> "الكتبِ")
    result = reconcile_case_ending_tool("على الكتبُ", word_index=1, target_harakah="kasra")
    assert result["success"] is True
    assert result["reconciled_text"] == "على الكتبِ"


def test_reconcile_rejects_unknown_harakah():
    result = reconcile_case_ending_tool("على الكتبُ", word_index=1, target_harakah="sukun")
    assert result["success"] is False


def test_reconcile_rejects_out_of_range_word_index():
    result = reconcile_case_ending_tool("على الكتبُ", word_index=5, target_harakah="kasra")
    assert result["success"] is False


def test_reconcile_reports_no_basic_mark_found():
    result = reconcile_case_ending_tool("على الكتب", word_index=1, target_harakah="kasra")
    assert result["success"] is False
    assert "no basic" in result["reason"]


def test_sanitizer_accepts_plain_arabic():
    result = sanitize_output_tool("فَعُولُنْ مَفَاعِيلُنْ")
    assert result["valid"] is True


def test_sanitizer_rejects_control_char():
    result = sanitize_output_tool("فَعُولُنْ\x00مَفَاعِيلُنْ")
    assert result["valid"] is False
    assert "control" in result["reason"]


def test_sanitizer_rejects_unexpected_latin_injection():
    result = sanitize_output_tool("فَعُولُنْ <script>")
    assert result["valid"] is False


def test_sanitizer_allows_common_punctuation():
    result = sanitize_output_tool("فَعُولُنْ، مَفَاعِيلُنْ؟")
    assert result["valid"] is True


# --- The two tests below require a REAL verification/arabic_prosody_feedback.py
# --- (the stub raises NotImplementedError by design). Skip until replaced.

@pytest.mark.skip(reason="requires real arabic_prosody_feedback.py, see README.md")
def test_commit_verse_rejects_failing_score():
    ...


@pytest.mark.skip(reason="requires real arabic_prosody_feedback.py, see README.md")
def test_verify_batch_splits_locked_and_broken_correctly():
    ...



def test_verify_batch_structural_incompatibility():
    from tools.prosody_tools import verify_batch_tool
    
    # Three verses verified to contain structural mora count deficits/excesses
    verses = [
        {"verse_id": "37926-0", "sadr": "لعينيك من جارة جائره ", "ajuz": " شقائي وآمالي العاثره"},
        {"verse_id": "37926-1", "sadr": "أتنأين عني وتجفينني ", "ajuz": " لإرضاء طائفة ماكره"},
        {"verse_id": "37926-2", "sadr": "برئنا إلي الحب لا ذنب لي ", "ajuz": " ولا لحبيبتي الهاجره"}
    ]
    
    result = verify_batch_tool(verses, "mutakareb", 1)
    
    assert "locked" in result
    assert "broken" in result
    assert "structurally_incompatible" in result
    
    assert len(result["locked"]) == 0
    assert set(result["structurally_incompatible"]) == {"37926-0", "37926-1", "37926-2"}
    assert len(result["broken"]) == 0
