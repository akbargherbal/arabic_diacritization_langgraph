"""
tests/test_prosody_split.py
============================
Phase 3 of docs/REFACTOR_PLAN.md: focused unit tests for the
verification/prosody/ split (scoring.py / analysis.py / reporting.py) and
the arabic_prosody_feedback.py backward-compatibility shim.

Not a re-test of pyarud's own correctness (see
test_pyarud_upstream_regressions.py for that) -- these tests exercise the
module boundaries introduced by the split: that scoring.py's pure functions
work standalone with no pyarud import, that analysis.py's enrichment
correctly consumes scoring.py, that reporting.py correctly consumes
analysis.py, and that the old import path (`from verification import
arabic_prosody_feedback as prosody`) still resolves every name
tools/prosody_tools.py depends on.
"""

from __future__ import annotations

import verification.prosody.analysis as analysis
import verification.prosody.reporting as reporting
import verification.prosody.scoring as scoring
from verification import arabic_prosody_feedback as prosody


# ---------------------------------------------------------------------------
# scoring.py -- pure, no-pyarud-dependency functions
# ---------------------------------------------------------------------------


def test_scoring_binary_ux_roundtrip():
    assert scoring.binary_to_ux("11010") == "UU_U_"
    assert scoring.ux_to_binary("UU_U_") == "11010"


def test_scoring_identify_zihaf_known_and_salim():
    assert scoring.identify_zihaf("1010110", "110110") == "Khaban"
    assert scoring.identify_zihaf("11010", "11010") == "Salim"


def test_scoring_identify_zihaf_unknown_structural_fallback():
    # Not in _ZIHAF_MAP, same length, different bits -> Taskeen fallback
    result = scoring.identify_zihaf("11010", "10010")
    assert result.startswith("Unknown (Taskeen")


def test_scoring_foot_health_levels():
    assert scoring.foot_health("ok", 1.0, "Salim") == "perfect"
    assert scoring.foot_health("ok", 0.8, "Khaban") == "valid_zihaf"
    assert scoring.foot_health("broken", 0.2, None) == "broken"
    assert scoring.foot_health("missing", 0.0, None) == "severe"
    assert scoring.foot_health("extra_bits", 0.0, None) == "severe"


def test_scoring_mora_diff_directions():
    assert scoring._mora_diff("UU_U_", "UU_U_")["direction"] == "match"
    assert scoring._mora_diff("UU_U_", "UU_")["direction"] == "too_short"
    assert scoring._mora_diff("UU_", "UU_U_")["direction"] == "too_long"
    assert scoring._mora_diff("UU_", "U_U")["direction"] == "wrong_weight"


def test_scoring_module_has_no_pyarud_dependency():
    # scoring.py must be importable/usable without pyarud installed -- the
    # whole point of putting it in its own module. We can't uninstall
    # pyarud mid-suite, so assert the module object itself never imported it.
    import sys

    assert "pyarud" not in scoring.__dict__


# ---------------------------------------------------------------------------
# analysis.py -- enrichment consumes scoring.py correctly
# ---------------------------------------------------------------------------


def test_analysis_enrich_foot_salim():
    raw = {
        "expected_pattern": "11010",
        "actual_segment": "11010",
        "status": "ok",
        "score": 1.0,
        "foot_index": 0,
    }
    foot = analysis._enrich_foot(raw, "Hashw", 1)
    assert foot.zihaf_name == "Salim"
    assert foot.health == "perfect"
    # Enriched fields are converted to U/_ notation by scoring.binary_to_ux
    assert foot.expected_pattern == "UU_U_"


def test_analysis_enrich_foot_applies_known_zihaf():
    raw = {
        "expected_pattern": "1010110",
        "actual_segment": "110110",
        "status": "ok",
        "score": 0.9,
        "foot_index": 1,
    }
    foot = analysis._enrich_foot(raw, "Hashw", 2)
    assert foot.zihaf_name == "Khaban"
    assert foot.health == "valid_zihaf"


def test_analysis_analyze_poem_sound_verse():
    result = analysis.analyze_poem(
        [
            (
                "أَنَامُ مِلْءَ جُفُونِي عَنْ شَوَارِدِهَا",
                "وَيَسْهَرُ الْخَلْقُ جَرَّاهَا وَيَخْتَصِمُ",
            )
        ],
        meter_name="baseet",
    )
    assert result.meter == "baseet"
    assert result.verses[0].sadr.is_sound


def test_analysis_resolve_meter_key_variants():
    assert analysis.resolve_meter_key("basit") == "baseet"
    assert analysis.resolve_meter_key("البسيط") == "baseet"
    assert analysis.resolve_meter_key("baseet") == "baseet"
    assert analysis.resolve_meter_key(None) is None


# ---------------------------------------------------------------------------
# reporting.py -- correction-report text generation consumes analysis.py
# ---------------------------------------------------------------------------


def test_reporting_verse_correction_sound_verse():
    verse = analysis.analyze_verse(
        "أَنَامُ مِلْءَ جُفُونِي عَنْ شَوَارِدِهَا",
        "وَيَسْهَرُ الْخَلْقُ جَرَّاهَا وَيَخْتَصِمُ",
        meter_name="baseet",
    )
    report = reporting.generate_verse_correction(verse)
    assert "SOUND" in report
    assert "100%" in report


def test_reporting_get_tafeela_mnemonic_known_and_fallback():
    assert reporting.get_tafeela_mnemonic("Mustafelon", "Khaban") == "مُتَفْعِلُنْ"
    # Unknown zihaf for a known foot class falls back to base + suffix
    assert "(NotAZihaf)" in reporting.get_tafeela_mnemonic(
        "Mustafelon", "NotAZihaf"
    )


# ---------------------------------------------------------------------------
# arabic_prosody_feedback.py -- backward-compat shim
# ---------------------------------------------------------------------------


def test_shim_reexports_everything_prosody_tools_needs():
    # These are exactly the three attributes tools/prosody_tools.py calls
    # via `prosody.<name>(...)`.
    assert prosody.analyze_poem is analysis.analyze_poem
    assert prosody.analyze_verse is analysis.analyze_verse
    assert prosody.generate_poem_correction_report is reporting.generate_poem_correction_report


def test_shim_end_to_end_analyze_and_report():
    result_dict, report = prosody.analyze_and_report(
        [
            (
                "أَنَامُ مِلْءَ جُفُونِي عَنْ شَوَارِدِهَا",
                "وَيَسْهَرُ الْخَلْقُ جَرَّاهَا وَيَخْتَصِمُ",
            )
        ],
        meter_name="baseet",
        print_summary=False,
    )
    assert result_dict["meter"] == "baseet"
    assert "POEM CORRECTION REPORT" in report
