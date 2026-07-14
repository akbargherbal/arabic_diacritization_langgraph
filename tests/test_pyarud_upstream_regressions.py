"""
tests/test_pyarud_upstream_regressions.py
============================================
Guards behavior of the *upstream* `pyarud` library that this codebase's
correctness depends on but doesn't control. Phase 1 of docs/REFACTOR_PLAN.md.

Background: verification/arabic_prosody_feedback.py's module docstring
historically documented an upstream pyarud converter bug affecting words
ending in a tanwīn fatḥ on an alif maqṣūra (e.g. 'أَسًى', 'هُدًى') -- the
converter allegedly turned the tanwīn into 'ن' but failed to skip the
trailing 'ى', appending a spurious silent unit and inflating the mora count.
No workaround for this was ever implemented in this codebase, only
documented as a known limitation.

Investigating during the Phase 0/1 refactor pass, this bug does **not**
reproduce against the now-pinned `pyarud==0.1.10` (see requirements.txt) --
conversion is correct. This test exists so that:

  1. The "isolated node" the dependency graph flagged
     (`Rule 6: tanwīn fatḥ bug on alif maqṣūra`) is now wired into something
     concrete and checkable, instead of being free-floating prose only a
     human reading the docstring would ever notice.
  2. If a future `pyarud` upgrade reintroduces this bug (or a regression in
     a *different* upstream release), this test fails loudly at the exact
     boundary where it would otherwise silently produce false "broken"
     metrical diagnostics downstream.

If this test starts failing after a `pyarud` version bump: do not "fix" it
by changing the assertion. That means the historical bug is back. Either
re-pin pyarud to a version where it's absent, or implement the phonetic
normalization workaround the old docstring described, upstream of
`ArudiConverter.prepare_text`.
"""

import pytest

pyarud = pytest.importorskip("pyarud")
from pyarud.arudi import ArudiConverter  # noqa: E402

# (input word, expected (skeleton, pattern)) -- expected values are the
# CORRECT conversion (tanwīn fatḥ + alif maqṣūra collapsing to a single
# 'ن' with no extra unit), i.e. the outcome the old docstring's proposed
# workaround was trying to force by hand.
TANWIN_FATHA_ALIF_MAQSURA_CASES = [
    ("أَسًى", ("أسن", "110")),
    ("هُدًى", ("هدن", "110")),
    ("فَتًى", ("فتن", "110")),
    ("رَحًى", ("رحن", "110")),
    ("مَعْنًى", ("معنن", "1010")),
]

# Control cases without the tanwin-fatha/alif-maqsura combination, to make
# sure the assertion isn't accidentally trivially true for all input.
CONTROL_CASES = [
    ("كِتَابٌ", ("كتابو", "11010")),
    ("قَلَمْ", ("قلم", "110")),
]


@pytest.mark.parametrize("word, expected", TANWIN_FATHA_ALIF_MAQSURA_CASES)
def test_tanwin_fatha_alif_maqsura_conversion_is_correct(word, expected):
    converter = ArudiConverter()
    result = converter.prepare_text(word)
    assert result == expected, (
        f"pyarud's tanwin-fatha/alif-maqsura conversion regressed for {word!r}: "
        f"got {result}, expected {expected}. This matches the historical bug "
        f"described in verification/arabic_prosody_feedback.py's module "
        f"docstring -- see that docstring and this file's module docstring "
        f"before changing this assertion."
    )


@pytest.mark.parametrize("word, expected", CONTROL_CASES)
def test_control_words_convert_as_expected(word, expected):
    converter = ArudiConverter()
    assert converter.prepare_text(word) == expected
