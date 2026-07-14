"""
arabic_prosody_feedback.py
==========================
LLM-actionable metrical correction feedback for Arabic poetry.

--- Phase 3 note (docs/REFACTOR_PLAN.md) ---
This module used to hold everything directly (data model, static lookup
tables, pyarud-driven analysis, and LLM-facing correction-report
formatting) in ~1392 lines -- the low-cohesion split GRAPH_REPORT.md
flagged (cohesion 0.06). That implementation now lives in the
`verification/prosody/` package, split by responsibility:

    verification/prosody/scoring.py    Pure U/_-pattern comparison and
                                        zihaf/health classification.
    verification/prosody/analysis.py   Data model + pyarud-driven
                                        analyze_poem/analyze_verse.
    verification/prosody/reporting.py  LLM-facing correction-report text
                                        generation.

This file re-exports the same public names for backward compatibility --
`tools/prosody_tools.py`'s `from verification import arabic_prosody_feedback
as prosody` and its `prosody.analyze_poem(...)` / `prosody.analyze_verse(...)`
/ `prosody.generate_poem_correction_report(...)` calls all still resolve
unchanged. **If you're patching internals in a test, patch them on the
`verification.prosody.<module>` they now live in, not here** -- e.g.
`patch.object(verification.prosody.analysis, "_enrich_foot", ...)`, not
`patch.object(arabic_prosody_feedback, "_enrich_foot", ...)`. Each function
resolves bare names via its OWN module's globals at call time, so a patch
applied only to this facade's re-exported binding has no effect on what the
function itself sees. (Same caveat langgraph_pipeline.py's Phase 2 facade
documents for the identical reason.)

This module is **fully standalone**: it embeds the subset of
`arabic_prosody_helpers` (data model, lookup tables, and analysis helpers)
that it depends on, so it can run without that file being present.  It still
requires **pyarud** (``pip install pyarud``) for the actual prosodic analysis.

**Historical Upstream Bug (status: not reproducible as of pyarud==0.1.10):**
An earlier version of `pyarud`'s text converter (`arudi.py`) reportedly
mis-scanned words ending with a tanwīn fatḥ on an alif maqṣūra (e.g.,
'أَسًى', 'هُدًى', 'فَتًى') -- turning the tanwīn into a 'ن' but failing to skip
the trailing 'ى', appending an extra silent/sākin unit (e.g., rewriting
'أَسًى' as 'أسنى' instead of the correct 'أسن'). That would inflate the mora
count and produce false "broken" diagnostics. No phonetic-normalization
workaround for this was ever implemented in this codebase -- only documented
here.

Verified against the currently pinned `pyarud==0.1.10`
(`ArudiConverter.prepare_text`), this conversion is now correct for all
sample words above ('أَسًى' -> ('أسن', '110'), matching the described fix, not
the described bug). See
`tests/test_pyarud_upstream_regressions.py::test_tanwin_fatha_alif_maqsura_conversion_is_correct`,
which pins this behavior as a regression guard: if a future `pyarud` upgrade
reintroduces the bug, that test fails loudly instead of silently corrupting
scansion results. Do not bump the `pyarud` pin without re-running that test.
"""

from __future__ import annotations

# ===========================================================================
# Re-exports for backward compatibility -- see module docstring above for
# where each of these now actually lives, and the patch-target warning.
# ===========================================================================

from verification.prosody.scoring import (  # noqa: F401
    CANONICAL_PATTERNS,
    FootStatus,
    HealthLevel,
    _ZIHAF_MAP,
    binary_to_ux,
    ux_to_binary,
    similarity,
    identify_zihaf,
    foot_health,
    get_canonical_pattern,
    _pattern_to_class,
    _mora_diff,
)

from verification.prosody.analysis import (  # noqa: F401
    _PYARUD_AVAILABLE,
    FootResult,
    HemistichResult,
    VerseResult,
    PoemResult,
    METER_ARABIC_NAMES,
    METER_TEMPLATES,
    _ALIASES,
    _METER_TABLE_TO_PYARUD,
    _require_pyarud,
    _get_processor,
    _enrich_foot,
    _enrich_hemistich,
    generate_diagnostics,
    analyze_poem,
    analyze_verse,
    _resolve_key,
    resolve_meter_key,
)

from verification.prosody.reporting import (  # noqa: F401
    _TAFEELA_MNEMONIC_MAP,
    get_tafeela_mnemonic,
    _render_diff,
    _tafeela_label,
    generate_verse_correction,
    generate_poem_correction_report,
    analyze_and_report,
)
