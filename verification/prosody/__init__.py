"""
verification/prosody/
======================
Phase 3 of docs/REFACTOR_PLAN.md: extraction target for
verification/arabic_prosody_feedback.py.

arabic_prosody_feedback.py started as a single 1392-line file mixing three
loosely-related concerns (cohesion 0.06 per GRAPH_REPORT.md's "LangGraph
Orchestration Pipeline"-style low-cohesion flag): pattern/health scoring,
pyarud-driven meter analysis, and LLM-facing correction-report formatting.
Split by responsibility:

    verification/prosody/scoring.py    Pure U/_-pattern comparison and
                                        zihaf/health classification. No
                                        pyarud import, no I/O.
    verification/prosody/analysis.py   Data model (FootResult/HemistichResult/
                                        VerseResult/PoemResult), meter-name
                                        resolution, and the pyarud-driving
                                        analyze_poem/analyze_verse entry
                                        points. Depends on scoring.py.
    verification/prosody/reporting.py  LLM-actionable correction-report
                                        text generation (ASCII grids,
                                        character-level diffs, numbered
                                        prescriptions). Depends on
                                        analysis.py and scoring.py.

Dependency direction is one-way: reporting -> analysis -> scoring. scoring.py
has no internal dependencies on the other two modules.

verification/arabic_prosody_feedback.py re-exports the full public surface
of all three modules for backward compatibility -- tools/prosody_tools.py's
`from verification import arabic_prosody_feedback as prosody` and its
`prosody.analyze_poem(...)` / `prosody.analyze_verse(...)` /
`prosody.generate_poem_correction_report(...)` calls all still resolve
unchanged. If a future session needs to patch internals in a test, patch
them on the `verification.prosody.<module>` they now live in, not on the
`arabic_prosody_feedback` facade -- e.g.
`patch.object(verification.prosody.analysis, "_enrich_foot", ...)`, not
`patch.object(arabic_prosody_feedback, "_enrich_foot", ...)`. Each function
resolves bare names via its OWN module's globals at call time, so a patch
applied only to the facade's re-exported binding has no effect on what the
function itself sees.

Design note (deviation from a prior handover's function grouping): a prior
session's handover suggested binary_to_ux/_enrich_foot/_enrich_hemistich as
the "UX-formatting cluster." Reading the actual call graph shows
_enrich_foot/_enrich_hemistich are called directly from analyze_poem to
build its structured result (VerseResult/PoemResult) -- they are core
analysis, not report formatting -- and putting them in reporting.py would
force analysis.py to import from reporting.py, inverting the intended
dependency direction. They live in analysis.py instead; binary_to_ux (a
pure 0/1 <-> U/_ notation converter with no formatting/layout logic) lives
in scoring.py, where identify_zihaf/foot_health/CANONICAL_PATTERNS/
_ZIHAF_MAP already live and where it's actually used.
"""
