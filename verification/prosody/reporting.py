"""
verification/prosody/reporting.py
==================================
LLM-actionable metrical correction feedback for Arabic poetry.

Provides three public functions:

    generate_verse_correction(verse)      → detailed feedback for one verse
    generate_poem_correction_report(poem) → full report for a poem
    analyze_and_report(verses, meter)     → convenience one-call entry point

Output format is designed to be fed directly back to an LLM as a correction
prompt: every broken foot is identified by its position, its expected U/_
pattern is shown alongside what was actually produced, a character-level diff
pinpoints the exact divergence, and a numbered prescription tells the model
precisely what to add, remove, or reweight.

Depends on verification/prosody/analysis.py (data model, analyze_poem,
resolve_meter_key) and verification/prosody/scoring.py (pattern comparison).

See verification/prosody/__init__.py for how this module fits into the
Phase 3 split.
"""

from __future__ import annotations

from dataclasses import asdict

from verification.prosody.analysis import (
    FootResult,
    HemistichResult,
    METER_ARABIC_NAMES,
    METER_TEMPLATES,
    PoemResult,
    VerseResult,
    analyze_poem,
    resolve_meter_key,
)
from verification.prosody.scoring import _mora_diff, _pattern_to_class

# ---------------------------------------------------------------------------
# Classical Arabic mnemonic spellings
# ---------------------------------------------------------------------------

#: Classical Arabic mnemonic spellings (Tafʿīla) keyed by
#: (canonical_foot_class, zihaf_name).
_TAFEELA_MNEMONIC_MAP: dict[tuple[str, str], str] = {
    # --- Fawlon (فَعُولُنْ) ---
    ("Fawlon", "Salim"): "فَعُولُنْ",
    ("Fawlon", "Qabadh"): "فَعُولُ",
    ("Fawlon", "Hadhf"): "فَعَلْ",
    ("Fawlon", "Batr"): "فَعْ",
    # --- Faelon (فَاعِلُنْ) ---
    ("Faelon", "Salim"): "فَاعِلُنْ",
    ("Faelon", "Khaban"): "فَعِلُنْ",
    # --- Faelaton (فَاعِلَاتُنْ) ---
    ("Faelaton", "Salim"): "فَاعِلَاتُنْ",
    ("Faelaton", "Khaban"): "فَعِلَاتُنْ",
    ("Faelaton", "Kaff"): "فَاعِلَاتُ",
    ("Faelaton", "Hadhf"): "فَاعِلُنْ",
    ("Faelaton", "Shakal"): "فَعِلَاتُ",
    # --- Mafaeelon (مَفَاعِيلُنْ) ---
    ("Mafaeelon", "Salim"): "مَفَاعِيلُنْ",
    ("Mafaeelon", "Qabadh"): "مَفَاعِلُنْ",
    ("Mafaeelon", "Kaff"): "مَفَاعِيلُ",
    ("Mafaeelon", "Hadhf"): "فَعُولُنْ",
    # --- Mustafelon (مُسْتَفْعِلُنْ) ---
    ("Mustafelon", "Salim"): "مُسْتَفْعِلُنْ",
    ("Mustafelon", "Khaban"): "مُتَفْعِلُنْ",  # fixed: was مُفَتْعِلُنْ
    ("Mustafelon", "Tay"): "مُفْتَعِلُنْ",
    ("Mustafelon", "Khabal"): "مُتَعِلُنْ",
    ("Mustafelon", "Kasf"): "مُسْتَفْعِلْ",
    # --- Mutafaelon (مُتَفَاعِلُنْ) ---
    ("Mutafaelon", "Salim"): "مُتَفَاعِلُنْ",
    ("Mutafaelon", "Edmaar"): "مُتْفَاعِلُنْ",
    ("Mutafaelon", "Idmar"): "مُتْفَاعِلُنْ",
    ("Mutafaelon", "Waqas"): "مُفَاعِلُنْ",
    ("Mutafaelon", "Khazal"): "مُتْفَعِلُنْ",
    # --- Mafaelaton (مُفَاعَلَتُنْ) ---
    ("Mafaelaton", "Salim"): "مُفَاعَلَتُنْ",
    ("Mafaelaton", "Akal"): "مُفَاعَلَتْ",
    ("Mafaelaton", "Asab"): "مُفَاعَلْتُنْ",
    ("Mafaelaton", "Qatf"): "فَعُولُنْ",
    # --- Mafoolato (مَفْعُولَاتُ) ---
    ("Mafoolato", "Salim"): "مَفْعُولَاتُ",
    ("Mafoolato", "Khaban"): "مَعُولَاتُ",
    ("Mafoolato", "Tay"): "مَفْعَلَاتُ",
    ("Mafoolato", "Kasf"): "مَفْعُولَا",
}


def get_tafeela_mnemonic(canonical_foot_class: str, zihaf_name: str | None) -> str:
    """
    Return the classical Arabic mnemonic spelling (Tafʿīla representation)
    for a given foot type and its active modification.

    Examples
    --------
    >>> get_tafeela_mnemonic("Mustafelon", "Khaban")
    'مُتَفْعِلُنْ'
    """
    if not zihaf_name:
        zihaf_name = "Salim"

    key = (canonical_foot_class, zihaf_name)
    if key in _TAFEELA_MNEMONIC_MAP:
        return _TAFEELA_MNEMONIC_MAP[key]

    default_key = (canonical_foot_class, "Salim")
    base_arabic = _TAFEELA_MNEMONIC_MAP.get(default_key, canonical_foot_class)

    if zihaf_name == "Salim":
        return base_arabic
    return f"{base_arabic} ({zihaf_name})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_diff(expected: str, actual: str, indent: str = "    ") -> str:
    """
    Three-line character-level diff of two U/_ patterns.

        Expected:  U _ U _ U U _
        Actual:    U _ U U _ · ·
        Diff:      | | | × × ^ ^

    Legend  | = match   × = wrong weight   ^ = missing in actual   v = extra
    """
    max_len = max(len(expected), len(actual))
    exp_p = expected.ljust(max_len, "·")
    act_p = actual.ljust(max_len, "·")

    markers = []
    for e, a in zip(exp_p, act_p):
        if e == a and e != "·":
            markers.append("|")
        elif e == "·":
            markers.append("v")  # extra in actual
        elif a == "·":
            markers.append("^")  # missing in actual
        else:
            markers.append("×")  # weight mismatch

    return (
        f"{indent}Expected:  {' '.join(exp_p)}\n"
        f"{indent}Actual:    {' '.join(act_p)}\n"
        f"{indent}Diff:      {' '.join(markers)}"
        f"    (| match  × wrong weight  ^ missing  v extra)"
    )


def _tafeela_label(f: FootResult) -> str:
    """Return the Arabic tafʿīla mnemonic for a foot, or its pattern if unknown."""
    cls = _pattern_to_class(f.canonical_pattern)
    if cls:
        return get_tafeela_mnemonic(cls, f.zihaf_name)
    return f.actual_segment


# ---------------------------------------------------------------------------
# Core: per-verse correction report
# ---------------------------------------------------------------------------


def generate_verse_correction(
    verse: VerseResult,
    *,
    include_meter_schema: bool = True,
) -> str:
    """
    Generate structured, LLM-actionable correction feedback for one verse.

    Compared with ``generate_diagnostics`` (which lists issues in prose), this
    function produces a compact, machine-parseable report with:

    - Expected vs actual U/_ pattern per hemistich, foot by foot in a grid
    - Character-level diff for every broken foot
    - Mora-count guidance ("2 units too short — extend with long + short")
    - Numbered correction prescriptions the LLM can act on directly
    - Optional meter schema reminder

    Parameters
    ----------
    verse:
        A :class:`VerseResult` from :func:`analyze_verse` or :func:`analyze_poem`.
    include_meter_schema:
        Append the canonical hemistich patterns at the end (useful for
        the first broken verse in a poem; can be suppressed for subsequent
        ones to keep the output concise).

    Returns
    -------
    str
        Multi-line feedback string.  Prepend directly to an LLM re-generation
        request.
    """
    out: list[str] = []
    BAR = "═" * 66

    pct = verse.combined_score * 100
    if verse.combined_score >= 1.0:
        tag = "✓ SOUND"
    elif verse.combined_score >= 0.90:
        tag = "~ NEAR-PERFECT"
    elif verse.combined_score >= 0.70:
        tag = "⚠ IRREGULAR"
    else:
        tag = "✗ BROKEN"

    meter_ar = METER_ARABIC_NAMES.get(verse.meter, verse.meter)

    out.append(BAR)
    out.append(f"  VERSE {verse.verse_index + 1}  ·  {tag}  ·  Score: {pct:.0f}%")
    out.append(BAR)
    out.append(f"  Meter : {meter_ar} ({verse.meter})")
    out.append(f"  Ṣadr  : {verse.sadr.text}")
    if verse.ajuz:
        out.append(f"  ʿAjuz : {verse.ajuz.text}")
    out.append("")

    # Collect broken feet for the detailed section
    all_broken: list[tuple[str, FootResult]] = []

    def _hemistich_grid(h: HemistichResult, label: str) -> None:
        """Render the expected / actual / status grid for one hemistich."""
        out.append(f"  ┌─ {label} {'─' * (56 - len(label))}┐")

        # Build aligned columns — one column per foot
        cols: list[dict] = []
        for f in h.feet:
            w = max(len(f.expected_pattern), len(f.actual_segment), 6)
            if f.status == "ok":
                if f.zihaf_name == "Salim":
                    st = "✓"
                    tafeela = _tafeela_label(f)
                else:
                    st = f"~{f.zihaf_name}"
                    tafeela = _tafeela_label(f)
            elif f.status == "broken":
                st = "✗ BROKEN"
                tafeela = "BROKEN"
                all_broken.append((label, f))
            elif f.status == "missing":
                st = "? MISSING"
                tafeela = "MISSING"
                all_broken.append((label, f))
            else:
                st = "! EXTRA"
                tafeela = "EXTRA"
                all_broken.append((label, f))

            cols.append(
                {
                    "w": w,
                    "exp": f.expected_pattern,
                    "act": f.actual_segment,
                    "st": st,
                    "tf": tafeela,
                }
            )

        def _row(key: str) -> str:
            parts = [f"[{c[key].center(c['w'])}]" for c in cols]
            return "  │  " + "  ".join(parts)

        out.append(_row("exp") + "   ← Expected")
        out.append(_row("act") + "   ← Actual")
        out.append(_row("st"))
        out.append(_row("tf"))

        # Mora totals
        exp_total = sum(len(c["exp"]) for c in cols)
        act_total = sum(len(c["act"]) for c in cols)
        mora_note = ""
        if exp_total != act_total:
            diff = act_total - exp_total
            mora_note = f"  ⚠ hemistich is {abs(diff)} mora(s) {'too long' if diff > 0 else 'too short'}"
        out.append(f"  │  Morae: expected {exp_total}, actual {act_total}{mora_note}")
        out.append(f"  └{'─' * 62}┘")
        out.append("")

    _hemistich_grid(verse.sadr, "ṢADR (صَدْر)")
    if verse.ajuz:
        _hemistich_grid(verse.ajuz, "ʿAJUZ (عَجُز)")

    # ── Detailed diagnosis ──────────────────────────────────────────────────
    if not all_broken:
        out.append("  ✓ All feet parsed correctly — verse is metrically sound.")
        out.append("")
    else:
        out.append("  ── DETAILED DIAGNOSIS " + "─" * 42)
        prescriptions: list[str] = []

        for hem_label, f in all_broken:
            out.append("")
            out.append(
                f"  ▸ [{hem_label}  |  Foot {f.foot_index + 1}  |  {f.position_label}]"
            )

            if f.status == "missing":
                out.append(f"    ✗ Foot is MISSING — hemistich is too short.")
                out.append(
                    f"    Expected : {f.expected_pattern}  ({len(f.expected_pattern)} morae)"
                )
                prescriptions.append(
                    f"[{hem_label}, Foot {f.foot_index + 1}]  Add text supplying "
                    f"the missing foot «{f.expected_pattern}» ({len(f.expected_pattern)} morae)."
                )
                continue

            if f.status == "extra_bits":
                out.append(f"    ✗ Extra material after all expected feet consumed.")
                out.append(
                    f"    Extra bits: {f.actual_segment}  ({len(f.actual_segment)} morae)"
                )
                prescriptions.append(
                    f"[{hem_label}]  Remove word(s) producing the trailing "
                    f"«{f.actual_segment}» ({len(f.actual_segment)} extra morae)."
                )
                continue

            # Broken foot — character-level diff + prescription
            info = _mora_diff(f.expected_pattern, f.actual_segment)
            out.append(
                f"    Expected : {f.expected_pattern}  ({len(f.expected_pattern)} morae)"
            )
            out.append(
                f"    Actual   : {f.actual_segment}  ({len(f.actual_segment)} morae)"
            )
            out.append("")
            out.append(
                _render_diff(f.expected_pattern, f.actual_segment, indent="    ")
            )
            out.append("")
            out.append(f"    → {info['suggestion']}")

            # Build the numbered prescription
            diff = info["len_diff"]
            if diff > 0:
                prescriptions.append(
                    f"[{hem_label}, Foot {f.foot_index + 1} ({f.position_label})]  "
                    f"Replace word(s) giving «{f.actual_segment}» with word(s) giving "
                    f"«{f.expected_pattern}» — need {diff} more mora(s)."
                )
            elif diff < 0:
                prescriptions.append(
                    f"[{hem_label}, Foot {f.foot_index + 1} ({f.position_label})]  "
                    f"Shorten word(s) giving «{f.actual_segment}» to produce "
                    f"«{f.expected_pattern}» — remove {abs(diff)} mora(s)."
                )
            else:
                prescriptions.append(
                    f"[{hem_label}, Foot {f.foot_index + 1} ({f.position_label})]  "
                    f"Adjust syllable weights in «{f.actual_segment}» to match "
                    f"«{f.expected_pattern}» (same length, wrong weight pattern)."
                )

        out.append("")
        out.append("  ── CORRECTION PRESCRIPTION " + "─" * 37)
        for i, p in enumerate(prescriptions, 1):
            out.append(f"  {i}. {p}")
        out.append("")

    # ── Meter reference ─────────────────────────────────────────────────────
    if include_meter_schema:
        out.append("  ── METER REFERENCE " + "─" * 45)
        out.append(f"  البحر  : {meter_ar} ({verse.meter})")
        template = METER_TEMPLATES.get(verse.meter, "")
        if template:
            out.append(f"  Tafāʿīl: {template}")
        sadr_pat = " | ".join(
            f.expected_pattern for f in verse.sadr.feet if f.status != "extra_bits"
        )
        if sadr_pat:
            out.append(f"  Ṣadr   : {sadr_pat}")
        if verse.ajuz:
            ajuz_pat = " | ".join(
                f.expected_pattern for f in verse.ajuz.feet if f.status != "extra_bits"
            )
            if ajuz_pat:
                out.append(f"  ʿAjuz  : {ajuz_pat}")
        out.append("")

    out.append(BAR)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Poem-level report
# ---------------------------------------------------------------------------


def generate_poem_correction_report(
    poem: PoemResult,
    *,
    only_broken: bool = True,
    score_threshold: float = 0.99,
    include_meter_schema: bool = True,
) -> str:
    """
    Generate a full correction report for all verses in a poem.

    Parameters
    ----------
    poem:
        A :class:`PoemResult` from :func:`analyze_poem`.
    only_broken:
        Skip verses whose ``combined_score >= score_threshold``.
        Default ``True``.
    score_threshold:
        Verses at or above this score are considered metrically sound.
        Default ``0.99`` (only perfectly-matching verses are skipped).
    include_meter_schema:
        Append the meter schema to the *first* broken verse only, to avoid
        repeating it on every verse.

    Returns
    -------
    str
        Full multi-line report.
    """
    out: list[str] = []
    W = "═" * 66
    meter_ar = METER_ARABIC_NAMES.get(poem.meter, poem.meter)
    pct = poem.overall_score * 100

    out.append("╔" + W + "╗")
    out.append(f"║  POEM CORRECTION REPORT  ·  {meter_ar} ({poem.meter})")
    out.append(f"║  {poem.total_verses} verses  ·  Overall score: {pct:.1f}%")
    out.append("╚" + W + "╝")
    out.append("")

    # ── Summary table ────────────────────────────────────────────────────────
    out.append("  VERSE SUMMARY")
    out.append("  " + "─" * 60)
    broken_indices: list[int] = []
    for v in poem.verses:
        is_ok = v.combined_score >= score_threshold
        flag = "✓" if is_ok else "✗"
        if not is_ok:
            broken_indices.append(v.verse_index)
        pv = v.combined_score * 100
        preview = v.sadr.text[:35] + ("…" if len(v.sadr.text) > 35 else "")
        out.append(f"  {flag}  Verse {v.verse_index + 1:>2}  [{pv:5.1f}%]  {preview}")
    out.append("  " + "─" * 60)
    out.append(f"  Broken / total: {len(broken_indices)} / {poem.total_verses}")
    out.append("")

    if not broken_indices:
        out.append("  ✓ All verses are metrically sound. Nothing to correct.")
        return "\n".join(out)

    # ── Per-verse detailed feedback ──────────────────────────────────────────
    schema_shown = False
    for v in poem.verses:
        if only_broken and v.combined_score >= score_threshold:
            continue
        show_schema = include_meter_schema and not schema_shown
        out.append(generate_verse_correction(v, include_meter_schema=show_schema))
        schema_shown = True
        out.append("")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def analyze_and_report(
    verses: list[tuple[str, str]],
    meter_name: str | None = None,
    *,
    only_broken: bool = True,
    score_threshold: float = 0.99,
    print_summary: bool = True,
) -> tuple[dict, str]:
    """
    Analyse a poem and produce a correction report in one call.

    This replaces the typical ``analyze_poem_to_dict(...)`` workflow when you
    want actionable feedback, not just accuracy percentages.

    Parameters
    ----------
    verses:
        List of ``(sadr, ajuz)`` pairs of fully-diacritized Arabic strings.
    meter_name:
        Any supported meter name variant (e.g. ``'khafif'``, ``'الخفيف'``,
        ``'khafeef'``).  ``None`` = auto-detect.
    only_broken:
        Include only problematic verses in the per-verse detail section.
    score_threshold:
        Verses at or above this score are considered sound. Default 0.99.
    print_summary:
        Print the one-line accuracy summary (same as ``analyze_poem_to_dict``).

    Returns
    -------
    tuple[dict, str]
        ``(result_dict, correction_report)``
        - ``result_dict`` mirrors the output of ``analyze_poem_to_dict``.
        - ``correction_report`` is the full LLM-ready feedback string.
    """
    pyarud_key = resolve_meter_key(meter_name)
    poem = analyze_poem(verses, meter_name=pyarud_key)

    if print_summary:
        for v in poem.verses:
            sadr = v.sadr.text
            ajuz = v.ajuz.text if v.ajuz else ""
            combined = f"{sadr} | {ajuz}" if ajuz else sadr
            pct = v.combined_score * 100
            print(f"Verse {v.verse_index + 1}: «{combined}» — Accuracy: {pct:.2f}%")

    report = generate_poem_correction_report(
        poem,
        only_broken=only_broken,
        score_threshold=score_threshold,
    )
    return asdict(poem), report
