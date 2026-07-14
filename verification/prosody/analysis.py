"""
verification/prosody/analysis.py
=================================
Data model (FootResult/HemistichResult/VerseResult/PoemResult), meter-name
resolution, and the pyarud-driving analyze_poem/analyze_verse entry points.

Depends on verification/prosody/scoring.py for pattern comparison and
health classification; has no dependency on reporting.py.

See verification/prosody/__init__.py for how this module fits into the
Phase 3 split, and for the rationale on why _enrich_foot/_enrich_hemistich
live here rather than in reporting.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verification.prosody.scoring import (
    CANONICAL_PATTERNS,
    FootStatus,
    HealthLevel,
    _ZIHAF_MAP,
    binary_to_ux,
    foot_health,
    identify_zihaf,
)

# ---------------------------------------------------------------------------
# Optional pyarud imports — fail clearly if library is absent
#
# NOTE: only ArudhProcessor is actually referenced below (via _get_processor).
# ArudiConverter and the Tafeela subclasses are unused dead imports carried
# over verbatim from the original arabic_prosody_feedback.py's "embedded
# subset" copy-paste. Confirmed no import-time side effects in pyarud.tafeela
# (plain class defs, no registration). Left in place rather than trimmed —
# this is a structural extraction, not a behavior/cleanup change; flagging
# here for a future dedicated cleanup pass instead.
# ---------------------------------------------------------------------------
try:
    from pyarud.processor import ArudhProcessor
    from pyarud.arudi import ArudiConverter
    from pyarud.tafeela import (
        Fawlon,
        Faelon,
        Faelaton,
        Mafaeelon,
        Mustafelon,
        Mutafaelon,
        Mafaelaton,
        Mafoolato,
        Fae_laton,
        Mustafe_lon,
    )

    _PYARUD_AVAILABLE = True
except ImportError:
    _PYARUD_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Data model
# ---------------------------------------------------------------------------


@dataclass
class FootResult:
    """
    Per-foot analysis result enriched with zihāf identification.

    All pattern fields use **U/_ notation** (U = short/mutaharrik,
    _ = long/sākin).

    Attributes
    ----------
    foot_index:
        Zero-based position of this foot within the hemistich.
    expected_pattern:
        U/_ string of the best-matched canonical (or zihāf) form,
        e.g. ``"UU_U_"``.
    actual_segment:
        U/_ string extracted from the input at this position.
    canonical_pattern:
        The pristine (Salīm / un-modified) pattern for this foot type.
    score:
        SequenceMatcher ratio raised to the 6th power, in ``[0.0, 1.0]``.
    status:
        ``"ok"`` · ``"broken"`` · ``"missing"`` · ``"extra_bits"``
    zihaf_name:
        Name of the applied zihāf/ʿilla, or ``"Salim"`` if pristine, or
        ``None`` when status is not ``"ok"``.
    health:
        Collapsed health label — see :func:`verification.prosody.scoring.foot_health`.
    position_label:
        Human-readable foot role: ``"Hashw"``, ``"ʿArūḍ"``, or ``"Ḍarb"``.
    """

    foot_index: int
    expected_pattern: str
    actual_segment: str
    canonical_pattern: str
    score: float
    status: FootStatus
    zihaf_name: str | None
    health: HealthLevel
    position_label: str


@dataclass
class HemistichResult:
    """Enriched analysis of one hemistich (Ṣadr or ʿAjuz)."""

    text: str
    pattern: str  # U/_ notation
    feet: list[FootResult]
    score: float  # average foot score
    is_sound: bool  # True ↔ all feet are "ok"
    broken_foot_indices: list[int]
    missing_foot_count: int
    extra_bits: str | None  # remaining bits (U/_ notation) after all feet consumed


@dataclass
class VerseResult:
    """Full analysis of one bayt (verse = Ṣadr + ʿAjuz)."""

    verse_index: int
    sadr: HemistichResult
    ajuz: HemistichResult | None  # None for single-hemistich input
    combined_score: float
    meter: str
    issues: list[str] = field(default_factory=list)  # plain-English diagnostics


@dataclass
class PoemResult:
    """Top-level result for a full poem."""

    meter: str
    meter_arabic: str
    total_verses: int
    verses: list[VerseResult]
    overall_score: float
    is_metrically_sound: bool
    candidate_meters: list[tuple[str, float]]  # top-3 (meter, score)


# ---------------------------------------------------------------------------
# 2. Static lookup tables
# ---------------------------------------------------------------------------

#: Arabic display names keyed by the pyarud meter string.
METER_ARABIC_NAMES: dict[str, str] = {
    "taweel": "الطويل",
    "madeed": "المديد",
    "baseet": "البسيط",
    "wafer": "الوافر",
    "kamel": "الكامل",
    "hazaj": "الهزج",
    "rajaz": "الرجز",
    "ramal": "الرمل",
    "saree": "السريع",
    "munsareh": "المنسرح",
    "khafeef": "الخفيف",
    "mudhare": "المضارع",
    "muqtadheb": "المقتضب",
    "mujtath": "المجتث",
    "mutakareb": "المتقارب",
    "mutadarak": "المتدارك",
}

#: Hashw template (canonical feet listing) for all 16 meters.
METER_TEMPLATES: dict[str, str] = {
    "taweel": "فَعُولُنْ مَفَاعِيلُنْ فَعُولُنْ مَفَاعِلُ",
    "madeed": "فَاعِلَاتُنْ فَاعِلُنْ فَاعِلَاتُ",
    "baseet": "مُسْتَفْعِلُنْ فَاعِلُنْ مُسْتَفْعِلُنْ فَعِلُ",
    "wafer": "مُفَاعَلَتُنْ مُفَاعَلَتُنْ فَعُولُ",
    "kamel": "مُتَفَاعِلُنْ مُتَفَاعِلُنْ مُتَفَاعِلُ",
    "hazaj": "مَفَاعِيلُنْ مَفَاعِيلُ",
    "rajaz": "مُسْتَفْعِلُنْ مُسْتَفْعِلُنْ مُسْتَفْعِلُ",
    "ramal": "فَاعِلَاتُنْ فَاعِلَاتُنْ فَاعِلَاتُ",
    "saree": "مُسْتَفْعِلُنْ مُسْتَفْعِلُنْ فَاعِلُ",
    "munsareh": "مُسْتَفْعِلُنْ مَفْعُولَاتُ مُفْتَعِلُ",
    "khafeef": "فَاعِلَاتُنْ مُسْتَفْعِلُنْ فَاعِلَاتُ",
    "mudhare": "مَفَاعِيلُ فَاعِلَاتُ",
    "muqtadheb": "مَفْعُولَاتُ مُفْتَعِلُ",
    "mujtath": "مُسْتَفْعِلُنْ فَاعِلَاتُ",
    "mutakareb": "فَعُولُنْ فَعُولُنْ فَعُولُنْ فَعُولُ",
    "mutadarak": "فَعِلُنْ فَعِلُنْ فَعِلُنْ فَعِلُ",
}

#: Resolution of any meter name variant (English / Arabic / transliterated)
#: to a canonical key used by :func:`_resolve_key`.
_ALIASES: dict[str, str] = {
    # Ṭawīl
    "tawil": "tawil",
    "al-tawil": "tawil",
    "ṭawīl": "tawil",
    "طويل": "tawil",
    "الطويل": "tawil",
    # Basīṭ
    "basit": "basit",
    "al-basit": "basit",
    "basīṭ": "basit",
    "بسيط": "basit",
    "البسيط": "basit",
    # Kāmil
    "kamil": "kamil",
    "al-kamil": "kamil",
    "kāmil": "kamil",
    "كامل": "kamil",
    "الكامل": "kamil",
    # Wāfir
    "wafir": "wafir",
    "al-wafir": "wafir",
    "wāfir": "wafir",
    "وافر": "wafir",
    "الوافر": "wafir",
    # Ramal
    "ramal": "ramal",
    "al-ramal": "ramal",
    "رمل": "ramal",
    "الرمل": "ramal",
    # Mutaqārib
    "mutaqarib": "mutaqarib",
    "al-mutaqarib": "mutaqarib",
    "mutaqārib": "mutaqarib",
    "متقارب": "mutaqarib",
    "المتقارب": "mutaqarib",
    # Mutadārak
    "mutadarak": "mutadarak",
    "al-mutadarak": "mutadarak",
    "mutadārak": "mutadarak",
    "متدارك": "mutadarak",
    "المتدارك": "mutadarak",
    "الخبب": "mutadarak",
    # Rajaz
    "rajaz": "rajaz",
    "al-rajaz": "rajaz",
    "رجز": "rajaz",
    "الرجز": "rajaz",
    # Khafīf
    "khafif": "khafif",
    "al-khafif": "khafif",
    "khafīf": "khafif",
    "خفيف": "khafif",
    "الخفيف": "khafif",
}

#: Translation from the ``_ALIASES`` canonical naming scheme (e.g.
#: "khafif", "basit") to the meter keys expected by pyarud's
#: ArudhProcessor / used in METER_ARABIC_NAMES / METER_TEMPLATES
#: (e.g. "khafeef", "baseet").
_METER_TABLE_TO_PYARUD: dict[str, str] = {
    "tawil": "taweel",
    "basit": "baseet",
    "kamil": "kamel",
    "wafir": "wafer",
    "ramal": "ramal",
    "mutaqarib": "mutakareb",
    "mutadarak": "mutadarak",
    "rajaz": "rajaz",
    "khafif": "khafeef",
}


# ---------------------------------------------------------------------------
# 3. Core analysis helpers
# ---------------------------------------------------------------------------


def _require_pyarud() -> None:
    if not _PYARUD_AVAILABLE:
        raise RuntimeError("pyarud is not installed. Run: pip install pyarud")


def _get_processor() -> "ArudhProcessor":
    _require_pyarud()
    return ArudhProcessor()


def _enrich_foot(raw: dict, position_label: str, num_feet: int) -> FootResult:
    """
    Convert a raw pyarud ``FootAnalysis`` dict to an enriched :class:`FootResult`.

    Internal computation uses binary strings for _ZIHAF_MAP lookups; the
    stored ``FootResult`` fields are converted to U/_ notation before return.
    """
    expected = raw.get("expected_pattern", "")
    actual = raw.get("actual_segment", "")
    status = raw.get("status", "broken")
    score = raw.get("score", 0.0)

    canonical_values = set(CANONICAL_PATTERNS.values())
    canonical = expected  # default

    zihaf_name: str | None = None
    if status == "ok":
        if expected in canonical_values and expected == actual:
            zihaf_name = "Salim"
        elif (expected, actual) in _ZIHAF_MAP:
            canonical = expected
            zihaf_name = _ZIHAF_MAP[(expected, actual)]
        else:
            found_canon: str | None = None
            for key in _ZIHAF_MAP:
                if key[1] == actual and key[0] in canonical_values:
                    # If expected pattern itself is canonical, do not map to a different canonical's zihāf
                    if expected in canonical_values and key[0] != expected:
                        continue
                    found_canon = key[0]
                    break
            if found_canon:
                canonical = found_canon
                zihaf_name = identify_zihaf(canonical, actual)
            else:
                zihaf_name = identify_zihaf(expected, actual)

    health = foot_health(status, score, zihaf_name)

    return FootResult(
        foot_index=raw.get("foot_index", 0),
        expected_pattern=binary_to_ux(expected),
        actual_segment=binary_to_ux(actual),
        canonical_pattern=binary_to_ux(canonical),
        score=score,
        status=status,
        zihaf_name=zihaf_name,
        health=health,
        position_label=position_label,
    )


def _enrich_hemistich(
    text: str,
    pattern: str,  # raw binary from pyarud
    raw_feet: list[dict],
    *,
    is_ajuz: bool = False,
) -> HemistichResult:
    """Build a :class:`HemistichResult` from raw pyarud foot dicts."""
    n = len(raw_feet)
    enriched: list[FootResult] = []
    for i, raw in enumerate(raw_feet):
        if raw.get("status") == "extra_bits":
            position_label = "Extra"
        elif i == n - 1 and n > 0:
            position_label = "Ḍarb" if is_ajuz else "ʿArūḍ"
        else:
            position_label = "Hashw"
        enriched.append(_enrich_foot(raw, position_label, n))

    broken = [f.foot_index for f in enriched if f.status == "broken"]
    missing = sum(1 for f in enriched if f.status == "missing")
    extra_bits_entry = next(
        (f.actual_segment for f in enriched if f.status == "extra_bits"), None
    )

    avg_score = sum(
        f.score for f in enriched if f.status not in ("missing", "extra_bits")
    ) / max(1, sum(1 for f in enriched if f.status not in ("missing", "extra_bits")))

    # Bug 1b fix: Python's all() vacuously returns True for an empty iterable.
    # An empty foot list means pyarud parsed nothing (e.g. unrecognised meter
    # key silently fell through), so we must treat it as *not* sound.
    is_sound = bool(enriched) and all(f.status == "ok" for f in enriched)

    return HemistichResult(
        text=text,
        pattern=binary_to_ux(pattern),  # converted to U/_ notation
        feet=enriched,
        score=avg_score,
        is_sound=is_sound,
        broken_foot_indices=broken,
        missing_foot_count=missing,
        extra_bits=extra_bits_entry,
    )


# ---------------------------------------------------------------------------
# 4. Public analysis API
# ---------------------------------------------------------------------------


def generate_diagnostics(
    sadr: HemistichResult,
    ajuz: HemistichResult | None,
    meter: str,
) -> list[str]:
    """
    Generate a list of plain-English diagnostic messages for a verse.

    The messages describe every structural issue found — broken feet, missing
    feet, leftover bits, and which specific zihāfāt are present.
    Pattern strings in messages use U/_ notation.
    """
    msgs: list[str] = []

    def _check(h: HemistichResult, label: str) -> None:
        for f in h.feet:
            loc = f"foot {f.foot_index + 1} ({f.position_label})"
            if f.status == "broken":
                msgs.append(
                    f"[{label}] {loc}: broken — "
                    f"observed «{f.actual_segment}» does not match any "
                    f"allowed form of the expected pattern «{f.expected_pattern}»."
                )
            elif f.status == "missing":
                msgs.append(
                    f"[{label}] {loc}: missing — "
                    "input is too short; the verse may be Majzūʾ (truncated)."
                )
            elif f.status == "extra_bits":
                msgs.append(
                    f"[{label}] extra bits after all feet consumed: "
                    f"«{f.actual_segment}» — verse may be too long."
                )
            elif f.status == "ok" and f.zihaf_name not in (None, "Salim"):
                msgs.append(
                    f"[{label}] {loc}: valid zihāf «{f.zihaf_name}» applied "
                    f"(canonical «{f.canonical_pattern}» → actual «{f.actual_segment}»)."
                )

        if h.missing_foot_count > 0:
            msgs.append(
                f"[{label}] {h.missing_foot_count} missing foot(s) — "
                "consider using a Majzūʾ / Mashtūr sub-variant."
            )

    _check(sadr, "Ṣadr")
    if ajuz:
        _check(ajuz, "ʿAjuz")

    if not msgs:
        msgs.append(
            f"Verse is metrically sound ({METER_ARABIC_NAMES.get(meter, meter)})."
        )

    return msgs


def analyze_poem(
    verses: list[tuple[str, str]],
    *,
    meter_name: str | None = None,
    top_n: int = 3,
) -> PoemResult:
    """
    Analyse a full Arabic poem and return a rich structured result.

    Parameters
    ----------
    verses:
        List of ``(sadr, ajuz)`` pairs of fully-diacritized Arabic strings.
        Pass ``ajuz=""`` for a single-hemistich line.
    meter_name:
        Force a specific meter (pyarud key, e.g. ``"baseet"``).  When
        ``None`` the meter is auto-detected.
    top_n:
        Number of candidate meters to include in ``PoemResult.candidate_meters``.

    Returns
    -------
    PoemResult

    Raises
    ------
    RuntimeError
        If pyarud is not installed.
    ValueError
        If ``verses`` is empty.

    Examples
    --------
    >>> result = analyze_poem([
    ...     ("أَنَامُ مِلْءَ جُفُونِي عَنْ شَوَارِدِهَا",
    ...      "وَيَسْهَرُ الْخَلْقُ جَرَّاهَا وَيَخْتَصِمُ"),
    ... ], meter_name="baseet")
    >>> result.meter
    'baseet'
    """
    if not verses:
        raise ValueError("verses must not be empty")
    _require_pyarud()

    # --- Bug 1a fix (part 1): auto-resolve any meter name variant before
    # passing to pyarud.  Without this, raw Arabic strings like 'الطويل' reach
    # ArudhProcessor unchanged, causing it to return per-verse error dicts
    # that silently cascade into vacuous-truth false-positives (is_sound=True,
    # feet=[], score=0.0).  resolve_meter_key() raises ValueError for truly
    # unknown names, so callers get a clear error instead of a silent bad result.
    if meter_name is not None and meter_name not in METER_ARABIC_NAMES:
        meter_name = resolve_meter_key(meter_name)

    proc = _get_processor()
    raw = proc.process_poem(verses, meter_name=meter_name)

    # --- Bug 1a fix (part 2): guard against a top-level error response. ---
    if "error" in raw:
        raise ValueError(
            f"pyarud returned an error for meter {meter_name!r}: {raw['error']}. "
            f"Pass a valid pyarud key (e.g. 'taweel', 'baseet') or use "
            f"resolve_meter_key() to translate any name variant first."
        )

    detected_meter: str = raw["meter"]
    verse_results: list[VerseResult] = []

    for i, verse_raw in enumerate(raw["verses"]):
        # --- Bug 1a fix (part 3): guard against per-verse error dicts. ---
        if "error" in verse_raw:
            raise ValueError(
                f"pyarud could not analyse verse {i + 1}: {verse_raw['error']}. "
                f"Ensure meter_name is a valid pyarud key (e.g. 'taweel') or "
                f"pass None for auto-detection."
            )
        sadr_text = verse_raw.get("sadr_text", verses[i][0])
        ajuz_text = verse_raw.get("ajuz_text", verses[i][1])
        full_pat = verse_raw.get("input_pattern", "")
        sadr_pat = full_pat[: len(full_pat) // 2]
        ajuz_pat = full_pat[len(full_pat) // 2 :]
        sadr_feet = verse_raw.get("sadr_analysis") or []
        ajuz_feet = verse_raw.get("ajuz_analysis")

        sadr_h = _enrich_hemistich(sadr_text, sadr_pat, sadr_feet, is_ajuz=False)
        ajuz_h = (
            _enrich_hemistich(ajuz_text, ajuz_pat, ajuz_feet, is_ajuz=True)
            if ajuz_feet is not None
            else None
        )

        combined_score = verse_raw.get("score", 0.0)
        issues = generate_diagnostics(sadr_h, ajuz_h, detected_meter)

        verse_results.append(
            VerseResult(
                verse_index=i,
                sadr=sadr_h,
                ajuz=ajuz_h,
                combined_score=combined_score,
                meter=detected_meter,
                issues=issues,
            )
        )

    overall = (
        sum(v.combined_score for v in verse_results) / len(verse_results)
        if verse_results
        else 0.0
    )

    candidates: list[tuple[str, float]] = [(detected_meter, overall)]

    return PoemResult(
        meter=detected_meter,
        meter_arabic=METER_ARABIC_NAMES.get(detected_meter, ""),
        total_verses=len(verse_results),
        verses=verse_results,
        overall_score=overall,
        is_metrically_sound=all(
            v.sadr.is_sound and (v.ajuz is None or v.ajuz.is_sound)
            for v in verse_results
        ),
        candidate_meters=candidates[:top_n],
    )


def analyze_verse(
    sadr: str,
    ajuz: str = "",
    *,
    meter_name: str | None = None,
) -> VerseResult:
    """
    Analyse a single verse and return a :class:`VerseResult`.

    Convenience wrapper around :func:`analyze_poem` for callers dealing with
    one verse at a time.
    """
    result = analyze_poem([(sadr, ajuz)], meter_name=meter_name)
    return result.verses[0]


# ---------------------------------------------------------------------------
# 5. Meter-name resolution
# ---------------------------------------------------------------------------


def _resolve_key(meter: str) -> str:
    """Resolve any meter name variant to a canonical alias-table key."""
    key = _ALIASES.get(meter.strip()) or _ALIASES.get(meter.strip().lower())
    if key is None:
        available = sorted(set(_ALIASES.values()))
        raise ValueError(
            f"Unknown meter: {meter!r}\n"
            f"Available keys: {available}\n"
            f"Arabic names: طويل، بسيط، كامل، وافر، رمل، متقارب، متدارك، رجز"
        )
    return key


def resolve_meter_key(meter: str | None) -> str | None:
    """
    Resolve any supported meter name variant (English, Arabic,
    transliterated, or pyarud's own spelling) to the meter key **pyarud**
    itself expects (e.g. ``"baseet"``, ``"khafeef"``) — the only spelling
    that matters, since that's what every downstream caller actually needs.

    Internally this first normalises *meter* to the alias-table's own
    canonical key (e.g. ``"basit"``, ``"khafif"``) via :func:`_resolve_key`,
    then translates that to pyarud's naming convention using
    :data:`_METER_TABLE_TO_PYARUD`. That intermediate key is never exposed;
    there is exactly one public resolution path now, and it always returns
    a pyarud-ready string.

    Raises ``ValueError`` if *meter* is a non-``None`` string that isn't
    recognised in any known spelling.

    Examples
    --------
    >>> resolve_meter_key("basit")
    'baseet'
    >>> resolve_meter_key("al-basit")
    'baseet'
    >>> resolve_meter_key("البسيط")
    'baseet'
    >>> resolve_meter_key("baseet")
    'baseet'
    >>> resolve_meter_key(None) is None
    True
    """
    if meter is None:
        return None

    # Already a valid pyarud-native key (e.g. "baseet", "khafeef") — pass
    # through unchanged rather than feeding it to _resolve_key, which only
    # knows the alias naming scheme and would raise ValueError.
    if meter in METER_ARABIC_NAMES:
        return meter

    key = _resolve_key(meter)
    return _METER_TABLE_TO_PYARUD.get(key, key)
