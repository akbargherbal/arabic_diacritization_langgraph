"""
arabic_prosody_feedback.py
==========================
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

This module is **fully standalone**: it embeds the subset of
`arabic_prosody_helpers` (data model, lookup tables, and analysis helpers)
that it depends on, so it can run without that file being present.  It still
requires **pyarud** (``pip install pyarud``) for the actual prosodic analysis.

**Upstream Bug / Known Limitation:**
Due to an upstream bug in `pyarud`'s text converter (`arudi.py`), words ending
with a tanwīn fatḥ on an alif maqṣūra (e.g., 'أَسًى', 'هُدًى', 'فَتًى') are scanned
incorrectly. The converter turns the tanwīn into a 'ن' but fails to skip the
trailing 'ى', appending an extra silent/sākin unit (e.g., rewriting 'أَسًى' as
'أسنى' instead of the correct 'أسن'). This results in an inflated mora count
and false "broken" diagnostics.

*Workaround:* Phonetically normalize such input words before analysis (e.g.,
rewrite 'أَسًى' to 'أَسَنْ' or 'هُدًى' to 'هُدَنْ') to neutralize this behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Literal, Optional

# ===========================================================================
# Embedded subset of arabic_prosody_helpers.py
# ===========================================================================

# ---------------------------------------------------------------------------
# Optional pyarud imports — fail clearly if library is absent
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

FootStatus = Literal["ok", "broken", "missing", "extra_bits"]
HealthLevel = Literal["perfect", "valid_zihaf", "broken", "severe"]


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
        Collapsed health label — see :func:`foot_health`.
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

#: Canonical binary patterns for all ten prosodic feet (tafāʿīl).
#: These remain as raw binary strings; they are used internally for
#: pyarud comparisons.  The public API converts them to U/_ via binary_to_ux.
CANONICAL_PATTERNS: dict[str, str] = {
    "Fawlon": "11010",  # فَعُولُنْ
    "Faelon": "10110",  # فَاعِلُنْ
    "Faelaton": "1011010",  # فَاعِلَاتُنْ
    "Mafaeelon": "1101010",  # مَفَاعِيلُنْ
    "Mustafelon": "1010110",  # مُسْتَفْعِلُنْ
    "Mutafaelon": "1110110",  # مُتَفَاعِلُنْ
    "Mafaelaton": "1101110",  # مُفَاعَلَتُنْ
    "Mafoolato": "1010101",  # مَفْعُولَاتُ
    "Mustafe_lon": "1010110",  # مُسْتَفْعِ لُنْ  (split, Khafif context)
    "Fae_laton": "1011010",  # فَاعِ لَاتُنْ    (split, Muḍāriʿ context)
}

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

#: Known zihāfāt: (canonical_binary, modified_binary) → zihāf name.
#: All patterns are raw binary strings matching pyarud output.
#: Covers all single and compound zihāfāt defined in pyarud/zihaf.py.
_ZIHAF_MAP: dict[tuple[str, str], str] = {
    # --- Fawlon (فَعُولُنْ) 11010 ---
    ("11010", "1101"): "Qabadh",  # drop نون (5th = sākin)
    ("11010", "110"): "Hadhf",  # drop last sabab (لُنْ = "10")
    ("11010", "10"): "Batr",  # Hadhf + Qataa (فَعْ)
    # --- Faelon (فَاعِلُنْ) 10110 ---
    ("10110", "1110"): "Khaban",  # drop ا (index 1)
    # --- Faelaton (فَاعِلَاتُنْ) 1011010 ---
    ("1011010", "111010"): "Khaban",  # drop ا (index 1)
    ("1011010", "101101"): "Kaff",  # drop نون (last sākin of sabab)
    ("1011010", "10110"): "Hadhf",  # drop last sabab (تُنْ)
    ("1011010", "11101"): "Shakal",  # Khaban + Kaff
    ("1011010", "1011"): "Waqf",  # quiet last bit of watad mafrūq
    # --- Mafaeelon (مَفَاعِيلُنْ) 1101010 ---
    ("1101010", "110110"): "Qabadh",  # drop ياء (index 4)
    ("1101010", "110101"): "Kaff",  # drop نون (last sākin)
    ("1101010", "11010"): "Hadhf",  # drop last sabab
    ("1101010", "11011"): "Shakl_alt",  # Qabdh + Kaff (rare)
    # --- Mustafelon (مُسْتَفْعِلُنْ) 1010110 ---
    ("1010110", "110110"): "Khaban",  # drop سين (index 1)
    ("1010110", "101110"): "Tay",  # drop فاء (index 3)
    ("1010110", "11110"): "Khabal",  # Khaban + Tay (drop 1 & 3)
    ("1010110", "101010"): "Kasf",  # drop نون (Saree context)
    # --- Mutafaelon (مُتَفَاعِلُنْ) 1110110 ---
    ("1110110", "1010110"): "Edmaar",  # taskeen تاء → مُتْفَاعِلُنْ (also: Idmar)
    ("1110110", "110110"): "Waqas",  # drop تاء (index 3)
    ("1110110", "101110"): "Khazal",  # Edmaar + Tay
    # --- Mafaelaton (مُفَاعَلَتُنْ) 1101110 ---
    ("1101110", "110110"): "Akal",  # drop تُنْ (Wafir context)
    ("1101110", "1101010"): "Asab",  # taskeen لام → مُفَاعَلْتُنْ
    ("1101110", "11010"): "Qatf",  # Asab + Hadhf → مُفَاعَلْ
    # --- Mafoolato (مَفْعُولَاتُ) 1010101 ---
    ("1010101", "110101"): "Khaban",  # drop فاء (index 1)
    ("1010101", "101101"): "Tay",  # drop و (index 3)
    ("1010101", "10101"): "Kasf",  # drop تاء (last sākin)
    # --- Mustafe_lon / Fae_laton: same bit-patterns; context distinguishes them ---
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


# ---------------------------------------------------------------------------
# 3. Low-level utilities
# ---------------------------------------------------------------------------


def binary_to_ux(pattern: str) -> str:
    """
    Convert a pyarud binary prosodic pattern to U/_ notation.

    Examples
    --------
    >>> binary_to_ux("11010")
    'UU_U_'
    """
    return pattern.replace("1", "U").replace("0", "_")


def ux_to_binary(pattern: str) -> str:
    """
    Convert a U/_ prosodic pattern back to pyarud binary.

    Examples
    --------
    >>> ux_to_binary("UU_U_")
    '11010'
    """
    return pattern.replace("U", "1").replace("_", "0")


def similarity(a: str, b: str) -> float:
    """
    Compute the SequenceMatcher ratio raised to the 6th power.

    This mirrors pyarud's internal ``_get_similarity()`` and penalises even
    small mismatches heavily (ratio ``1.00`` → ``1.00``; ``0.95`` → ``~0.74``;
    ``0.80`` → ``~0.26``).
    """
    import math

    raw = SequenceMatcher(None, a, b).ratio()
    return math.pow(raw, 6)


def identify_zihaf(canonical: str, actual: str) -> str:
    """
    Identify the zihāf or ʿilla applied to a prosodic foot by comparing its
    canonical binary pattern against the observed pattern.

    The function first checks the static lookup table ``_ZIHAF_MAP``.  If no
    exact match is found it falls back to a structural delta comparison
    (counting dropped vs quieted bits) and returns a descriptive label.

    Examples
    --------
    >>> identify_zihaf("1010110", "110110")
    'Khaban'
    >>> identify_zihaf("11010", "11010")
    'Salim'
    """
    if canonical == actual:
        return "Salim"

    key = (canonical, actual)
    if key in _ZIHAF_MAP:
        return _ZIHAF_MAP[key]

    # Structural fallback -------------------------------------------------
    len_diff = len(canonical) - len(actual)
    if len_diff > 0:
        return f"Unknown ({len_diff} bit{'s' if len_diff > 1 else ''} dropped)"
    if len_diff < 0:
        return f"Unknown ({abs(len_diff)} bit{'s' if abs(len_diff) > 1 else ''} added)"

    # Same length but different bits → a Taskīn (quieting) operation
    diffs = sum(c != a for c, a in zip(canonical, actual))
    return f"Unknown (Taskeen, {diffs} bit{'s' if diffs > 1 else ''} changed)"


def foot_health(
    status: FootStatus, score: float, zihaf_name: str | None
) -> HealthLevel:
    """
    Collapse foot status into a four-level health label.

    - ``perfect``     : Foot is Salīm (canonical pattern, no modification).
    - ``valid_zihaf``: Foot is ``"ok"`` with a recognised classical modification.
    - ``broken``      : Foot cannot be mapped to any canonical or modified form.
    - ``severe``      : Foot is missing from the input or leftover bits remain.
    """
    if status in ("missing", "extra_bits"):
        return "severe"
    if status == "broken":
        return "broken"
    # status == "ok"
    if zihaf_name == "Salim":
        return "perfect"
    return "valid_zihaf"


def get_canonical_pattern(foot_class_name: str) -> str | None:
    """
    Return the canonical binary pattern for a named prosodic foot class.

    Examples
    --------
    >>> get_canonical_pattern("Fawlon")
    '11010'
    """
    return CANONICAL_PATTERNS.get(foot_class_name)


# ---------------------------------------------------------------------------
# 4. Core analysis helpers
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
# 5. Public analysis API
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
# 6. Meter-name resolution & Tafʿīla mnemonics
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


# ===========================================================================
# End of embedded arabic_prosody_helpers subset — feedback-specific code below
# ===========================================================================


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pattern_to_class(ux: str) -> Optional[str]:
    """Reverse-map a U/_ canonical pattern to its pyarud foot class name."""
    binary = ux_to_binary(ux)
    for name, pat in CANONICAL_PATTERNS.items():
        if pat == binary:
            return name
    return None


def _mora_diff(expected: str, actual: str) -> dict:
    """
    Compute diff metrics between two U/_ prosodic patterns.

    Returns
    -------
    dict
        len_diff    : int   — positive → actual is too short
        direction   : str   — "match" | "too_short" | "too_long" | "wrong_weight"
        first_div   : int   — index of first diverging character (-1 if none)
        suggestion  : str   — concise plain-English fix
    """
    if expected == actual:
        return {
            "len_diff": 0,
            "direction": "match",
            "first_div": -1,
            "suggestion": "Pattern matches exactly.",
        }

    len_diff = len(expected) - len(actual)
    min_len = min(len(expected), len(actual))
    first_div = min_len  # diverges at the length boundary by default
    for i in range(min_len):
        if expected[i] != actual[i]:
            first_div = i
            break

    if len_diff > 0:
        missing = expected[len(actual) :]
        units = [("long (_)" if c == "_" else "short (U)") for c in missing]
        sug = (
            f"Foot is {len_diff} mora(s) too short. "
            f"Extend with: {' + '.join(units)}  →  target «{expected}»."
        )
        return {
            "len_diff": len_diff,
            "direction": "too_short",
            "first_div": first_div,
            "suggestion": sug,
        }

    if len_diff < 0:
        extra = abs(len_diff)
        sug = (
            f"Foot is {extra} mora(s) too long ({len(actual)} units given, "
            f"{len(expected)} needed). Trim to produce «{expected}»."
        )
        return {
            "len_diff": len_diff,
            "direction": "too_long",
            "first_div": first_div,
            "suggestion": sug,
        }

    # Same length, wrong weights
    wrongs = [
        (i, expected[i], actual[i])
        for i in range(len(expected))
        if expected[i] != actual[i]
    ]
    fixes = []
    for pos, e, a in wrongs:
        e_lbl = "long (_)" if e == "_" else "short (U)"
        a_lbl = "long (_)" if a == "_" else "short (U)"
        fixes.append(f"pos {pos + 1}: {a_lbl} → {e_lbl}")
    sug = "Wrong syllable weight(s): " + "; ".join(fixes) + "."
    return {
        "len_diff": 0,
        "direction": "wrong_weight",
        "first_div": first_div,
        "suggestion": sug,
    }


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

    # ── Consolidated "what to fix" list ─────────────────────────────────────
    out.append("╔" + "═" * 66 + "╗")
    out.append("║  CONSOLIDATED FIX LIST")
    out.append("╚" + "═" * 66 + "╝")
    item = 1
    for v in poem.verses:
        if v.combined_score >= score_threshold:
            continue
        for h, label in [(v.sadr, "Ṣadr"), (v.ajuz, "ʿAjuz")]:
            if h is None:
                continue
            for f in h.feet:
                if f.status == "ok":
                    continue
                if f.status == "missing":
                    out.append(
                        f"  {item}. Verse {v.verse_index + 1} [{label}, Foot {f.foot_index + 1}]  "
                        f"ADD text for missing foot «{f.expected_pattern}»."
                    )
                elif f.status == "extra_bits":
                    out.append(
                        f"  {item}. Verse {v.verse_index + 1} [{label}]  "
                        f"REMOVE extra «{f.actual_segment}»."
                    )
                else:
                    info = _mora_diff(f.expected_pattern, f.actual_segment)
                    d = info["len_diff"]
                    verb = (
                        f"ADD {d} mora(s) to"
                        if d > 0
                        else f"TRIM {abs(d)} mora(s) from" if d < 0 else "REWEIGHT"
                    )
                    out.append(
                        f"  {item}. Verse {v.verse_index + 1} [{label}, Foot {f.foot_index + 1} "
                        f"({f.position_label})]  {verb}  «{f.actual_segment}»"
                        f"  →  «{f.expected_pattern}»."
                    )
                item += 1
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
