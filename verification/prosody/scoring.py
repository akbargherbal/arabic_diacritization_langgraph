"""
verification/prosody/scoring.py
================================
Pure U/_ prosodic-pattern comparison, zihaf identification, and health
classification. No pyarud import, no I/O, no LLM-facing formatting -- every
function here takes strings/patterns in and returns data out, so it's
independently testable without a pyarud install or a full VerseResult.

See verification/prosody/__init__.py for how this module fits into the
Phase 3 split.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Literal, Optional

FootStatus = Literal["ok", "broken", "missing", "extra_bits"]
HealthLevel = Literal["perfect", "valid_zihaf", "broken", "severe"]

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
