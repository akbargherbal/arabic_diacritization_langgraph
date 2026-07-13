"""
tools/fidelity_tools.py
=========================
Fidelity axis: the diacritized output's consonant/letter skeleton must
exactly match the skeleton of the undiacritized input verse it was derived
from. This is a FOURTH independent, deciding axis alongside Structural
(pyarud) and Security (sanitizer) -- see dataset_tools.py and README.md,
which previously named only those two as "genuinely independent, deciding"
(as opposed to advisory, e.g. irab_checker / naturalness_critic).

Why this exists: pyarud only scores the U/_ prosodic pattern of whatever
text it's handed. It has no way to know whether that text is the verse it
was asked to diacritize, a different (also metrically valid) canonical
verse the model recalled instead, or a single-letter substitution that
changes meaning while preserving syllable weight. Diacritic marks alone
(harakat/shadda/tanwin/sukun) can't drift the skeleton by definition --
this check is specifically about the base letters underneath them.

Trust boundary: the ground-truth input is loaded HERE, from
dataset/inputs/*.jsonl, by verse_id -- it is NOT accepted as a parameter
from the caller. If it were a parameter, a hallucinating (or lying) agent
turn could simply hand back whatever text it already produced as "the
original" and the check would be vacuous. dataset/inputs/** must remain
in the same permission-denied class as config/meter_tables.py: read-only
to the agent, so this file is the actual source of record for what
verse_id maps to what input text.
"""

import json
import pathlib
from difflib import SequenceMatcher

INPUTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "dataset" / "inputs"

# Same skeleton alphabet as sanitization_tools.py's normalize_text.
# TODO: import from sanitization_tools instead of duplicating, once that
# module exposes AR_CHARS/normalize_text as public names rather than
# inlining them locally. Until then, keep the two lists in sync by hand.
AR_CHARS = frozenset(" ءأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي")


def normalize_text(text: str) -> str:
    """Strip everything except base letters and spaces -- diacritics are
    dropped because none of them are in AR_CHARS. Hamza seat
    (أ/إ/ؤ/ئ/ء/ا) and alif maqsura vs ya (ى/ي) are deliberately kept
    DISTINCT, not folded together: a hamza-seat or alif-maqsura swap is
    exactly the class of silent error this check exists to catch.
    Whitespace is collapsed so incidental spacing differences between
    input and output don't register as false mismatches.
    """
    filtered = "".join(ch for ch in text if ch in AR_CHARS)
    return " ".join(filtered.split())


_input_index_cache: dict[str, dict] | None = None
_input_index_mtime: float | None = None


def _load_input_index() -> dict[str, dict]:
    """Build (and cache) verse_id -> {sadr, ajuz} from every *.jsonl under
    dataset/inputs/. Cache invalidates on any file's mtime change, so a
    new batch file dropped in mid-run is picked up without a restart.
    """
    global _input_index_cache, _input_index_mtime

    files = sorted(INPUTS_DIR.glob("*.jsonl"))
    latest_mtime = max((f.stat().st_mtime for f in files), default=0.0)

    if _input_index_cache is not None and latest_mtime == _input_index_mtime:
        return _input_index_cache

    index: dict[str, dict] = {}
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                index[row["verse_id"]] = {"sadr": row["sadr"], "ajuz": row["ajuz"]}

    _input_index_cache = index
    _input_index_mtime = latest_mtime
    return index


def verify_skeleton_fidelity_tool(verse_id: str, sadr: str, ajuz: str) -> dict:
    """Compare the normalized (diacritics-stripped) skeleton of the
    proposed committed text against the normalized skeleton of the
    trusted original input for this verse_id.

    Deliberately does NOT accept the "original" as a parameter -- see
    module docstring. Looks it up itself.
    """
    index = _load_input_index()
    original = index.get(verse_id)

    if original is None:
        return {
            "match": False,
            "verse_id": verse_id,
            "found_input": False,
            "sadr_match": None,
            "ajuz_match": None,
            "sadr_diff": None,
            "ajuz_diff": None,
            "similarity": None,
            "reason": f"verse_id {verse_id!r} not found in dataset/inputs/*.jsonl",
        }

    norm_sadr_in = normalize_text(original["sadr"])
    norm_ajuz_in = normalize_text(original["ajuz"])
    norm_sadr_out = normalize_text(sadr)
    norm_ajuz_out = normalize_text(ajuz)

    sadr_match = norm_sadr_in == norm_sadr_out
    ajuz_match = norm_ajuz_in == norm_ajuz_out

    full_in = f"{norm_sadr_in} {norm_ajuz_in}"
    full_out = f"{norm_sadr_out} {norm_ajuz_out}"
    similarity = SequenceMatcher(None, full_in, full_out).ratio()

    return {
        "match": sadr_match and ajuz_match,
        "verse_id": verse_id,
        "found_input": True,
        "sadr_match": sadr_match,
        "ajuz_match": ajuz_match,
        "sadr_diff": None if sadr_match else _diff(norm_sadr_in, norm_sadr_out),
        "ajuz_diff": None if ajuz_match else _diff(norm_ajuz_in, norm_ajuz_out),
        # Low similarity (~<0.3) => probably a different verse entirely.
        # High similarity but match=False (~>0.9) => probably a 1-2
        # letter substitution. Useful for triage in the rejection log.
        "similarity": round(similarity, 4),
    }


def _diff(expected: str, actual: str) -> str:
    """Char-level diff for logs/disagreements records, not for display --
    lets a human see exactly what drifted without re-deriving it."""
    ops = SequenceMatcher(None, expected, actual).get_opcodes()
    parts = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        parts.append(
            f"{tag}: expected[{i1}:{i2}]={expected[i1:i2]!r} -> actual[{j1}:{j2}]={actual[j1:j2]!r}"
        )
    return (
        "; ".join(parts)
        if parts
        else "(no char-level diff -- check whitespace/segmentation)"
    )
