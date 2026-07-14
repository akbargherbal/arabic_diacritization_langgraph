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

# NOTE: this alphabet is specific to the Fidelity axis's skeleton-comparison
# use case (hamza-seat and alif-maqsura/ya variants deliberately kept
# distinct -- see normalize_text below). sanitization_tools.py solves a
# different problem (rejecting disallowed Unicode code points/control chars
# via ALLOWED_RANGES) and does not define AR_CHARS or normalize_text, so
# there's nothing to import from it. A prior version of this comment
# claimed the two modules duplicated this list; that's no longer accurate
# and the two are not expected to converge -- leave this list local to this
# module.
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
    """
    index = _load_input_index()
    ref = index.get(verse_id)
    if not ref:
        return {"match": False, "reason": f"unknown verse_id: {verse_id}"}

    ref_sadr = normalize_text(ref["sadr"])
    ref_ajuz = normalize_text(ref["ajuz"])
    prop_sadr = normalize_text(sadr)
    prop_ajuz = normalize_text(ajuz)

    sadr_match = ref_sadr == prop_sadr
    ajuz_match = ref_ajuz == prop_ajuz

    if sadr_match and ajuz_match:
        return {"match": True, "reason": None}

    # Compute a helpful character-level diff if they mismatch
    sadr_diff = ""
    if not sadr_match:
        matcher = SequenceMatcher(None, ref_sadr, prop_sadr)
        diffs = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            orig = ref_sadr[i1:i2]
            prop = prop_sadr[j1:j2]
            diffs.append(f"[{orig} -> {prop}]")
        sadr_diff = "Sadr mismatch: " + " ".join(diffs)

    ajuz_diff = ""
    if not ajuz_match:
        matcher = SequenceMatcher(None, ref_ajuz, prop_ajuz)
        diffs = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            orig = ref_ajuz[i1:i2]
            prop = prop_ajuz[j1:j2]
            diffs.append(f"[{orig} -> {prop}]")
        ajuz_diff = "Ajuz mismatch: " + " ".join(diffs)

    reason = "; ".join(filter(None, [sadr_diff, ajuz_diff]))
    return {"match": False, "reason": reason}
