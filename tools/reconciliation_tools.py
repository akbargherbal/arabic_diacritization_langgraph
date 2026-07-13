"""
tools/reconciliation_tools.py
================================
Handles the specific, common case where pyarud and إعراب "disagree" but
aren't actually in conflict: a case-ending (إعراب) error where the required
fix is swapping among fatha/damma/kasra -- or, now, among the corresponding
tanwin marks (fathatayn/dammatayn/kasratayn) on an indefinite noun.

Why this is mechanically safe: pyarud's structural representation encodes
whether a letter carries a short vowel at all (mutaharrik vs sakin) — it
does NOT distinguish which short vowel, and a final tanwin mark is a
single mutaharrik unit exactly like a plain short vowel. Swapping between
any of these six marks cannot change the metrical (U/_) pattern in the
underlying model.

This is a claim about Arabic prosody in general, not a guarantee about this
specific pyarud build's converter (which has at least one documented
upstream bug already — see verification/arabic_prosody_feedback.py's
module docstring). So this tool does NOT assume immunity: every caller
MUST re-verify with verify_single_verse_tool after applying a swap, and
only treat it as reconciled if that re-check still passes. If it doesn't,
this was not actually a free fix, and the orchestrator should fall back
to the poetic-license precedence rule instead (see AGENTS.md).

Known limitation of this mechanical implementation: it targets the LAST
basic harakah or tanwin mark (fatha/damma/kasra/fathatayn/dammatayn/
kasratayn) in a whitespace-delimited word as "the case ending." It does
NOT handle a shadda with no following vowel mark (bare pausal/sukun
ending) -- if no mark is found, it reports failure rather than guessing,
and the orchestrator should route that case to the standard precedence
rule / disagreement log instead.

Orthographic side-effect handled: swapping TO fathatayn on an indefinite
noun conventionally requires appending the accusative alif seat (e.g.
كِتَابٍ -> كِتَاباً), except when the word already ends in tاء مربوطة, a
hamza already seated on alif (e.g. سَمَاء), or an alif/alif maqsura. See
_needs_accusative_alif below. This is purely orthographic (it does not
add a new syllable/mora -- the alif seat here is silent, carrying the
tanwin's own single mutaharrik unit) and does not touch the skeleton
fidelity check's letter-count expectations any differently than the
model producing that alif itself would have.
"""

FATHA = "\u064E"
DAMMA = "\u064F"
KASRA = "\u0650"
FATHATAYN = "\u064B"
DAMMATAYN = "\u064C"
KASRATAYN = "\u064D"

ALIF = "\u0627"
TAA_MARBUTA = "\u0629"
ALIF_MAQSURA = "\u0649"
HAMZA_CHARS = {"\u0621", "\u0624", "\u0626", "\u0623", "\u0625", "\u0622"}
SHADDA = "\u0651"

HARAKAT = {
    "fatha": FATHA,
    "damma": DAMMA,
    "kasra": KASRA,
    "fathatayn": FATHATAYN,
    "dammatayn": DAMMATAYN,
    "kasratayn": KASRATAYN,
}
TANWIN_KEYS = {"fathatayn", "dammatayn", "kasratayn"}
BASIC_MARKS = set(HARAKAT.values())


def _needs_accusative_alif(word: str, mark_pos: int) -> bool:
    """True if placing fathatayn at mark_pos on this word requires
    appending the orthographic accusative alif seat (indefinite
    accusative tanwin), False for taa-marbuta, hamza-on-alif-seat, or an
    already-alif/alif-maqsura ending where no extra seat letter is used.
    """
    stem = word[:mark_pos]
    if not stem:
        return False
    last_letter = stem[-1]
    if last_letter == TAA_MARBUTA:
        return False
    if last_letter == ALIF_MAQSURA:
        return False
    if last_letter == ALIF:
        return False  # already ends in alif
    if last_letter in HAMZA_CHARS and len(stem) >= 2 and stem[-2] == ALIF:
        return False  # e.g. سَمَاء -- hamza already seated on alif
    return True


def reconcile_case_ending_tool(hemistich_text: str, word_index: int, target_harakah: str) -> dict:
    """
    Mechanically swap the final basic harakah or tanwin mark on the word at
    word_index (0-indexed, whitespace-split, whitespace-normalized) to
    target_harakah ("fatha"|"damma"|"kasra"|"fathatayn"|"dammatayn"|"kasratayn").

    Returns {"success": bool, "reconciled_text": str | None, "reason": str | None}.

    This function makes NO pyarud call itself — the orchestrator must call
    verify_single_verse_tool on the result and only accept it as reconciled
    if the score still meets threshold. If it drops, treat this as a genuine
    structural conflict, not a free fix, and fall back to the precedence rule.
    """
    if target_harakah not in HARAKAT:
        return {"success": False, "reconciled_text": None,
                "reason": f"unknown target_harakah '{target_harakah}', expected one of {sorted(HARAKAT)}"}

    # split() (not split(" ")) collapses runs of whitespace so a stray
    # double-space in drafted/reconciled text can't silently shift every
    # subsequent word_index onto the wrong word.
    words = hemistich_text.split()
    if word_index < 0 or word_index >= len(words):
        return {"success": False, "reconciled_text": None, "reason": "word_index out of range"}

    word = words[word_index]
    positions = [i for i, ch in enumerate(word) if ch in BASIC_MARKS]
    if not positions:
        has_shadda = SHADDA in word
        reason = (
            "no basic harakah/tanwin mark found on this word's ending "
            + ("(shadda present with no following vowel mark — likely a "
               "pausal/sukun reading) " if has_shadda else "(sukun/pausal case) ")
            + "— not handled by this mechanical tool, route to the standard "
              "precedence rule instead"
        )
        return {"success": False, "reconciled_text": None, "reason": reason}

    last_pos = positions[-1]
    new_mark = HARAKAT[target_harakah]
    new_word = word[:last_pos] + new_mark + word[last_pos + 1:]

    if target_harakah == "fathatayn" and _needs_accusative_alif(word, last_pos):
        new_word += ALIF

    words[word_index] = new_word
    return {"success": True, "reconciled_text": " ".join(words), "reason": None}
