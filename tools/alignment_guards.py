"""
tools/alignment_guards.py
==========================
Programmatic safety layer to prevent LLM ID drift during batched audits.
"""

from tools.advisory_ledger import read_ledger_tool
from tools.fidelity_tools import normalize_text


def validate_advisory_batch_alignment(
    batch_verdicts: list[dict], fix_type_field: str = "case_ending_swap"
) -> tuple[bool, str | None]:
    """
    Programmatically asserts 100% ID and text alignment of batched LLM advisory output.

    Returns:
        (True, None) if alignment is mathematically guaranteed.
        (False, error_reason) if any drift, hallucination, or mismatch is detected.
    """
    # 1. Sourced input_verses from read_ledger_tool
    res = read_ledger_tool(clear=False)
    input_verses = res.get("verses", [])

    # Structural Verification: Cardinally equal counts
    if len(input_verses) != len(batch_verdicts):
        return (
            False,
            f"Count mismatch: sent {len(input_verses)} verses, but received {len(batch_verdicts)} verdicts.",
        )

    # Map trusted inputs by ID for O(1) lookups
    input_map = {v["verse_id"]: v for v in input_verses}
    seen_ids = set()

    for verdict in batch_verdicts:
        vid = verdict.get("verse_id")
        if not vid:
            return False, "Malformed verdict: missing 'verse_id' field."
        if vid not in input_map:
            return False, f"LLM hallucinated a non-existent 'verse_id': {vid!r}."
        if vid in seen_ids:
            return False, f"LLM returned duplicate records for 'verse_id': {vid!r}."
        seen_ids.add(vid)

        # Content Alignment Verification (The Anchor Guard)
        if verdict.get("flag") and verdict.get("fix_type") == fix_type_field:
            word_index = verdict.get("word_index")
            target_word_skeleton = verdict.get("target_word_skeleton")
            hemistich = verdict.get("hemistich")

            # Assert schema completeness
            if (
                word_index is None
                or not target_word_skeleton
                or hemistich not in ("sadr", "ajuz")
            ):
                return (
                    False,
                    f"Verse {vid}: 'case_ending_swap' requested but missing index, skeleton, or hemistich.",
                )

            # Retrieve the trusted original verse components
            original_verse = input_map[vid]
            original_hemistich_text = original_verse.get(hemistich, "")
            words = original_hemistich_text.split()

            if word_index < 0 or word_index >= len(words):
                return (
                    False,
                    f"Verse {vid}: word_index {word_index} is out of bounds for hemistich {hemistich!r}.",
                )

            # Strip diacritics and compare base consonant skeletons
            expected_consonant_skeleton = normalize_text(words[word_index])
            provided_consonant_skeleton = normalize_text(target_word_skeleton)

            # CRITICAL SHIELD: If the skeletons diverge, alignment has failed
            if expected_consonant_skeleton != provided_consonant_skeleton:
                return False, (
                    f"DATA CONTAMINATION SHIELD TRIGGERED! LLM proposed 'case_ending_swap' on "
                    f"verse {vid} ({hemistich}) at word_index {word_index}. Expected consonant skeleton "
                    f"'{expected_consonant_skeleton}', but LLM output targeted '{provided_consonant_skeleton}'."
                )

    # Completeness Verification
    if len(seen_ids) != len(input_verses):
        missing_ids = set(input_map.keys()) - seen_ids
        return False, f"LLM omitted verdicts for the following verses: {missing_ids}"

    return True, None


def validate_naturalness_batch_alignment(
    batch_verdicts: list[dict],
) -> tuple[bool, str | None]:
    """
    Simple alignment check for naturalness critic batched output (no skeleton validation).

    Returns:
        (True, None) if alignment is structurally valid.
        (False, error_reason) if count mismatch, hallucination, duplicate, or omission is found.
    """
    # 1. Sourced input_verses from read_ledger_tool
    res = read_ledger_tool(clear=False)
    input_verses = res.get("verses", [])

    # Structural Verification: Cardinally equal counts
    if len(input_verses) != len(batch_verdicts):
        return (
            False,
            f"Count mismatch: sent {len(input_verses)} verses, but received {len(batch_verdicts)} verdicts.",
        )

    # Map trusted inputs by ID
    input_map = {v["verse_id"]: v for v in input_verses}
    seen_ids = set()

    for verdict in batch_verdicts:
        vid = verdict.get("verse_id")
        if not vid:
            return False, "Malformed verdict: missing 'verse_id' field."
        if vid not in input_map:
            return False, f"LLM hallucinated a non-existent 'verse_id': {vid!r}."
        if vid in seen_ids:
            return False, f"LLM returned duplicate records for 'verse_id': {vid!r}."
        seen_ids.add(vid)

    # Completeness Verification
    if len(seen_ids) != len(input_verses):
        missing_ids = set(input_map.keys()) - seen_ids
        return False, f"LLM omitted verdicts for the following verses: {missing_ids}"

    return True, None
