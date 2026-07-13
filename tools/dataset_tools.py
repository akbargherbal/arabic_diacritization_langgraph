"""
tools/dataset_tools.py
========================
commit_verse_tool is the ONLY path into dataset/. Permission rules deny the
agent's built-in write_file/edit_file against dataset/**, but that layer
does NOT cover custom tools -- a custom tool is just a Python function the
agent can call, invisible to the declarative permissions system. So the
real enforcement lives here, in code:

  1. re-run the sanitizer (security axis, deciding)
  2. re-run the skeleton fidelity check (fidelity axis, deciding) --
     see tools/fidelity_tools.py
  3. re-run the pyarud check (never trust a stale "it passed earlier")
     (structural axis, deciding)
  4. append-only write, never overwrite/delete -- durably (flush + fsync)
     and under an OS-level advisory lock (see _append_locked)

This is design principle 3 made concrete: the gate's enforcement point is
code the agent cannot edit (verification/ is permission-denied), not
agent discretion, and not the permission system either.

Nothing is silently discarded. Every verse that fails any of the three
deciding gates -- sanitize, fidelity, pyarud -- is written to
dataset/verses_rejected.jsonl with full diagnostic detail (which gate
failed, the pyarud score if one was computed, the fidelity diff if one was
computed) so a human reviewer can see *why* it was rejected. pyarud is not
assumed infallible -- a rejection here is a candidate for human review,
not a verdict.

--- Duplicate detection (added) ---
A verse whose verse_id has already been committed, or whose
(verse_id, sadr, ajuz, meter) tuple is already present under it, is treated
as an idempotent no-op rather than appended a second time. This matters
because main.py's checkpoint-resume path (agent.invoke(None, ...) on an
interrupted thread) can, in principle, replay a tool call whose side effect
already landed before the interruption. The seen-set is loaded once from
the existing file(s) on first use in this process and updated on every
successful commit.

--- Cross-process write safety (added) ---
Appends to verses.jsonl / verses_rejected.jsonl go through _append_locked,
which takes an OS-level advisory lock (fcntl.flock on POSIX,
msvcrt.locking on Windows) around the write, and flush()+fsync()s before
releasing it. This protects against two things a bare `open("a").write()`
does not: (a) two processes' writes interleaving into a corrupted line,
and (b) a half-flushed line surviving a crash. _iter_valid_records is the
matching tolerant reader -- it skips (and reports) an unparseable trailing
line instead of raising, since a truncated final line is the expected
shape of an interrupted write even with fsync in place (fsync guarantees
durability of *completed* writes, not that a write in progress at the
moment of a crash left a complete line).
"""

import hashlib
import json
import os
import pathlib
import sys
import threading
from datetime import datetime, timezone

from tools.prosody_tools import verify_single_verse_tool
from tools.sanitization_tools import sanitize_output_tool
from tools.fidelity_tools import verify_skeleton_fidelity_tool

DATASET_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "dataset" / "verses.jsonl"
)
REJECTED_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "dataset" / "verses_rejected.jsonl"
)
SCORE_THRESHOLD = 0.99  # placeholder — tune deliberately against real data


# ---------------------------------------------------------------------------
# Cross-process file locking (C2)
# ---------------------------------------------------------------------------
# A single in-process threading.Lock (_commit_lock, below) is NOT sufficient
# on its own: nothing stops a second `python main.py` process (a manual
# retry, or a second input file/meter run concurrently) from writing the
# same dataset file at the same time. These helpers take an OS-level
# advisory lock scoped to the file handle for the duration of one append.

if sys.platform == "win32":
    import msvcrt

    def _lock_file(f) -> None:
        f.seek(0, 2)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock_file(f) -> None:
        f.seek(0, 2)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

else:
    import fcntl

    def _lock_file(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    def _unlock_file(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _append_locked(path: pathlib.Path, line: str) -> None:
    """Append one line to `path` under an OS-level advisory lock, flushing
    and fsyncing before releasing it -- see module docstring's
    "Cross-process write safety" section for why both the lock and the
    fsync matter (C2/C3)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        _lock_file(f)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            _unlock_file(f)


def _iter_valid_records(path: pathlib.Path):
    """Yield parsed dict records from a JSONL file, skipping (and warning
    about) any unparseable line. A corrupt LAST line is treated as the
    expected shape of an interrupted write and warned about quietly; a
    corrupt line anywhere else is warned about more insistently since
    that's not the expected failure shape and deserves investigation."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            is_last = i == len(lines) - 1
            if is_last:
                print(
                    f"[!] Warning: truncated trailing line in {path.name}, "
                    f"likely from an interrupted write — skipping it."
                )
            else:
                print(
                    f"[!] Warning: corrupt line {i} in {path.name} (not the "
                    f"last line — investigate, this isn't the expected "
                    f"interrupted-write shape) — skipping it."
                )


# ---------------------------------------------------------------------------
# Duplicate detection (C1)
# ---------------------------------------------------------------------------

_commit_lock = threading.Lock()
_seen_verse_ids: set[str] | None = None
_seen_text_hashes: set[str] | None = None


def _verse_hash(sadr: str, ajuz: str, meter: str) -> str:
    # Deliberately does NOT include verse_id: this hash exists specifically
    # to catch identical (sadr, ajuz, meter) committed under a DIFFERENT
    # verse_id, which is a distinct duplicate scenario from an exact
    # verse_id repeat (that's what _seen_verse_ids is for). Including
    # verse_id here would make every hash unique by construction and
    # silently defeat this check.
    payload = f"{sadr}\u241f{ajuz}\u241f{meter}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_seen() -> None:
    global _seen_verse_ids, _seen_text_hashes
    _seen_verse_ids, _seen_text_hashes = set(), set()
    for rec in _iter_valid_records(DATASET_PATH):
        vid = rec.get("verse_id")
        if vid is None:
            continue
        _seen_verse_ids.add(vid)
        _seen_text_hashes.add(
            _verse_hash(rec.get("sadr", ""), rec.get("ajuz", ""), rec.get("meter", ""))
        )


def _reset_seen_cache_for_tests() -> None:
    """Test-only helper to force _load_seen() to re-run (e.g. after a test
    fixture rewrites DATASET_PATH). Not used by the runtime code path."""
    global _seen_verse_ids, _seen_text_hashes
    _seen_verse_ids, _seen_text_hashes = None, None


def commit_verse_tool(
    verse_id: str,
    sadr: str,
    ajuz: str,
    meter: str,
    irab_flag: bool = False,
    naturalness_flag: bool = False,
    reconciled: bool = False,
    original_sadr: str | None = None,
    original_ajuz: str | None = None,
    notes: str = "",
    fix_type: str | None = None,
    word_index: int | None = None,
    target_harakah: str | None = None,
) -> dict:
    """Commit one verified verse to the dataset. Re-checks sanitize,
    skeleton fidelity, and pyarud itself before writing -- do not treat an
    earlier verify_batch_tool pass as sufficient, this call re-verifies.

    Idempotent against duplicates: if verse_id was already committed, or an
    identical (verse_id, sadr, ajuz, meter) tuple was already committed
    under a different verse_id, this returns committed=False with
    duplicate=True rather than writing a second row. See module docstring.

    A verse that fails any of the three checks is NOT discarded: it's
    written to dataset/verses_rejected.jsonl with the failing stage, the
    pyarud score (if computed), and the fidelity diff (if computed), for
    human review. pyarud has known false positives/negatives -- a
    rejection here is a lead for a reviewer, not a final judgment.

    If reconciled=True, sadr/ajuz are the POST-reconciliation text (after a
    deterministic case-ending swap resolved an إعراب flag without touching
    the meter — see tools/reconciliation_tools.py) and original_sadr/
    original_ajuz preserve what pyarud originally verified, for audit.
    Reconciled verses do NOT need needs_review — the fix was mechanically
    applied and re-verified, not left as an open disagreement.

    Verses that pass all three gates but were flagged by the advisory
    إعراب (structural, non-reconcilable) or naturalness checks are still
    committed to verses.jsonl, but with needs_review=True and a
    disagreement log entry — flagged, not silently dropped, not silently
    "cleaned."
    """
    with _commit_lock:
        global _seen_verse_ids, _seen_text_hashes
        if _seen_verse_ids is None:
            _load_seen()

        this_hash = _verse_hash(sadr, ajuz, meter)
        if verse_id in _seen_verse_ids:
            return {
                "committed": False,
                "reason": "duplicate verse_id already committed",
                "duplicate": True,
            }
        if this_hash in _seen_text_hashes:
            return {
                "committed": False,
                "reason": "identical (verse_id, sadr, ajuz, meter) already committed "
                "under a different verse_id",
                "duplicate": True,
            }

        timestamp = datetime.now(timezone.utc).isoformat()
        full_text = f"{sadr} {ajuz}".strip()

        sanitize_result = sanitize_output_tool(full_text)
        if not sanitize_result["valid"]:
            reason = f"sanitization failed: {sanitize_result['reason']}"
            _append_rejected(
                _rejection_record(
                    verse_id,
                    sadr,
                    ajuz,
                    meter,
                    stage="sanitize",
                    reason=reason,
                    pyarud_score=None,
                    pyarud_detail=None,
                    fidelity=None,
                    irab_flag=irab_flag,
                    naturalness_flag=naturalness_flag,
                    notes=notes,
                    timestamp=timestamp,
                )
            )
            return {"committed": False, "reason": reason, "logged_for_review": True}

        fidelity_result = verify_skeleton_fidelity_tool(verse_id, sadr, ajuz)
        if not fidelity_result["match"]:
            reason = (
                "skeleton fidelity check failed -- output letters diverge "
                "from the input verse, not just its diacritics"
            )
            _append_rejected(
                _rejection_record(
                    verse_id,
                    sadr,
                    ajuz,
                    meter,
                    stage="fidelity",
                    reason=reason,
                    pyarud_score=None,
                    pyarud_detail=None,
                    fidelity=fidelity_result,
                    irab_flag=irab_flag,
                    naturalness_flag=naturalness_flag,
                    notes=notes,
                    timestamp=timestamp,
                )
            )
            return {
                "committed": False,
                "reason": reason,
                "fidelity": fidelity_result,
                "logged_for_review": True,
            }

        verify_result = verify_single_verse_tool(sadr, ajuz, meter)
        if not verify_result["is_sound"]:
            reason = "failed pyarud re-check at commit time"
            _append_rejected(
                _rejection_record(
                    verse_id,
                    sadr,
                    ajuz,
                    meter,
                    stage="pyarud_commit",
                    reason=reason,
                    pyarud_score=verify_result.get("combined_score"),
                    pyarud_detail=verify_result,
                    fidelity=fidelity_result,
                    irab_flag=irab_flag,
                    naturalness_flag=naturalness_flag,
                    notes=notes,
                    timestamp=timestamp,
                )
            )
            return {
                "committed": False,
                "reason": reason,
                "score": verify_result.get("combined_score"),
                "logged_for_review": True,
            }

        # Reconciled verses resolved their disagreement mechanically; only a
        # non-reconciled flag is a genuine open disagreement worth review.
        needs_review = (irab_flag or naturalness_flag) and not reconciled
        record = {
            "verse_id": verse_id,
            "meter": meter,
            "sadr": sadr,
            "ajuz": ajuz,
            "combined_score": verify_result["combined_score"],
            "needs_review": needs_review,
            "irab_flag": irab_flag,
            "naturalness_flag": naturalness_flag,
            "reconciled": reconciled,
            "original_sadr": original_sadr,
            "original_ajuz": original_ajuz,
            "notes": notes,
            "fix_type": fix_type,
            "word_index": word_index,
            "target_harakah": target_harakah,
            "committed_at": timestamp,
        }

        _append_locked(DATASET_PATH, json.dumps(record, ensure_ascii=False) + "\n")
        _seen_verse_ids.add(verse_id)
        _seen_text_hashes.add(this_hash)

        if needs_review:
            _log_disagreement(verse_id, record)

        return {"committed": True, "needs_review": needs_review}


def _rejection_record(
    verse_id,
    sadr,
    ajuz,
    meter,
    *,
    stage,
    reason,
    pyarud_score,
    pyarud_detail,
    fidelity,
    irab_flag,
    naturalness_flag,
    notes,
    timestamp,
) -> dict:
    """Uniform schema for every rejected verse, regardless of which gate
    rejected it, so a human reviewer can scan verses_rejected.jsonl with
    one mental model instead of three different shapes."""
    return {
        "verse_id": verse_id,
        "meter": meter,
        "sadr": sadr,
        "ajuz": ajuz,
        "stage": stage,  # "sanitize" | "fidelity" | "pyarud_commit" | "unresolved_max_passes"
        "reason": reason,
        "pyarud_score": pyarud_score,  # combined_score if pyarud ran this time, else null
        "pyarud_detail": pyarud_detail,  # full verify_single_verse_tool() dict if it ran, else null
        "fidelity": fidelity,  # full verify_skeleton_fidelity_tool() dict if it ran, else null
        "irab_flag": irab_flag,
        "naturalness_flag": naturalness_flag,
        "notes": notes,
        "rejected_at": timestamp,
    }


def _append_rejected(record: dict) -> None:
    _append_locked(REJECTED_PATH, json.dumps(record, ensure_ascii=False) + "\n")


def _log_disagreement(verse_id: str, record: dict) -> None:
    log_dir = pathlib.Path(__file__).resolve().parent.parent / "logs" / "disagreements"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{verse_id}.json"
    log_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def log_unresolved_tool(
    verse_id: str,
    sadr: str,
    ajuz: str,
    meter: str,
    last_report: str,
    stage: str = "unresolved_max_passes",
    reason: str | None = None,
) -> dict:
    """Log a verse that will never enter the dataset for this run.
    Excluded from dataset/verses.jsonl -- never force a further pass,
    never auto-accept -- but still appended to verses_rejected.jsonl (not
    just the per-verse audit file under logs/disagreements/) so it shows
    up in the same human-review sweep as sanitize/fidelity/pyarud
    rejections, instead of needing a separate lookup path.

    stage/reason distinguish WHY this verse never made it in -- do not
    default every call to "unresolved_max_passes". Two genuinely different
    situations both route through this tool and must not look identical in
    the logs:
      - stage="unresolved_max_passes" (default): the verse was actually
        diacritized and re-checked across the full pass budget and still
        didn't converge.
      - stage="structurally_incompatible": verify_batch_tool determined,
        from actual diacritized output, that the letter skeleton cannot
        satisfy the target meter's mora count regardless of diacritics --
        pass an explicit reason describing the mora/foot mismatch so a
        human reviewer isn't left thinking this exhausted 3 real passes
        when it may have been excluded earlier.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    resolved_reason = reason or (
        "did not converge within max correction passes"
        if stage == "unresolved_max_passes"
        else f"excluded at stage={stage!r}"
    )

    log_dir = pathlib.Path(__file__).resolve().parent.parent / "logs" / "disagreements"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{verse_id}_unresolved.json"
    detail_record = {
        "verse_id": verse_id,
        "sadr": sadr,
        "ajuz": ajuz,
        "meter": meter,
        "status": "unresolved",
        "stage": stage,
        "reason": resolved_reason,
        "last_correction_report": last_report,
        "logged_at": timestamp,
    }
    log_path.write_text(
        json.dumps(detail_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _append_rejected(
        _rejection_record(
            verse_id,
            sadr,
            ajuz,
            meter,
            stage=stage,
            reason=resolved_reason,
            pyarud_score=None,
            pyarud_detail={"last_correction_report": last_report},
            fidelity=None,
            irab_flag=False,
            naturalness_flag=False,
            notes="",
            timestamp=timestamp,
        )
    )

    return {"logged": True}
