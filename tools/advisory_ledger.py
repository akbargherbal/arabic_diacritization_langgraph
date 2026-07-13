"""
tools/advisory_ledger.py
=========================
Advisory ledger for tracking locked verses within a batch before sending
them in a batched advisory request.
"""

import json
import pathlib
import threading
from datetime import datetime, timezone

from tools.fidelity_tools import verify_skeleton_fidelity_tool
from tools.prosody_tools import verify_single_verse_tool
from tools.tracing import current_trace

_ledger_lock = threading.Lock()


def _get_ledger_path() -> pathlib.Path:
    trace = current_trace()
    thread_id = (
        trace.langgraph_thread_id if trace and trace.langgraph_thread_id else "unknown"
    )
    safe_thread = thread_id.replace(":", "_")

    # Project root is parent of tools/
    project_root = pathlib.Path(__file__).resolve().parent.parent
    return project_root / "workspace" / safe_thread / "advisory_ledger.json"


def record_locked_verse_tool(verse_id: str, sadr: str, ajuz: str, meter: str) -> dict:
    """
    Independently re-verify the supplied text before persisting it.
    If either check fails, do NOT write anything and return failure.
    If both pass, append to the advisory_ledger.json file.
    """
    # 1. Re-verify skeleton fidelity
    fidelity_result = verify_skeleton_fidelity_tool(verse_id, sadr, ajuz)
    if not fidelity_result.get("match"):
        return {
            "recorded": False,
            "reason": f"fidelity check failed: {fidelity_result.get('reason', 'output letters diverge from the input verse')}",
        }

    # 2. Re-verify meter
    verify_result = verify_single_verse_tool(sadr, ajuz, meter)
    if not verify_result.get("is_sound"):
        return {
            "recorded": False,
            "reason": f"pyarud check failed: score={verify_result.get('combined_score')}",
        }

    # 3. Append to the ledger under a thread-safe lock
    with _ledger_lock:
        path = _get_ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        verses = []
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    verses = json.load(f)
            except (json.JSONDecodeError, ValueError):
                verses = []

        # Guard against duplicate verse_id
        for v in verses:
            if v.get("verse_id") == verse_id:
                return {
                    "recorded": False,
                    "reason": "duplicate verse_id already in ledger",
                    "duplicate": True,
                }

        # Success path
        timestamp = datetime.now(timezone.utc).isoformat()
        new_entry = {
            "verse_id": verse_id,
            "sadr": sadr,
            "ajuz": ajuz,
            "meter": meter,
            "recorded_at": timestamp,
        }
        verses.append(new_entry)

        with path.open("w", encoding="utf-8") as f:
            json.dump(verses, f, ensure_ascii=False, indent=2)

    return {"recorded": True}


def read_ledger_tool(clear: bool = False) -> dict:
    """
    Read the ledger file for the current thread_id.
    If clear=True, delete the ledger file after reading.
    """
    with _ledger_lock:
        path = _get_ledger_path()
        if not path.exists():
            return {"verses": []}

        try:
            with path.open("r", encoding="utf-8") as f:
                verses = json.load(f)
        except (json.JSONDecodeError, ValueError):
            verses = []

        if clear:
            try:
                path.unlink()
            except OSError:
                pass

        return {"verses": verses}
