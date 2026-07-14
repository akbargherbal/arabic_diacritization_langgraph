"""
facades/ledger_client.py
===========================
Phase 4 of PHASED_PLAN.md: "Put a face on the cross-cutting infra".

Before this facade, pipeline/verify_stage.py and pipeline/advisory_stage.py
each imported `record_locked_verse_tool`, `read_ledger_tool`,
`commit_verse_tool`, `log_unresolved_tool`, and
`build_batched_advisory_payload_tool` directly from tools/advisory_ledger.py,
tools/dataset_tools.py, and tools/advisory_batch.py -- three modules'
internals leaking into two pipeline stages.

LedgerClient is the single narrow interface both stages depend on instead:
"record this verse as locked", "log this verse as unresolved", "commit this
verse to the dataset", "give me the batched advisory payload", "reset the
ledger". Nothing in pipeline/ needs to know the ledger is a JSON file, that
commits go through sanitize/fidelity/pyarud gates, or which module owns
which function.

Patch `facades.ledger_client.LedgerClient.<method>` (e.g.
`patch.object(LedgerClient, "commit", ...)`) when testing pipeline code
that goes through this facade -- both verify_stage and advisory_stage
import the same LedgerClient class, so patching it here affects both call
sites consistently.
"""

from __future__ import annotations

from typing import Optional

from tools.advisory_batch import build_batched_advisory_payload_tool
from tools.advisory_ledger import read_ledger_tool, record_locked_verse_tool
from tools.dataset_tools import commit_verse_tool, log_unresolved_tool


class LedgerClient:
    """Facade over the advisory-ledger + dataset-commit cross-cutting infra.
    All methods are staticmethods -- this is a namespace, not a stateful
    object (the underlying tools already manage their own locking/state)."""

    # -- ledger (tools/advisory_ledger.py) ---------------------------------

    @staticmethod
    def record_locked(verse_id: str, sadr: str, ajuz: str, meter: str) -> dict:
        """Independently re-verify and append a newly-locked verse to the
        current thread's advisory_ledger.json."""
        return record_locked_verse_tool(
            verse_id=verse_id, sadr=sadr, ajuz=ajuz, meter=meter
        )

    @staticmethod
    def read_ledger(clear: bool = False) -> dict:
        return read_ledger_tool(clear=clear)

    @staticmethod
    def reset_ledger() -> dict:
        """Clear the current thread's ledger file. Called once at the end
        of advisory_stage, after every locked verse has been resolved."""
        return read_ledger_tool(clear=True)

    @staticmethod
    def build_advisory_payload() -> dict:
        """Compile everything currently in the ledger into the pure-JSON
        payload the batched advisory subagents expect."""
        return build_batched_advisory_payload_tool()

    # -- dataset commits (tools/dataset_tools.py) --------------------------

    @staticmethod
    def commit(
        verse_id: str,
        sadr: str,
        ajuz: str,
        meter: str,
        irab_flag: bool = False,
        naturalness_flag: bool = False,
        reconciled: bool = False,
        original_sadr: Optional[str] = None,
        original_ajuz: Optional[str] = None,
        notes: str = "",
        fix_type: Optional[str] = None,
        word_index: Optional[int] = None,
        target_harakah: Optional[str] = None,
    ) -> dict:
        """Re-verify (sanitize/fidelity/pyarud) and, if all gates pass,
        append one verse to dataset/verses.jsonl."""
        return commit_verse_tool(
            verse_id=verse_id,
            sadr=sadr,
            ajuz=ajuz,
            meter=meter,
            irab_flag=irab_flag,
            naturalness_flag=naturalness_flag,
            reconciled=reconciled,
            original_sadr=original_sadr,
            original_ajuz=original_ajuz,
            notes=notes,
            fix_type=fix_type,
            word_index=word_index,
            target_harakah=target_harakah,
        )

    @staticmethod
    def log_unresolved(
        verse_id: str,
        sadr: str,
        ajuz: str,
        meter: str,
        last_report: str,
        stage: str = "unresolved_max_passes",
        reason: Optional[str] = None,
    ) -> dict:
        """Log a verse that will never enter the dataset for this run --
        excluded from verses.jsonl, appended to verses_rejected.jsonl."""
        return log_unresolved_tool(
            verse_id=verse_id,
            sadr=sadr,
            ajuz=ajuz,
            meter=meter,
            last_report=last_report,
            stage=stage,
            reason=reason,
        )
