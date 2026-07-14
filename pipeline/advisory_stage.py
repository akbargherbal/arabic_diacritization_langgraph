"""
pipeline/advisory_stage.py
=============================
Task 2.2/2.3/2.4's advisory subagent calls, reconciliation/commit precedence
(resolve_and_commit), and the advisory_stage terminal node, extracted from
langgraph_pipeline.py (Phase 2b of docs/REFACTOR_PLAN.md).

Patch names HERE (`pipeline.advisory_stage.<name>`) when testing this
stage -- `run_irab_checker_batch`, `run_naturalness_critic_batch`,
`run_irab_checker_single`, `run_naturalness_critic_single`,
`reconcile_case_ending_tool`, and `verify_single_verse_tool` are all
called as bare names resolved via this module's globals, not via
`langgraph_pipeline`'s re-exported bindings.

Phase 4 of PHASED_PLAN.md: committing verses, logging unresolved ones,
building the batched advisory payload, and resetting the ledger all go
through the LedgerClient facade (facades/ledger_client.py) instead of
importing commit_verse_tool/log_unresolved_tool/read_ledger_tool/
build_batched_advisory_payload_tool directly. Patch
`pipeline.advisory_stage.LedgerClient` (e.g.
`patch.object(LedgerClient, "commit", ...)`) when testing those paths.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from runtime import MAX_CORRECTION_PASSES, PROJECT_ROOT
from subagents.irab_checker_agent import IRAB_SYSTEM_PROMPT, IRAB_BATCH_SYSTEM_PROMPT
from subagents.naturalness_critic import (
    NATURALNESS_SYSTEM_PROMPT,
    NATURALNESS_BATCH_SYSTEM_PROMPT,
)
from tools.alignment_guards import (
    validate_advisory_batch_alignment,
    validate_naturalness_batch_alignment,
)
from tools.prosody_tools import verify_single_verse_tool
from tools.reconciliation_tools import reconcile_case_ending_tool

from facades.ledger_client import LedgerClient
from pipeline.json_utils import _extract_json
from pipeline.state import BatchState

# ===========================================================================
# Task 2.2: irab_checker / naturalness_critic batch + single-verse calls
# ===========================================================================


def _call_advisory_model(
    model, system_prompt: str, user_payload: str, agent_tag: str
) -> Any:
    ai_msg = model.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_payload)],
        config={"tags": [f"agent:{agent_tag}"]},
    )
    return _extract_json(ai_msg.content)


def run_irab_checker_batch(model, payload_json: str) -> list[dict]:
    return _call_advisory_model(
        model, IRAB_BATCH_SYSTEM_PROMPT, payload_json, "irab_checker_batch"
    )


def run_naturalness_critic_batch(model, payload_json: str) -> list[dict]:
    return _call_advisory_model(
        model, NATURALNESS_BATCH_SYSTEM_PROMPT, payload_json, "naturalness_critic_batch"
    )


def run_irab_checker_single(model, verse: dict) -> dict:
    payload = json.dumps(
        {
            "verse_id": verse["verse_id"],
            "sadr": verse["sadr"],
            "ajuz": verse.get("ajuz", ""),
        },
        ensure_ascii=False,
    )
    return _call_advisory_model(model, IRAB_SYSTEM_PROMPT, payload, "irab_checker")


def run_naturalness_critic_single(model, verse: dict) -> dict:
    payload = json.dumps(
        {
            "verse_id": verse["verse_id"],
            "sadr": verse["sadr"],
            "ajuz": verse.get("ajuz", ""),
        },
        ensure_ascii=False,
    )
    return _call_advisory_model(
        model, NATURALNESS_SYSTEM_PROMPT, payload, "naturalness_critic"
    )


# ===========================================================================
# Task 2.4: reconciliation / precedence / commit, per verse
# ===========================================================================


def resolve_and_commit(
    verse_id: str,
    sadr: str,
    ajuz: str,
    meter: str,
    irab_verdict: Optional[dict],
    naturalness_verdict: Optional[dict],
) -> dict:
    """Literal port of AGENTS.md rule 1 / ORCHESTRATOR_SYSTEM_PROMPT's
    "Handling a pyarud/إعراب disagreement" section. Precedence order is
    fixed: attempt reconciliation first; pyarud-wins fallback second. Do
    not reorder (Task 2.4's Minimum Change Rule).
    """
    irab_verdict = irab_verdict or {}
    naturalness_verdict = naturalness_verdict or {}

    irab_flag = bool(irab_verdict.get("flag"))
    fix_type = irab_verdict.get("fix_type")
    naturalness_flag = not naturalness_verdict.get("natural", True)

    if irab_flag and fix_type == "case_ending_swap":
        word_index = irab_verdict.get("word_index")
        target_harakah = irab_verdict.get("target_harakah")
        hemistich = irab_verdict.get("hemistich", "sadr")
        hemistich_text = sadr if hemistich == "sadr" else ajuz

        recon = reconcile_case_ending_tool(hemistich_text, word_index, target_harakah)
        if recon["success"]:
            new_text = recon["reconciled_text"]
            new_sadr = new_text if hemistich == "sadr" else sadr
            new_ajuz = new_text if hemistich == "ajuz" else ajuz
            verify = verify_single_verse_tool(new_sadr, new_ajuz, meter)
            if verify["is_sound"]:
                # Resolved grammar fix, no metrical cost -- not a disagreement.
                return LedgerClient.commit(
                    verse_id=verse_id,
                    sadr=new_sadr,
                    ajuz=new_ajuz,
                    meter=meter,
                    reconciled=True,
                    original_sadr=sadr,
                    original_ajuz=ajuz,
                    naturalness_flag=naturalness_flag,
                    notes=(
                        naturalness_verdict.get("note", "") if naturalness_flag else ""
                    ),
                )
            # Reconciled text failed pyarud -- fall through to precedence
            # rule below using the ORIGINAL (pre-swap) text.
        # Reconciliation not applicable/failed -- fall through.

    if irab_flag:
        # fix_type == "structural", or reconciliation was attempted and failed.
        return LedgerClient.commit(
            verse_id=verse_id,
            sadr=sadr,
            ajuz=ajuz,
            meter=meter,
            irab_flag=True,
            naturalness_flag=naturalness_flag,
            notes=irab_verdict.get("note", ""),
            fix_type=fix_type,
            word_index=irab_verdict.get("word_index"),
            target_harakah=irab_verdict.get("target_harakah"),
        )

    if naturalness_flag:
        return LedgerClient.commit(
            verse_id=verse_id,
            sadr=sadr,
            ajuz=ajuz,
            meter=meter,
            naturalness_flag=True,
            notes=naturalness_verdict.get("note", ""),
        )

    # Clean on both advisory axes.
    return LedgerClient.commit(verse_id=verse_id, sadr=sadr, ajuz=ajuz, meter=meter)


# ===========================================================================
# Task 2.3 (+2.5's terminal branch): advisory_stage node
# ===========================================================================


def make_advisory_stage(model):
    def advisory_stage(state: BatchState) -> dict:
        """Terminal node. Reached only once, after route_after_verify decides
        no further pass is warranted (broken is empty, or pass budget is
        exhausted). Literal 8-step port of ORCHESTRATOR_SYSTEM_PROMPT's
        "Ledger recording and batched advisory triggers" section, MINUS
        step 1 (already handled per-pass inside verify_pass -- see that
        function's docstring for why step 1 cannot be deferred to here
        without breaking the original "after each pass" cadence).
        """
        meter = state["meter_name"]
        drafts = state.get("drafts", {})
        locked_ids = set(state.get("locked", []))
        broken_ids = list(state.get("broken", []))
        pass_number = state.get("pass_number", 1)

        # Task 2.5's terminal branch: any verse STILL broken (not structurally
        # incompatible -- those were already excluded/logged in verify_pass)
        # once the loop exits must be pass-budget-exhausted, never anything
        # else (route_after_verify only reaches this node when either broken
        # is empty or pass_number > MAX_CORRECTION_PASSES).
        report_path = state.get("report_path")
        report_text = ""
        if report_path:
            p = PROJECT_ROOT / report_path
            if p.exists():
                report_text = p.read_text(encoding="utf-8")

        for vid in broken_ids:
            draft = drafts.get(vid) or next(
                v for v in state["verses"] if v["verse_id"] == vid
            )
            LedgerClient.log_unresolved(
                verse_id=vid,
                sadr=draft["sadr"],
                ajuz=draft.get("ajuz", ""),
                meter=meter,
                last_report=report_text,
                stage="unresolved_max_passes",
                reason=(
                    f"still broken after pass {pass_number - 1} "
                    f"(budget={MAX_CORRECTION_PASSES})"
                ),
            )

        # Step 2: once resolved, build the batched advisory payload from
        # everything record_locked_verse_tool wrote during verify_pass.
        payload_result = LedgerClient.build_advisory_payload()
        payload = payload_result.get("payload")

        # Step 3
        if payload is None:
            return {}

        ledger_verses = json.loads(payload)  # [{verse_id, sadr, ajuz}, ...]

        # Step 4: dispatch batched irab_checker_batch / naturalness_critic_batch.
        irab_verdicts = run_irab_checker_batch(model, payload)
        naturalness_verdicts = run_naturalness_critic_batch(model, payload)

        # Step 5
        irab_ok, irab_reason = validate_advisory_batch_alignment(irab_verdicts)
        nat_ok, nat_reason = validate_naturalness_batch_alignment(naturalness_verdicts)

        irab_by_id: dict[str, dict] = {}
        nat_by_id: dict[str, dict] = {}

        if irab_ok and nat_ok:
            # Step 7: proceed with batch verdicts.
            irab_by_id = {v["verse_id"]: v for v in irab_verdicts}
            nat_by_id = {v["verse_id"]: v for v in naturalness_verdicts}
        else:
            # Step 6: fall back to per-verse single dispatch for EVERY verse
            # in this batch, and log which guard failed and why.
            if not irab_ok:
                print(
                    f"[!] irab_checker_batch alignment guard failed: {irab_reason} -- falling back to per-verse dispatch."
                )
            if not nat_ok:
                print(
                    f"[!] naturalness_critic_batch alignment guard failed: {nat_reason} -- falling back to per-verse dispatch."
                )
            for lv in ledger_verses:
                irab_by_id[lv["verse_id"]] = run_irab_checker_single(model, lv)
                nat_by_id[lv["verse_id"]] = run_naturalness_critic_single(model, lv)

        # Steps 6/7 continued: resolve/commit every verse in the ledger.
        commit_results = {}
        for lv in ledger_verses:
            vid = lv["verse_id"]
            commit_results[vid] = resolve_and_commit(
                verse_id=vid,
                sadr=lv["sadr"],
                ajuz=lv["ajuz"],
                meter=meter,
                irab_verdict=irab_by_id.get(vid),
                naturalness_verdict=nat_by_id.get(vid),
            )

        # Step 8: reset the ledger for the next input batch.
        LedgerClient.reset_ledger()

        return {}

    return advisory_stage
