"""
langgraph_pipeline.py
======================
Raw langgraph.graph.StateGraph replacement for the DeepAgents orchestrator
in main.py. See PHASED_PLAN_v3_LangGraph_Migration.md for the full spec
this file implements (Phases 1-3, Tasks 1.1-3.3's pipeline half).

Core idea (Locked Decision, plan Section 2): the dispatch -> verify -> route
pass-loop is enforced by the SHAPE of the graph's edges, not by an LLM's
willingness to follow a "check your work before dispatching again"
instruction. route_after_verify is a plain Python function; it cannot be
"not followed" the way a system-prompt rule could.

Every deciding-axis tool (verify_batch_tool, verify_single_verse_tool,
sanitize_output_tool, verify_skeleton_fidelity_tool via commit_verse_tool)
is reused unchanged from tools/. Every subagent prompt (diacritizer,
irab_checker, naturalness_critic, and their _batch variants) is reused
verbatim from subagents/ -- only the execution wrapper (CompiledSubAgent
-> plain model.invoke()/tool-loop) changes, per the plan's locked
"Subagent prompts" decision.

--- Design note on Task 1.2's "Send" instruction ---
The pre-coding checklist confirms `from langgraph.types import Send` is
importable (langgraph 1.2.9). However Task 1.4's success criteria requires
`dispatch_diacritizer` to appear as an actual graph NODE with exactly one
outgoing edge, fixed, to `verify_pass` -- which is incompatible with using
Send as a conditional-edge router (Send-based fan-out routes to a
DIFFERENT node per invocation, not through a single node with one fixed
outgoing edge). Given that conflict, and that per-verse parallelism is
achievable equally well inside a single node via ThreadPoolExecutor
(each verse still gets its own model call, still all in parallel, still
zero conditional dispatch logic inside the node), this implementation
dispatches per-verse calls in parallel from INSIDE the `dispatch_diacritizer`
node body rather than via graph-level Send. This satisfies every stated
Task 1.2/1.4/1.5 success criterion (one call per verse per pass, no
pass_number/verify_batch_tool/looping construct in the node, exactly one
fixed edge dispatch_diacritizer -> verify_pass). Flagged explicitly here
and in the session handover rather than silently picked -- if a future
session wants true graph-level Send fan-out (splitting this into a router
+ a `diacritize_single_verse` node), the model-call logic below
(`_diacritize_one_verse`) already isolates cleanly for that refactor.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from backends.model_provider import get_model
from tools.prosody_tools import (
    verify_batch_tool,
    verify_single_verse_tool,
    meter_schema_tool,
)
from tools.dataset_tools import commit_verse_tool, log_unresolved_tool
from tools.reconciliation_tools import reconcile_case_ending_tool
from tools.advisory_ledger import record_locked_verse_tool, read_ledger_tool
from tools.advisory_batch import build_batched_advisory_payload_tool
from tools.alignment_guards import (
    validate_advisory_batch_alignment,
    validate_naturalness_batch_alignment,
)
from tools.tracing import trace_run, current_trace

from subagents.diacritizer import DIACRITIZER_SYSTEM_PROMPT
from subagents.irab_checker_agent import IRAB_SYSTEM_PROMPT, IRAB_BATCH_SYSTEM_PROMPT
from subagents.naturalness_critic import (
    NATURALNESS_SYSTEM_PROMPT,
    NATURALNESS_BATCH_SYSTEM_PROMPT,
)

from runtime import MAX_CORRECTION_PASSES, PROJECT_ROOT

DIACRITIZER_MODEL_KWARGS = dict(
    max_completion_tokens=65536,
    max_tokens=65536,
    # Session 4 finding: model_provider.py's _REASONING_SUPPRESSION table
    # defaults deepseek to thinking DISABLED, which was Bug 2's fix (Session
    # 3) -- but for THIS callsite specifically, disabling the thinking
    # channel causes the model to relocate its scansion reasoning into
    # visible `content` instead of skipping it (Session 3's Finding 3).
    # Explicitly re-enabling thinking here routes that reasoning into
    # `additional_kwargs["reasoning_content"]` instead, confirmed via
    # _inspect_reasoning_content.py and _inspect_real_verses_thinking_enabled.py
    # against real verses. get_model()'s explicit-kwarg-wins-over-setdefault
    # contract (see model_provider.py docstring) means this overrides the
    # module default for this call only -- every other provider/callsite is
    # unaffected.
    #
    # Phase 1 (PHASED_PLAN_v4_Diacritizer_Refactor.md) note: this cap was
    # 24576, tuned for ONE verse per call (~20.5k observed reasoning tokens
    # for a single verse in the pre-refactor per-verse dispatch). Dispatch
    # is now one call per BATCH (up to ~12 verses per dataset/inputs/*.jsonl
    # file), so this is raised to 65536 as a starting estimate, not a
    # validated number -- Phase 1's checkpoint requires confirming via
    # `python -m tools.trace_report` that real batch calls aren't hitting
    # this ceiling (truncated JSON output is the symptom to watch for).
    # Adjust up (or down, if cost/latency is a concern and it's never come
    # close) once real telemetry exists.
    extra_body={"thinking": {"type": "enabled"}},
)


# ===========================================================================
# Task 1.1: State schema
# ===========================================================================


def _merge_dicts(a: Optional[dict], b: Optional[dict]) -> dict:
    """Reducer for BatchState["drafts"]: per-verse parallel writers each
    touch a disjoint verse_id key, so a plain last-writer-wins update per
    key is safe -- there is no cross-verse key collision by construction
    (each verse is dispatched exactly once per pass, see dispatch_diacritizer).
    """
    merged = dict(a) if a else {}
    if b:
        merged.update(b)
    return merged


def _append_unique(a: Optional[list], b: Optional[list]) -> list:
    """Reducer for cumulative id lists (locked / structurally_incompatible):
    append only ids not already present, preserving order."""
    out = list(a) if a else []
    seen = set(out)
    for item in b or []:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


class BatchState(TypedDict, total=False):
    verses: list[dict]  # [{verse_id, sadr, ajuz}, ...] -- original input, immutable
    meter_name: str
    pass_number: int  # starts at 1; incremented ONLY in verify_pass (Task 1.3)
    locked: Annotated[list[str], _append_unique]  # cumulative across passes
    broken: list[str]  # THIS pass's still-broken verse_ids (overwritten each pass)
    structurally_incompatible: Annotated[list[str], _append_unique]  # cumulative
    drafts: Annotated[dict, _merge_dicts]  # verse_id -> {"sadr", "ajuz"}
    report_path: Optional[str]  # most recent pass's correction-report file
    thread_id: str  # for workspace/ledger file scoping (mirrors trace context)


# ===========================================================================
# Shared helpers
# ===========================================================================


def _cleanup_json_text(text: str) -> str:
    text = text.strip()
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _extract_json(text: str) -> Any:
    """Parse a model's JSON output, tolerating deviations from the "return
    ONLY a JSON array" instruction that models sometimes make anyway:
    ```json ... ``` fencing, and/or conversational preamble/postamble
    around the fenced or unfenced JSON (e.g. "Here is the diacritized
    output ...\n\n```json\n[...]\n```").

    Tolerates leading conversational text that contains bracket/brace
    characters by scanning all candidates and picking the largest valid JSON structure.
    Also handles trailing commas cleanly.
    """
    stripped = text.strip()

    # 1. Try to find fenced code blocks first
    for pattern in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        for match in re.finditer(pattern, stripped, re.DOTALL):
            block = match.group(1).strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass
            cleaned = _cleanup_json_text(block)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

    # 2. No clean fenced block worked. Let's find any JSON structure (object or array)
    # by scanning all potential starting brackets/braces and finding the largest valid span.
    for i, char in enumerate(stripped):
        if char in ("{", "["):
            target_char = "}" if char == "{" else "]"
            # Search from the end of the text backwards for the matching character
            # to prioritize larger spans
            for j in range(len(stripped) - 1, i, -1):
                if stripped[j] == target_char:
                    candidate = stripped[i : j + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, (dict, list)):
                            return parsed
                    except json.JSONDecodeError:
                        try:
                            cleaned = _cleanup_json_text(candidate)
                            parsed = json.loads(cleaned)
                            if isinstance(parsed, (dict, list)):
                                return parsed
                        except json.JSONDecodeError:
                            pass

    # 3. If everything failed, raise a helpful JSONDecodeError
    raise json.JSONDecodeError(
        "Could not find or decode any valid JSON array/object in model output",
        stripped,
        0,
    )


@tool
def read_workspace_file(file_path: str) -> str:
    """Read a text file (e.g. a pass's correction report) given a path
    relative to the project root -- e.g. 'workspace/<thread>/pass_2_report.json'.
    This is the diacritizer's only filesystem access, mirroring the single
    `read_file` tool it had under DeepAgents' FilesystemMiddleware (see
    subagents/diacritizer.py); it cannot write anything.
    """
    workspace_root = (PROJECT_ROOT / "workspace").resolve()
    p = (PROJECT_ROOT / file_path).resolve()
    try:
        p.relative_to(workspace_root)
    except ValueError:
        return "ERROR: read_workspace_file only permits files under workspace/"
    if not p.exists():
        return f"ERROR: file not found: {file_path}"
    return p.read_text(encoding="utf-8")


def _current_thread_id(state: BatchState) -> str:
    trace = current_trace()
    if trace and trace.langgraph_thread_id:
        return trace.langgraph_thread_id
    return state.get("thread_id", "unknown")


# ===========================================================================
# Task 1.2 / 2.1: dispatch_diacritizer node
# ===========================================================================


def _diacritize_batch(
    model,
    targets: list[dict],
    meter_name: str,
    report_path: Optional[str],
    pass_number: int,
    config: Optional[RunnableConfig] = None,
) -> dict:
    """Phase 1 (PHASED_PLAN_v4_Diacritizer_Refactor.md): ONE model call for
    the entire pass's target-verse array (plus any tool-calls it makes to
    meter_schema_tool / read_workspace_file), not one call per verse.

    This replaces the prior `_diacritize_one_verse` + `ThreadPoolExecutor`
    fan-out, which was the direct cause of the refactor brief's core
    objection ("here is a JSON object of undiacritized verses -- not one
    verse at a time!"). Bound tools are unchanged from the per-verse version
    (`meter_schema_tool`, `read_workspace_file`) -- only the unit of
    dispatch changed, from one verse to the whole batch.

    Carries forward two bugfixes from the per-verse version, now scoped to
    the batch: (1) meter_name is always substituted into the prompt on
    every pass, never left as a literal placeholder; (2) `config` is
    threaded through so this call attaches to the active trace (see
    tools/tracing.py -- though note PHASED_PLAN_v4 Phase 8 flags that
    per-agent attribution itself needs a separate fix under the current
    LangGraph dispatch shape).
    """
    bound = model.bind_tools([meter_schema_tool, read_workspace_file])

    verses_payload = json.dumps(
        [
            {"verse_id": v["verse_id"], "sadr": v["sadr"], "ajuz": v.get("ajuz", "")}
            for v in targets
        ],
        ensure_ascii=False,
    )

    if pass_number == 1:
        user_content = (
            f"Diacritize every verse below for meter '{meter_name}' (pass 1, "
            f"first attempt -- no prior correction report).\n\n"
            f"Input verses (JSON array, {len(targets)} verse(s)):\n{verses_payload}\n\n"
            f"Return a JSON array, same order, one object per verse_id: "
            f'[{{"verse_id": "...", "sadr": "...", "ajuz": "..."}}, ...] '
            f"-- diacritized text only, no commentary, no markdown fences."
        )
    else:
        user_content = (
            f"Correct every verse below (pass {pass_number}) for meter "
            f"'{meter_name}'. Each failed verify_batch_tool last pass. Read "
            f"the correction report at report_path via read_workspace_file "
            f"before drafting -- it names, per verse_id, exactly which foot "
            f"diverged and the prescribed fix. Do not re-diacritize any "
            f"verse from scratch while ignoring the report's guidance for it.\n\n"
            f"report_path: {report_path}\n\n"
            f"Input verses (JSON array, {len(targets)} verse(s)):\n{verses_payload}\n\n"
            f"Return a JSON array, same order, one object per verse_id: "
            f'[{{"verse_id": "...", "sadr": "...", "ajuz": "..."}}, ...] '
            f"-- no commentary, no markdown fences."
        )

    messages = [
        SystemMessage(content=DIACRITIZER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    # Bounded tool-calling loop -- same pattern as before; capped defensively
    # so a misbehaving model can't loop forever inside one dispatch.
    for _ in range(6):
        ai_msg = bound.invoke(messages, config=config)
        messages.append(ai_msg)
        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            raw_text = ai_msg.content
            try:
                parsed = _extract_json(raw_text)
                if not isinstance(parsed, list):
                    raise ValueError("Parsed JSON is not a list")

                # Heuristic salvage: align IDs and fill missing items deterministically
                from subagents.formatter import heuristic_salvage

                salvaged = heuristic_salvage(parsed, targets)
                if salvaged is not None:
                    return salvaged

                drafts: dict = {}
                for item in parsed:
                    drafts[item["verse_id"]] = {
                        "sadr": item["sadr"],
                        "ajuz": item.get("ajuz", ""),
                    }
                return drafts
            except Exception as err:
                from subagents.formatter import call_salvage_agent

                error_info = f"{type(err).__name__}: {str(err)}"
                print(
                    f"[*] Diacritizer output formatting check failed ({error_info}). Invoking Formatter/Salvage agent..."
                )
                return call_salvage_agent(model, raw_text, targets, error_info)
        for tc in tool_calls:
            if tc["name"] == "meter_schema_tool":
                result = meter_schema_tool(**tc["args"])
            elif tc["name"] == "read_workspace_file":
                result = read_workspace_file.invoke(tc["args"])
            else:
                result = f"ERROR: unknown tool {tc['name']}"
            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=tc["id"],
                )
            )

    raise RuntimeError(
        f"diacritizer tool-loop did not converge for a batch of {len(targets)} "
        f"verse(s) (pass {pass_number})"
    )


def make_dispatch_diacritizer(diacritizer_model):
    def dispatch_diacritizer(state: BatchState, config: RunnableConfig) -> dict:
        """Dispatch exactly ONE diacritizer call per pass, carrying every
        target verse as a single JSON array (Phase 1). Pass 1 dispatches
        ALL input verses unconditionally (Structural Incompatibility Rule
        -- pyarud requires diacritized text, so nothing can be pre-filtered
        before a first draft exists). From pass 2 on, dispatch only verses
        currently `broken` (never `locked`, never
        `structurally_incompatible` -- both are excluded by construction).

        Contains NO reference to pass_number-driven looping, verify_batch_tool,
        or any decision about whether to dispatch again -- that's still
        route_after_verify's job alone, unchanged by this refactor.

        Declares `config` so LangGraph injects the active RunnableConfig
        (which carries run_one_batch's trace.callback) into the single
        underlying model call.
        """
        pass_number = state.get("pass_number", 1)
        if pass_number == 1:
            targets = list(state["verses"])
        else:
            broken_ids = set(state.get("broken", []))
            targets = [v for v in state["verses"] if v["verse_id"] in broken_ids]

        if not targets:
            return {"drafts": {}}

        meter_name = state["meter_name"]
        report_path = state.get("report_path")
        drafts = _diacritize_batch(
            diacritizer_model, targets, meter_name, report_path, pass_number, config
        )

        return {"drafts": drafts}

    return dispatch_diacritizer


# ===========================================================================
# Task 1.3 / 2.5: verify_pass node
# ===========================================================================


def verify_pass(state: BatchState) -> dict:
    """Task 1.3: call verify_batch_tool unchanged; this is the ONLY place
    pass_number is ever incremented (see Task 1.3's dangerous-zone note and
    Task 3.5's hard-cap verification).

    Also folds in Task 2.5's immediate-logging requirement: any verse newly
    structurally_incompatible THIS pass is logged via log_unresolved_tool
    right away (it must never reach dispatch_diacritizer again -- excluding
    it here, rather than deferring to the terminal node, is what makes
    Task 3.4's exclusion guarantee airtight), and any newly-locked verse is
    recorded via record_locked_verse_tool immediately (matching the original
    ORCHESTRATOR_SYSTEM_PROMPT's "after each verify_batch_tool pass" cadence
    -- not deferred to the end of the batch).
    """
    prior_locked = set(state.get("locked", []))
    prior_incompatible = set(state.get("structurally_incompatible", []))
    drafts = state.get("drafts", {})
    pass_number = state.get("pass_number", 1)

    # Build the current text for every verse still in play (exclude verses
    # already excluded as structurally_incompatible in an earlier pass).
    verses_to_verify = []
    for v in state["verses"]:
        vid = v["verse_id"]
        if vid in prior_incompatible:
            continue
        draft = drafts.get(vid)
        text = (
            {"verse_id": vid, "sadr": draft["sadr"], "ajuz": draft.get("ajuz", "")}
            if draft
            else v
        )
        verses_to_verify.append(text)

    result = verify_batch_tool(verses_to_verify, state["meter_name"], pass_number)

    newly_locked = [vid for vid in result["locked"] if vid not in prior_locked]
    newly_incompatible = [
        vid
        for vid in result["structurally_incompatible"]
        if vid not in prior_incompatible
    ]

    report_text = None
    rp = result.get("report_path")
    if rp:
        p = PROJECT_ROOT / rp
        if p.exists():
            report_text = p.read_text(encoding="utf-8")

    for vid in newly_locked:
        draft = drafts.get(vid) or next(
            v for v in state["verses"] if v["verse_id"] == vid
        )
        record_locked_verse_tool(
            verse_id=vid,
            sadr=draft["sadr"],
            ajuz=draft.get("ajuz", ""),
            meter=state["meter_name"],
        )

    for vid in newly_incompatible:
        draft = drafts.get(vid) or next(
            v for v in state["verses"] if v["verse_id"] == vid
        )
        log_unresolved_tool(
            verse_id=vid,
            sadr=draft["sadr"],
            ajuz=draft.get("ajuz", ""),
            meter=state["meter_name"],
            last_report=report_text or "",
            stage="structurally_incompatible",
            reason=(
                f"verify_batch_tool pass {pass_number}: mora/foot mismatch that "
                f"persists even in diacritized output -- see report_path={rp}"
            ),
        )

    return {
        "locked": newly_locked,  # merged via _append_unique reducer
        "broken": result["broken"],
        "structurally_incompatible": newly_incompatible,  # merged via reducer
        "report_path": rp,
        "pass_number": pass_number + 1,
    }


def route_after_verify(state: BatchState) -> str:
    """Task 1.4: the single edge this migration exists to get right. Pure
    Python, no LLM call, no branch that can fire dispatch_diacritizer twice
    without an intervening verify_pass -- guaranteed by the graph SHAPE
    (dispatch_diacritizer's only outgoing edge is the fixed one to
    verify_pass; this function is the only thing that can route back to
    dispatch_diacritizer, and it's only reachable AFTER verify_pass runs).
    """
    if state.get("broken") and state.get("pass_number", 1) <= MAX_CORRECTION_PASSES:
        return "dispatch_diacritizer"
    return "advisory_stage"


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
                return commit_verse_tool(
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
        return commit_verse_tool(
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
        return commit_verse_tool(
            verse_id=verse_id,
            sadr=sadr,
            ajuz=ajuz,
            meter=meter,
            naturalness_flag=True,
            notes=naturalness_verdict.get("note", ""),
        )

    # Clean on both advisory axes.
    return commit_verse_tool(verse_id=verse_id, sadr=sadr, ajuz=ajuz, meter=meter)


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
            log_unresolved_tool(
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
        payload_result = build_batched_advisory_payload_tool()
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
        read_ledger_tool(clear=True)

        return {}

    return advisory_stage


# ===========================================================================
# Task 1.4/1.1: graph construction
# ===========================================================================


def build_graph(diacritizer_model, advisory_model, checkpointer=None):
    graph = StateGraph(BatchState)

    graph.add_node("dispatch_diacritizer", make_dispatch_diacritizer(diacritizer_model))
    graph.add_node("verify_pass", verify_pass)
    graph.add_node("advisory_stage", make_advisory_stage(advisory_model))

    graph.set_entry_point("dispatch_diacritizer")
    graph.add_edge("dispatch_diacritizer", "verify_pass")
    graph.add_conditional_edges(
        "verify_pass",
        route_after_verify,
        {
            "dispatch_diacritizer": "dispatch_diacritizer",
            "advisory_stage": "advisory_stage",
        },
    )
    graph.add_edge("advisory_stage", END)

    return graph.compile(checkpointer=checkpointer)


# ===========================================================================
# Task 3.1: SqliteSaver checkpointing (mirrors main.py::build_agent verbatim)
# ===========================================================================


def build_langgraph_pipeline(use_checkpointer: bool = True):
    """Analogous to main.py::build_agent() -- returns (compiled_graph,
    checkpoint_conn, checkpoint_db_path) so main.py's CLI runner can treat
    both engines uniformly (Task 3.3).
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    model = get_model()
    diacritizer_model = get_model(**DIACRITIZER_MODEL_KWARGS)

    checkpointer = None
    checkpoint_conn = None
    checkpoint_db_path = None

    if use_checkpointer:
        checkpoint_db_path = PROJECT_ROOT / "checkpoints.sqlite"
        checkpoint_conn = sqlite3.connect(
            str(checkpoint_db_path), check_same_thread=False
        )
        checkpoint_conn.execute("PRAGMA busy_timeout = 30000")
        checkpoint_conn.execute("PRAGMA synchronous = NORMAL")
        try:
            cursor = checkpoint_conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            if integrity_result != "ok":
                print(
                    f"[!] Warning: Checkpoint database corrupted. PRAGMA integrity_check returned: {integrity_result}"
                )
        except Exception as exc:
            print(f"[-] Pre-run checkpoint diagnostic failed: {exc}")
        checkpointer = SqliteSaver(checkpoint_conn)

    graph = build_graph(diacritizer_model, model, checkpointer=checkpointer)
    return graph, checkpoint_conn, checkpoint_db_path


def build_studio_graph():
    """LangGraph Studio factory; Studio owns checkpoint persistence itself."""
    graph, _, _ = build_langgraph_pipeline(use_checkpointer=False)
    return graph


# ===========================================================================
# Task 3.2: tracing integration + a convenience single-batch runner
# ===========================================================================


def run_one_batch(graph, verses_batch: list[dict], meter: str, thread_id: str) -> dict:
    """Invoke the graph for one (input file, meter) batch under trace_run,
    with a caller-supplied LangGraph-namespaced thread id.
    """
    with trace_run(label=meter, langgraph_thread_id=thread_id) as trace:
        run_config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [trace.callback],
        }
        print(
            f"[*] trace_id='{trace.trace_id}' (inspect with: python -m tools.trace_report --trace {trace.trace_id})"
        )

        try:
            existing_state = graph.get_state(run_config)
        except Exception:
            existing_state = None

        initial_state: BatchState = {
            "verses": verses_batch,
            "meter_name": meter,
            "pass_number": 1,
            "locked": [],
            "broken": [],
            "structurally_incompatible": [],
            "drafts": {},
            "report_path": None,
            "thread_id": thread_id,
        }

        if existing_state and existing_state.next:
            print(
                f"[*] Resuming interrupted LangGraph thread '{thread_id}' (next step: {existing_state.next})...."
            )
            result = graph.invoke(None, config=run_config)
        else:
            result = graph.invoke(initial_state, config=run_config)

        print(
            f"[*] trace summary: python -m tools.trace_report --trace {trace.trace_id}"
        )
        return result
