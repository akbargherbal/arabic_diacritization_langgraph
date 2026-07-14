"""
pipeline/diacritize_stage.py
==============================
Task 1.2/2.1's dispatch_diacritizer node, extracted from langgraph_pipeline.py
(Phase 2b of docs/REFACTOR_PLAN.md).

IMPORTANT for anyone patching this module in tests: `_diacritize_batch` is
called as a bare name from inside `dispatch_diacritizer`'s closure (built by
`make_dispatch_diacritizer`), so it resolves via *this module's* globals at
call time. Patch it here --
`unittest.mock.patch.object(pipeline.diacritize_stage, "_diacritize_batch", ...)`
-- not on `langgraph_pipeline`. `langgraph_pipeline.py` re-exports the name
for import-compatibility only; patching the re-exported binding there does
not affect what this module's own functions see.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from runtime import PROJECT_ROOT
from subagents.diacritizer import DIACRITIZER_SYSTEM_PROMPT
from tools.prosody_tools import meter_schema_tool

from pipeline.json_utils import _extract_json
from pipeline.state import BatchState


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
        (which carries run_one_batch's trace.callback) into the single\n        underlying model call.
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
