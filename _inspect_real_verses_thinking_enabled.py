"""
_inspect_real_verses_thinking_enabled.py

Minimal-diff variant of _inspect_real_verses.py (Session 3's script that reproduced
Finding 3 -- see Session_3_Handover.md). Only three changes from the original:

  1. Override DIACRITIZER_MODEL_KWARGS's extra_body to re-enable thinking (opposite of
     Bug 2's fix, which set thinking to disabled) and raise the token cap, since
     reasoning_content and content share the same completion budget.
  2. Print reasoning_content alongside content on every iteration.
  3. On the final iteration (no more tool_calls), try json.loads(ai_msg.content) to check
     whether content is clean JSON on its own -- the actual thing Finding 3 needs answered.

Everything else -- the tool loop, message construction, meter name, prompt text -- is
identical to the original script. Read-only: no dataset/checkpoint writes.
"""

import json
from backends.model_provider import get_model
from subagents.diacritizer import DIACRITIZER_SYSTEM_PROMPT
from tools.prosody_tools import meter_schema_tool
from langgraph_pipeline import read_workspace_file, DIACRITIZER_MODEL_KWARGS
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

with open("dataset/inputs/3VERSES_1919_batch_00.jsonl", "r", encoding="utf-8") as f:
    verses = [json.loads(line) for line in f if line.strip()]

# --- CHANGE 1: override thinking + token cap, keep everything else from DIACRITIZER_MODEL_KWARGS ---
overridden_kwargs = dict(DIACRITIZER_MODEL_KWARGS)
overridden_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
overridden_kwargs["max_tokens"] = 16384
overridden_kwargs["max_completion_tokens"] = 16384

print(f"Original DIACRITIZER_MODEL_KWARGS: {DIACRITIZER_MODEL_KWARGS}")
print(f"Overridden for this probe: {overridden_kwargs}")
print("=" * 60)

m = get_model(**overridden_kwargs)
bound = m.bind_tools([meter_schema_tool, read_workspace_file])

for verse in verses:
    print(f"=========== VERSE {verse['verse_id']} ===========")
    messages = [
        SystemMessage(content=DIACRITIZER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Diacritize this verse for meter 'ramal' (pass 1, no prior "
            f"correction report -- first attempt):\n"
            f"verse_id: {verse['verse_id']}\nsadr: {verse['sadr']}\najuz: {verse.get('ajuz','')}\n\n"
            f'Return ONLY a JSON object: {{"verse_id": "{verse["verse_id"]}", "sadr": "...", "ajuz": "..."}} '
            f"-- the corrected/diacritized text, no commentary, no markdown fences."
        )),
    ]
    for i in range(6):
        ai_msg = bound.invoke(messages)
        messages.append(ai_msg)
        tool_calls = getattr(ai_msg, "tool_calls", None) or []

        # --- CHANGE 2: dump reasoning_content alongside content ---
        reasoning_content = ai_msg.additional_kwargs.get("reasoning_content")

        print(f"--- iteration {i} ---")
        print("CONTENT:", repr(ai_msg.content))
        print("REASONING_CONTENT length:", len(reasoning_content) if reasoning_content else 0)
        print("TOOL_CALLS:", [tc["name"] for tc in tool_calls])
        print("FINISH_REASON:", ai_msg.response_metadata.get("finish_reason"))
        print("TOKEN_USAGE:", ai_msg.response_metadata.get("token_usage"))

        if not tool_calls:
            # --- CHANGE 3: check if content is clean JSON on its own ---
            try:
                parsed = json.loads(ai_msg.content)
                print("RESULT: content is valid, clean JSON.")
                print(json.dumps(parsed, ensure_ascii=False, indent=2))
            except json.JSONDecodeError as e:
                print(f"RESULT: content is NOT valid JSON on its own ({e}).")
            break

        for tc in tool_calls:
            if tc["name"] == "meter_schema_tool":
                result = meter_schema_tool(**tc["args"])
            elif tc["name"] == "read_workspace_file":
                result = read_workspace_file.invoke(tc["args"])
            else:
                result = f"ERROR: unknown tool {tc['name']}"
            messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tc["id"]))
    print()
