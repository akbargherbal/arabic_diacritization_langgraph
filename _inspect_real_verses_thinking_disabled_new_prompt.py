"""
_inspect_real_verses_thinking_disabled_new_prompt.py

Cheaper test than _inspect_real_verses_thinking_enabled.py: uses the ORIGINAL
Bug-2 config (thinking DISABLED via model_provider.py's default
_REASONING_SUPPRESSION, i.e. no extra_body override here at all) but with the
NEW tightened DIACRITIZER_SYSTEM_PROMPT (no fences, no filler, trust the tool).

Hypothesis: Finding 3's root cause was the prompt never having an explicit
output contract, not thinking being disabled per se. If this run produces
clean JSON quickly (few hundred to low-thousands of tokens, no length
truncation), thinking-enabled is unnecessary -- cheaper and faster, matching
the Gemini-speed comparison. If content still gets contaminated with
narration, thinking-enabled is genuinely required despite its cost.

Sequential by design (same reason as the original script -- clean per-verse
diagnostic output, not a preview of production parallel dispatch).
Read-only: no dataset/checkpoint writes.
"""

import json
from backends.model_provider import get_model
from subagents.diacritizer import DIACRITIZER_SYSTEM_PROMPT
from tools.prosody_tools import meter_schema_tool
from langgraph_pipeline import read_workspace_file
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

with open("dataset/inputs/3VERSES_1919_batch_00.jsonl", "r", encoding="utf-8") as f:
    verses = [json.loads(line) for line in f if line.strip()]

# Deliberately NOT importing DIACRITIZER_MODEL_KWARGS here -- this test uses
# thinking DISABLED (model_provider.py's default for deepseek), the opposite
# of what's currently in DIACRITIZER_MODEL_KWARGS after the Session 4 patch.
# A generous but much smaller cap than the thinking-enabled test, since a
# clean-JSON-only response shouldn't need anywhere near 16k/24k tokens.
kwargs = dict(max_completion_tokens=4096, max_tokens=4096)

print(f"Testing with: {kwargs} (thinking left at model_provider.py's default -- disabled)")
print("=" * 60)

m = get_model(**kwargs)
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
        reasoning_content = ai_msg.additional_kwargs.get("reasoning_content")

        print(f"--- iteration {i} ---")
        print("CONTENT:", repr(ai_msg.content))
        print("REASONING_CONTENT length:", len(reasoning_content) if reasoning_content else 0)
        print("TOOL_CALLS:", [tc["name"] for tc in tool_calls])
        print("FINISH_REASON:", ai_msg.response_metadata.get("finish_reason"))
        print("TOKEN_USAGE:", ai_msg.response_metadata.get("token_usage"))

        if not tool_calls:
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
