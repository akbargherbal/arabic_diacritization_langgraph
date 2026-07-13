from backends.model_provider import get_model
from subagents.diacritizer import DIACRITIZER_SYSTEM_PROMPT
from tools.prosody_tools import meter_schema_tool
from langgraph_pipeline import read_workspace_file, DIACRITIZER_MODEL_KWARGS
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
import json

m = get_model(**DIACRITIZER_MODEL_KWARGS)
bound = m.bind_tools([meter_schema_tool, read_workspace_file])

messages = [
    SystemMessage(content=DIACRITIZER_SYSTEM_PROMPT),
    HumanMessage(content=(
        "Diacritize this verse for meter 'ramal' (pass 1, no prior "
        "correction report -- first attempt):\n"
        "verse_id: TEST-1\nsadr: على الكتب\najuz: عجز تجريبي\n\n"
        "Return ONLY a JSON object: {\"verse_id\": \"TEST-1\", \"sadr\": \"...\", \"ajuz\": \"...\"} "
        "-- the corrected/diacritized text, no commentary, no markdown fences."
    )),
]

for i in range(6):
    ai_msg = bound.invoke(messages)
    messages.append(ai_msg)
    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    print(f"--- iteration {i} ---")
    print("CONTENT:", repr(ai_msg.content))
    print("TOOL_CALLS:", [tc["name"] for tc in tool_calls])
    print("FINISH_REASON:", ai_msg.response_metadata.get("finish_reason"))
    print("TOKEN_USAGE:", ai_msg.response_metadata.get("token_usage"))
    if not tool_calls:
        print("WOULD PARSE:", repr(ai_msg.content.strip()))
        break
    for tc in tool_calls:
        if tc["name"] == "meter_schema_tool":
            result = meter_schema_tool(**tc["args"])
        elif tc["name"] == "read_workspace_file":
            result = read_workspace_file.invoke(tc["args"])
        else:
            result = f"ERROR: unknown tool {tc['name']}"
        messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tc["id"]))
