from backends.model_provider import get_model
from subagents.diacritizer import DIACRITIZER_SYSTEM_PROMPT
from tools.prosody_tools import meter_schema_tool
from langgraph_pipeline import read_workspace_file, DIACRITIZER_MODEL_KWARGS
from langchain_core.messages import SystemMessage, HumanMessage

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

resp = bound.invoke(messages)
print("CONTENT:", repr(resp.content))
print("TOOL_CALLS:", resp.tool_calls)
print("FINISH_REASON:", resp.response_metadata.get("finish_reason"))
print("TOKEN_USAGE:", resp.response_metadata.get("token_usage"))
