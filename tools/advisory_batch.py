"""
tools/advisory_batch.py
========================
Helper tool to compile recorded locked verses into a pure JSON payload.
"""

import json
from tools.advisory_ledger import read_ledger_tool


def build_batched_advisory_payload_tool() -> dict:
    """
    Build a pure JSON string payload of locked verses for batched advisory subagents.
    """
    res = read_ledger_tool(clear=False)
    verses = res.get("verses", [])
    if not verses:
        return {"payload": None, "reason": "ledger is empty"}

    # Serialize only the three required keys: verse_id, sadr, ajuz
    payload_list = []
    for v in verses:
        payload_list.append(
            {"verse_id": v["verse_id"], "sadr": v["sadr"], "ajuz": v["ajuz"]}
        )

    # Pure JSON string, unmodified, parseable by json.loads
    payload_json = json.dumps(payload_list, ensure_ascii=False)
    return {"payload": payload_json}
