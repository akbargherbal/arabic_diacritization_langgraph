import json
import traceback
from langgraph_pipeline import build_langgraph_pipeline, run_one_batch

graph, conn, db_path = build_langgraph_pipeline(use_checkpointer=False)

with open("dataset/inputs/3VERSES_1919_batch_00.jsonl", "r", encoding="utf-8") as f:
    verses = [json.loads(line) for line in f if line.strip()]

try:
    result = run_one_batch(graph, verses, "ramal", "diag_run_no_checkpoint")
    print("SUCCESS")
    print(result)
except Exception:
    print("FAILED -- full traceback below:")
    traceback.print_exc()
