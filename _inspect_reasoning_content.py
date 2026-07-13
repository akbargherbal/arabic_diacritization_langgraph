"""
_inspect_reasoning_content.py

Purpose (Session 4 investigation, per handover item 2 candidate):
Confirm whether re-enabling DeepSeek's thinking channel (extra_body={"thinking":{"type":"enabled"}})
routes the scansion narration into `reasoning_content` instead of `content`, and whether `content`
alone then contains clean, parseable JSON — without touching subagents/diacritizer.py's prompt.

This is a read-only diagnostic: one direct model call, no graph, no tool loop, no dataset/checkpoint
writes. Matches the pattern used for _inspect_real_verses.py last session.

ADJUST BEFORE RUNNING (only remaining unconfirmed item):
  - The import of the diacritizer system prompt below assumes a module-level constant named
    DIACRITIZER_SYSTEM_PROMPT. If subagents/diacritizer.py names it something else, fix the
    import.

get_model()'s real signature (confirmed against backends/model_provider.py, Session 4):
    get_model(provider: str | None = None, model_name: str | None = None, **kwargs) -> BaseChatModel
It wraps langchain's init_chat_model and returns a retry-wrapped LangChain BaseChatModel, NOT a
raw OpenAI-SDK client -- so this script calls model.invoke(messages), not
model.chat.completions.create(...).
"""

import json
import sys

# --- ADJUST: match your actual import paths ---
try:
    from subagents.diacritizer import DIACRITIZER_SYSTEM_PROMPT
except ImportError:
    print("Could not import DIACRITIZER_SYSTEM_PROMPT from subagents.diacritizer — "
          "open that file and copy the exact constant name here.")
    sys.exit(1)

try:
    from backends.model_provider import get_model
except ImportError:
    print("Could not import get_model from backends.model_provider.")
    sys.exit(1)

# --- Load one real verse from the 3-verse fixture ---
FIXTURE_PATH = "dataset/inputs/3VERSES_1919_batch_00.jsonl"

with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f if l.strip()]

verse = lines[0]  # first verse only, for this probe
print(f"Testing verse_id={verse.get('verse_id')}")
print(f"sadr: {verse.get('sadr')}")
print(f"ajuz: {verse.get('ajuz')}")
print("-" * 60)

# --- Build the user message the same way dispatch_diacritizer would ---
user_content = json.dumps(
    {"verse_id": verse.get("verse_id"), "sadr": verse.get("sadr"), "ajuz": verse.get("ajuz")},
    ensure_ascii=False,
)

messages = [
    {"role": "system", "content": DIACRITIZER_SYSTEM_PROMPT},
    {"role": "user", "content": user_content},
]

# --- Direct call, thinking ENABLED (opposite of Bug 2's fix) ---
# get_model() returns a LangChain BaseChatModel built via init_chat_model, retry-wrapped.
# **kwargs are forwarded to init_chat_model -> the underlying provider class's __init__.
# extra_body passed explicitly here wins over _REASONING_SUPPRESSION's setdefault
# (deepseek's default is thinking DISABLED -- see model_provider.py's _REASONING_SUPPRESSION
# table -- so we must override it explicitly to test the enabled case).
#
# NOTE: raise the token cap well above 2048 -- reasoning_content and content share the same
# completion budget, so a tight cap will truncate reasoning before content even starts.
model = get_model(
    provider="deepseek",
    model_name="deepseek-v4-pro",
    max_tokens=16384,
    max_completion_tokens=16384,
    extra_body={"thinking": {"type": "enabled"}},
)

result = model.invoke(messages)

content = result.content

# Don't guess a single key name -- dump both dicts so we can see exactly where
# DeepSeek's reasoning trace actually landed under this langchain integration.
print("=== result.additional_kwargs (raw) ===")
print(result.additional_kwargs)
print("=== result.response_metadata (raw) ===")
print(result.response_metadata)
print("-" * 60)

reasoning_content = (
    result.additional_kwargs.get("reasoning_content")
    or result.additional_kwargs.get("reasoning")
    or result.response_metadata.get("reasoning_content")
)
finish_reason = result.response_metadata.get("finish_reason")
usage = result.response_metadata.get("token_usage") or getattr(result, "usage_metadata", None)

print(f"finish_reason: {finish_reason}")
print(f"usage: {usage}")
print("-" * 60)
print(f"reasoning_content length: {len(reasoning_content) if reasoning_content else 0} chars")
print(f"content length: {len(content) if content else 0} chars")
print("-" * 60)
print("=== content (should be JSON only, no scansion narration) ===")
print(content)
print("-" * 60)

# --- The actual thing we're checking ---
try:
    parsed = json.loads(content)
    print("RESULT: content is valid, clean JSON. Hypothesis CONFIRMED.")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
except json.JSONDecodeError as e:
    print(f"RESULT: content is NOT valid JSON on its own ({e}). Hypothesis NOT confirmed as-is.")
    print("First 500 chars of content for inspection:")
    print(content[:500] if content else "(empty)")

print("-" * 60)
print("=== reasoning_content (first 500 chars, for reference only — not parsed) ===")
print((reasoning_content or "(none — check if thinking was actually honored)")[:500])
