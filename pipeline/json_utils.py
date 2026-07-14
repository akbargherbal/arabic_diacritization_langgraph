"""
pipeline/json_utils.py
========================
Model-output JSON parsing helpers, extracted verbatim from
langgraph_pipeline.py (Phase 2 of docs/REFACTOR_PLAN.md). Pure text
processing, no model/tool calls, no test patches this file's names
directly -- safe to relocate without touching any patch.object(lp, ...)
target.
"""

from __future__ import annotations

import json
import re
from typing import Any


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
