"""
tests/test_pipeline_json_utils.py
====================================
Unit tests for pipeline/json_utils.py (Phase 2 of docs/REFACTOR_PLAN.md).

_extract_json is the tolerant parser standing between "the model was asked
to return ONLY a JSON array" and "the model returned that, plus a paragraph
of preamble, inside a markdown fence, with a trailing comma." Splitting it
out of langgraph_pipeline.py's 890-line monolith exposed that it had almost
no direct test coverage (15% -- see docs/REFACTOR_PLAN.md Phase 2 notes)
despite being exactly the kind of multi-branch heuristic parsing that's
most likely to silently misbehave. These tests pin its documented fallback
behavior down.
"""

import json

import pytest

from pipeline.json_utils import _cleanup_json_text, _extract_json


class TestCleanupJsonText:
    def test_strips_surrounding_whitespace(self):
        assert _cleanup_json_text("  [1, 2]  ") == "[1, 2]"

    def test_removes_trailing_comma_before_closing_bracket(self):
        assert _cleanup_json_text("[1, 2, 3,]") == "[1, 2, 3]"

    def test_removes_trailing_comma_before_closing_brace(self):
        assert _cleanup_json_text('{"a": 1,}') == '{"a": 1}'

    def test_leaves_valid_json_unchanged_besides_stripping(self):
        assert _cleanup_json_text('{"a": 1}') == '{"a": 1}'


class TestExtractJson:
    def test_plain_json_array(self):
        assert _extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_plain_json_object(self):
        assert _extract_json('{"verse_id": "V1", "flag": false}') == {
            "verse_id": "V1",
            "flag": False,
        }

    def test_json_fenced_with_language_tag(self):
        text = 'Here is the result:\n```json\n[{"a": 1}]\n```'
        assert _extract_json(text) == [{"a": 1}]

    def test_json_fenced_without_language_tag(self):
        text = '```\n{"a": 1}\n```'
        assert _extract_json(text) == {"a": 1}

    def test_fenced_json_with_trailing_comma_is_cleaned(self):
        text = '```json\n[{"a": 1},]\n```'
        assert _extract_json(text) == [{"a": 1}]

    def test_conversational_preamble_and_postamble_around_unfenced_json(self):
        text = 'Sure, here is the array:\n[{"verse_id": "V1"}]\nLet me know if you need anything else!'
        assert _extract_json(text) == [{"verse_id": "V1"}]

    def test_picks_largest_valid_span_when_multiple_bracket_pairs_present(self):
        # The preamble sentence itself contains a bracket pair that isn't
        # valid JSON on its own; the real payload is the larger span.
        text = 'Note [see below] for details: [{"a": 1}, {"b": 2}]'
        assert _extract_json(text) == [{"a": 1}, {"b": 2}]

    def test_unfenced_json_with_trailing_comma_is_cleaned(self):
        text = 'Result: [{"a": 1}, {"b": 2},]'
        assert _extract_json(text) == [{"a": 1}, {"b": 2}]

    def test_raises_json_decode_error_when_nothing_parses(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("This response contains no JSON at all.")

    def test_raises_json_decode_error_on_empty_string(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("")
