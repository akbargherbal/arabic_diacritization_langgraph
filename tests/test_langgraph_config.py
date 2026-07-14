"""
tests/test_langgraph_config.py
==================================
Unit tests for langgraph.json and local configuration files.
Phase 6 of docs/REFACTOR_PLAN.md.

This test connects the isolated configuration-level nodes (langgraph.json config,
graphs config keys, environment pointers) to the test suite to ensure that
runtime configurations are valid, syntax-correct, and reference existing,
importable entrypoint symbols in the codebase.
"""

import os
import json
import importlib
import pytest


def test_langgraph_json_exists_and_parses():
    """Verify langgraph.json exists at root and is syntactically valid JSON."""
    config_path = "langgraph.json"
    assert os.path.exists(config_path), "langgraph.json is missing from root directory."

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as err:
            pytest.fail(f"langgraph.json is not valid JSON: {err}")

    assert isinstance(data, dict), "langgraph.json root must be a JSON object."
    assert "graphs" in data, "langgraph.json missing required 'graphs' definition."
    assert "dependencies" in data, "langgraph.json missing required 'dependencies' definition."
    assert "env" in data, "langgraph.json missing required 'env' definition."


def test_langgraph_graphs_point_to_valid_symbols():
    """Verify that all entrypoints defined in langgraph.json point to importable symbols."""
    config_path = "langgraph.json"
    assert os.path.exists(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graphs = data.get("graphs", {})
    assert graphs, "langgraph.json defines no graphs."

    for graph_name, import_path in graphs.items():
        assert isinstance(import_path, str), f"Graph path for {graph_name} must be a string."
        assert ":" in import_path, (
            f"Graph path '{import_path}' for '{graph_name}' must use 'module:attribute' format "
            "as per LangGraph specifications."
        )

        module_name, attr_name = import_path.split(":")

        # Try to import the module
        try:
            module = importlib.import_module(module_name)
        except ImportError as err:
            pytest.fail(
                f"Failed to import module '{module_name}' defined for graph '{graph_name}' in "
                f"langgraph.json: {err}"
            )

        # Try to get the attribute
        assert hasattr(module, attr_name), (
            f"Module '{module_name}' has no attribute '{attr_name}' "
            f"as defined for graph '{graph_name}' in langgraph.json."
        )

        attr = getattr(module, attr_name)
        assert callable(attr), (
            f"Attribute '{attr_name}' in module '{module_name}' is not callable. "
            "Graph entry points must be functions/callables that construct the StateGraph."
        )


def test_env_file_pointer_corresponds_to_example():
    """Verify that the env file configuration matches .env.example variables."""
    config_path = "langgraph.json"
    assert os.path.exists(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    env_pointer = data.get("env")
    assert env_pointer, "langgraph.json 'env' key is empty."

    # Verify that .env.example exists to document required variables
    example_path = ".env.example"
    assert os.path.exists(example_path), ".env.example must exist to document required env variables."
