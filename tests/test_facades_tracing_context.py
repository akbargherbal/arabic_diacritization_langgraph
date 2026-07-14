"""
tests/test_facades_tracing_context.py
========================================
Unit tests for the TracingContext facade (Phase 4 of PHASED_PLAN.md).

Proves: `start()` opens exactly one real trace (via tools.tracing.trace_run,
so TraceStore/TokenTracingCallback plumbing is exercised end-to-end, not
mocked away), `trace_id` reflects that trace, and `run_config()` builds the
same shape pipeline/runner.py used to build by hand -- a
`{"configurable": {"thread_id": ...}, "callbacks": [...]}` dict with the
trace's own callback wired in.
"""

from facades.tracing_context import TracingContext
from tools.tracing import TokenTracingCallback


def test_start_yields_a_tracing_context_with_a_trace_id():
    with TracingContext.start(label="ramal", thread_id="facade_test_1") as tracing:
        assert isinstance(tracing, TracingContext)
        assert isinstance(tracing.trace_id, str)
        assert tracing.trace_id  # non-empty


def test_run_config_has_expected_shape_and_callback():
    with TracingContext.start(label="ramal", thread_id="facade_test_2") as tracing:
        config = tracing.run_config("facade_test_2")

    assert config["configurable"] == {"thread_id": "facade_test_2"}
    assert len(config["callbacks"]) == 1
    assert isinstance(config["callbacks"][0], TokenTracingCallback)


def test_each_start_call_gets_a_fresh_trace_id():
    with TracingContext.start(
        label="ramal", thread_id="facade_test_3"
    ) as tracing_a:
        id_a = tracing_a.trace_id

    with TracingContext.start(
        label="ramal", thread_id="facade_test_3"
    ) as tracing_b:
        id_b = tracing_b.trace_id

    # Same thread_id (simulating a resume), but trace_run guarantees a
    # fresh trace_id per attempt -- see tools/tracing.py's module docstring
    # ("IMPORTANT -- trace_id vs. LangGraph's thread_id").
    assert id_a != id_b
