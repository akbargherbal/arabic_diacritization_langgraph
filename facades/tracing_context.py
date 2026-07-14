"""
facades/tracing_context.py
=============================
Phase 4 of PHASED_PLAN.md: "Put a face on the cross-cutting infra".

Before this facade, pipeline/runner.py called `tools.tracing.trace_run()`
directly and reached into the returned `Trace` object's `.callback` /
`.trace_id` attributes to assemble LangGraph's run config. That's fine for
one call site, but it means runner.py depends on tools.tracing's internal
shape (TraceStore, TokenTracingCallback, the Trace dataclass) rather than
on a narrow, stable interface.

TracingContext is that narrow interface: "start a trace for this batch,
give me the run config to invoke the graph with, and expose the trace_id
for logging." Nothing outside this module and tools/tracing.py needs to
know TraceStore or TokenTracingCallback exist.

Patch `facades.tracing_context.trace_run` (not `tools.tracing.trace_run`)
when testing code that goes through this facade -- it's imported into this
module's namespace and resolved as a bare name from here.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

from tools.tracing import Trace, trace_run


class TracingContext:
    """Thin wrapper around one `tools.tracing.Trace`. Construct via
    `TracingContext.start(...)`, not directly."""

    def __init__(self, trace: Trace):
        self._trace = trace

    @property
    def trace_id(self) -> str:
        return self._trace.trace_id

    def run_config(self, thread_id: str) -> dict:
        """Build the LangGraph `config=` dict for this trace: the
        caller-supplied thread_id plus the tracing callback wired in.
        Callers never touch TokenTracingCallback directly."""
        return {
            "configurable": {"thread_id": thread_id},
            "callbacks": [self._trace.callback],
        }

    @staticmethod
    @contextmanager
    def start(label: Optional[str] = None, thread_id: Optional[str] = None):
        """Open a trace for one `run_one_batch` attempt. Mirrors
        `tools.tracing.trace_run`'s contract (fresh trace_id every call,
        including resumes) but yields a TracingContext instead of a raw
        Trace."""
        with trace_run(label=label, langgraph_thread_id=thread_id) as trace:
            yield TracingContext(trace)
