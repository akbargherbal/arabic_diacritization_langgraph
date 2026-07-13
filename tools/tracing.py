"""
tools/tracing.py

Lightweight, dependency-free-ish observability for the diacritization
DeepAgent, built specifically around how *this* repo dispatches work:

    agent (deepagents) --tools--> orchestrator's own tools (verify_batch_tool,
    commit_verse_tool, ...) AND the `task` tool, whose call args include
    `subagent_type` ("diacritizer" | "irab_checker" | "naturalness_critic").

That `subagent_type` argument is the ground truth for "who did this LLM
call" — more reliable than guessing from LangGraph node names, since
deepagents' internal graph shape is exactly the kind of thing its own
README says moves fast between versions. So this tracer:

  1. Watches every tool call. When a tool named "task" starts, it reads
     `subagent_type` out of the call args and remembers it.
  2. Watches every LLM call. To attribute it, it walks up the chain of
     parent_run_ids (which LangChain always provides) until it finds an
     ancestor "task" call — that's the subagent. If none is found, the
     call belongs to the orchestrator itself.
  3. Records input/output/cached tokens + latency for every LLM call, and
     wall-clock duration for every tool call, into a local SQLite file
     (traces.sqlite) — same pattern as this repo's existing
     checkpoints.sqlite.

IMPORTANT — trace_id vs. LangGraph's thread_id:
  main.py's `thread_id` is deliberately STABLE per (input file, meter) so a
  Ctrl+C'd run can resume the same checkpoint. That's the opposite of what
  you want for observability (you want *this specific attempt* to be
  distinguishable from the retry after a crash). So this module introduces
  a separate `trace_id` — fresh and unique every time `trace_run()` is
  called, i.e. every `agent.invoke(...)` attempt, including resumes. The
  LangGraph thread_id is recorded alongside it purely for cross-reference.

IMPORTANT — concurrency:
  deepagents/LangGraph can and does execute multiple tool calls from a
  single AIMessage concurrently (e.g. the orchestrator dispatching
  `irab_checker` and `naturalness_critic` as two `task` calls together) —
  each running its own on_tool_start/on_llm_start/on_tool_end/on_llm_end
  sequence, potentially from different threads at the same time.
  sqlite3.Connection(check_same_thread=False) permits a connection to be
  *used* from a different thread than the one that created it, but it does
  NOT make concurrent execute()/commit() calls from multiple threads safe
  -- that's a separate guarantee sqlite3 does not provide on its own. So
  TraceStore serializes all writes through a single threading.Lock. Without
  it, concurrent writers can raise sqlite3.InterfaceError('bad parameter or
  other API misuse') -- this is not hypothetical, it's what happened the
  first time this traced a batch with parallel subagent dispatch.

IMPORTANT — cross-process safety (added):
  WAL mode alone lets readers avoid blocking on a writer's snapshot, but it
  does NOT eliminate SQLITE_BUSY between two *writers* (e.g. this process
  writing while `tools/trace_report.py` is invoked, or two `main.py`
  processes pointed at the same file). A `busy_timeout` pragma is set
  immediately after connecting so SQLite retries internally for a bounded
  window instead of raising immediately -- mirroring the same defense
  main.py already applies to checkpoints.sqlite.

IMPORTANT — connection lifetime (changed):
  Previously, `trace_run()` constructed a brand-new TraceStore (and thus a
  brand-new sqlite3 connection) for every batch, closing it at the end of
  the `with` block. That's needless connect/close churn across a
  multi-meter input file processed in one `main.py` run. `trace_run()` now
  reuses a single module-level TraceStore per (process, db_path), created
  lazily on first use and left open for the life of the process.

---------------------------------------------------------------------------
USAGE (see main.py for the wired-in version)
---------------------------------------------------------------------------

    from tools.tracing import trace_run

    with trace_run(label=meter, langgraph_thread_id=thread_id) as trace:
        response = agent.invoke(
            {"messages": [...]},
            config={**run_config, "callbacks": [trace.callback]},
        )
        print(f"[trace] trace_id={trace.trace_id}")
        print(f"        -> python -m tools.trace_report --trace {trace.trace_id}")

No changes are needed inside subagents/*.py or tools/*.py — attribution is
derived entirely from the `task` tool's own call args, which the
orchestrator already passes correctly.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:  # pragma: no cover - lets this file be imported/tested
    # standalone before langchain is installed in the environment.
    class BaseCallbackHandler:  # type: ignore
        pass


DEFAULT_DB_PATH = Path("traces.sqlite")
DISPATCH_TOOL_NAME = "task"  # deepagents' subagent-dispatch tool
SUBAGENT_ARG_KEYS = ("subagent_type", "subagent", "agent_type")  # defensive
ORCHESTRATOR = "orchestrator"
BUSY_TIMEOUT_MS = 30000  # mirrors main.py's checkpoint_conn setting


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_trace_id() -> str:
    """Human-sortable + unique: 2026-07-08T14-32-01Z_a3f9c2d1"""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class TraceStore:
    """Thin wrapper around a SQLite file. Safe to share across a process --
    including across the multiple threads deepagents/LangGraph may use to
    run parallel tool calls -- because every write is serialized through
    self._lock. check_same_thread=False alone is not sufficient for that;
    see the module docstring's "IMPORTANT — concurrency" section.

    Also sets a busy_timeout pragma so cross-PROCESS writer collisions
    (e.g. this process vs. a concurrently-run trace_report.py, or two
    main.py processes against the same db_path) retry for a bounded window
    instead of raising sqlite3.OperationalError("database is locked")
    immediately. See "IMPORTANT — cross-process safety" above.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS};")
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._init_schema()

    def _init_schema(self) -> None:
        # Called only from __init__, already inside self._lock -- do not
        # acquire the lock again here (RLock isn't used, so a second
        # acquire on a plain Lock from the same thread would deadlock).
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                trace_id            TEXT PRIMARY KEY,
                langgraph_thread_id TEXT,
                label               TEXT,
                started_at          TEXT NOT NULL,
                ended_at            TEXT,
                meta                TEXT
            );

            CREATE TABLE IF NOT EXISTS calls (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id        TEXT NOT NULL,
                run_id          TEXT,
                parent_run_id   TEXT,
                agent           TEXT,       -- 'orchestrator' | 'diacritizer' | 'irab_checker' | 'naturalness_critic' | ...
                model           TEXT,
                kind            TEXT,       -- 'llm' | 'tool'
                tool_name       TEXT,
                started_at      TEXT,
                ended_at        TEXT,
                latency_ms      REAL,
                input_tokens    INTEGER DEFAULT 0,
                output_tokens   INTEGER DEFAULT 0,
                cached_tokens   INTEGER DEFAULT 0,
                total_tokens    INTEGER DEFAULT 0,
                error           TEXT
            );

            -- Chronological lookups (list_runs / _latest_trace_id in
            -- trace_report.py) sort runs by started_at -- index it so that's
            -- not a full-table scan + in-memory sort as the table grows.
            CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);

            -- dump_calls() in trace_report.py filters by trace_id AND sorts
            -- by started_at -- a composite index satisfies both in one
            -- lookup instead of filtering then sorting separately. This
            -- also covers every query that previously relied on
            -- idx_calls_trace (trace_id alone is a prefix of this index).
            CREATE INDEX IF NOT EXISTS idx_calls_trace_started ON calls(trace_id, started_at);

            CREATE INDEX IF NOT EXISTS idx_calls_agent ON calls(trace_id, agent);
            """)
        self._conn.commit()

    def start_run(
        self,
        trace_id: str,
        langgraph_thread_id: Optional[str],
        label: Optional[str],
        meta: str = "",
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs (trace_id, langgraph_thread_id, label, started_at, meta) "
                "VALUES (?, ?, ?, ?, ?)",
                (trace_id, langgraph_thread_id, label, _utcnow_iso(), meta),
            )
            self._conn.commit()

    def end_run(self, trace_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET ended_at = ? WHERE trace_id = ?",
                (_utcnow_iso(), trace_id),
            )
            self._conn.commit()

    def log_call(self, **fields: Any) -> None:
        cols = ", ".join(fields.keys())
        placeholders = ", ".join("?" for _ in fields)
        with self._lock:
            self._conn.execute(
                f"INSERT INTO calls ({cols}) VALUES ({placeholders})",
                tuple(fields.values()),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Callback handler
# ---------------------------------------------------------------------------


class TokenTracingCallback(BaseCallbackHandler):
    """
    Attributes every LLM call to the agent that made it by walking up the
    parent_run_id chain to find the nearest ancestor `task` tool call and
    reading its `subagent_type` argument. Falls back to "orchestrator" if
    no such ancestor exists (i.e. the call is the top-level agent's own).
    """

    def __init__(self, trace_id: str, store: TraceStore):
        self.trace_id = trace_id
        self.store = store
        self._starts: dict[str, dict[str, Any]] = {}
        self._parent_of: dict[str, Optional[str]] = {}
        self._agent_of: dict[str, str] = (
            {}
        )  # run_id -> subagent name, set on `task` tool start

    # -- attribution ------------------------------------------------------

    def _resolve_agent(
        self, run_id: str, tags: Optional[list], metadata: Optional[dict] = None
    ) -> str:
        node = (metadata or {}).get("langgraph_node")
        if node == "dispatch_diacritizer":
            return "diacritizer"
        if node == "advisory_stage":
            return "advisory"
        for t in tags or []:
            if isinstance(t, str) and t.startswith("agent:"):
                return t[len("agent:") :]
        cursor = self._parent_of.get(run_id)
        seen = set()
        while cursor and cursor not in seen:
            seen.add(cursor)
            if cursor in self._agent_of:
                return self._agent_of[cursor]
            cursor = self._parent_of.get(cursor)
        return ORCHESTRATOR

    @staticmethod
    def _extract_subagent_type(input_str: Any, kwargs: dict) -> Optional[str]:
        candidates = []
        # Newer LangChain versions may pass structured args via kwargs["inputs"].
        for source in (kwargs.get("inputs"), input_str):
            if isinstance(source, dict):
                candidates.append(source)
            elif isinstance(source, str):
                try:
                    parsed = json.loads(source)
                    if isinstance(parsed, dict):
                        candidates.append(parsed)
                except (json.JSONDecodeError, TypeError):
                    pass
        for cand in candidates:
            for key in SUBAGENT_ARG_KEYS:
                if key in cand and cand[key]:
                    return str(cand[key])
        return None

    @staticmethod
    def _resolve_model(
        serialized: Optional[dict], invocation_params: Optional[dict]
    ) -> str:
        if invocation_params:
            for key in ("model_name", "model"):
                if invocation_params.get(key):
                    return str(invocation_params[key])
        if serialized:
            for key in ("name", "id"):
                val = serialized.get(key)
                if val:
                    return str(val if isinstance(val, str) else val[-1])
        return "unknown"

    # -- LLM lifecycle ----------------------------------------------------

    def on_llm_start(
        self,
        serialized,
        prompts,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ) -> None:
        self._record_llm_start(run_id, parent_run_id, tags, metadata, serialized, kwargs)

    def on_chat_model_start(
        self,
        serialized,
        messages,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ) -> None:
        self._record_llm_start(run_id, parent_run_id, tags, metadata, serialized, kwargs)

    def _record_llm_start(
        self, run_id, parent_run_id, tags, metadata, serialized, kwargs
    ) -> None:
        rid, pid = str(run_id), str(parent_run_id) if parent_run_id else None
        self._parent_of[rid] = pid
        self._starts[rid] = {
            "t0": time.monotonic(),
            "started_at": _utcnow_iso(),
            "agent": self._resolve_agent(rid, tags, metadata),
            "model": self._resolve_model(serialized, kwargs.get("invocation_params")),
            "parent_run_id": pid,
        }

    def on_llm_end(
        self, response, *, run_id, parent_run_id=None, tags=None, **kwargs
    ) -> None:
        rid = str(run_id)
        start = self._starts.pop(rid, None)
        latency_ms = (time.monotonic() - start["t0"]) * 1000 if start else None

        input_tokens = output_tokens = cached_tokens = total_tokens = 0
        try:
            usage = None
            llm_output = getattr(response, "llm_output", None) or {}
            usage = llm_output.get("token_usage") or llm_output.get("usage")
            if usage is None:
                gen = response.generations[0][0]
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None) if msg else None
            if usage:
                input_tokens = (
                    usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                )
                output_tokens = (
                    usage.get("output_tokens") or usage.get("completion_tokens") or 0
                )
                total_tokens = usage.get("total_tokens") or (
                    input_tokens + output_tokens
                )
                details = (
                    usage.get("input_token_details")
                    or usage.get("prompt_tokens_details")
                    or {}
                )
                cached_tokens = (
                    (details or {}).get("cache_read")
                    or (details or {}).get("cached_tokens")
                    or 0
                )
        except Exception:
            pass  # tracing must never break the actual run

        self.store.log_call(
            trace_id=self.trace_id,
            run_id=rid,
            parent_run_id=(start or {}).get("parent_run_id"),
            agent=(start or {}).get("agent", ORCHESTRATOR),
            model=(start or {}).get("model", "unknown"),
            kind="llm",
            tool_name=None,
            started_at=(start or {}).get("started_at"),
            ended_at=_utcnow_iso(),
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            error=None,
        )

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._log_llm_error(run_id, error)

    def on_chat_model_error(self, error, *, run_id, **kwargs) -> None:
        self._log_llm_error(run_id, error)

    def _log_llm_error(self, run_id, error) -> None:
        rid = str(run_id)
        start = self._starts.pop(rid, None)
        latency_ms = (time.monotonic() - start["t0"]) * 1000 if start else None
        self.store.log_call(
            trace_id=self.trace_id,
            run_id=rid,
            parent_run_id=(start or {}).get("parent_run_id"),
            agent=(start or {}).get("agent", ORCHESTRATOR),
            model=(start or {}).get("model", "unknown"),
            kind="llm",
            tool_name=None,
            started_at=(start or {}).get("started_at"),
            ended_at=_utcnow_iso(),
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            total_tokens=0,
            error=str(error),
        )

    # -- Chain lifecycle ----------------------------------------------------

    def on_chain_start(
        self,
        serialized,
        inputs,
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ) -> None:
        rid, pid = str(run_id), str(parent_run_id) if parent_run_id else None
        self._parent_of[rid] = pid

    # -- Tool lifecycle -----------------------------------------------------
    # Every tool call is timed. Calls to DISPATCH_TOOL_NAME ("task") are
    # additionally parsed for subagent_type, which every LLM call nested
    # underneath (found by walking parent_run_id) will inherit.

    def on_tool_start(
        self,
        serialized,
        input_str,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ) -> None:
        rid, pid = str(run_id), str(parent_run_id) if parent_run_id else None
        self._parent_of[rid] = pid
        tool_name = (serialized or {}).get("name", "")

        agent_for_this_call = self._resolve_agent(
            rid, tags, metadata
        )  # who *dispatched* this tool
        if tool_name == DISPATCH_TOOL_NAME:
            subagent = self._extract_subagent_type(input_str, kwargs)
            if subagent:
                self._agent_of[rid] = (
                    subagent  # children of THIS run belong to the subagent
                )

        self._starts[rid] = {
            "t0": time.monotonic(),
            "started_at": _utcnow_iso(),
            "agent": agent_for_this_call,
            "model": None,
            "tool_name": tool_name,
            "parent_run_id": pid,
        }

    def on_tool_end(self, output, *, run_id, **kwargs) -> None:
        self._finish_tool(run_id, error=None)

    def on_tool_error(self, error, *, run_id, **kwargs) -> None:
        self._finish_tool(run_id, error=str(error))

    def _finish_tool(self, run_id, error: Optional[str]) -> None:
        rid = str(run_id)
        start = self._starts.pop(rid, None)
        latency_ms = (time.monotonic() - start["t0"]) * 1000 if start else None
        self.store.log_call(
            trace_id=self.trace_id,
            run_id=rid,
            parent_run_id=(start or {}).get("parent_run_id"),
            agent=(start or {}).get("agent", ORCHESTRATOR),
            model=None,
            kind="tool",
            tool_name=(start or {}).get("tool_name"),
            started_at=(start or {}).get("started_at"),
            ended_at=_utcnow_iso(),
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            total_tokens=0,
            error=error,
        )


# ---------------------------------------------------------------------------
# Public context-manager API
# ---------------------------------------------------------------------------


@dataclass
class Trace:
    trace_id: str
    langgraph_thread_id: Optional[str]
    label: Optional[str]
    store: TraceStore
    callback: TokenTracingCallback = field(init=False)

    def __post_init__(self) -> None:
        self.callback = TokenTracingCallback(self.trace_id, self.store)


_current_trace: ContextVar[Optional[Trace]] = ContextVar("_current_trace", default=None)

# Module-level, lazily-initialized store shared across all trace_run() calls
# in this process, keyed by resolved db_path. Avoids a connect()/close() pair
# per batch (see "IMPORTANT — connection lifetime" above). Guarded by its own
# lock since trace_run() may itself be entered from multiple threads.
_stores: dict[str, TraceStore] = {}
_stores_lock = threading.Lock()


def _get_store(db_path: Path) -> TraceStore:
    key = str(Path(db_path).resolve())
    with _stores_lock:
        store = _stores.get(key)
        if store is None:
            store = TraceStore(db_path)
            _stores[key] = store
        return store


def current_trace() -> Optional[Trace]:
    """Fetch the Trace for the currently-running `trace_run` block, if any."""
    return _current_trace.get()


@contextmanager
def trace_run(
    label: Optional[str] = None,
    langgraph_thread_id: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
):
    """
    Open a new trace for one `agent.invoke(...)` attempt. Generates a fresh,
    unique trace_id every call (including resumed attempts on the same
    LangGraph thread_id) so each attempt is independently inspectable.

    The underlying TraceStore/connection is shared across calls in this
    process (see _get_store) rather than opened and closed per call; it is
    intentionally NOT closed when this context manager exits, since another
    trace_run() may follow immediately (e.g. the next meter in main.py's
    batch loop). It closes when the process exits.
    """
    store = _get_store(db_path)
    trace_id = new_trace_id()
    store.start_run(trace_id, langgraph_thread_id, label)
    trace = Trace(
        trace_id=trace_id,
        langgraph_thread_id=langgraph_thread_id,
        label=label,
        store=store,
    )
    token = _current_trace.set(trace)
    try:
        yield trace
    finally:
        store.end_run(trace_id)
        _current_trace.reset(token)
        # No store.close() here -- the store outlives any single trace_run()
        # call; see module docstring's "IMPORTANT — connection lifetime".
