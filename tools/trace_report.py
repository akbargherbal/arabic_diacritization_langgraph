"""
tools/trace_report.py

CLI to inspect traces recorded by tools/tracing.py.

Examples:

    # list recent runs (most recent first)
    python -m tools.trace_report --list

    # per-agent token/latency summary for the most recent run
    python -m tools.trace_report

    # same, for a specific trace
    python -m tools.trace_report --trace 2026-07-08T14-32-01Z_a3f9c2d1

    # raw chronological call timeline (great for spotting where a pass
    # looped / burned tokens without producing a usable result)
    python -m tools.trace_report --trace 2026-07-08T14-32-01Z_a3f9c2d1 --calls

    # all trace_ids that share a LangGraph thread_id (e.g. a batch that was
    # interrupted and resumed 3 times)
    python -m tools.trace_report --for-thread batch_01:taweel
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from tools.tracing import DEFAULT_DB_PATH

# Mirrors tools/tracing.py's BUSY_TIMEOUT_MS -- this CLI opens its own,
# separate connection to the same traces.sqlite file that the live agent
# run may be writing to concurrently. Without a timeout, running this
# report mid-batch can raise sqlite3.OperationalError("database is locked")
# instead of just waiting briefly for the writer to finish its commit.
BUSY_TIMEOUT_MS = 30000


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise SystemExit(f"No trace database found at {db_path}. Run something with trace_run() first.")
    conn = sqlite3.connect(db_path, timeout=BUSY_TIMEOUT_MS / 1000.0)
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS};")
    conn.row_factory = sqlite3.Row
    return conn


def _latest_trace_id(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT trace_id FROM runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        raise SystemExit("No runs recorded yet.")
    return row["trace_id"]


def _fmt_table(headers, rows) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
             "  ".join("-" * w for w in widths)]
    for row in rows:
        lines.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


def list_runs(conn: sqlite3.Connection, limit: int = 20) -> None:
    rows = conn.execute(
        """
        SELECT r.trace_id, r.langgraph_thread_id, r.label, r.started_at, r.ended_at,
               COALESCE(SUM(c.total_tokens), 0) AS total_tokens,
               COUNT(c.id) AS calls
        FROM runs r
        LEFT JOIN calls c ON c.trace_id = r.trace_id
        GROUP BY r.trace_id
        ORDER BY r.started_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    headers = ["trace_id", "thread_id", "label", "started_at", "ended_at", "total_tok", "calls"]
    table_rows = [
        (r["trace_id"], r["langgraph_thread_id"] or "-", r["label"] or "-",
         r["started_at"], r["ended_at"] or "(running)", r["total_tokens"], r["calls"])
        for r in rows
    ]
    print(_fmt_table(headers, table_rows))


def list_for_thread(conn: sqlite3.Connection, thread_id: str) -> None:
    rows = conn.execute(
        """
        SELECT r.trace_id, r.label, r.started_at, r.ended_at,
               COALESCE(SUM(c.total_tokens), 0) AS total_tokens,
               COUNT(c.id) AS calls
        FROM runs r
        LEFT JOIN calls c ON c.trace_id = r.trace_id
        WHERE r.langgraph_thread_id = ?
        GROUP BY r.trace_id
        ORDER BY r.started_at ASC
        """,
        (thread_id,),
    ).fetchall()
    if not rows:
        print(f"(no trace attempts recorded for langgraph thread_id={thread_id!r})")
        return
    headers = ["trace_id", "label", "started_at", "ended_at", "total_tok", "calls"]
    table_rows = [
        (r["trace_id"], r["label"] or "-", r["started_at"], r["ended_at"] or "(running)",
         r["total_tokens"], r["calls"])
        for r in rows
    ]
    print(f"attempts for langgraph thread_id={thread_id!r} (in chronological order — "
          f"more than one row usually means an interrupted/resumed run):")
    print(_fmt_table(headers, table_rows))


def summarize(conn: sqlite3.Connection, trace_id: str) -> None:
    run = conn.execute("SELECT * FROM runs WHERE trace_id = ?", (trace_id,)).fetchone()
    if not run:
        raise SystemExit(f"No run found with trace_id={trace_id!r}")

    print(f"trace_id   : {trace_id}")
    print(f"thread_id  : {run['langgraph_thread_id'] or '-'}")
    print(f"label      : {run['label'] or '-'}")
    print(f"started    : {run['started_at']}")
    print(f"ended      : {run['ended_at'] or '(still running / not closed)'}")
    print()

    rows = conn.execute(
        """
        SELECT agent,
               kind,
               COUNT(*) AS calls,
               COALESCE(SUM(input_tokens), 0) AS input_tokens,
               COALESCE(SUM(output_tokens), 0) AS output_tokens,
               COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
               COALESCE(SUM(total_tokens), 0) AS total_tokens,
               COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
               COALESCE(SUM(latency_ms), 0) AS sum_latency_ms,
               SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors
        FROM calls
        WHERE trace_id = ?
        GROUP BY agent, kind
        ORDER BY total_tokens DESC, sum_latency_ms DESC
        """,
        (trace_id,),
    ).fetchall()

    if not rows:
        print("(no calls recorded for this trace)")
        return

    headers = ["agent", "kind", "calls", "in_tok", "out_tok", "cached", "total_tok",
               "avg_ms", "sum_ms", "errors"]
    table_rows = [
        (r["agent"], r["kind"], r["calls"], r["input_tokens"], r["output_tokens"],
         r["cached_tokens"], r["total_tokens"], round(r["avg_latency_ms"]),
         round(r["sum_latency_ms"]), r["errors"])
        for r in rows
    ]
    print(_fmt_table(headers, table_rows))

    totals = conn.execute(
        """
        SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
               COALESCE(SUM(total_tokens),0), COUNT(*)
        FROM calls WHERE trace_id = ? AND kind = 'llm'
        """,
        (trace_id,),
    ).fetchone()
    print()
    print(f"TOTAL (llm calls only): {totals[3]} calls, "
          f"{totals[0]} input tokens, {totals[1]} output tokens, {totals[2]} total tokens")


def dump_calls(conn: sqlite3.Connection, trace_id: str) -> None:
    rows = conn.execute(
        """
        SELECT started_at, agent, kind, tool_name, model, input_tokens, output_tokens,
               total_tokens, round(latency_ms) AS ms, error
        FROM calls
        WHERE trace_id = ?
        ORDER BY started_at ASC
        """,
        (trace_id,),
    ).fetchall()
    if not rows:
        print("(no calls recorded for this trace)")
        return
    headers = ["started_at", "agent", "kind", "tool/model", "in_tok", "out_tok", "total_tok", "ms", "error"]
    table_rows = [
        (r["started_at"], r["agent"], r["kind"], r["tool_name"] or r["model"] or "-",
         r["input_tokens"], r["output_tokens"], r["total_tokens"], r["ms"], r["error"] or "")
        for r in rows
    ]
    print(_fmt_table(headers, table_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="path to traces.sqlite")
    parser.add_argument("--trace", type=str, default=None, help="trace_id to inspect (default: most recent)")
    parser.add_argument("--for-thread", type=str, default=None,
                         help="list all trace_ids recorded under a given LangGraph thread_id")
    parser.add_argument("--list", action="store_true", help="list recent runs and exit")
    parser.add_argument("--calls", action="store_true", help="dump raw chronological call timeline")
    args = parser.parse_args()

    conn = _connect(args.db)
    try:
        if args.list:
            list_runs(conn)
            return
        if args.for_thread:
            list_for_thread(conn, args.for_thread)
            return
        trace_id = args.trace or _latest_trace_id(conn)
        if args.calls:
            dump_calls(conn, trace_id)
        else:
            summarize(conn, trace_id)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
