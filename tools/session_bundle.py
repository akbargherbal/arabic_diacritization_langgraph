"""
tools/session_bundle.py

Assembles a single Markdown "evidence bundle" for one traced run, meant to
be pasted (after docs/AGENT_DEBUG_PERSONA.md) into a session with a
behavior-debugging LLM. Deliberately does NOT touch any source code —
only trace data, per-pass workspace reports, and disagreement logs.

Usage:

    python -m tools.session_bundle --trace 2026-07-08T14-32-01Z_a3f9c2d1
    python -m tools.session_bundle --trace 2026-07-08T14-32-01Z_a3f9c2d1 --out bundle.md

If --out is omitted, prints to stdout.
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
from contextlib import redirect_stdout
from pathlib import Path

from tools.tracing import DEFAULT_DB_PATH
from tools import trace_report

WORKSPACE_DIR = Path("workspace")
DISAGREEMENTS_DIR = Path("logs/disagreements")


def _capture(fn, *args, **kwargs) -> str:
    """Run a trace_report print-based function and capture its stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n```\n{body.strip()}\n```\n"


def build_bundle(trace_id: str, db_path: Path) -> str:
    conn = trace_report._connect(db_path)
    try:
        parts = [f"# Session Bundle — trace {trace_id}\n"]

        parts.append(_section("Trace summary", _capture(trace_report.summarize, conn, trace_id)))
        parts.append(_section("Call timeline", _capture(trace_report.dump_calls, conn, trace_id)))

        run = conn.execute("SELECT langgraph_thread_id FROM runs WHERE trace_id = ?", (trace_id,)).fetchone()
        thread_id = run["langgraph_thread_id"] if run else None
        if thread_id:
            parts.append(
                _section(
                    f"Attempt history for thread_id={thread_id}",
                    _capture(trace_report.list_for_thread, conn, thread_id),
                )
            )
    finally:
        conn.close()

    # Per-pass workspace reports (these get overwritten each run, so this
    # only makes sense to bundle immediately after the run you're
    # inspecting -- copy them out first if you need to keep several
    # runs' reports side by side).
    if WORKSPACE_DIR.exists():
        for report_file in sorted(WORKSPACE_DIR.glob("pass_*_report.json")):
            try:
                data = json.loads(report_file.read_text(encoding="utf-8"))
                body = json.dumps(data, ensure_ascii=False, indent=2)
            except (json.JSONDecodeError, OSError) as exc:
                body = f"(could not read {report_file}: {exc})"
            parts.append(_section(f"Workspace report: {report_file.name}", body))

    # Disagreement logs
    if DISAGREEMENTS_DIR.exists():
        disagreement_files = sorted(DISAGREEMENTS_DIR.glob("*.json"))
        if disagreement_files:
            for log_file in disagreement_files:
                try:
                    body = log_file.read_text(encoding="utf-8")
                except OSError as exc:
                    body = f"(could not read {log_file}: {exc})"
                parts.append(_section(f"Disagreement log: {log_file.name}", body))
        else:
            parts.append("## Disagreement logs\n\n(none present)\n")

    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="trace_id to bundle")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--out", type=Path, default=None, help="write to this file instead of stdout")
    args = parser.parse_args()

    bundle = build_bundle(args.trace, args.db)
    if args.out:
        args.out.write_text(bundle, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(bundle)


if __name__ == "__main__":
    main()
