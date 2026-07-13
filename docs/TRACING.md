# Tracing & Observability

This document details how to run offline tracing analysis and inspect execution details using the trace CLI.

---

## 1. Overview

`tools/tracing.py` and `tools/trace_report.py` provide deep observability for every `agent.invoke(...)` attempt:

- Generates a fresh, unique **`trace_id`** every time (format: `2026-07-08T14-32-01Z_a3f9c2d1`).
- Logs input, output, and cached token totals **per agent** (`orchestrator`, `diacritizer`, `irab_checker_batch`, `naturalness_critic_batch`, `irab_checker`, `naturalness_critic`) automatically from the dispatch mechanism without manual tagging.
- Tracks latency per LLM call and per tool call.
- Persists everything in a local `traces.sqlite` database.

---

## 2. Trace Attribution Mechanics

Subagents are dispatched through the orchestrator’s `task` tool, whose arguments include `subagent_type` (e.g., `{'name': 'task', 'args': {..., 'subagent_type': 'irab_checker_batch'}}`).

The callback inside `tools/tracing.py` functions as follows:

1.  **Tool Call Interception**: Watches for a tool named `task` starting, reads the `subagent_type` from the arguments, and tracks it.
2.  **LLM Call Attribution**: When an LLM call occurs, the tracer traverses the parent-run-ID chain until it encounters the ancestor `task` call to identify the active subagent. If no ancestor is found, the call is attributed to the `orchestrator` itself.

Because the trace attribution logic reads the `subagent_type` parameter dynamically from the `task` tool invocation, both batched audits (`irab_checker_batch` / `naturalness_critic_batch`) and sequential fallback audits (`irab_checker` / `naturalness_critic`) are captured and categorized under their correct telemetry labels automatically, requiring no database schema changes.

### Trace IDs vs. LangGraph Thread IDs

- **`thread_id`** (e.g., `batch_01:taweel`): Stable identifier to safely resume interrupted executions from checkpoint logs.
- **`trace_id`**: Generated uniquely for each invocation. It allows developers to distinguish between attempt #1 and retry #2 of the same thread.

---

## 3. CLI Usage

Run the following commands from the project root to analyze pipeline performance:

### List Recent Executions

```bash
python -m tools.trace_report --list
```

### View Summary of the Most Recent Attempt

```bash
python -m tools.trace_report
```

### View Summary of a Specific Trace ID

```bash
python -m tools.trace_report --trace 2026-07-08T14-32-01Z_a3f9c2d1
```

The output summary table dynamically groups execution costs by agent name, letting you compare the token and latency savings of `irab_checker_batch` vs. sequential fallback runs directly:

```
agent                    kind  calls  in_tok  out_tok  cached  total_tok  avg_ms  sum_ms  errors
-----------------------  ----  -----  ------  -------  ------  ---------  ------  ------  ------
orchestrator             llm   3      31024   512      0       31536      1841    5523    0
diacritizer              llm   1      8412    1120     0       9532       2450    2450    0
irab_checker_batch       llm   1      4150    820      0       4970       1980    1980    0
naturalness_critic_batch llm   1      3110    210      0       3320       1240    1240    0
```

### View Chronological Event Timeline

Useful for identifying loop bottlenecks, long-running tools, or repetitive LLM correction queries:

```bash
python -m tools.trace_report --trace 2026-07-08T14-32-01Z_a3f9c2d1 --calls
```

### View Attempt History of a Specific Thread

```bash
python -m tools.trace_report --for-thread batch_01:taweel
```

---

## 4. LangSmith Integration

LangSmith is used alongside local SQLite tracking to support hosted-dashboard debugging and prototyping during development. Enable it by setting the following environment variables:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-langsmith-key"
export LANGCHAIN_PROJECT="arabic-diacritization-deepagent"
```
