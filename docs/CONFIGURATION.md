# Configuration, Security & Checkpointing

This document details the configuration parameters of the multi-provider model
loader and the local LangGraph checkpointer database. File access is now
implemented by narrowly scoped graph helpers rather than an agent filesystem.

---

## 1. LLM Provider Configuration

The pipeline routes LLM requests through `backends/model_provider.py`, which integrates directly with LangChain's `init_chat_model` and handles transient network failures automatically.

### Supported Providers & Models

| Provider (`MODEL_PROVIDER`) | Default Model (`MODEL_NAME`) | Required Env Vars                                    |
| :-------------------------- | :--------------------------- | :--------------------------------------------------- |
| `deepseek` (Default)        | `deepseek-chat`              | `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` (optional)   |
| `anthropic`                 | `claude-sonnet-4-5`          | `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` (optional) |
| `openai`                    | `gpt-4.1`                    | `OPENAI_API_KEY`, `OPENAI_BASE_URL` (optional)       |
| `nvidia`                    | `z-ai/glm-5.2`               | `NVIDIA_API_KEY`, `NVIDIA_MODEL` (optional)          |

To run the pipeline with a custom model configuration:

```bash
export MODEL_PROVIDER="anthropic"
export MODEL_NAME="claude-3-5-sonnet-latest"
export ANTHROPIC_API_KEY="your-api-key"
python main.py
```

### Execution Parameters

- **Timeout**: Requests default to a **300-second (5-minute)** timeout to support complex reasoning models. Override this globally by setting the `MODEL_TIMEOUT_SECONDS` environment variable.
- **Retry Policy**: Implements bounded exponential backoff with jitter. On encountering typical transient errors (rate limits, overloaded servers, HTTP 5xx, or timeouts), it will back off up to **5 times** before failing.
- **NVIDIA NIM Specifics**: Direct implementation via `ChatNVIDIA`. Includes custom patches to prevent client gateway failures from being masked as unreadable JSON decoding errors.

---

## 2. Subagent Security Model

To prevent subagents from self-modifying instruction documents, editing the metrics validator, or directly tampering with raw dataset output, protection is enforced by **tool grants**, not by a runtime filesystem sandbox. Under the current `langgraph_pipeline.py` architecture, each subagent's model call is bound to an explicit, minimal tool list via `model.bind_tools([...])`; a subagent simply has no path to a filesystem operation it wasn't given a tool for. Concretely:

- **Diacritizer**: bound to `meter_schema_tool` (read-only lookup against `config/meter_tables.py`) and `read_workspace_file` (hard-restricted at the code level to paths under `workspace/`, and read-only — see `read_workspace_file`'s own `.relative_to(workspace_root)` guard in `langgraph_pipeline.py`). It has no write tool of any kind.
- **`irab_checker` / `naturalness_critic`** (batch and single-verse variants): bound to no tools at all — they receive a JSON payload, return a JSON verdict, and cannot touch the filesystem.
- **Dataset writes**: exclusively through `commit_verse_tool`, which re-runs the pyarud check itself before writing (`AGENTS.md` rule 4). No subagent calls a generic `write_file` against `dataset/`.
- **`verification/`, `config/meter_tables.py`, `tests/`**: no subagent is ever bound to a tool capable of writing to these paths, so the "deny write" guarantee here is structural (no tool exists) rather than a permission check that could theoretically be bypassed or misconfigured.

### Historical note: the retired DeepAgents sandbox

An earlier version of this project (built on the `deepagents` framework, since retired — see `PHASED_PLAN_v4_Diacritizer_Refactor.md`) implemented this protection differently, via a declarative `CompositeBackend` with per-path permission gateways:

```python
BACKEND = CompositeBackend(
    default=StateBackend(),  # In-memory, starts empty on every invoke
    routes={
        "/workspace/": FilesystemBackend(root_dir="workspace", virtual_mode=True),
        "/dataset/":   FilesystemBackend(root_dir="dataset", virtual_mode=True),
        "/logs/":      FilesystemBackend(root_dir="logs", virtual_mode=True),
        "/verification/": FilesystemBackend(root_dir="verification", virtual_mode=True),
        "/config/":    FilesystemBackend(root_dir="config", virtual_mode=True),
        "/tests/":     FilesystemBackend(root_dir="tests", virtual_mode=True),
        "/skills/":    FilesystemBackend(root_dir="skills", virtual_mode=True),
    },
)
```

with top-down, first-match-wins rules such as `Deny Write/Edit` on `/verification/**` and `/config/meter_tables.py`, `Deny Write/Edit/Delete` on `/dataset/**` and `/skills/**`, `Deny Write/Edit` on `/tests/**`, and `Allow` on `/logs/**` and `/workspace/**`. **This code no longer exists in the current pipeline** — it's kept here only because it's the direct ancestor/rationale of `AGENTS.md` rule 3 ("verification code is read-only to every agent"), which the tool-grant model above now enforces by a different, simpler mechanism.

### Ephemeral Ledger Security

During the diacritization correction passes, newly metrically sound verses are saved incrementally to an advisory ledger file:
`/workspace/{safe_thread}/advisory_ledger.json`

Because this file resides inside the sandboxed `/workspace/` mount point, the orchestrator retains complete read and write access to log inputs [10]. To protect this shared boundary, the file is accessed under strict thread locks (`_ledger_lock`) [3] and is deleted via `read_ledger_tool(clear=True)` [3] as soon as the advisory phase for the batch is finalized.

---

## 3. local Checkpoint Engine

The orchestrator integrates local checkpointing through `checkpoints.sqlite` using LangGraph's native `SqliteSaver` checkpointer.

- **Scope**: Checkpointing is active only for the standard CLI entrypoint (`python main.py`, which calls `build_langgraph_pipeline(use_checkpointer=True)` — the default). LangGraph Studio's graph factory (`build_studio_graph`) runs with checkpointing disabled, letting Studio manage execution state in its own hosting layer instead.
- **Thread Stability**: Executions use a stable thread identifier mapped to the input batch name (e.g., `batch_01:taweel`). Rerunning the script after an interruption resumes from the exact state saved in the checkpoint rather than restarting the batch.
- **Diagnostic Integrity Checks**: On startup, `main.py` executes a database validation scan:
  ```sql
  PRAGMA integrity_check;
  ```
  If any corruption is found, warnings are surfaced to prevent faulty checkpoint states from compromising execution integrity.
- **Performance Pragmas**:
  ```sql
  PRAGMA busy_timeout = 30000;
  PRAGMA synchronous = NORMAL;
  ```
