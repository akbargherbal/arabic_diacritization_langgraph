# Configuration, Security & Checkpointing

This document details the configuration parameters of the multi-provider model loader, the security and sandboxing permission schema of the DeepAgents composite filesystem, and the local checkpointer database.

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

## 2. Sandbox Filesystem & Security

To prevent subagents from self-modifying instruction documents, editing the metrics validator, or directly tampering with raw dataset output, the orchestrator employs a declarative sandboxing schema configured in `main.py`.

### Composite Filesystem Backend

Instead of giving agents open access to the host machine, the orchestrator mounts persistent routes to a `CompositeBackend` while isolating ephemeral agent operations:

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

### Filesystem Permission Gateways

Top-down, first-match-wins permission filters are enforced globally. Unmatched paths default to `ALLOW` defensively:

- **`Deny Write/Edit`** on `/verification/**`: Protects structural and metric valuation engines.
- **`Deny Write/Edit`** on `/config/meter_tables.py`: Protects canonical meter blueprints.
- **`Deny Write/Edit/Delete`** on `/dataset/**`: Prevents direct file manipulation (writes must pass programmatically through `commit_verse_tool`).
- **`Deny Write/Edit`** on `/tests/**`: Prevents agents from altering validation tests.
- **`Deny Write/Edit/Delete`** on `/skills/**`: Prevents subagents from altering their own instruction guidelines.
- **`Allow Write`** on `/logs/**`: Allows error and disagreement exporting.
- **`Allow Read/Write/Edit`** on `/workspace/**`: Provides a sandbox for raw processing.

### Ephemeral Ledger Security

During the diacritization correction passes, newly metrically sound verses are saved incrementally to an advisory ledger file:
`/workspace/{safe_thread}/advisory_ledger.json`

Because this file resides inside the sandboxed `/workspace/` mount point, the orchestrator retains complete read and write access to log inputs [10]. To protect this shared boundary, the file is accessed under strict thread locks (`_ledger_lock`) [3] and is deleted via `read_ledger_tool(clear=True)` [3] as soon as the advisory phase for the batch is finalized.

---

## 3. local Checkpoint Engine

The orchestrator integrates local checkpointing through `checkpoints.sqlite` using LangGraph's native `SqliteSaver` checkpointer.

- **Scope**: Checkpointing is active only for the standard CLI entrypoint (`python main.py`, which calls `build_agent(use_checkpointer=True)` — the default). LangGraph Studio's orchestrator entry point (`build_studio_batch_agent`) and the single-verse playground agent both run with checkpointing disabled, letting Studio manage execution state in its own hosting layer instead.
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
