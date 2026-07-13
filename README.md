# Arabic Prosody Diacritization — LangGraph Dataset Builder

This project is a dataset generation pipeline designed to produce metrically sound, letter-faithful, and grammatically plausible diacritized Arabic poetry. It leverages Large Language Models (LLMs) as generative drafters while enforcing deterministic programmatic quality criteria through automated validation gates to assemble high-fidelity training data.

---

## 1. Project Purpose & Scope

The primary purpose of this system is to compile training corpora of classical Arabic verses. It processes normalized (undiacritized) inputs and outputs fully diacritized pairs (Sadr and Ajuz) matching specific poetic meters. The system operates on a hybrid verification model where generative AI proposes drafts, but deterministic programmatic logic holds absolute veto power over what is ultimately committed.

---

## 2. Core Objectives

- **Metrical Compliance**: Guarantee that all committed verses strictly adhere to the target prosodic meter (rhythm) as evaluated by the `pyarud` engine.
- **Skeleton Fidelity**: Ensure that the underlying consonant structure (the letter skeleton) of the input verse remains completely unchanged during the diacritization process, preventing hallucinated verse substitutions or character-level drift.
- **Security & Sanitization**: Protect downstream consumers by filtering out invalid character injections, control characters, or non-Arabic Unicode blocks.
- **Linguistic Coherence**: Identify and document potential grammatical (`إعراب`) or stylistic anomalies using LLM-based advisory judgment.

---

## 3. High-Level Architecture

The system uses an explicit LangGraph state machine, segregating generation
from verification to maintain strict quality boundaries.

```
       [ Raw Undiacritized Inputs ]
                   │
                   ▼
         ┌───────────────────┐
         │   Orchestrator    │◄─────────────────────────┐
         └───────────────────┘                          │
           │               │                            │ [Up to 3 correction passes]
           ▼               ▼                            │
     [Subagents]     [Verification] ────────────────────┘
     - Diacritizer   - Sanitizer (Security) [Deciding]
                     - Fidelity Check (Consonants) [Deciding]
                     - pyarud (Structure/Meter) [Deciding]
                           │
                           ├──► [All Deciding Gates Pass]
                           │             │
                           │             ▼
                           │    [Advisory Alignment Guard]
                           │       ├──► [Pass] ──► [Batched Advisory Audits]
                           │       │                 - irab_checker_batch
                           │       │                 - naturalness_critic_batch
                           │       │
                           │       └──► [Fail] ──► [Sequential Fallback Audits]
                           │                         - irab_checker
                           │                         - naturalness_critic
                           │
                           ▼
                 [dataset/verses.jsonl] (Needs review if flagged)
```

### The Orchestrator & Validation Core

- **Orchestration**: Manages the execution pipeline over raw verse batches, enforcing a lock-on-success rule, a maximum of 3 correction passes, and automated state-saving checkpointing.
- **Incremental Ledger Logging**: Locked verses (those passing metrical checks) are saved to an isolated local ledger (`workspace/{safe_thread}/advisory_ledger.json`) on every pass, shielding them from modification.
- **Batched Advisory & Alignment Guards**: Once a batch is fully resolved, locked verses are compiled into a single JSON array payload and audited in parallel sweeps (`irab_checker_batch` and `naturalness_critic_batch`). Programmatic validation guards verify the count, IDs, completeness, and word-level skeletons of the LLM outputs to prevent LLM ID Drift or data contamination.
- **Single-Verse Fallback**: If the programmatic alignment check fails (indicating the LLM struggled with structured batch output), the orchestrator falls back to executing sequential single-verse audits to protect system stability.
- **Four-Axis Validation**: To qualify for the final dataset, every proposed verse must clear three Deciding gates (Security, Fidelity, and Structural) and is annotated by a fourth Advisory axis (Linguistic) to determine if it should be committed with a review flag.

For a detailed walkthrough of the pipeline flow, subagents, and the mechanical case-ending reconciliation loop, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 4. Quick Start

### Installation & Setup

1. Clone the repository and navigate to the project root:

   ```bash
   cd arabic_diacritization_deepagent
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your target model provider API keys and optional custom base URLs in your environment:
   ```bash
   export MODEL_PROVIDER="deepseek"
   export DEEPSEEK_API_KEY="your-api-key"
   ```

### Running the Pipeline

To run the diacritization pipeline against the standard inputs:

```bash
python main.py
```

- **Checkpoint & Resume**: If interrupted mid-flight (e.g., via `Ctrl+C`), re-running the script will automatically resume from the last saved state in `checkpoints.sqlite` using LangGraph's native checkpointer.

---

## 5. Directory Structure

```
.
├── AGENTS.md                 # Agent charter, persistent rules, and design rationale
├── README.md                 # This file (Project Overview & Quick Start)
├── checkpoints.sqlite        # LangGraph execution-state checkpoints (CLI runs only)
├── langgraph.json            # LangGraph Studio/server configuration (graph entries, deps, env)
├── last_bundle.md            # Generated Markdown evidence bundle for the most recent run
├── main.py                   # Main pipeline entrypoint and orchestrator agent
├── requirements.txt          # Python dependencies
├── traces.sqlite             # Local telemetry database (latency, tokens, per-agent/tool logs)
├── backends/
│   └── model_provider.py     # Multi-provider model loader with retry and timeout layers
├── config/
│   └── meter_tables.py       # Ground-truth poetic meter templates & canonical patterns
├── dataset/
│   ├── inputs/               # Untrusted raw input batches (e.g., batch_01.jsonl)
│   ├── verses.jsonl          # High-fidelity committed dataset (Deciding gates passed)
│   └── verses_rejected.jsonl # Uniform diagnostic log of all rejected attempts
├── docs/                     # Detailed technical and execution documentation
│   ├── ARCHITECTURE.md       # Pipeline flows, subagents, and reconciliation details
│   ├── CONFIGURATION.md      # Sandbox permissions, model provider parameters, & checkpointing
│   └── TRACING.md            # Execution token tracing & offline reporting
├── logs/disagreements/       # JSON exports of advisory flags and unresolved failures
├── skills/                   # Guidance and rules loaded dynamically by subagents
├── subagents/                # Advisory and generative agent prompt declarations
├── tests/                    # Test suite and fixtures (protected from agent writes/edits)
├── tools/                    # Core programmatic validation and debugging utilities
└── verification/             # pyarud meter feedback engine (protected from agent writes/edits)
```

---

## 6. Documentation Index

For advanced configuration, auditing, and architectural specifics, refer to the following documents:

- **[AGENTS.md](AGENTS.md)**: Persistent memory, non-negotiable behavior constraints, and project rules.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Deep dive into the orchestrator loop, subagents, validation axes, and the case-ending swap reconciliation tool.
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**: Technical breakdown of DeepAgents security paths, local database checkpointing settings, and supported LLM backends (Anthropic, OpenAI, DeepSeek, and NVIDIA NIM).
- **[docs/TRACING.md](docs/TRACING.md)**: Guide to using `traces.sqlite` and the trace reporting CLI to audit latency, token counts, and execution runs.
