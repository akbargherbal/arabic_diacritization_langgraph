"""
main.py
========
Entrypoint. Constructs the orchestrator agent. Domain logic (real pyarud
implementation, real irab rules, real batch-invocation loop reading your
input format) is intentionally left for you to wire in — see README.md.

NOTE: deepagents' exact create_deep_agent signature, the permissions= rule
schema, and interrupt_on condition syntax move fast on this framework.
Confirm this against docs.langchain.com/oss/python/deepagents before
relying on it in production. What's below reflects the framework's
documented shape at design time, not a guarantee of the current API.

CHANGE (A4): agent/checkpointer construction used to happen at MODULE
IMPORT time, which meant merely importing this module (from a test, a
notebook, a future second entrypoint) opened a real sqlite3 connection to
checkpoints.sqlite and ran the integrity_check query as a side effect.
That construction now lives in build_agent(), called from main() under the
__main__ guard, so importing this module has no side effects.

CHANGE (A4): the resume path (agent.invoke(None, ...)) previously shared
the exact same broad except-Exception branch as a fresh invoke, printing
an identical generic message either way. It now has its own except branch
that names the corruption possibility explicitly and suggests a concrete
next step (fresh thread_id) rather than leaving the operator to guess
whether a resume failure is an ordinary run-time error or a checkpoint
integrity problem.
"""

import json
import pathlib
import os
import sqlite3
import sys

from deepagents import create_deep_agent, FilesystemPermission
from deepagents.backends import CompositeBackend, StateBackend, FilesystemBackend
from backends.model_provider import get_model
from langgraph.checkpoint.sqlite import SqliteSaver

from tools.prosody_tools import (
    verify_batch_tool,
    meter_schema_tool,
    verify_single_verse_tool,
)
from tools.dataset_tools import commit_verse_tool, log_unresolved_tool
from tools.sanitization_tools import sanitize_output_tool
from tools.reconciliation_tools import reconcile_case_ending_tool
from tools.context_tools import summarize_correction_report_tool
from tools.tracing import trace_run

# --- Task 1, 2, and 3 Tool and Guard Imports ---
from tools.advisory_ledger import (
    record_locked_verse_tool,
    read_ledger_tool,
)
from tools.advisory_batch import build_batched_advisory_payload_tool
from tools.alignment_guards import (
    validate_advisory_batch_alignment,
    validate_naturalness_batch_alignment,
)

from subagents.diacritizer import build_diacritizer_subagent
from subagents.irab_checker_agent import (
    build_irab_checker_subagent,
    build_irab_checker_batch_subagent,
)
from subagents.naturalness_critic import (
    build_naturalness_critic_subagent,
    build_naturalness_critic_batch_subagent,
)

# ===========================================================================
# FIX (Phase 1, Task 1.3 — see docs/FINDINGS_general_purpose_dispatch.md):
# Without this, create_deep_agent() silently auto-adds a "general-purpose"
# subagent that inherits the orchestrator's FULL tool set (including
# commit_verse_tool) plus TodoListMiddleware/FilesystemMiddleware/
# SkillsMiddleware -- deepagents==0.6.12's own documented default whenever
# no subagent named "general-purpose" is supplied and no harness profile
# disables it. Neither create_deep_agent() call site below ever requested
# this, and it consumed 66% of the reference 3-verse batch's wall-clock
# time (trace 2026-07-12T11-16-29Z_26d00760). Registering this profile
# BEFORE either create_deep_agent(...) call runs suppresses it. Keyed to
# MODEL_PROVIDER because deepagents' harness-profile lookup matches the
# resolved model's provider (see backends/model_provider.py's own
# MODEL_PROVIDER resolution) -- confirmed empirically against the real
# ChatDeepSeek model this project builds by default.
# ===========================================================================
from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfileConfig,
    register_harness_profile,
)

register_harness_profile(
    os.environ.get("MODEL_PROVIDER", "deepseek"),
    HarnessProfileConfig(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
    ),
)

# ===========================================================================
# TECH LEAD DIAGNOSTIC MONKEY-PATCH (Functools Wraps Version)
# ===========================================================================
try:
    import functools
    import deepagents.middleware.subagents as ms
    from langchain_core.runnables.config import var_child_runnable_config

    original_build_task_tool = ms._build_task_tool

    def patched_build_task_tool(*args, **kwargs):
        # 1. Build the tool using the original factory function
        tool = original_build_task_tool(*args, **kwargs)

        # 2. Intercept the synchronous tool runner while preserving signature
        original_func = tool.func

        @functools.wraps(original_func)
        def wrapped_func(*w_args, **w_kwargs):
            subagent_config = {"configurable": {"ls_agent_type": "subagent"}}

            # Print the exact diagnostic values requested by the tech lead
            print("\n" + "=" * 70)
            print(
                f"[DIAGNOSTIC] callbacks in subagent_config: {subagent_config.get('callbacks')}"
            )
            print(
                f"[DIAGNOSTIC] var_child_runnable_config at this point: {var_child_runnable_config.get()}"
            )
            print("=" * 70 + "\n")

            return original_func(*w_args, **w_kwargs)

        tool.func = wrapped_func

        # 3. Intercept the asynchronous tool runner while preserving signature
        if hasattr(tool, "coroutine") and tool.coroutine:
            original_coroutine = tool.coroutine

            @functools.wraps(original_coroutine)
            async def wrapped_coroutine(*w_args, **w_kwargs):
                subagent_config = {"configurable": {"ls_agent_type": "subagent"}}
                print("\n" + "=" * 70)
                print(
                    f"[DIAGNOSTIC] callbacks in subagent_config: {subagent_config.get('callbacks')}"
                )
                print(
                    f"[DIAGNOSTIC] var_child_runnable_config at this point: {var_child_runnable_config.get()}"
                )
                print("=" * 70 + "\n")
                return await original_coroutine(*w_args, **w_kwargs)

            tool.coroutine = wrapped_coroutine

        return tool

    ms._build_task_tool = patched_build_task_tool
    print(
        "[DIAGNOSTIC] Applied signature-preserving _build_task_tool patch successfully."
    )
except Exception as e:
    print("[DIAGNOSTIC] Failed to apply monkey-patch:", e)
# ===========================================================================

MAX_CORRECTION_PASSES = (
    3  # hard cap — do not raise without re-reading the design rationale
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

PERMISSIONS = [
    # Order matters: first-match-wins, top-down. Unmatched paths default to
    # ALLOW in deepagents — every deny below is written defensively because
    # of that default, not because deny-by-default can be assumed.
    FilesystemPermission(
        paths=["/verification/**"], operations=["write", "edit"], mode="deny"
    ),
    FilesystemPermission(
        paths=["/config/meter_tables.py"], operations=["write", "edit"], mode="deny"
    ),
    FilesystemPermission(
        paths=["/dataset/**"], operations=["write", "edit", "delete"], mode="deny"
    ),
    FilesystemPermission(
        paths=["/tests/**"], operations=["write", "edit"], mode="deny"
    ),
    # Deny agents from altering their own instruction/guideline documents
    FilesystemPermission(
        paths=["/skills/**"], operations=["write", "edit", "delete"], mode="deny"
    ),
    FilesystemPermission(paths=["/logs/**"], operations=["write"], mode="allow"),
    FilesystemPermission(
        paths=["/workspace/**"], operations=["read", "write", "edit"], mode="allow"
    ),
]

# ---------------------------------------------------------------------------
# Backend wiring
# ---------------------------------------------------------------------------
BACKEND = CompositeBackend(
    default=StateBackend(),
    routes={
        "/workspace/": FilesystemBackend(
            root_dir=str(PROJECT_ROOT / "workspace"), virtual_mode=True
        ),
        "/dataset/": FilesystemBackend(
            root_dir=str(PROJECT_ROOT / "dataset"), virtual_mode=True
        ),
        "/logs/": FilesystemBackend(
            root_dir=str(PROJECT_ROOT / "logs"), virtual_mode=True
        ),
        "/verification/": FilesystemBackend(
            root_dir=str(PROJECT_ROOT / "verification"), virtual_mode=True
        ),
        "/config/": FilesystemBackend(
            root_dir=str(PROJECT_ROOT / "config"), virtual_mode=True
        ),
        "/tests/": FilesystemBackend(
            root_dir=str(PROJECT_ROOT / "tests"), virtual_mode=True
        ),
        "/skills/": FilesystemBackend(
            root_dir=str(PROJECT_ROOT / "skills"), virtual_mode=True
        ),
    },
)

ORCHESTRATOR_SYSTEM_PROMPT = f"""
You coordinate diacritization of a batch of normalized Arabic verses against
a target meter, producing dataset records for a downstream training set.

The batch of verses and the target meter name are given to you directly in
the user message as JSON — read them from there. Do not go looking for them
on the filesystem; if a batch is already in your context, use it as-is.

You do NOT diacritize verses yourself — delegate via
task(subagent_type="diacritizer", ...). This exact string,
"diacritizer", is the only valid subagent_type for drafting/correcting
diacritics; never invent, abbreviate, or substitute a different
subagent_type value for this dispatch. You DO call verify_batch_tool
yourself between passes; never let a subagent call it. The entity
deciding pass/fail must not be the entity trying to pass.

Locking rule: once a verse's pyarud scan is sound (from verify_batch_tool),
mark it locked and never resubmit it to the diacritizer for regeneration,
even in later passes or later batches.

Pass budget: maximum {MAX_CORRECTION_PASSES} correction passes per batch.
Verses still broken after that are logged via log_unresolved_tool and
EXCLUDED from the dataset — do not force a further pass, do not auto-accept.

--- Structural Incompatibility Rule ---
NEVER call verify_batch_tool against the raw, undiacritized input verses
straight from the user message. pyarud's mora/weight analysis requires
actual short-vowel marks (fatha/damma/kasra/sukun) to compute a meaningful
pattern -- verification/arabic_prosody_feedback.py's own docstring states
its inputs must be "fully-diacritized Arabic strings". Run against bare
consonant skeletons, it will misreport morae counts on every verse and
false-flag verses as "structurally_incompatible" before they have ever
been diacritized -- that is a false incompatibility signal, not a real one.

So: Pass 1 begins by dispatching ALL initial verses via
task(subagent_type="diacritizer", ...) for their first diacritization
attempt -- there is no pre-diacritization filtering step. Only AFTER that
first draft do you call
verify_batch_tool (pass_number=1) on the diacritized text; this is the
first point at which "structurally_incompatible" is ever evaluated, and it
is evaluated against real diacritized output, not raw input. From this
pass onward, whenever verify_batch_tool returns any verse_id in
"structurally_incompatible", do NOT send it to the diacritizer subagent in
any further pass (diacritics cannot modify base consonant structures or
resolve syllable count deficits/excesses that persist even once fully
diacritized). Instead, immediately log it via log_unresolved_tool with
stage="structurally_incompatible" and a reason describing the mora/foot
mismatch you observed (e.g. "sadr hemistich N morae short even after
diacritization"), and exclude it from future passes. Only the remaining,
compatible broken verses continue into pass 2/3.

A verse that instead reaches the end of pass {MAX_CORRECTION_PASSES} still
in "broken" (not "structurally_incompatible") should be logged via
log_unresolved_tool with its default stage="unresolved_max_passes" -- do
not use that stage for a verse that was actually excluded earlier as
structurally incompatible; the two must stay distinguishable in the logs.

--- Context discipline across correction passes ---

verify_batch_tool's correction_report is a complete, per-foot diagnostic
text. Do not let it accumulate unpruned in your own reasoning across
passes — that is the single largest token-cost driver in this system.
Concretely:

  1. verify_batch_tool now writes the full correction report to disk itself
     and returns a "report_path" instead of the report text -- you never
     receive or need to quote the full report. Pass an incrementing
     pass_number (starting at 1) on every call. Still call
     summarize_correction_report_tool on the returned poem_result_json for
     your own terse "verse_id: score" bookkeeping across passes.
  2. When dispatching via task(subagent_type="diacritizer", ...) for a
     given pass, hand it ONLY that pass's "report_path" (the diacritizer
     will read it itself)
     plus the verse's original input text. Never embed report text
     directly in the dispatch, and never re-include a verse's prior-pass
     rejected draft text -- it already failed verify_batch_tool and
     carries no diagnostic value the fresh report doesn't already contain
     better.
  3. Locked verses need no report at all in any pass — they are not
     resubmitted (see Locking rule above), so nothing about them belongs
     in a dispatch to the diacritizer.

--- Pass execution discipline ---

Each correction pass is a single, strictly bounded turn — not an open-ended
back-and-forth with the diacritizer:

  1. Enforce strict sequential pass advancing. Do not begin pass N+1 work
     (dispatching, verifying, or logging) until pass N has been fully
     closed out via verify_batch_tool and the resulting locked/broken/
     structurally_incompatible verses have been handled per the rules
     above.
  2. For a given pass, batch ALL diacritizer dispatches for that pass's
     broken/unlocked verses into a single parallel turn — issue every
     needed task(subagent_type="diacritizer", ...) call together, not
     staggered one at a time across multiple turns.
  3. NEVER dispatch a second task(subagent_type="diacritizer", ...) call
     for the same verse_id within the same pass. One dispatch per verse
     per pass, full stop — if a draft comes back incomplete or unusable,
     that is what the NEXT pass (and its own fresh correction_report) is
     for, not a same-pass re-dispatch.
  4. As soon as that turn's diacritizer subagent tasks have all returned,
     call verify_batch_tool immediately. Do not loop back to the
     diacritizer again within the same pass for any reason — no
     intra-pass retries, no "just one more attempt" before verifying.

--- Ledger recording and batched advisory triggers ---

  1. After each verify_batch_tool pass, for every verse_id in that pass's returned "locked" array, call record_locked_verse_tool with that verse's current sadr/ajuz/meter.
  2. Once the batch reaches full resolution (every verse is either locked across any pass, or unresolved after pass 3 — i.e. no verse remains pending), call build_batched_advisory_payload_tool.
  3. If it returns payload=None, skip advisory entirely and proceed to final reporting/exit.
  4. Otherwise, dispatch the returned payload string verbatim, unmodified, as the description argument to task(subagent_type="irab_checker_batch", ...) and separately to task(subagent_type="naturalness_critic_batch", ...). Do not re-transcribe or summarize the verse text by hand for these calls.
  5. Parse each subagent's response as a JSON array of verdicts. Call validate_advisory_batch_alignment on the irab_checker verdicts and validate_naturalness_batch_alignment on the naturalness_critic verdicts.
  6. If either guard returns False: fall back to the existing single-verse dispatch path for every verse in this batch — call task(subagent_type="irab_checker", ...) and task(subagent_type="naturalness_critic", ...) per-verse, exactly these two strings, as described below. Log which guard failed and why.
  7. If both guards pass: proceed per-verse through the existing reconciliation/precedence/commit logic described below, using each verse's individual verdict from the batch array in place of what a single-verse call would have returned.
  8. After the batch's advisory round completes (batched or fallback), call read_ledger_tool(clear=True) to reset the ledger for the next input batch.

--- Handling a pyarud/إعراب disagreement (two-step, in this order) ---

If single-verse fallback is active, or once batched verdicts are validated, process each verse's individual verdict. If a verdict has flag=true with fix_type="case_ending_swap":
  1. This is NOT automatically poetic license — attempt reconciliation
     FIRST. Call reconcile_case_ending_tool with the word_index and
     target_harakah it proposed (target_harakah may now be a tanwin mark —
     fathatayn/dammatayn/kasratayn — in addition to the plain short
     vowels). This performs a mechanical vowel/tanwin swap, which cannot
     change the metrical (U/_) pattern in the
     underlying model.
  2. Re-run verify_single_verse_tool on the reconciled text.
  3. If it still passes: this was a genuine grammar fix with no metrical
     cost. Commit the reconciled text via commit_verse_tool with
     reconciled=True and original_sadr/original_ajuz set to the pre-swap
     text. Do NOT mark needs_review — this is resolved, not an open
     disagreement.
  4. If the reconciled text FAILS pyarud (rare, but the underlying
     converter has known quirks — see verification/arabic_prosody_feedback.py's
     docstring): the swap was not actually free in this instance. Fall
     back to the precedence rule below, using the ORIGINAL (pre-swap) text.

If irab_checker returns flag=true with fix_type="structural", or
reconciliation was attempted and failed per step 4 above:
  Precedence rule: pyarud decides. Commit the ORIGINAL pyarud-verified
  text via commit_verse_tool with irab_flag=True, needs_review will be set
  automatically. This is presumed poetic license (الضرورات الشعرية) unless
  a human reviewer says otherwise later — log it, don't block on it.

naturalness_critic flags follow the same non-blocking pattern as the
"structural" branch above — advisory only, feeds needs_review, never
triggers reconciliation (there is no mechanical fix for "reads unnatural").

commit_verse_tool re-verifies pyarud and sanitization itself before
writing — treat its "committed": false response as authoritative, not a
bug to route around. A "duplicate": true response means this exact verse
(or an identical text under a different verse_id) was already committed —
treat that as already handled, not as a failure to retry.

When calling commit_verse_tool with irab_flag=True, also pass the optional
parameters: fix_type, word_index, and target_harakah using the exact values
the irab_checker subagent returned for that verse, and populate notes with
the irab_checker's own "note" field verbatim (or the naturalness_critic's
"note" field, if the naturalness_critic was the one that flagged it) --
never leave notes empty when an advisory flag is present.
"""


def build_agent(use_checkpointer=True):
    """Construct the checkpointer and the deep agent. Called from main()
    under the __main__ guard so that merely importing this module (e.g.
    from a test) has no side effects -- no sqlite connection is opened and
    no model client is constructed at import time (A4)."""

    model = get_model()

    # -----------------------------------------------------------------
    # Checkpointing (Optional inside LangGraph Studio to avoid DB locks)
    # -----------------------------------------------------------------
    checkpointer = None
    checkpoint_conn = None
    checkpoint_db_path = None

    if use_checkpointer:
        checkpoint_db_path = PROJECT_ROOT / "checkpoints.sqlite"
        checkpoint_conn = sqlite3.connect(
            str(checkpoint_db_path), check_same_thread=False
        )

        # Optimization Pragma Settings (PRAGMA setup runs prior to SqliteSaver)
        checkpoint_conn.execute("PRAGMA busy_timeout = 30000")
        checkpoint_conn.execute("PRAGMA synchronous = NORMAL")

        # Startup Diagnostics: database file integrity check
        try:
            cursor = checkpoint_conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            if integrity_result != "ok":
                print(
                    f"[!] Warning: Checkpoint database corrupted. PRAGMA integrity_check returned: {integrity_result}"
                )
        except Exception as exc:
            print(f"[-] Pre-run checkpoint diagnostic failed: {exc}")

        checkpointer = SqliteSaver(checkpoint_conn)

    # -----------------------------------------------------------------
    # Subagent construction (CompiledSubAgent, see FIX_PLAN.md Divergence A)
    # -----------------------------------------------------------------
    # Built here rather than at module level because each needs the real
    # `model` and `BACKEND` bound in. Compiling as a CompiledSubAgent
    # ("runnable" key) bypasses create_deep_agent's default
    # TodoListMiddleware/FilesystemMiddleware prepend -- which is why the
    # "skills" dict-key assignment that used to live here is gone: a
    # CompiledSubAgent bypasses the framework's SkillsMiddleware injection
    # too, so diacritizer's and irab_checker's skill content is folded
    # directly into their system prompts at compile time instead (see
    # subagents/diacritizer.py and subagents/irab_checker_agent.py).
    diacritizer_model = get_model(max_completion_tokens=2048, max_tokens=2048, extra_body={"thinking_budget": 0})
    DIACRITIZER_SUBAGENT = build_diacritizer_subagent(diacritizer_model, BACKEND)
    IRAB_SUBAGENT = build_irab_checker_subagent(model)
    NATURALNESS_CRITIC_SUBAGENT = build_naturalness_critic_subagent(model)
    IRAB_BATCH_SUBAGENT = build_irab_checker_batch_subagent(model)
    NATURALNESS_CRITIC_BATCH_SUBAGENT = build_naturalness_critic_batch_subagent(model)

    agent = create_deep_agent(
        model=model,
        tools=[
            verify_batch_tool,
            meter_schema_tool,
            verify_single_verse_tool,
            commit_verse_tool,
            log_unresolved_tool,
            sanitize_output_tool,
            reconcile_case_ending_tool,
            summarize_correction_report_tool,
            record_locked_verse_tool,
            read_ledger_tool,
            build_batched_advisory_payload_tool,
        ],
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        permissions=PERMISSIONS,
        backend=BACKEND,
        subagents=[
            DIACRITIZER_SUBAGENT,
            IRAB_SUBAGENT,
            NATURALNESS_CRITIC_SUBAGENT,
            IRAB_BATCH_SUBAGENT,
            NATURALNESS_CRITIC_BATCH_SUBAGENT,
        ],
        checkpointer=checkpointer,
        skills=["/skills/"],
        interrupt_on={
            "finalize_batch": {
                "mode": "approve",
                "condition": "disagreement_rate > 0.25",
            },
        },
    )

    return agent, checkpoint_conn, checkpoint_db_path


# ===========================================================================
# INTERACTIVE PLAYGROUND TOOLS & FACTORIES
# ===========================================================================


def verify_playground_fidelity_tool(
    verse_id: str,
    sadr: str,
    ajuz: str,
    original_sadr: str = "",
    original_ajuz: str = "",
) -> dict:
    """Compare the normalized (diacritics-stripped) skeleton of the
    proposed committed text against the normalized skeleton of the
    trusted original input for this verse_id.

    If verse_id is custom/unregistered, it falls back to comparing against
    original_sadr and original_ajuz directly, allowing custom sandbox testing.
    """
    from tools.fidelity_tools import verify_skeleton_fidelity_tool

    # If verse_id exists in input database, run standard verification
    official_res = verify_skeleton_fidelity_tool(verse_id, sadr, ajuz)
    if official_res.get("found_input"):
        return official_res

    # Fallback to direct validation of custom provided values
    from tools.fidelity_tools import normalize_text, _diff
    from difflib import SequenceMatcher

    norm_sadr_in = normalize_text(original_sadr)
    norm_ajuz_in = normalize_text(original_ajuz)
    norm_sadr_out = normalize_text(sadr)
    norm_ajuz_out = normalize_text(ajuz)

    sadr_match = norm_sadr_in == norm_sadr_out
    ajuz_match = norm_ajuz_in == norm_ajuz_out

    full_in = f"{norm_sadr_in} {norm_ajuz_in}"
    full_out = f"{norm_sadr_out} {norm_ajuz_out}"
    similarity = SequenceMatcher(None, full_in, full_out).ratio()

    return {
        "match": sadr_match and ajuz_match,
        "verse_id": verse_id or "playground_custom",
        "found_input": False,
        "sadr_match": sadr_match,
        "ajuz_match": ajuz_match,
        "sadr_diff": None if sadr_match else _diff(norm_sadr_in, norm_sadr_out),
        "ajuz_diff": None if ajuz_match else _diff(norm_ajuz_in, norm_ajuz_out),
        "similarity": round(similarity, 4),
    }


def build_studio_batch_agent():
    """Factory function for LangGraph Studio to load the batch orchestrator."""
    agent, _, _ = build_agent(use_checkpointer=False)
    return agent


def build_studio_playground_agent():
    """Factory function to build the interactive Single-Verse Playground agent for Studio."""
    model = get_model()

    # See build_agent()'s comment above -- compiled subagents fold their
    # skill content into their own system prompts, so no "skills" dict-key
    # assignment is needed here anymore.
    diacritizer_model = get_model(max_completion_tokens=2048, max_tokens=2048, extra_body={"thinking_budget": 0})
    DIACRITIZER_SUBAGENT = build_diacritizer_subagent(diacritizer_model, BACKEND)
    IRAB_SUBAGENT = build_irab_checker_subagent(model)
    NATURALNESS_CRITIC_SUBAGENT = build_naturalness_critic_subagent(model)
    IRAB_BATCH_SUBAGENT = build_irab_checker_batch_subagent(model)
    NATURALNESS_CRITIC_BATCH_SUBAGENT = build_naturalness_critic_batch_subagent(model)

    playground_prompt = """You coordinate the diacritization and validation of a single Arabic verse against a target meter.

The input parameters are given directly inside the user's message as a structured format or text. Ensure you extract:
- verse_id (optional, default to "playground_custom")
- original_sadr (the Sadr text)
- original_ajuz (the Ajuz text)
- meter (the poetic meter name)

Your execution workflow:
1. Dispatch the `diacritizer` subagent (via the task tool) to diacritize both Sadr and Ajuz. Hand it the original_sadr and original_ajuz, and target meter.
2. Verify the diacritized output using your three deciding gates:
   a. Call sanitize_output_tool on Sadr and Ajuz to ensure security unicode limits are met.
   b. Call verify_playground_fidelity_tool with original_sadr and original_ajuz to ensure consonant skeleton remains identical.
   c. Call verify_single_verse_tool to ensure prosodic (pyarud) metrical rhythm matches.
3. If all deciding gates pass, run parallel advisory reviews using:
   - irab_checker (via task tool)
   - naturalness_critic (via task tool)
4. If irab_checker flags 'case_ending_swap':
   - Call reconcile_case_ending_tool with the returned word_index and target_harakah.
   - Immediately verify the reconciled text with verify_single_verse_tool.
   - If the metrical score remains >= 0.99, accept the swap. Otherwise, fall back to the pre-swap text.
5. Provide a complete, formatted result report to the user showing whether each gate passed/failed, listing any advisory criticism, and printing the final diacritized verse clearly.
"""

    agent = create_deep_agent(
        model=model,
        tools=[
            verify_single_verse_tool,
            sanitize_output_tool,
            reconcile_case_ending_tool,
            verify_playground_fidelity_tool,
            record_locked_verse_tool,
            read_ledger_tool,
            build_batched_advisory_payload_tool,
        ],
        system_prompt=playground_prompt,
        permissions=PERMISSIONS,
        backend=BACKEND,
        subagents=[
            DIACRITIZER_SUBAGENT,
            IRAB_SUBAGENT,
            NATURALNESS_CRITIC_SUBAGENT,
            IRAB_BATCH_SUBAGENT,
            NATURALNESS_CRITIC_BATCH_SUBAGENT,
        ],
        skills=["/skills/"],
    )
    return agent


# ===========================================================================
# CLI RUNNER ENTRYPOINT
# ===========================================================================


def _extract_engine_flag(argv: list[str]) -> tuple[str, list[str]]:
    """Task 3.3: pull an optional '--engine=deepagents|langgraph' flag out of
    argv, wherever it appears, without disturbing the existing positional
    filename/--all argument parsing below (which only ever looks at
    sys.argv[1]). Defaults to 'deepagents' -- the existing engine's behavior
    is unchanged if this flag is never passed (Minimum Change Rule: strictly
    additive, per Task 3.3)."""
    engine = "deepagents"
    remaining = []
    for a in argv:
        if a.startswith("--engine="):
            engine = a.split("=", 1)[1].strip().lower()
        else:
            remaining.append(a)
    if engine not in ("deepagents", "langgraph"):
        print(f"[-] Unknown --engine value '{engine}'; expected 'deepagents' or 'langgraph'. Defaulting to 'deepagents'.")
        engine = "deepagents"
    return engine, remaining


def main() -> None:
    engine, argv = _extract_engine_flag(sys.argv)
    print(f"[*] Engine: {engine}")

    if engine == "langgraph":
        # Deferred import: langgraph_pipeline.py itself imports main.py (for
        # MAX_CORRECTION_PASSES/PROJECT_ROOT), so importing it eagerly at
        # module level here would be a circular import. Safe to import
        # lazily inside main() since Python caches the partially-initialized
        # main module by the time this line runs.
        from langgraph_pipeline import build_langgraph_pipeline

        agent, checkpoint_conn, checkpoint_db_path = build_langgraph_pipeline(use_checkpointer=True)
    else:
        agent, checkpoint_conn, checkpoint_db_path = build_agent(use_checkpointer=True)

    try:
        # Determine the input paths dynamically
        inputs_dir = PROJECT_ROOT / "dataset" / "inputs"
        jsonl_files = []

        # Check if input file(s) are passed as a command-line argument
        if len(argv) > 1:
            arg = argv[1]
            if arg == "--all":
                jsonl_files = sorted(list(inputs_dir.glob("*.jsonl")))
                print(
                    f"[*] Processing ALL {len(jsonl_files)} input files in dataset/inputs/."
                )
            else:
                arg_path = pathlib.Path(arg)
                if arg_path.is_absolute() or arg_path.exists():
                    jsonl_files = [arg_path]
                else:
                    # Check if it's a filename inside the inputs directory
                    specific_file = inputs_dir / arg_path.name
                    if specific_file.exists():
                        jsonl_files = [specific_file]
                    else:
                        print(f"[-] Input file not found: {arg}")
                        if inputs_dir.exists():
                            print("[*] Available input files in dataset/inputs/:")
                            for f in sorted(inputs_dir.glob("*.jsonl")):
                                print(f"    - {f.name}")
                        return
        else:
            # No argument provided; list and default to all files
            jsonl_files = sorted(list(inputs_dir.glob("*.jsonl")))
            if jsonl_files:
                print(
                    f"[*] No input file specified. Found {len(jsonl_files)} input files in dataset/inputs/."
                )
                print(
                    "[*] Defaulting to processing ALL files. To run a single file, pass its name as an argument:"
                )
                print("    python main.py <filename.jsonl>")
                print("[*] Starting the processing loop...")
            else:
                print(f"[-] No .jsonl files found in {inputs_dir}")
                return

        # Loop through all resolved input files
        for input_path in jsonl_files:
            print(f"\n" + "=" * 60)
            print(f"[*] Reading inputs from: {input_path.name}")
            print("=" * 60)

            # 1. Read input verses
            raw_verses = []
            with input_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        raw_verses.append(json.loads(line))

            if not raw_verses:
                print(f"[-] No verses found in {input_path.name}. Skipping.")
                continue

            # 2. Group verses by meter to process them in coherent batches
            batches_by_meter = {}
            for v in raw_verses:
                meter = v.get("meter", "taweel")  # default fallback if not specified
                batches_by_meter.setdefault(meter, []).append(
                    {
                        "verse_id": v["verse_id"],
                        "sadr": v["sadr"],
                        "ajuz": v.get("ajuz", ""),
                    }
                )

            # 3. Invoke the DeepAgent orchestrator for each batch
            for meter, verses_batch in batches_by_meter.items():
                print(
                    f"[*] Processing batch of {len(verses_batch)} verses for meter: '{meter}' from '{input_path.name}'..."
                )

                # The graph's default state schema only recognizes "messages"
                # (plus "files"/"todos"). Passing raw "input"/"verses"/
                # "meter_name" top-level keys here silently drops them -- the
                # model never sees the batch. So the verses + meter are
                # embedded directly into the user message content as JSON
                # instead.
                verses_json = json.dumps(verses_batch, ensure_ascii=False, indent=2)
                user_message = (
                    f"Diacritize the following batch of verses against the meter "
                    f"'{meter}'.\n\n"
                    f"verses (JSON array of {{verse_id, sadr, ajuz}} objects):\n"
                    f"{verses_json}"
                )

                # Stable per-(input file, meter) thread_id: rerunning this
                # script resumes the SAME checkpointed thread instead of
                # silently starting a fresh, unrelated one each time.
                # Task 3.1's dangerous-zone note: the langgraph engine's graph
                # shape is completely different from DeepAgents' internal
                # graph, so the two must never attempt to resume each other's
                # checkpoints under the same nominal thread_id -- a ':lg'
                # suffix keeps them permanently distinct in checkpoints.sqlite.
                thread_id = f"{input_path.stem}:{meter}"
                if engine == "langgraph":
                    thread_id = f"{thread_id}:lg"
                run_config = {"configurable": {"thread_id": thread_id}}

                try:
                    existing_state = agent.get_state(run_config)
                except Exception:
                    existing_state = None

                # trace_run() opens a fresh, unique trace_id for THIS invoke
                # attempt (deliberately NOT the same thing as the LangGraph
                # thread_id above, which stays stable across resumes on
                # purpose — see tools/tracing.py's module docstring for why).
                # The trace_id lets you inspect token usage / latency per
                # agent (orchestrator, diacritizer, irab_checker,
                # naturalness_critic) for this specific attempt, even if the
                # same thread_id gets resumed multiple times.
                with trace_run(label=meter, langgraph_thread_id=thread_id) as trace:
                    traced_config = {**run_config, "callbacks": [trace.callback]}
                    print(
                        f"[*] trace_id='{trace.trace_id}' "
                        f"(inspect with: python -m tools.trace_report --trace {trace.trace_id})"
                    )

                    try:
                        if existing_state and existing_state.next:
                            # A previous run was interrupted (e.g. Ctrl+C)
                            # partway through this exact thread_id, with a
                            # pending next step. Passing None as input resumes
                            # from the last completed checkpoint instead of
                            # re-sending the original message and starting the
                            # batch over.
                            print(
                                f"[*] Resuming interrupted thread '{thread_id}' "
                                f"(next step: {existing_state.next})...."
                            )
                            try:
                                response = agent.invoke(None, config=traced_config)
                            except Exception as resume_exc:
                                # A4: distinct branch from the fresh-invoke
                                # path below -- a failure HERE specifically
                                # means the checkpoint we tried to resume from
                                # may itself be the problem, not just an
                                # ordinary mid-run LLM/tool error. Name that
                                # possibility explicitly rather than printing
                                # the same generic message either way.
                                print(
                                    f"[-] Resume failed for thread '{thread_id}': {resume_exc}"
                                )
                                print(
                                    "    The checkpoint may be corrupted or its state "
                                    "incompatible with a fresh run. Run `PRAGMA "
                                    f"integrity_check` against {checkpoint_db_path} to "
                                    "confirm, or re-run with a modified thread_id "
                                    "(e.g. append a suffix to input_path.stem) to "
                                    "reprocess this batch from scratch instead of "
                                    "resuming."
                                )
                                raise
                        elif engine == "langgraph":
                            # BatchState (langgraph_pipeline.py), not the
                            # DeepAgents "messages" shape -- the langgraph
                            # engine's nodes read verses/meter_name directly
                            # off state rather than parsing them back out of
                            # a JSON-embedded chat message.
                            response = agent.invoke(
                                {
                                    "verses": verses_batch,
                                    "meter_name": meter,
                                    "pass_number": 1,
                                    "locked": [],
                                    "broken": [],
                                    "structurally_incompatible": [],
                                    "drafts": {},
                                    "report_path": None,
                                    "thread_id": thread_id,
                                },
                                config=traced_config,
                            )
                        else:
                            response = agent.invoke(
                                {
                                    "messages": [
                                        {"role": "user", "content": user_message}
                                    ]
                                },
                                config=traced_config,
                            )
                        print(
                            f"[+] Batch execution complete. Response status: {response}"
                        )
                    except Exception as e:
                        print(
                            f"[-] Execution failed for batch under meter '{meter}': {str(e)}"
                        )
                        print(
                            f"    Checkpointed state for this run is saved under "
                            f"thread_id='{thread_id}' in {checkpoint_db_path}. "
                            f"Re-running this script will attempt to resume it."
                        )
                    finally:
                        print(
                            f"[*] trace summary: python -m tools.trace_report --trace {trace.trace_id}"
                        )
    finally:
        # Guarantee resources are cleaned up and SQLite is not left with
        # dangling file descriptors
        if checkpoint_conn is not None:
            print("[*] Terminating run. Closing checkpoint DB stream connection...")
            checkpoint_conn.close()


if __name__ == "__main__":
    main()
