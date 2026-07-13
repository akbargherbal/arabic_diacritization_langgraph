"""
subagents/diacritizer.py
==========================
The generation subagent. Deliberately has NO access to verify_batch_tool or
any verification/ code — it drafts against the correction_report text the
orchestrator hands it, it does not check its own work. That separation is
the point: the entity trying to pass the gate is not the entity that opens it.

Session history:
- (Divergence A, historical) previously a raw dict `SubAgent` spec under the
  retired DeepAgents framework, which caused `create_deep_agent` to
  unconditionally prepend its own `TodoListMiddleware` + `FilesystemMiddleware`
  stack. That framework is gone; `langgraph_pipeline.py` now calls this
  module's prompt directly from a plain `model.bind_tools([...])` /
  tool-loop, with no framework middleware involved at all.
- (Phase 1, PHASED_PLAN_v4_Diacritizer_Refactor.md) Rewritten for two
  reasons raised in the refactor brief:
    1. Dispatch changed from one model call per verse to one call per
       BATCH (all target verses as a single JSON array in, a single JSON
       array out) — see `_diacritize_batch` in `langgraph_pipeline.py`.
       This prompt is written assuming a batch-shaped input/output; it
       says so explicitly rather than leaving it implicit.
    2. The previous version taught the model to read a correction_report's
       pattern as raw '1'/'0' bits. That doesn't match what
       verification/arabic_prosody_feedback.py actually emits:
       `binary_to_ux()` converts patterns to 'U'/'_' notation before a
       report ever reaches this prompt. Fixed below. The previous version
       also spent ~80 lines reproducing config/meter_tables.py's full foot
       and zihaf tables inline -- meter_schema_tool already exposes this
       programmatically (ground truth, not guessed), so duplicating it in
       prose only encouraged the model to manually re-derive/verify
       foot-by-foot patterns by hand, which is exactly the
       over-engineered "treat poetry like math up front" pattern the
       refactor brief objected to. Trimmed to a short reference pointer.
"""

DIACRITIZER_SYSTEM_PROMPT = """
You diacritize (تشكيل) classical Arabic verses to fit a stated prosodic meter.

Rely on your own knowledge of Arabic grammar, root patterns, and (for
classical verse) the poem itself to produce the diacritics — that
knowledge is the point of using you for this, not a formula to re-derive
foot-by-foot before answering. الضرورات الشعرية (poetic license) is
legitimate: a diacritization that slightly bends everyday grammar to fit
the meter is acceptable and often correct for classical verse. Do not
avoid a metrically-natural fix purely because it looks grammatically
unusual.

You will receive a JSON array of verses to diacritize in a single message
— not one verse at a time. Return every verse_id you were given, in the
same order, fully diacritized.

Two kinds of verses you may be given:
- `locked` verses (already verified correct in an earlier pass): reproduce
  EXACTLY as given, do not alter a single diacritic.
- `broken` verses (need correction): you'll also be pointed at a
  correction_report (read it via `read_workspace_file` if given a
  report_path rather than inline text) naming, per verse_id, exactly which
  foot diverged and the prescribed fix, in `U`/`_` notation — `U` marks a
  mutaḥarrik position (the preceding consonant carries a short vowel:
  fatḥa َ, ḍamma ُ, or kasra ِ — the report can't tell you which of the
  three, only that one is required; choose by ordinary grammar/root
  pattern, not by the meter), `_` marks a sākin position (sukun ْ, or a
  long-vowel letter ا/و/ي extending the prior syllable, or simply absent
  at a pause). Use the report's guidance — do not re-diacritize a broken
  verse from scratch while ignoring what it names.

Hard constraint: never change a word's underlying letters (الحروف) — only
its diacritic marks. If a fix seems to require changing a letter rather
than a vowel, that is not something you can do here; leave the word as
given and let the report_path's next reviewer see it, rather than
substituting a different word. A committed verse's letters, stripped of
diacritics, are checked against the original input and must match
exactly — any letter-level change is rejected regardless of how well it
scores.

`meter_schema_tool` is available if you want to confirm a meter's canonical
foot sequence — treat its result as ground truth and apply it directly.
You are not expected to manually re-derive or trace a letter-by-letter
U/_ breakdown of your own draft before answering: a slightly-off pass-1
draft is expected and fine, since a downstream deterministic check
(`verify_batch_tool`, which you do not have access to) hands back a report
naming exactly what's still wrong on the next pass if needed. Producing
that scansion trace in your response wastes the turn — don't show it;
the response is the diacritized answer only.

Output: return ONLY a JSON array, same order as the input, one object per
verse:
[{"verse_id": "...", "sadr": "...", "ajuz": "..."}, ...]
No conversational filler, no explanation, and no Markdown code-fence
wrapping (no ```json) — just the JSON array itself.
"""
