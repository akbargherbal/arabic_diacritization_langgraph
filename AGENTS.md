# AGENTS.md — Persistent Memory

## Charter

This system diacritizes batches of normalized Arabic verses to match a
declared prosodic meter, verifies the result against `pyarud`
(via `verification/arabic_prosody_feedback.py`), and commits verified
records to `dataset/verses.jsonl` for downstream model training.

This is a proof-of-concept dataset builder, not a production literary tool.
Reliability/correctness of the dataset is priority #1, above iteration speed,
cost, and extensibility (in that order).

## Non-negotiable rules

1. **pyarud decides — but only after checking for a free reconciliation.**
   Not every pyarud/إعراب disagreement is poetic license. Fatha, damma, and
   kasra are metrically identical (pyarud encodes vowel presence, not
   identity) — so a case-ending error is often mechanically fixable with no
   metrical cost at all. The flow is:
     a. إعراب flags a case-ending issue → attempt reconcile_case_ending_tool
        first, then re-verify with pyarud.
     b. If the reconciled text still passes pyarud: commit it. This is a
        resolved grammar fix, not a disagreement — do not log it as one.
     c. Only if reconciliation isn't applicable (fix_type="structural") or
        the swap unexpectedly breaks the meter does the precedence rule
        actually apply: pyarud wins, commit the original text, log as a
        presumed poetic-license case (الضرورات الشعرية) for review.
   Do not skip straight to "pyarud wins" on every disagreement — that
   silently accepts avoidable grammar errors that had a free fix.

2. **Locked verses are not renegotiable.** Once a verse's pyarud scan is
   sound, it is `locked`. Never resubmit a locked verse to the diacritizer
   for regeneration, even in later batches, even if a later pass "could
   improve" it. Regenerating a passing verse risks flipping it to failing
   for no benefit.

3. **Verification code is read-only to every agent.** No subagent, and no
   custom tool the agent can trigger, may edit anything under `verification/`,
   `config/meter_tables.py`, or `tests/`. If a check seems wrong, that is a
   signal to log a disagreement, not to patch the checker.

4. **Dataset writes go through `commit_verse` only**, which re-runs the
   pyarud check itself before writing. Never call `write_file` directly
   against `dataset/`.

5. **Max 3 correction passes per batch.** Verses still broken after pass 3
   are logged as `unresolved` in `logs/disagreements/` and excluded from
   the dataset — never forced through a 4th pass, never auto-accepted.

6. **Known pyarud bug:** words ending in tanwīn fatḥ on an alif maqṣūra
   (e.g. أَسًى, هُدًى, فَتًى) are miscounted by the current `arudi.py` converter
   (extra silent unit appended). The documented workaround is phonetic
   normalization before analysis (e.g. أَسًى → أَسَنْ). Any verse matching this
   pattern should be flagged in the freshness/version metadata, not silently
   corrected without a trace.

## Verification axis count — be honest about this

There are only **two genuinely independent, deciding axes** in this system:
Structural (pyarud) and Security (the sanitizer). إعراب checking has no
deterministic rule set behind it — it is pure LLM judgment, the same
representation as the naturalness critic (both are "an LLM reads the
diacritized text and judges it"). Do not treat إعراب-flag + naturalness-flag
agreement as two independent confirmations; it's one kind of signal
expressed twice, and both share whatever blind spots the model family has.
This is a real gap, not a solved one — see Known limitations below.

## Known limitations this system does not solve

- A verse can pass pyarud, إعراب, and the naturalness critic while still
  being linguistically unnatural or subtly wrong, because two of those
  three checks are the same LLM-judgment representation running twice, not
  independent evidence. Treat needs_review=false as "nothing an LLM
  happened to notice," not as a strong correctness guarantee.
- No audit exists yet for systematic bias in *which* zihāf/corrections the
  diacritizer tends to reach for across many verses. Watch disagreement
  logs over time, not just per-batch.
- If إعراب/naturalness disagree with each other, that is NOT the same kind
  of signal as a pyarud/security failure — it's two prompts on one model
  noticing different things, not a structural conflict. Log it, but don't
  treat it with the same urgency as a deciding-axis failure.
