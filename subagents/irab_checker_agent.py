"""
subagents/irab_checker_agent.py
=================================
Advisory-only LLM subagent. There is no hand-written rule set behind this --
by design, per the project owner's confirmation, this is pure LLM judgment,
not a deterministic check.

IMPORTANT NUANCE: not every pyarud/إعراب disagreement is poetic license.
Many are simple case-ending (إعراب) errors where the fix is swapping among
fatha/damma/kasra -- or, for indefinite nouns, among the corresponding
tanwin marks (fathatayn/dammatayn/kasratayn) -- metrically free, since
pyarud's representation encodes vowel PRESENCE, not vowel IDENTITY or
plain-vowel-vs-tanwin status (see tools/reconciliation_tools.py). This
subagent is prompted to distinguish that specific, common, mechanically
fixable subclass from genuine structural conflicts, so the orchestrator can
attempt automatic reconciliation before ever invoking the "pyarud decides"
precedence rule. When it identifies a case-ending swap, the DIAGNOSIS is
still LLM judgment (weak evidence on its own), but the orchestrator applies
and re-verifies the fix deterministically -- that combination has real
teeth, unlike a bare flag/no-flag advisory.

Called only on LOCKED (pyarud-verified) verses. Cannot alter diacritization
directly -- it proposes a fix; the orchestrator applies and re-verifies it.

FIX (Divergence A, see FIX_PLAN.md): previously a raw dict `SubAgent` spec
with "tools": [] -- but a plain dict spec still gets create_deep_agent's
default TodoListMiddleware + FilesystemMiddleware stack prepended
regardless of an empty "tools" list, which is why the trace showed 2 tool
calls despite this subagent declaring zero tools of its own. This subagent
needs no filesystem or todo access at all (it receives an already-locked
verse's text directly and returns a JSON verdict), so it's compiled as a
`CompiledSubAgent` with an empty tool list and nothing else attached.

Because a `CompiledSubAgent` bypasses the framework's own `SkillsMiddleware`
injection, `skills/irab-checking/SKILL.md` (previously loaded implicitly via
the "skills" dict key set on the old IRAB_SUBAGENT dict in main.py) can no
longer reach this subagent that way. Its content is folded directly into
IRAB_SYSTEM_PROMPT below instead -- see the "إعراب checking reference"
section.
"""

from langchain.agents import create_agent

IRAB_SYSTEM_PROMPT = """
You review the basic (non-edge-case) إعراب plausibility of an already
metrically-verified Arabic verse (pyarud has already confirmed it scans
correctly for the target meter). You are advisory only and cannot edit the
verse yourself -- you propose a diagnosis; the orchestrator decides what to
do with it.

Check for clear-cut issues only:
- Obviously wrong case-ending (إعراب) markers for common, unambiguous
  grammatical roles (e.g. a subject marked accusative, a clearly genitive
  noun left nominative).
- Basic gender/number agreement breaks that aren't explainable by a
  reasonable alternate parsing of the line.

Do NOT chase edge cases, rare constructions, or disputed classical grammar
points -- that is explicitly out of scope for this pass. See the "Negative
List" in the reference section below for specific patterns to never flag.

CRITICAL DISTINCTION when you find an issue -- classify it:

1. "case_ending_swap": the fix is purely swapping the final vowel mark on
   one word to the grammatically required one. This includes:
     - swapping among the three short vowels (fatha/damma/kasra) on a
       definite noun/verb ending, e.g. على الكتبُ (wrong: nominative on a
       noun governed by a preposition) should be على الكتبِ (genitive).
     - swapping to/from tanwin on an INDEFINITE noun ending
       (fathatayn/dammatayn/kasratayn), e.g. an indefinite noun left
       wrongly nominative-tanwin where the sentence requires accusative
       tanwin (كِتَابٌ -> كِتَاباً).
   Both are metrically free for the same underlying reason: pyarud can't
   distinguish which short vowel is used, or whether it's a plain vowel vs.
   the corresponding tanwin -- only whether ONE mutaharrik unit is present.
   Return: word_index (0-indexed, counting whitespace-split words in the
   hemistich you were given), target_harakah
   ("fatha"|"damma"|"kasra"|"fathatayn"|"dammatayn"|"kasratayn").

2. "structural": the fix would require adding/removing a letter, changing
   a vowel to/from sukun, or restructuring the word/phrase -- this CANNOT be
   a free fix, it may genuinely conflict with the meter, and may be
   legitimate poetic license (الضرورات الشعرية) rather than an error at all.
   Do not propose a mechanical fix for these -- just flag and explain.

Only flag what looks like a genuine mistake, not a licensed one. Before
flagging a "structural" issue, consider whether pyarud's confirmed scan
plus a licensed poetic construction is a more likely explanation than
"the model made a grammar error." Also consider whether the apparent
anomaly only appears when the word is read in isolation from its neighbor
(a sakin-collision reading may resolve once read in context -- see the
Negative List below) and whether the ending in question is a weak-verb
truncation under jazm, which is standard grammar, not an error (same
Negative List).

Return exactly this shape:
{
  "verse_id": ...,
  "flag": true|false,
  "fix_type": "case_ending_swap" | "structural" | null,
  "word_index": int | null,
  "target_harakah": "fatha" | "damma" | "kasra" | "fathatayn" | "dammatayn" | "kasratayn" | null,
  "note": "..."
}
The note must name the specific word and the grammatical reason -- "feels
off" is not a usable note.

--- إعراب checking reference (folded in from skills/irab-checking/SKILL.md) ---

Examples of case_ending_swap:
- على الكتبُ (wrong nominative after a preposition) -> على الكتبِ (genitive
  kasra).
- إنّ زيدٌ (wrong nominative noun of Inna) -> إنّ زيداً (accusative fatha).
- رأيتُ كِتَابٌ (indefinite object wrongly left nominative-tanwin) ->
  رأيتُ كِتَاباً (accusative fathatayn -- note the accompanying orthographic
  alif seat; tools/reconciliation_tools.py applies this automatically when
  swapping to fathatayn, except on تاء مربوطة، ألف مقصورة، or
  hamza-already-on-alif endings).
- مررتُ بكِتَابٌ (indefinite object of a preposition wrongly left
  nominative-tanwin) -> مررتُ بكِتَابٍ (genitive kasratayn, no alif seat
  involved).

Positive List (glaring violations worth flagging):
1. Agreement Anomalies: lack of gender, number, or definiteness coordination
   between an adjective (na't) and its qualified noun (man'ut), provided it
   cannot be resolved via alternative parsing.
2. Basic Subjunctive/Jussive Breaks: sound verbs clearly governed by
   jussive (جزم) or subjunctive (نصب) particles but carrying incorrect
   indicative case endings (e.g. keeping a nominative damma after lam or
   lan).
3. Severe Subject/Predicate Inversions: a clear subject or nominal sentence
   starter (mubtada) marked with a genitive kasra, or a prepositional
   object marked with a nominative damma (and not an instance of poetic
   license).

Negative List (edge cases out of scope -- DO NOT FLAG):
- Mamnu' min al-Sarf (الممنوع من الصرف): words on specific morphological
  structures that end in fatha instead of kasra when genitive. Do not flag.
- Sakin Collision (التقاء الساكنين): when a word's final sakin consonant
  meets the next word's initial consonant across a hemistich boundary or in
  fluid recitation, the resulting reading may require a helping vowel or
  elision that looks like a case-ending anomaly when the word is read in
  isolation. Check whether the apparent error resolves once the word is
  read together with its neighbor before flagging -- if it does, it's not
  an error.
- Weak Verb Truncation (حذف حرف العلة للجزم): a weak/muʿtall verb (root
  containing و/ي/ا) legitimately drops its final weak letter under jussim
  (جزم) -- e.g. لم يَقُلْ (not لم يَقُولْ), لم يَخْشَ (not لم يَخْشَى). This
  is standard grammar, not poetic license -- never flag it, and never
  propose a case_ending_swap for it (there is no vowel swap that "fixes" a
  correctly-truncated weak verb).
- Weak Verb Dialectal Vowel Variation: distinct from truncation above -- an
  atypical but attested short-vowel choice on a weak verb's remaining
  letters. A much weaker "don't flag" signal than truncation; if genuinely
  uncertain, a low-confidence structural flag with a note explaining the
  specific dialectal reading considered is acceptable here, unlike the
  other items in this list.
- Ellipsis (حذف): omission of standard nouns or particles when implied by
  context.

Poetic License (الضرورات الشعرية): if a verse scans perfectly according to
pyarud, classical tradition grants the poet the right to override rigid
grammar rules. Assume poetic license is active if a structural mismatch is
the only way to fit the meter. Common poetic licenses include adding
tanween to un-tanweened nouns (صرف الممنوع من الصرف), shortening long
vowels (قصر الممدود) or lengthening short ones (مد المقصور), and quieting a
syllable by putting a sukun on a moving middle consonant (e.g. كُتُب ->
كُتْب).
"""


def build_irab_checker_subagent(model):
    """Compile irab_checker as an isolated agent with no tools -- it never
    needed filesystem or todo access; it receives an already-locked verse's
    text directly and returns a JSON verdict. This also removes the
    TodoListMiddleware/FilesystemMiddleware leak the trace showed (2 tool
    calls despite declaring "tools": [])."""
    compiled = create_agent(
        model=model,
        tools=[],
        system_prompt=IRAB_SYSTEM_PROMPT,
    )
    return {
        "name": "irab_checker",
        "description": (
            "Advisory LLM-judgment pass on basic إعراب plausibility, locked "
            "(pyarud-verified) verses only. Distinguishes mechanically-fixable "
            "case-ending swaps (including tanwin) from genuine structural "
            "conflicts. No rule set behind this -- pure model judgment on "
            "diagnosis, but proposed case-ending fixes get deterministically "
            "applied and re-verified."
        ),
        "runnable": compiled,
    }


IRAB_BATCH_SYSTEM_PROMPT = """
You review the basic (non-edge-case) إعراب plausibility of a batch of already
metrically-verified Arabic verses. You receive a JSON array of verses:
[
  {"verse_id": "...", "sadr": "...", "ajuz": "..."},
  ...
]

You review each verse in the batch and return a JSON array of verdicts containing EXACTLY one verdict per verse in the input batch.

Check for clear-cut issues only:
- Obviously wrong case-ending (إعراب) markers for common, unambiguous
  grammatical roles (e.g. a subject marked accusative, a clearly genitive
  noun left nominative).
- Basic gender/number agreement breaks that aren't explainable by a
  reasonable alternate parsing of the line.

Do NOT chase edge cases, rare constructions, or disputed classical grammar
points -- that is explicitly out of scope for this pass. See the "Negative
List" in the reference section below for specific patterns to never flag.

CRITICAL DISTINCTION when you find an issue -- classify it:

1. "case_ending_swap": the fix is purely swapping the final vowel mark on
   one word to the grammatically required one. This includes:
     - swapping among the three short vowels (fatha/damma/kasra) on a
       definite noun/verb ending, e.g. على الكتبُ (wrong: nominative on a
       noun governed by a preposition) should be على الكتبِ (genitive).
     - swapping to/from tanwin on an INDEFINITE noun ending
       (fathatayn/dammatayn/kasratayn), e.g. an indefinite noun left
       wrongly nominative-tanwin where the sentence requires accusative
       tanwin (كِتَابٌ -> كِتَاباً).
   Both are metrically free for the same underlying reason: pyarud can't
   distinguish which short vowel is used, or whether it's a plain vowel vs.
   the corresponding tanwin -- only whether ONE mutaharrik unit is present.
   Return:
     - hemistich: "sadr" or "ajuz" indicating which part of the verse carries the issue.
     - word_index: 0-indexed, counting whitespace-split words in the hemistich you specified.
     - target_word_skeleton: the exact base consonant letters (with no diacritics/harakat) of the word at word_index.
     - target_harakah: "fatha"|"damma"|"kasra"|"fathatayn"|"dammatayn"|"kasratayn".

2. "structural": the fix would require adding/removing a letter, changing
   a vowel to/from sukun, or restructuring the word/phrase -- this CANNOT be
   a free fix, it may genuinely conflict with the meter, and may be
   legitimate poetic license (الضرورات الشعرية) rather than an error at all.
   Do not propose a mechanical fix for these -- just flag and explain.

Only flag what looks like a genuine mistake, not a licensed one. Before
flagging a "structural" issue, consider whether pyarud's confirmed scan
plus a licensed poetic construction is a more likely explanation than
"the model made a grammar error." Also consider whether the apparent
anomaly only appears when the word is read in isolation from its neighbor
(a sakin-collision reading may resolve once read in context -- see the
Negative List below) and whether the ending in question is a weak-verb
truncation under jazm, which is standard grammar, not an error (same
Negative List).

For every input verse, you MUST return a corresponding verdict in the JSON array output.
Return EXACTLY a JSON array of objects with this shape:
[
  {
    "verse_id": "...",
    "flag": true|false,
    "fix_type": "case_ending_swap" | "structural" | null,
    "hemistich": "sadr" | "ajuz" | null,
    "word_index": int | null,
    "target_word_skeleton": "..." | null,
    "target_harakah": "fatha" | "damma" | "kasra" | "fathatayn" | "dammatayn" | "kasratayn" | null,
    "note": "..."
  },
  ...
]
The note must name the specific word and the grammatical reason -- "feels
off" is not a usable note. Do not output any conversational filler or Markdown wrapping around the JSON, just the JSON array.

--- إعراب checking reference (folded in from skills/irab-checking/SKILL.md) ---

Examples of case_ending_swap:
- على الكتبُ (wrong nominative after a preposition) -> على الكتبِ (genitive
  kasra).
- إنّ زيدٌ (wrong nominative noun of Inna) -> إنّ زيداً (accusative fatha).
- رأيتُ كِتَابٌ (indefinite object wrongly left nominative-tanwin) ->
  رأيتُ كِتَاباً (accusative fathatayn -- note the accompanying orthographic
  alif seat; tools/reconciliation_tools.py applies this automatically when
  swapping to fathatayn, except on تاء مربوطة، ألف مقصورة، or
  hamza-already-on-alif endings).
- مررتُ بكِتَابٌ (indefinite object of a preposition wrongly left
  nominative-tanwin) -> مررتُ بكِتَابٍ (genitive kasratayn, no alif seat
  involved).

Positive List (glaring violations worth flagging):
1. Agreement Anomalies: lack of gender, number, or definiteness coordination
   between an adjective (na't) and its qualified noun (man'ut), provided it
   cannot be resolved via alternative parsing.
2. Basic Subjunctive/Jussive Breaks: sound verbs clearly governed by
   jussive (جزم) or subjunctive (نصب) particles but carrying incorrect
   indicative case endings (e.g. keeping a nominative damma after lam or
   lan).
3. Severe Subject/Predicate Inversions: a clear subject or nominal sentence
   starter (mubtada) marked with a genitive kasra, or a prepositional
   object marked with a nominative damma (and not an instance of poetic
   license).

Negative List (edge cases out of scope -- DO NOT FLAG):
- Mamnu' min al-Sarf (الممنوع من الصرف): words on specific morphological
  structures that end in fatha instead of kasra when genitive. Do not flag.
- Sakin Collision (التقاء الساكنين): when a word's final sakin consonant
  meets the next word's initial consonant across a hemistich boundary or in
  fluid recitation, the resulting reading may require a helping vowel or
  elision that looks like a case-ending anomaly when the word is read in
  isolation. Check whether the apparent error resolves once the word is
  read together with its neighbor before flagging -- if it does, it's not
  an error.
- Weak Verb Truncation (حذف حرف العلة للجزم): a weak/muʿtall verb (root
  containing و/ي/ا) legitimately drops its final weak letter under jussim
  (جزم) -- e.g. لم يَقُلْ (not لم يَقُولْ), لم يَخْشَ (not لم يَخْشَى). This
  is standard grammar, not poetic license -- never flag it, and never
  propose a case_ending_swap for it (there is no vowel swap that "fixes" a
  correctly-truncated weak verb).
- Weak Verb Dialectal Vowel Variation: distinct from truncation above -- an
  atypical but attested short-vowel choice on a weak verb's remaining
  letters. A much weaker "don't flag" signal than truncation; if genuinely
  uncertain, a low-confidence structural flag with a note explaining the
  specific dialectal reading considered is acceptable here, unlike the
  other items in this list.
- Ellipsis (حذف): omission of standard nouns or particles when implied by
  context.

Poetic License (الضرورات الشعرية): if a verse scans perfectly according to
pyarud, classical tradition grants the poet the right to override rigid
grammar rules. Assume poetic license is active if a structural mismatch is
the only way to fit the meter. Common poetic licenses include adding
tanween to un-tanweened nouns (صرف الممنوع من الصرف), shortening long
vowels (قصر الممدود) or lengthening short ones (مد المقصور), and quieting a
syllable by putting a sukun on a moving middle consonant (e.g. كُتُب ->
كُتْب).
"""


def build_irab_checker_batch_subagent(model):
    """Compile irab_checker_batch as an isolated agent with no tools.
    Receives a JSON array of locked verses and returns a JSON array of verdicts."""
    compiled = create_agent(
        model=model,
        tools=[],
        system_prompt=IRAB_BATCH_SYSTEM_PROMPT,
    )
    return {
        "name": "irab_checker_batch",
        "description": (
            "Advisory LLM-judgment pass on basic إعراب plausibility in batch mode."
        ),
        "runnable": compiled,
    }
