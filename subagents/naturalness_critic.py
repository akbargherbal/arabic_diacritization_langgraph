"""
subagents/naturalness_critic.py
=================================
LLM-based advisory pass targeting the specific Goodhart gap: a verse can
satisfy pyarud's binary scan while being phonologically unnatural (extra
vowels, unnatural elongations, silent-letter abuse) in a way no human would
actually vocalize. This check exists BECAUSE the pyarud proxy is gameable,
not as a generic quality pass.

Same model family as the diacritizer (per project constraint: DeepSeek for
both) -- treat its flags as weaker evidence than the deterministic axes.
Never used as a gate, never used as a tiebreaker for a pyarud/إعراب conflict.

Because this subagent shares a model family with the one that DRAFTED the
verse, a plain "sounds natural to me" read is weak evidence specifically
against whatever failure modes that model family shares with itself. The
prompt below does not (and cannot, given the fixed-model-provider
constraint) manufacture true independence -- instead it points the one
genuinely distinct thing this pass can do (actively search for an
alternative, more natural reading of the same letters) at the specific
failure shape a meter-fitting model is prone to, rather than leaving the
check generic.

FIX (Divergence A, see FIX_PLAN.md): previously a raw dict `SubAgent` spec
with "tools": [] -- but a plain dict spec still gets create_deep_agent's
default TodoListMiddleware + FilesystemMiddleware stack prepended
regardless of an empty "tools" list, which is why the trace showed 1 tool
call despite this subagent declaring zero tools of its own. This subagent
needs no filesystem or todo access -- it receives an already-locked verse's
text directly and returns a JSON verdict -- so it's compiled as a
`CompiledSubAgent` with an empty tool list and nothing else attached.

Unlike diacritizer/irab_checker, this subagent's prompt never referenced a
skills/ file (main.py never set a "skills" key on the old
NATURALNESS_CRITIC_SUBAGENT dict either), so there is no skill content to
fold in here.
"""

from langchain.agents import create_agent

NATURALNESS_SYSTEM_PROMPT = """
You review an already metrically-verified (pyarud-passing) Arabic verse for
phonological naturalness: would a fluent speaker actually vocalize it this
way, or does it read as artificially stretched/padded to force a metrical
match (e.g. unnatural vowel lengthening, implausible silent-letter choices,
an unlikely reading of an ambiguous word chosen only because it scans)?

You share a model family with the subagent that drafted this verse, which
means your blind spots likely overlap with whatever produced it -- do not
treat a fluent-sounding surface reading as sufficient on its own. Actively
check for the specific failure shape a meter-fitting model is prone to: a
choice that scans correctly ONLY under one unlikely reading of an ambiguous
word/root, chosen because that reading happens to fit, not because it's the
reading a listener would default to. Concretely: for each word whose vowel
choice is not the single obvious one, try to construct a more natural
alternative reading of the SAME letters and check whether that alternative
would break the meter. If it would, that is the strongest signal this check
exists to catch -- stronger than a general "sounds a bit off" -- and you
should flag it explicitly, naming the alternative reading you considered.

You are advisory only. You do not have access to any verification tool and
cannot alter the verse. Return {"verse_id": ..., "natural": bool, "note": str}.

Be specific in your note about WHAT reads as unnatural if you flag one --
"feels off" is not useful, "the elongation on X requires reading it as Y
which no fluent speaker would default to, whereas the natural reading Z
would break the meter" is.
"""


def build_naturalness_critic_subagent(model):
    """Compile naturalness_critic as an isolated agent with no tools -- it
    never needed filesystem or todo access; it receives an already-locked
    verse's text directly and returns a JSON verdict. This also removes the
    TodoListMiddleware/FilesystemMiddleware leak the trace showed (1 tool
    call despite declaring "tools": [])."""
    compiled = create_agent(
        model=model,
        tools=[],
        system_prompt=NATURALNESS_SYSTEM_PROMPT,
    )
    return {
        "name": "naturalness_critic",
        "description": (
            "Advisory LLM pass flagging pyarud-passing verses that read as "
            "phonologically unnatural. Same model family as the diacritizer -- "
            "treat flags as weaker evidence than deterministic axes. Actively "
            "checks for alternative, more-natural readings that would break "
            "the meter, to partially offset the shared-model-family blind spot."
        ),
        "runnable": compiled,
    }


NATURALNESS_BATCH_SYSTEM_PROMPT = """
You review a batch of already metrically-verified (pyarud-passing) Arabic verses for
phonological naturalness: would a fluent speaker actually vocalize each verse this
way, or does it read as artificially stretched/padded to force a metrical
match (e.g. unnatural vowel lengthening, implausible silent-letter choices,
an unlikely reading of an ambiguous word chosen only because it scans)?

You receive a JSON array of verses:
[
  {"verse_id": "...", "sadr": "...", "ajuz": "..."},
  ...
]

You review each verse in the batch and return a JSON array of verdicts containing EXACTLY one verdict per verse in the input batch.

You share a model family with the subagent that drafted these verses, which
means your blind spots likely overlap with whatever produced them -- do not
treat a fluent-sounding surface reading as sufficient on its own. Actively
check for the specific failure shape a meter-fitting model is prone to: a
choice that scans correctly ONLY under one unlikely reading of an ambiguous
word/root, chosen because that reading happens to fit, not because it's the
reading a listener would default to. Concretely: for each word whose vowel
choice is not the single obvious one, try to construct a more natural
alternative reading of the SAME letters and check whether that alternative
would break the meter. If it would, that is the strongest signal this check
exists to catch -- stronger than a general "sounds a bit off" -- and you
should flag it explicitly, naming the alternative reading you considered.

You are advisory only. Return EXACTLY a JSON array of objects with this shape wrapped in a markdown code block:
```json
[
  {
    "verse_id": "...",
    "natural": bool,
    "note": "..."
  },
  ...
]
```
Do not output any conversational filler or text outside the code block. Start directly with the code block, and close it immediately after the array.

Be specific in your note about WHAT reads as unnatural if you flag one --
"feels off" is not useful, "the elongation on X requires reading it as Y
which no fluent speaker would default to, whereas the natural reading Z
would break the meter" is.
"""


def build_naturalness_critic_batch_subagent(model):
    """Compile naturalness_critic_batch as an isolated agent with no tools.
    Receives a JSON array of locked verses and returns a JSON array of verdicts."""
    compiled = create_agent(
        model=model,
        tools=[],
        system_prompt=NATURALNESS_BATCH_SYSTEM_PROMPT,
    )
    return {
        "name": "naturalness_critic_batch",
        "description": (
            "Advisory LLM pass flagging phonologically unnatural verses in batch mode."
        ),
        "runnable": compiled,
    }
