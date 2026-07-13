"""
tools/sanitization_tools.py
=============================
Security verification axis: confirms diacritized output is composed only of
expected Arabic script + diacritic Unicode ranges, with no injected control
characters or non-Arabic payload. This is a DECIDING check (blocks commit),
unlike إعراب/naturalness which are advisory — malformed byte content isn't
a judgment call, and a dataset consumed by downstream training code should
never have to defend against injected control characters.

Extend ALLOWED_RANGES if you need to support additional legitimate marks
(e.g. Quranic annotation marks) — but extend deliberately, don't widen this
just to make a stubborn verse pass.
"""

import unicodedata

# Arabic block, Arabic Supplement, Arabic Presentation Forms, Arabic diacritics,
# plus basic whitespace/punctuation commonly present in verse text.
ALLOWED_RANGES = [
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic Supplement
    (0xFB50, 0xFDFF),   # Arabic Presentational Forms-A
    (0xFE70, 0xFEFF),   # Arabic Presentational Forms-B
]
ALLOWED_EXTRA_CHARS = set(" \t\n\r.,؛؟!—-()«»\"'")


def sanitize_output_tool(text: str) -> dict:
    """Validate a diacritized verse string before it's eligible for commit.

    Returns {"valid": bool, "reason": str | None}. commit_verse_tool calls
    this itself as a final gate — this is enforcement, not just advisory
    linting, because it sits inside the custom tool's own code rather than
    relying on the declarative permission layer (which doesn't cover
    custom-tool side effects at all).
    """
    for ch in text:
        if ch in ALLOWED_EXTRA_CHARS:
            continue
        cp = ord(ch)
        if not any(lo <= cp <= hi for lo, hi in ALLOWED_RANGES):
            category = unicodedata.category(ch)
            if category.startswith("C"):  # control/format/surrogate/private-use
                return {"valid": False, "reason": f"disallowed control char U+{cp:04X}"}
            return {"valid": False, "reason": f"unexpected char '{ch}' (U+{cp:04X}, category {category})"}
    return {"valid": True, "reason": None}
