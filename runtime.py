"""Framework-neutral runtime settings for the diacritization pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# This is a correctness boundary, not a tuning knob.  AGENTS.md requires that
# verses still failing after this many correction passes are logged unresolved.
MAX_CORRECTION_PASSES = 3
