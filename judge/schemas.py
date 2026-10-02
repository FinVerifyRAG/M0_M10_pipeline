"""
judge/schemas.py
-----------------
Lightweight data schemas for the M8 judge — no LLM or openai dependency.
These can be imported anywhere without triggering openai installation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Verdict constants
VERIFIED     = "VERIFIED"
NOT_VERIFIED = "NOT_VERIFIED"


@dataclass
class JudgeResult:
    """Outcome from the V3 judge for a single atom."""
    verdict:        str             # VERIFIED | NOT_VERIFIED
    rationale:      str             # One-sentence explanation
    evidence_quote: str             # Verbatim evidence quote, or "none"
    atom_id:        str             # Which atom was judged
    risk:           float           # Risk score from M6
    latency_ms:     float = 0.0     # LLM round-trip time in milliseconds
    model_id:       str   = ""      # Which model was used
    raw_response:   str   = ""      # Raw LLM output (for debugging)
    parse_error:    bool  = False   # True if JSON parsing failed
