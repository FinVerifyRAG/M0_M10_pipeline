"""
decision/answer_rewrite.py
---------------------------
M8 Answer Rewrite: trims or flags the answer based on atom decisions.

When some atoms are ABSTAINED or NOT_VERIFIED, the original answer cannot be
shown as-is because it contains unverified claims.  This module provides two
strategies:

  1. **Flag mode** (default): Wrap the original answer with a disclaimer and
     annotate the problematic atoms inline (e.g., [⚠ UNVERIFIED]).
     This is safe and preserves all information.

  2. **Trim mode**: Remove sentences containing ABSTAINED/NOT_VERIFIED atoms
     from the answer text.  More aggressive; may leave incomplete sentences.

  3. **Stub mode**: Replace the answer with a standard abstention message
     when ALL atoms are abstained.

When all atoms are SUPPORTED or VERIFIED, the original answer is returned
unchanged (no rewrite needed).

Public API
----------
    from decision.answer_rewrite import rewrite_answer, RewrittenAnswer

    rw = rewrite_answer(answer, answer_decision, mode="flag")
    print(rw.text)           # The rewritten answer text
    print(rw.rewrite_mode)   # "none" | "flag" | "trim" | "stub"
    print(rw.disclaimer)     # Prepended disclaimer (if any)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from common.schemas import GeneratedAnswer
from decision.combine import AnswerDecision, AtomDecision, FULLY_SUPPORTED, ANSWER_ABSTAINED

logger = logging.getLogger("decision.answer_rewrite")

# Statuses that indicate an atom needs flagging or removal
_UNVERIFIED_STATUSES = {"ABSTAINED", "NOT_VERIFIED", "UNCERTAIN", "REJECTED"}
# OUTDATED atoms are shown with a special warning (they passed verification but are superseded)
_OUTDATED_STATUS = "OUTDATED"

# Standard disclaimers
_DISCLAIMER_PARTIAL = (
    "⚠ Note: This answer contains one or more claims that could not be fully "
    "verified against the available evidence. Flagged claims are marked with [⚠]."
)
_DISCLAIMER_ABSTAIN = (
    "⚠ This answer could not be verified. The claims made are outside the "
    "system's certified reliability threshold and should not be relied upon."
)
# Phase 8: New disclaimers for OUTDATED and REJECTED statuses
_DISCLAIMER_OUTDATED = (
    "⚠ Note: This answer may reference superseded regulatory text. "
    "Outdated claims are marked with [OUTDATED]. Always verify against the latest circular."
)
_DISCLAIMER_REJECTED = (
    "⚠ Critical: This answer contains claims that were explicitly rejected as contradicted "
    "by the evidence. Rejected claims are marked with [REJECTED]."
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RewrittenAnswer:
    """The rewritten (or unchanged) answer with provenance."""
    text:         str                  # Final answer text to show
    original:     str                  # Original answer_text (unchanged)
    rewrite_mode: str                  # "none" | "flag" | "trim" | "stub"
    disclaimer:   str = ""             # Prepended disclaimer (empty if FULLY_SUPPORTED)
    flagged_atoms: List[str] = field(default_factory=list)  # atom_ids that were flagged
    removed_atoms: List[str] = field(default_factory=list)  # atom_ids removed in trim


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _flag_mode(
    original_text: str,
    bad_atoms:     List[AtomDecision],
) -> tuple:
    """
    Annotate problematic atom spans with [⚠] markers.
    Phase 8: OUTDATED atoms get [OUTDATED] tag, REJECTED get [REJECTED] tag.
    Returns (rewritten_text, flagged_atom_ids).
    """
    text = original_text
    flagged_ids: List[str] = []

    # Sort by text length descending to avoid nested replacements
    sorted_atoms = sorted(bad_atoms, key=lambda a: len(a.atom_text), reverse=True)

    for atom in sorted_atoms:
        snippet = atom.atom_text.strip()
        if not snippet or snippet not in text:
            # Try the claim as fallback
            if atom.atom_claim and atom.atom_claim.strip() in text:
                snippet = atom.atom_claim.strip()
            else:
                continue

        # Phase 8: choose tag based on final status
        if atom.final_status == "OUTDATED":
            tag = f"[OUTDATED: {snippet}]"
        elif atom.final_status == "REJECTED":
            tag = f"[REJECTED: {snippet}]"
        else:
            tag = f"[⚠ UNVERIFIED: {snippet}]"

        text = text.replace(snippet, tag, 1)
        flagged_ids.append(atom.atom_id)

    return text, flagged_ids



def _trim_mode(
    original_text: str,
    bad_atoms:     List[AtomDecision],
) -> tuple:
    """
    Remove sentences from the answer that contain ABSTAINED/NOT_VERIFIED atoms.
    Returns (trimmed_text, removed_atom_ids).
    """
    # Split into sentences (simple period/newline heuristic)
    sentences = re.split(r'(?<=[.!?])\s+|\n', original_text)
    removed_ids: List[str] = []
    kept: List[str] = []

    for sent in sentences:
        sent_stripped = sent.strip()
        if not sent_stripped:
            continue

        # Check if any bad atom's text is in this sentence
        sentence_is_bad = False
        for atom in bad_atoms:
            if atom.atom_text.strip() in sent_stripped or (
                atom.atom_claim and atom.atom_claim.strip() in sent_stripped
            ):
                sentence_is_bad = True
                removed_ids.append(atom.atom_id)
                break

        if not sentence_is_bad:
            kept.append(sent_stripped)

    trimmed = " ".join(kept).strip()
    return trimmed, list(set(removed_ids))


def _stub_mode() -> str:
    return (
        "This answer cannot be provided. The system was unable to verify any "
        "of the claims against the available regulatory evidence. "
        "Please consult the original regulatory documents directly."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rewrite_answer(
    answer:          GeneratedAnswer,
    answer_decision: AnswerDecision,
    mode:            str = "flag",
) -> RewrittenAnswer:
    """
    Rewrite (or pass through) the answer based on atom-level decisions.

    Parameters
    ----------
    answer          : Original GeneratedAnswer from M3.
    answer_decision : AnswerDecision from decision.combine.
    mode            : "flag" | "trim" | "stub_on_abstain" | "none".
                      "flag"            - inline [⚠] markers (default, safe).
                      "trim"            - remove bad sentences.
                      "stub_on_abstain" - replace fully abstained answers.
                      "none"            - return original, only add disclaimer.

    Returns
    -------
    RewrittenAnswer.
    """
    original_text = answer.answer_text
    answer_status = answer_decision.answer_status

    # No rewrite needed if fully supported
    if answer_status == FULLY_SUPPORTED:
        return RewrittenAnswer(
            text=original_text,
            original=original_text,
            rewrite_mode="none",
            disclaimer="",
        )

    # Identify bad atoms — Phase 8: include OUTDATED and REJECTED
    bad_atoms = [
        ad for ad in answer_decision.atom_decisions
        if ad.final_status in _UNVERIFIED_STATUSES or ad.final_status == _OUTDATED_STATUS
    ]

    # Choose disclaimer: prefer OUTDATED or REJECTED-specific messages if present
    has_outdated  = any(ad.final_status == "OUTDATED"  for ad in bad_atoms)
    has_rejected  = any(ad.final_status == "REJECTED"  for ad in bad_atoms)
    if has_rejected:
        active_disclaimer = _DISCLAIMER_REJECTED
    elif has_outdated:
        active_disclaimer = _DISCLAIMER_OUTDATED
    else:
        active_disclaimer = _DISCLAIMER_PARTIAL

    # Full abstention
    if answer_status == ANSWER_ABSTAINED or (
        len(bad_atoms) == len(answer_decision.atom_decisions) and bad_atoms
    ):
        stub_text = _stub_mode()
        return RewrittenAnswer(
            text=stub_text,
            original=original_text,
            rewrite_mode="stub",
            disclaimer=_DISCLAIMER_ABSTAIN,
        )

    if mode == "flag":
        rewritten, flagged = _flag_mode(original_text, bad_atoms)
        result = RewrittenAnswer(
            text=active_disclaimer + "\n\n" + rewritten,
            original=original_text,
            rewrite_mode="flag",
            disclaimer=active_disclaimer,
            flagged_atoms=flagged,
        )

    elif mode == "trim":
        trimmed, removed = _trim_mode(original_text, bad_atoms)
        if not trimmed.strip():
            trimmed = _stub_mode()
            mode    = "stub"
        result = RewrittenAnswer(
            text=active_disclaimer + "\n\n" + trimmed,
            original=original_text,
            rewrite_mode=mode,
            disclaimer=active_disclaimer,
            removed_atoms=removed,
        )

    elif mode == "stub_on_abstain":
        stub_text = _stub_mode()
        result = RewrittenAnswer(
            text=_DISCLAIMER_ABSTAIN + "\n\n" + stub_text,
            original=original_text,
            rewrite_mode="stub",
            disclaimer=_DISCLAIMER_ABSTAIN,
        )

    else:  # mode == "none" or unknown
        result = RewrittenAnswer(
            text=active_disclaimer + "\n\n" + original_text,
            original=original_text,
            rewrite_mode="none",
            disclaimer=active_disclaimer,
        )

    logger.info(
        "Rewrite: mode=%s  bad_atoms=%d  flagged=%d  removed=%d",
        result.rewrite_mode, len(bad_atoms),
        len(result.flagged_atoms), len(result.removed_atoms),
    )
    return result
