"""
verify/v1_deterministic.py
--------------------------
V1 literal verifier: exact / normalised matching of an atom against
the retrieved evidence chunks.

Decision labels
    MATCH       The evidence explicitly states the same value/date/reference.
    MISMATCH    The evidence mentions the same quantity but with a different value.
    NOT_FOUND   The evidence does not mention the quantity at all.
    NA          The atom type is not covered by V1 (ENTITY, APPLICABILITY).

Atom types handled:
    RATE        Percentage value  →  norm_pct()
    THRESHOLD   Rupee / time amount  →  norm_amount()
    DATE        Any date  →  norm_date()
    SECTION     Regulation/section reference  →  norm_section()
    ENTITY / APPLICABILITY  →  NA  (handed to V2)
"""
from __future__ import annotations

import re
import logging
from typing import List, Optional

from common.schemas import Atom, Chunk, RetrievalResult
from verify.normalizers import (
    norm_pct, norm_amount, norm_date, norm_section,
    numerically_equal, dates_equal, sections_equal,
    PCT, SEC, _BPS_RE,
)

logger = logging.getLogger("verify.v1_deterministic")

# V1 status constants
MATCH = "MATCH"
MISMATCH = "MISMATCH"
NOT_FOUND = "NOT_FOUND"
NA = "NA"

# Atom types that V1 cannot verify deterministically
_V2_ONLY_TYPES = {"ENTITY", "APPLICABILITY"}


# ---------------------------------------------------------------------------
# Helper: scan a piece of text for all values of a given type
# ---------------------------------------------------------------------------

# BUG 1 FIX: use corrected PCT pattern (no word-boundary after %)
_PCT_SCAN = PCT
_AMT_SCAN = re.compile(
    r"(?:₹|`|Rs\.?\s*|INR\s*)?\s*\d[\d,]*(?:\.\d+)?\s*"
    r"(?:lakh|lakhs?|lac|lacs?|crore|crores?|cr\.?|thousand|thousands?|million|billion)?"
    r"(?:\s*(?:rupees?|/-))?" ,
    re.IGNORECASE,
)
# BUG 2 FIX: use corrected SEC pattern (preserves 194-IA, u/s, lists)
_SEC_SCAN = SEC
_DATE_SCAN = re.compile(
    r"\b(?:FY\s*\d{4}[-–]\d{2,4}|"
    r"AY\s*\d{4}[-–]\d{2,4}|"
    r"\d{1,2}[/\-.](\d{1,2})[/\-.](\d{4})|"
    r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|"
    r"\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)


def _scan_values(text: str, atom_type: str) -> list:
    """
    Return all raw matched strings for the given atom type found in text.
    """
    if atom_type == "RATE":
        return [m.group(0) for m in _PCT_SCAN.finditer(text)]
    if atom_type == "THRESHOLD":
        return [m.group(0) for m in _AMT_SCAN.finditer(text) if m.group(0).strip()]
    if atom_type == "DATE":
        return [m.group(0) for m in _DATE_SCAN.finditer(text)]
    if atom_type == "SECTION":
        return [m.group(0) for m in _SEC_SCAN.finditer(text)]
    return []


def _normalize(value: str, atom_type: str):
    """Normalize a raw value string for the given atom type."""
    if atom_type == "RATE":
        return norm_pct(value)
    if atom_type == "THRESHOLD":
        return norm_amount(value)
    if atom_type == "DATE":
        return norm_date(value)
    if atom_type == "SECTION":
        return norm_section(value)
    return None


def _values_equal(a, b, atom_type: str) -> bool:
    if a is None or b is None:
        return False
    if atom_type in ("RATE", "THRESHOLD"):
        result = numerically_equal(a, b)
        return result is True
    if atom_type == "DATE":
        return a == b
    if atom_type == "SECTION":
        return a == b
    return a == b


# ---------------------------------------------------------------------------
# Per-chunk check
# ---------------------------------------------------------------------------

def _check_chunk(atom: Atom, chunk_text: str) -> str:
    """
    Return MATCH, MISMATCH, or NOT_FOUND for a single chunk.
    """
    atom_norm = _normalize(atom.text, atom.type)
    if atom_norm is None:
        return NOT_FOUND

    evidence_values = _scan_values(chunk_text, atom.type)
    if not evidence_values:
        return NOT_FOUND

    for ev_raw in evidence_values:
        ev_norm = _normalize(ev_raw, atom.type)
        if ev_norm is None:
            continue
        if _values_equal(atom_norm, ev_norm, atom.type):
            return MATCH

    # Evidence mentions the same type of quantity but with a different value
    return MISMATCH


# ---------------------------------------------------------------------------
# Main V1 entry point
# ---------------------------------------------------------------------------

def v1_check(atom: Atom, rr: RetrievalResult) -> str:
    """
    Run V1 deterministic check on one atom against all retrieved chunks.

    Strategy:
    1. If atom.cited_chunk is set, check that chunk first.
    2. If the cited chunk returns NOT_FOUND, scan ALL chunks for any match.
    3. Return MATCH if any chunk matches, MISMATCH if any disagrees, NOT_FOUND if silent.

    Returns: 'MATCH' | 'MISMATCH' | 'NOT_FOUND' | 'NA'
    """
    if atom.type in _V2_ONLY_TYPES:
        logger.debug("v1_check: atom %s is type %s → NA", atom.atom_id, atom.type)
        return NA

    atom_norm = _normalize(atom.text, atom.type)
    if atom_norm is None:
        logger.warning("v1_check: could not normalize atom '%s' (%s)", atom.text, atom.type)
        return NOT_FOUND

    chunk_map: dict[str, Chunk] = {c.chunk_id: c for c in rr.chunks}

    # --- Priority: cited chunk first ---
    results: List[str] = []

    if atom.cited_chunk and atom.cited_chunk in chunk_map:
        cited_result = _check_chunk(atom, chunk_map[atom.cited_chunk].text)
        results.append(cited_result)
        if cited_result == MATCH:
            logger.debug("v1_check: MATCH in cited chunk %s", atom.cited_chunk)
            return MATCH

    # --- Scan all remaining chunks ---
    for chunk in rr.chunks:
        if chunk.chunk_id == atom.cited_chunk:
            continue
        r = _check_chunk(atom, chunk.text)
        results.append(r)
        if r == MATCH:
            logger.debug("v1_check: MATCH in chunk %s", chunk.chunk_id)
            return MATCH

    # Aggregate: if any chunk said MISMATCH, escalate
    if MISMATCH in results:
        logger.debug("v1_check: MISMATCH for atom '%s'", atom.text)
        return MISMATCH

    logger.debug("v1_check: NOT_FOUND for atom '%s'", atom.text)
    return NOT_FOUND
