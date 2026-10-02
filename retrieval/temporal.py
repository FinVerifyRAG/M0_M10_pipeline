"""
retrieval/temporal.py
----------------------
Phase 6: Temporal filtering with entropy-based scoring.

Two-stage filtering:
1. Hard temporal filter: drop chunks outside effective date window.
2. Soft temporal entropy scoring: rank remaining chunks by temporal
   coverage entropy (prefer unambiguous, recent, non-superseded chunks).

Public API
----------
    from retrieval.temporal import temporal_filter, temporal_entropy_score

    # Hard filter (as before)
    valid = temporal_filter(chunks, graph, query_date)

    # Phase 6: entropy-scored sort (use this after hard filter)
    ranked = sorted(valid, key=lambda c: temporal_entropy_score(c, query_date))
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import List, Optional

from common.schemas import Chunk
from ingest.version_graph.build import VersionGraph
from ingest.version_graph.query import in_force

logger = logging.getLogger("retrieval.temporal")

# Maximum age in days for the recency bonus calculation
_MAX_RECENCY_DAYS = 3650   # 10 years


def temporal_filter(
    chunks: List[Chunk],
    graph: VersionGraph,
    query_date: Optional[str] = None,
) -> List[Chunk]:
    """
    Hard temporal filter: drop chunks that are outside their effective
    date window or superseded by a different active version.

    Phase 6: also drops chunks whose effective_from is strictly after
    query_date (not yet in force) and those whose effective_to is strictly
    before query_date (already expired).

    Parameters
    ----------
    chunks     : Candidate chunks from retrieval.
    graph      : VersionGraph from M1 for supersession queries.
    query_date : ISO date string.  If None, uses currently active chunks only.

    Returns
    -------
    List of chunks that pass the hard filter (order preserved).
    """
    valid_chunks = []

    for chunk in chunks:
        section_id = chunk.metadata.get("section_id")

        # ── Version-graph supersession check ──────────────────────────────────
        if query_date and section_id:
            active_doc_id = in_force(graph, section_id, query_date)
            # If the graph knows the active version and this chunk isn't it → skip
            if active_doc_id and chunk.metadata.get("doc_id") != active_doc_id:
                logger.debug(
                    "Temporal filter: chunk %s superseded by %s on %s",
                    chunk.chunk_id, active_doc_id, query_date,
                )
                continue

        # ── Hard date bounds check ────────────────────────────────────────────
        if query_date:
            # Not yet effective: effective_from is in the future relative to query_date
            if chunk.effective_from and chunk.effective_from > query_date:
                logger.debug(
                    "Temporal filter: chunk %s not yet effective (from=%s, query=%s)",
                    chunk.chunk_id, chunk.effective_from, query_date,
                )
                continue

            # Already expired: effective_to is in the past relative to query_date
            if chunk.effective_to and chunk.effective_to < query_date:
                logger.debug(
                    "Temporal filter: chunk %s expired (to=%s, query=%s)",
                    chunk.chunk_id, chunk.effective_to, query_date,
                )
                continue

        valid_chunks.append(chunk)

    logger.debug(
        "Temporal filter: %d/%d chunks passed (query_date=%s)",
        len(valid_chunks), len(chunks), query_date,
    )
    return valid_chunks


def temporal_entropy_score(
    chunk: Chunk,
    query_date: Optional[str] = None,
) -> float:
    """
    Phase 6: Temporal entropy score for a chunk.

    Combines two components:
    1. Recency score:   newer chunks score lower (0 = most recent).
    2. Ambiguity score: chunks without effective_to (still active) score 0;
                        chunks with effective_to set score proportionally higher.

    The total score is in [0, 1].  Chunks should be sorted ascending by this
    score — lower score = more temporally relevant.

    Usage: sort retrieved chunks after hard temporal filter:
        ranked = sorted(valid, key=lambda c: temporal_entropy_score(c, query_date))
    """
    recency   = _recency_component(chunk, query_date)
    ambiguity = _ambiguity_component(chunk)
    # Weighted combination: 60% recency, 40% ambiguity
    score = 0.6 * recency + 0.4 * ambiguity
    return round(score, 4)


def _recency_component(chunk: Chunk, query_date: Optional[str]) -> float:
    """
    Lower score = more recent.

    Computes age in days from query_date to chunk.issue_date and normalises
    by _MAX_RECENCY_DAYS.  Uses an exponential decay to de-emphasise very
    old documents smoothly.
    """
    if not query_date:
        return 0.0

    date_str = chunk.issue_date or chunk.effective_from or ""
    if not date_str:
        return 0.5  # Unknown date → neutral

    try:
        query_dt = datetime.strptime(query_date[:10], "%Y-%m-%d")
        chunk_dt  = datetime.strptime(date_str[:10], "%Y-%m-%d")
    except ValueError:
        return 0.5

    age_days = max(0, (query_dt - chunk_dt).days)
    # Exponential decay: score → 0 for recent; → 1 for very old
    return round(1.0 - math.exp(-age_days / _MAX_RECENCY_DAYS), 4)


def _ambiguity_component(chunk: Chunk) -> float:
    """
    Lower score = less ambiguous (chunk is still active).

    A chunk with effective_to set is bounded → ambiguity proportional to
    how short the window is.  A still-active chunk has ambiguity 0.
    """
    if chunk.effective_to is None:
        return 0.0   # Active (no expiry) → no ambiguity

    if chunk.effective_from is None:
        return 0.5   # Unknown window → moderate ambiguity

    try:
        t_from = datetime.strptime(chunk.effective_from[:10], "%Y-%m-%d")
        t_to   = datetime.strptime(chunk.effective_to[:10],   "%Y-%m-%d")
    except ValueError:
        return 0.5

    window_days = max(1, (t_to - t_from).days)
    # Short windows are more ambiguous (the rule was only valid briefly)
    # Normalise against 365 days: windows < 1 year are moderately ambiguous
    return round(min(1.0, 365.0 / window_days), 4)
