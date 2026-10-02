"""
signals/s_ret.py  —  Retrieval Strength Signal
-----------------------------------------------
s_ret captures how strongly the retrieved evidence supports the atom's
cited chunk.

Computation
-----------
The reranker scores the retrieved chunks against the query. The score for
the chunk cited in the atom (atom.cited_chunk) is taken as the raw signal,
then min-max normalised over all chunks in the retrieval result.

If the atom has no cited_chunk, or the cited chunk is not in the result,
we use the MINIMUM reranker score among all chunks (worst case assumption).

If no reranker scores are present (metadata["rerank_scores"] absent), we
fall back to the chunk's RRF rank position, normalised to [0, 1].

Signal polarity:
    s_ret → 0 = weak retrieval (evidence may not actually support atom)
    s_ret → 1 = strong retrieval (evidence firmly supports atom)
"""
from __future__ import annotations

import logging
from typing import Optional

from common.schemas import VerifiedAtom, RetrievalResult

logger = logging.getLogger("signals.s_ret")


def compute_s_ret(
    verified_atom: VerifiedAtom,
    rr: RetrievalResult,
) -> float:
    """
    Compute retrieval-strength signal for one atom.

    Looks for rerank scores stored in rr.metadata["rerank_scores"] as a
    dict {chunk_id: float}.  Falls back to rank-based normalisation.

    Returns float in [0, 1].
    """
    atom = verified_atom.atom
    cited = atom.cited_chunk

    # ------------------------------------------------------------------
    # Option A: use stored reranker scores
    # ------------------------------------------------------------------
    rerank_scores: dict = rr.metadata.get("rerank_scores", {})

    if rerank_scores:
        scores_list = list(rerank_scores.values())
        s_min = min(scores_list)
        s_max = max(scores_list)

        if cited and cited in rerank_scores:
            raw = rerank_scores[cited]
        else:
            # No cited chunk → pessimistic: use min score
            raw = s_min
            logger.debug("s_ret: no cited_chunk or not in reranker scores, using min.")

        # Min-max normalise
        if s_max == s_min:
            return 1.0 if cited else 0.0
        return round((raw - s_min) / (s_max - s_min), 4)

    # ------------------------------------------------------------------
    # Option B: fallback — rank-based (position in rr.chunks list)
    # Chunk at position 0 gets score 1.0, last chunk gets score → 0.
    # ------------------------------------------------------------------
    chunk_ids = [c.chunk_id for c in rr.chunks]
    n = len(chunk_ids)

    if n == 0:
        return 0.5

    if cited and cited in chunk_ids:
        rank = chunk_ids.index(cited)
        return round(1.0 - rank / n, 4)

    # Atom's cited chunk not in retrieved set → weakest signal
    logger.debug("s_ret: cited_chunk '%s' not in retrieval result.", cited)
    return 0.0
