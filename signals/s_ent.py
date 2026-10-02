"""
signals/s_ent.py  —  Model Uncertainty Signal (Token Entropy)
--------------------------------------------------------------
s_ent measures how uncertain the LLM was when generating the atom's tokens,
using the token-level log-probabilities stored in GeneratedAnswer.token_logprobs.

High entropy → model was uncertain → atom may be hallucinated → higher risk.
Low entropy  → model was confident → atom likely grounded   → lower risk.

Computation
-----------
Given a per-token logprob list (log p_i for each token i), we compute:
    token entropy_i = -log(p_i)  (= -lp_i, since lp_i ≤ 0)

Then for the atom's token span (approximated from character offsets):
    s_ent_mean = mean(-lp_i)  over span
    s_ent_max  = max(-lp_i)   over span

Both are normalised to [0, 1] by clipping to a max entropy of
`max_entropy_clip` (default 5.0, corresponding to ~150-way uncertainty).

If no logprobs are available, the signal is missing (NaN) and
s_ent_skipped is 1. The value 0.5 is not written in its place.

Returned features
-----------------
    s_ent      : float — mean entropy (primary feature for aggregator)
    s_ent_max  : float — max entropy over span
    s_ent_skipped: 0/1 — flag when logprobs were unavailable
"""
from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple

from common.schemas import VerifiedAtom, GeneratedAnswer

logger = logging.getLogger("signals.s_ent")

_MAX_CLIP = 5.0         # max -log(p) we normalise against (p ≈ 0.67%)
_CHARS_PER_TOKEN = 4    # rough heuristic for char→token span mapping


def _token_span(char_span: Optional[Tuple[int, int]], answer_text: str) -> Tuple[int, int]:
    """
    Convert a character span to a rough token index span.
    Uses the heuristic that ~4 characters correspond to 1 token.
    """
    if char_span is None:
        return (0, max(1, len(answer_text) // _CHARS_PER_TOKEN))
    start = max(0, char_span[0] // _CHARS_PER_TOKEN)
    end   = max(start + 1, char_span[1] // _CHARS_PER_TOKEN)
    return (start, end)


def _entropy_stats(
    logprobs: List[float],
    tok_start: int,
    tok_end: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (mean_entropy, max_entropy) over [tok_start, tok_end].
    Entropy = -log_prob (since logprobs are natural log or log2 probabilities).
    Returns (None, None) if span is empty or out of range.
    """
    span = logprobs[tok_start: tok_end]
    if not span:
        return None, None
    entropies = [-lp for lp in span]   # -log(p) ≥ 0
    return sum(entropies) / len(entropies), max(entropies)


def compute_s_ent(
    verified_atom: VerifiedAtom,
    answer: GeneratedAnswer,
    max_clip: float = _MAX_CLIP,
) -> float:
    """
    Return the mean-entropy signal for one atom, normalised to [0, 1].
    Returns NaN if logprobs are unavailable. The skip flag carries that fact.
    """
    feats = s_ent_features(verified_atom, answer, max_clip=max_clip)
    return feats["s_ent"]


def s_ent_features(
    verified_atom: VerifiedAtom,
    answer: GeneratedAnswer,
    max_clip: float = _MAX_CLIP,
) -> dict:
    """
    Return:
        {
          "s_ent":          float — normalised mean entropy over atom span,
          "s_ent_max":      float — normalised max entropy over span,
          "s_ent_skipped":  0/1  — flag when logprobs unavailable,
        }
    """
    if not answer.token_logprobs:
        logger.debug("s_ent: no token_logprobs; marking missing.")
        return {"s_ent": float("nan"), "s_ent_max": float("nan"), "s_ent_skipped": 1}

    tok_start, tok_end = _token_span(verified_atom.atom.span, answer.answer_text)
    mean_ent, max_ent = _entropy_stats(answer.token_logprobs, tok_start, tok_end)

    if mean_ent is None:
        return {"s_ent": float("nan"), "s_ent_max": float("nan"), "s_ent_skipped": 1}

    return {
        "s_ent":         round(min(1.0, mean_ent / max_clip), 4),
        "s_ent_max":     round(min(1.0, max_ent  / max_clip), 4),
        "s_ent_skipped": 0,
    }
