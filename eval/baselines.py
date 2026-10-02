"""
eval/baselines.py
------------------
M10 Evaluation: Baseline systems for comparison against RegGuard.

Baselines (as per implementation plan)
---------------------------------------
1. plain_rag         — No verification; every atom is "SUPPORTED".
2. selfcheck         — Self-consistency / SelfCheckGPT proxy:
                       Uses V2 NLI entail prob as the only signal;
                       accepts if entail_prob >= 0.5.
3. nli_only          — NLI verifier without cascade/guarantee:
                       No thresholds, just NLI entail prob thresholded at 0.5.
4. llm_judge_all     — Call the judge on EVERY atom (most expensive).
5. conformal_no_mondrian — Split-conformal with a global threshold (no
                            Mondrian stratification, no drift).

Each baseline implements the same interface:
    def run(scored_atoms, labels, thresholds=None, judge=None, rr=None)
        -> Dict[atom_id -> decision_str]

Public API
----------
    from eval.baselines import (
        plain_rag, selfcheck, nli_only,
        llm_judge_all, conformal_no_mondrian,
        run_baseline, BASELINE_REGISTRY,
    )
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger("eval.baselines")

# ── Decision constants (mirror decision.router) ───────────────────────────────
SUPPORTED    = "SUPPORTED"
ABSTAINED    = "ABSTAINED"
UNCERTAIN    = "UNCERTAIN"
VERIFIED     = "VERIFIED"
NOT_VERIFIED = "NOT_VERIFIED"


# ══════════════════════════════════════════════════════════════════════════════
# Baseline 1: Plain RAG (no verification)
# ══════════════════════════════════════════════════════════════════════════════

def plain_rag(
    scored_atoms: list,
    **_kwargs,
) -> Dict[str, str]:
    """
    Baseline 1: Accept every atom unconditionally.
    Represents a plain RAG system with no factual verification.
    """
    return {sa.verified.atom.atom_id: SUPPORTED for sa in scored_atoms}


# ══════════════════════════════════════════════════════════════════════════════
# Baseline 2: Self-consistency / SelfCheckGPT proxy
# ══════════════════════════════════════════════════════════════════════════════

def selfcheck(
    scored_atoms: list,
    entail_threshold: float = 0.50,
    **_kwargs,
) -> Dict[str, str]:
    """
    Baseline 2: Use V2 NLI entail prob as the single signal.

    If v2_entail_prob >= entail_threshold -> SUPPORTED.
    If v2_entail_prob is None (NLI skipped) -> fallback to ABSTAINED.
    Else -> ABSTAINED.

    This approximates SelfCheckGPT's consistency-based filtering.
    """
    results = {}
    for sa in scored_atoms:
        atom_id  = sa.verified.atom.atom_id
        p_entail = sa.verified.v2_entail_prob
        if p_entail is None:
            results[atom_id] = ABSTAINED
        elif p_entail >= entail_threshold:
            results[atom_id] = SUPPORTED
        else:
            results[atom_id] = ABSTAINED
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Baseline 3: NLI-only verifier (no cascade, no guarantee)
# ══════════════════════════════════════════════════════════════════════════════

def nli_only(
    scored_atoms: list,
    entail_threshold: float = 0.50,
    **_kwargs,
) -> Dict[str, str]:
    """
    Baseline 3: NLI-only verifier without the V1 cascade or LTT guarantee.

    Unlike selfcheck, also uses v1_status=MATCH as a fallback when NLI is absent.
    No risk thresholds are applied — purely NLI-based.
    """
    results = {}
    for sa in scored_atoms:
        atom_id  = sa.verified.atom.atom_id
        p_entail = sa.verified.v2_entail_prob
        v1       = sa.verified.v1_status

        if p_entail is not None:
            results[atom_id] = SUPPORTED if p_entail >= entail_threshold else ABSTAINED
        elif v1 == "MATCH":
            results[atom_id] = SUPPORTED
        elif v1 == "MISMATCH":
            results[atom_id] = ABSTAINED
        else:
            results[atom_id] = ABSTAINED
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Baseline 4: LLM-as-judge on every atom
# ══════════════════════════════════════════════════════════════════════════════

def llm_judge_all(
    scored_atoms: list,
    judge=None,
    rr=None,
    **_kwargs,
) -> Dict[str, str]:
    """
    Baseline 4: Call the V3 judge on EVERY atom (most expensive, highest quality).

    If judge is None (unavailable), falls back to plain_rag.
    """
    if judge is None:
        logger.warning("llm_judge_all: judge not provided; falling back to plain_rag.")
        return plain_rag(scored_atoms)

    results = {}
    try:
        judge.reset_call_counter()
        judge_results = judge.judge_batch(scored_atoms, rr)
        for atom_id, jr in judge_results.items():
            from judge.schemas import VERIFIED as JV
            results[atom_id] = VERIFIED if jr.verdict == JV else NOT_VERIFIED
        # Fill any atoms the judge missed (call cap)
        for sa in scored_atoms:
            aid = sa.verified.atom.atom_id
            if aid not in results:
                results[aid] = ABSTAINED
    except Exception as exc:
        logger.warning("llm_judge_all failed (%s); falling back to plain_rag.", exc)
        return plain_rag(scored_atoms)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Baseline 5: Split-conformal without Mondrian or drift
# ══════════════════════════════════════════════════════════════════════════════

def conformal_no_mondrian(
    scored_atoms:  list,
    global_threshold: float = 0.5,
    **_kwargs,
) -> Dict[str, str]:
    """
    Baseline 5: Global conformal threshold (no Mondrian, no drift adaptation).

    Accept all atoms with risk <= global_threshold.
    Represents a naïve split-conformal approach without stratification.
    """
    results = {}
    for sa in scored_atoms:
        atom_id = sa.verified.atom.atom_id
        results[atom_id] = SUPPORTED if sa.risk <= global_threshold else ABSTAINED
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Registry and runner
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_REGISTRY: Dict[str, Callable] = {
    "plain_rag":              plain_rag,
    "selfcheck":              selfcheck,
    "nli_only":               nli_only,
    "llm_judge_all":          llm_judge_all,
    "conformal_no_mondrian":  conformal_no_mondrian,
}


def run_baseline(
    name:          str,
    scored_atoms:  list,
    labels:        Optional[List[str]] = None,
    judge=None,
    rr=None,
    **kwargs,
) -> Dict[str, str]:
    """
    Run a named baseline and return {atom_id -> decision} dict.

    Parameters
    ----------
    name         : Key in BASELINE_REGISTRY.
    scored_atoms : List[ScoredAtom] from M6.
    labels       : Optional ground-truth labels (not used by baselines,
                   but passed through for callers that want them).
    judge        : Optional JudgeLLM for llm_judge_all.
    rr           : Optional RetrievalResult for llm_judge_all.

    Returns
    -------
    Dict[atom_id -> decision_str].
    """
    fn = BASELINE_REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"Unknown baseline {name!r}. Choose from: {list(BASELINE_REGISTRY)}")
    return fn(scored_atoms, judge=judge, rr=rr, **kwargs)
