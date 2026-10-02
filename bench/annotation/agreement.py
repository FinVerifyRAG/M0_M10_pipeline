"""
bench/annotation/agreement.py
-------------------------------
M10 Benchmark: Inter-annotator agreement for double-annotated atoms.

Implements Cohen's kappa for the 4-way label set
{supported, unsupported, outdated, uncertain}.

Public API
----------
    from bench.annotation.agreement import cohen_kappa, agreement_stats

    kappa = cohen_kappa(labels_a, labels_b)
    stats = agreement_stats(records_double_annotated)
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("bench.annotation.agreement")

LABEL_ORDER = ["supported", "unsupported", "outdated", "uncertain"]


def confusion_matrix(
    labels_a: List[str],
    labels_b: List[str],
) -> np.ndarray:
    """Build a confusion matrix for two annotators."""
    n = len(LABEL_ORDER)
    idx = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
    mat = np.zeros((n, n), dtype=float)
    for a, b in zip(labels_a, labels_b):
        ia = idx.get(a, -1)
        ib = idx.get(b, -1)
        if ia >= 0 and ib >= 0:
            mat[ia, ib] += 1
    return mat


def cohen_kappa(
    labels_a: List[str],
    labels_b: List[str],
) -> float:
    """
    Compute Cohen's kappa between two annotators.

    Parameters
    ----------
    labels_a, labels_b : Parallel lists of label strings.

    Returns
    -------
    kappa : float in [-1, 1]. kappa >= 0.6 is considered acceptable agreement.
            Returns 0.0 if inputs are empty or degenerate.
    """
    if not labels_a or len(labels_a) != len(labels_b):
        logger.warning("cohen_kappa: mismatched or empty label lists.")
        return 0.0

    mat = confusion_matrix(labels_a, labels_b)
    n   = mat.sum()
    if n == 0:
        return 0.0

    po = mat.diagonal().sum() / n                         # Observed agreement

    row_sums = mat.sum(axis=1) / n
    col_sums = mat.sum(axis=0) / n
    pe = float(np.dot(row_sums, col_sums))                # Expected agreement

    if abs(1.0 - pe) < 1e-9:
        return 1.0 if abs(po - pe) < 1e-9 else 0.0

    return round((po - pe) / (1.0 - pe), 4)


def percent_agreement(labels_a: List[str], labels_b: List[str]) -> float:
    """Raw percent agreement (0–1)."""
    if not labels_a:
        return 0.0
    return round(sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a), 4)


def agreement_stats(
    paired_labels: List[Tuple[str, str]],
) -> dict:
    """
    Compute agreement statistics for a list of (label_a, label_b) pairs.

    Returns
    -------
    dict with keys: kappa, percent_agreement, n_pairs, n_agree, confusion_matrix
    """
    if not paired_labels:
        return {"kappa": 0.0, "percent_agreement": 0.0, "n_pairs": 0}

    labels_a = [p[0] for p in paired_labels]
    labels_b = [p[1] for p in paired_labels]

    kappa  = cohen_kappa(labels_a, labels_b)
    pct    = percent_agreement(labels_a, labels_b)
    mat    = confusion_matrix(labels_a, labels_b)
    n_agree = int(mat.diagonal().sum())

    return {
        "kappa":              kappa,
        "percent_agreement":  pct,
        "n_pairs":            len(paired_labels),
        "n_agree":            n_agree,
        "kappa_interpretation": _interpret_kappa(kappa),
        "confusion_matrix": {
            "labels": LABEL_ORDER,
            "values": mat.tolist(),
        },
    }


def _interpret_kappa(kappa: float) -> str:
    if kappa < 0:       return "poor (below chance)"
    if kappa < 0.20:    return "slight"
    if kappa < 0.40:    return "fair"
    if kappa < 0.60:    return "moderate"
    if kappa < 0.80:    return "substantial"
    return "almost perfect"
