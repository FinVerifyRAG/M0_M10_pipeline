"""Paired bootstrap and McNemar tests for ablations."""
from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """McNemar's test on paired correctness. Returns chi-square and p-value."""
    if len(correct_a) != len(correct_b):
        raise ValueError("paired inputs must have the same length")
    b = c = 0
    for a, d in zip(correct_a, correct_b):
        if a and not d:
            b += 1
        elif d and not a:
            c += 1
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)  # continuity correction
    # Survival function of chi-square with 1 df: erfc(sqrt(x/2))
    p = math.erfc(math.sqrt(stat / 2.0))
    return {"b": b, "c": c, "statistic": stat, "p_value": p}


def paired_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """
    Bootstrap the mean paired difference scores_a - scores_b.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("paired inputs must have the same length")
    n = len(scores_a)
    if n == 0:
        return {"mean_diff": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_resamples):
        acc = 0.0
        for _i in range(n):
            j = rng.randrange(n)
            acc += scores_a[j] - scores_b[j]
        diffs.append(acc / n)
    diffs.sort()
    mean = sum(scores_a[i] - scores_b[i] for i in range(n)) / n
    lo_i = int((alpha / 2) * (n_resamples - 1))
    hi_i = int((1 - alpha / 2) * (n_resamples - 1))
    return {
        "mean_diff": mean,
        "ci_low": diffs[lo_i],
        "ci_high": diffs[hi_i],
        "n": n,
        "n_resamples": n_resamples,
    }
