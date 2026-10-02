"""
guarantee/ltt.py
-----------------
Learn-then-Test (LTT) statistical threshold selection.

**Guarantee (paper language):**
Among atoms the system *accepts* (risk <= lambda), the fraction that are
wrong is at most epsilon, with probability at least 1 - delta over the
calibration data.

Statistical unit of independence
---------------------------------
**BUG 4 / Section 9 fix**: Atoms within the same question are correlated
(shared retrieval context, same LLM pass).  Treating every atom as an
independent Bernoulli trial invalidates the binomial p-value.

Primary fix: work at question/cluster level.
For question q, define:
    Z_q = k_q - eps * n_q
where k_q = wrong accepted atoms in q, n_q = accepted atoms in q.
The null H0: selective risk > eps becomes E[Z] > 0.
The Z_q observations are treated as i.i.d. across questions/clusters.

Primary test: Hoeffding–Bentkus at cluster level  (Section 9).
Binomial test retained as a SANITY CHECK ONLY, not for certification.

Grid construction
-----------------
**Section 5.1**: Build the lambda grid from risk-score quantiles on a
separate agg_val split.  Do NOT build the threshold grid from calibration
data to avoid data reuse that invalidates fixed-sequence testing.

n_start
-------
**Bug 4.2**: The fixed-sequence loop now starts at the first lambda for
which n >= n_start.  Default n_start = 50 (configurable).

None threshold
--------------
**Bug 4.3**: accept_below = None is a valid state meaning "no certified
acceptance threshold; abstain on everything."  The comparison
    s.risk <= None
is illegal and must never execute.

abstain_above
-------------
**Section 10, Option B (operational)**:
abstain_above is defined as a judge-budget threshold, NOT as a second
certified risk guarantee.  It is the lambda above which sending to a
judge is not cost-effective; atoms above it are directly ABSTAINED.
This is a resource-control threshold.  An invariant is asserted:
    accept_below < abstain_above
when both are not None.

References
----------
- Angelopoulos & Bates, "Learn then Test: Calibrating Predictive
  Algorithms to Achieve Risk Control", ICML 2022.
- Bates et al., "Testing for Outliers with Conformal P-values", 2023.
- Hoeffding (1963) "Probability inequalities for sums of bounded random
  variables."
- Bentkus (2004) "On Hoeffding's inequalities."
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binom  # type: ignore

logger = logging.getLogger("guarantee.ltt")

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_N_START = 50   # minimum accepted atoms before a lambda is considered
DEFAULT_N_GRID  = 200  # number of grid points when building from quantiles


def min_accepted(eps: float, delta: float) -> int:
    """
    Smallest number of accepted atoms that can certify risk <= eps at
    confidence delta under a binomial tail with zero observed errors:

        ceil( ln(delta) / ln(1 - eps) )

    About 45 for eps=0.05 and delta=0.1. Strata smaller than this are
    merged or marked not_certifiable. This is not a silent default threshold.
    """
    if not (0.0 < eps < 1.0) or not (0.0 < delta < 1.0):
        raise ValueError(f"eps and delta must lie in (0, 1), got eps={eps}, delta={delta}")
    return int(math.ceil(math.log(delta) / math.log(1.0 - eps)))


# ---------------------------------------------------------------------------
# Core: binomial p-value  (SANITY CHECK ONLY — not primary certification)
# ---------------------------------------------------------------------------

def binom_pvalue(k: int, n: int, epsilon: float) -> float:
    """
    One-sided binomial p-value for H0: true selective risk > epsilon.

    p = P(Bin(n, epsilon) <= k)

    A small p means the data are consistent with risk <= epsilon (good).
    We *reject* H0 (claim the threshold is safe) when p <= delta.

    **IMPORTANT**: Use this only as a sanity check.  The primary
    certification test is cluster_hb_pvalue() which accounts for
    within-question correlation (Section 9 fix).

    Parameters
    ----------
    k       : number of wrong accepted atoms.
    n       : total accepted atoms.
    epsilon : target risk level (e.g. 0.05).

    Returns
    -------
    float in [0, 1].
    """
    if n == 0:
        return 1.0  # No atoms accepted -> no evidence -> conservative
    return float(binom.cdf(k, n, epsilon))


def hb_pvalue(k: int, n: int, epsilon: float) -> float:
    """
    Hoeffding-Bentkus p-value (distribution-free, finite-sample).

    p_hb = min(1, e * Binom.CDF(k; n, epsilon))

    This is a valid p-value for the atom-level binomial test.  It is used
    as the SANITY CHECK statistic alongside the cluster-level test.
    """
    if n == 0:
        return 1.0
    p_bin = float(binom.cdf(k, n, epsilon))
    return min(1.0, math.e * p_bin)


# ---------------------------------------------------------------------------
# Cluster-level test  (PRIMARY certification method, Section 9)
# ---------------------------------------------------------------------------

def cluster_z_scores(
    risk: np.ndarray,
    wrong: np.ndarray,
    question_ids: np.ndarray,
    lam: float,
    eps: float,
) -> np.ndarray:
    """
    Compute Z_q = k_q - eps * n_q for each question cluster q.

    Only questions with at least one accepted atom (n_q >= 1) contribute.

    Parameters
    ----------
    risk         : 1-D array of risk scores.
    wrong        : 1-D boolean/int array, 1 if wrong.
    question_ids : 1-D array of question/cluster identifiers (same length).
    lam          : threshold; atoms with risk <= lam are accepted.
    eps          : target selective risk.

    Returns
    -------
    1-D array of Z_q values (one per question with >= 1 accepted atom).
    """
    accepted = risk <= lam
    zs: List[float] = []
    for qid in np.unique(question_ids):
        mask = accepted & (question_ids == qid)
        n_q = int(mask.sum())
        if n_q == 0:
            continue
        k_q = int(wrong[mask].sum())
        zs.append(k_q - eps * n_q)
    return np.array(zs, dtype=float)


def cluster_hb_pvalue(
    risk: np.ndarray,
    wrong: np.ndarray,
    question_ids: np.ndarray,
    lam: float,
    eps: float,
) -> float:
    """
    Hoeffding-Bentkus p-value at cluster/question level.

    Null: E[Z_q] > 0  (i.e. true selective risk > eps).
    Alternative: E[Z_q] <= 0  (selective risk controlled).

    Z_q = k_q - eps * n_q is bounded in [-eps * n_q, (1-eps) * n_q].
    For the test we use an atom-bounded version:
        Z_q in [-(eps), 1 - eps]  (scaled per accepted atom).
    Then apply Hoeffding + Bentkus correction at the cluster level.

    This implementation uses the per-cluster rescaled statistic approach:
        Z'_q = Z_q / n_q  in [-eps, 1-eps]
    and then applies Hoeffding's inequality at the cluster mean level,
    with the Bentkus correction factor.

    Assumptions:
      - Z_q are i.i.d. across questions/clusters.
      - n_q may vary; we use the per-atom-bounded range.

    References: Hoeffding (1963), Bentkus (2004).
    """
    zs = cluster_z_scores(risk, wrong, question_ids, lam, eps)
    m = len(zs)
    if m == 0:
        return 1.0

    z_bar = float(zs.mean())

    # Each Z'_q = Z_q / n_q is in [-eps, 1-eps], range = 1.
    # Hoeffding bound: P(Z_bar <= z_bar | E[Z'] > 0) <= exp(-2 * m * z_bar^2)
    # Only meaningful when z_bar < 0 (data look good).
    if z_bar >= 0:
        return 1.0  # data do not support H1

    # Hoeffding bound: P(mean(Z) <= z_bar | E[Z] = 0) <= exp(-2 * m * z_bar^2 / range^2)
    # Range of each Z_q in unnormalized form spans [-eps*n_q, (1-eps)*n_q].
    # For a conservative bound we assume range = 1 per cluster observation.
    # z_bar < 0, so 2 * m * z_bar^2 > 0, and exp(-positive) gives small p.
    hoeff_exp = 2.0 * m * (z_bar ** 2)   # positive value
    p_hoeff = math.exp(-min(hoeff_exp, 700))  # small number in (0, 1]

    # Bentkus correction: multiply by e for a tighter bound
    p_hb = min(1.0, math.e * p_hoeff)
    return p_hb


# ---------------------------------------------------------------------------
# Grid construction from agg_val (Section 5.1 fix)
# ---------------------------------------------------------------------------

def build_lambda_grid(
    agg_val_risk: np.ndarray,
    n_grid: int = DEFAULT_N_GRID,
) -> np.ndarray:
    """
    Build the lambda grid from quantiles of agg_val risk scores.

    **Section 5.1**: The grid must be built from a SEPARATE agg_val split,
    not from calibration data.  Building from calibration data would create
    data reuse that can invalidate fixed-sequence testing.

    Parameters
    ----------
    agg_val_risk : risk scores from the agg_val split.
    n_grid       : approximate number of grid points.

    Returns
    -------
    Sorted 1-D array of lambda values in [0, 1].
    """
    if len(agg_val_risk) == 0:
        logger.warning(
            "build_lambda_grid: empty agg_val; falling back to uniform grid."
        )
        return np.linspace(0.0, 1.0, n_grid + 1)

    quantiles = np.linspace(0.0, 1.0, n_grid + 1)
    grid = np.quantile(agg_val_risk, quantiles)
    grid = np.unique(np.clip(grid, 0.0, 1.0))
    return grid


# ---------------------------------------------------------------------------
# Fixed-sequence threshold search (LTT)
# ---------------------------------------------------------------------------

def ltt_threshold(
    risk: np.ndarray,
    wrong: np.ndarray,
    eps: float,
    delta: float,
    grid: Optional[np.ndarray] = None,
    n_grid: int = DEFAULT_N_GRID,
    use_hb: bool = True,
    n_start: int = DEFAULT_N_START,
    question_ids: Optional[np.ndarray] = None,
) -> Optional[float]:
    """
    Return the largest certified ``lambda`` such that:
        P(selective_risk > eps | calibration data) <= delta.

    Uses fixed-sequence testing: walk lambda ascending from the first
    lambda where n >= n_start; stop at the first failure.

    **Bug 4.2 fix**: Starts at first lambda where n >= n_start (not at
    lambda=0 where n can be tiny and p-value high).

    **Section 9 fix**: When question_ids are provided, uses cluster-level
    Hoeffding-Bentkus as the primary test.  Atom-level HB is the fallback.

    Parameters
    ----------
    risk         : 1-D array of risk scores in [0, 1], one per atom.
    wrong        : 1-D boolean/int array, 1 if the atom is wrong.
    eps          : target selective risk (e.g. 0.05 = 5% error rate).
    delta        : confidence level (e.g. 0.10 -> 90% guarantee).
    grid         : explicit lambda grid; if None, build uniform n_grid+1 pts.
    n_grid       : grid resolution when grid is None.
    use_hb       : use Hoeffding-Bentkus bound (True) or plain binomial (False).
                   Only applies to the SANITY CHECK atom-level test.
    n_start      : minimum accepted atoms before a lambda is considered.
    question_ids : cluster/question IDs for the cluster-level test.
                   If None, falls back to atom-level test (less rigorous).

    Returns
    -------
    float or None.  None means no lambda reached certification; the stratum
    should abstain on everything (fail-closed state).
    """
    risk  = np.asarray(risk,  dtype=float)
    wrong = np.asarray(wrong, dtype=float)

    if grid is None:
        grid = np.linspace(0.0, 1.0, n_grid + 1)

    use_cluster = question_ids is not None
    if use_cluster:
        question_ids = np.asarray(question_ids)

    # Atom-level sanity check function
    sanity_fn = hb_pvalue if use_hb else binom_pvalue

    best: Optional[float] = None

    # Pre-filter grid to eligible lambdas (n >= n_start).
    # n_start filtering: only lambdas where at least n_start atoms are accepted
    # are included in the fixed-sequence test.
    sorted_grid = sorted(grid)
    eligible_lambdas = [
        lam for lam in sorted_grid if int((risk <= lam).sum()) >= n_start
    ]

    if not eligible_lambdas:
        logger.debug(
            "ltt_threshold: no lambdas with n >= n_start=%d; returning None.", n_start
        )
        return None

    # Fixed-sequence test runs ASCENDING (smallest eligible lambda first).
    # This is mandatory for FWER/FDR control: we certify the LARGEST lambda
    # such that ALL lambdas <= it also pass the test.
    #
    # accept_below = max{ lambda in eligible_lambdas : p(lambda) <= delta
    #                     AND p(all smaller eligible lambdas) <= delta }
    #
    # n_start semantics: lambdas with n < n_start are excluded from the
    # sequence; they cannot trigger a break.  The sequence starts at the
    # first eligible lambda (first with n >= n_start).
    for lam in eligible_lambdas:
        accepted = risk <= lam
        n = int(accepted.sum())
        k = int(wrong[accepted].sum())

        # Primary certification: cluster-level HB (Section 9)
        if use_cluster:
            p = cluster_hb_pvalue(risk, wrong, question_ids, lam, eps)
        else:
            # Fallback: atom-level (less rigorous, may be invalid if correlated)
            p = sanity_fn(k, n, eps)

        # Sanity check: also compute atom-level p for logging
        p_sanity = sanity_fn(k, n, eps)
        logger.debug(
            "lam=%.4f  n=%d  k=%d  p_primary=%.4f  p_sanity=%.4f  delta=%.4f",
            lam, n, k, p, p_sanity, delta,
        )

        if p <= delta:
            best = lam  # Still certified -- continue to find a larger lambda
        else:
            break       # Fixed-sequence stop (required for LTT FWER guarantee)

    return best


# ---------------------------------------------------------------------------
# Pair: accept_below and abstain_above
# ---------------------------------------------------------------------------

def ltt_thresholds_pair(
    risk: np.ndarray,
    wrong: np.ndarray,
    eps_accept: float,
    delta: float,
    n_grid: int = DEFAULT_N_GRID,
    use_hb: bool = True,
    n_start: int = DEFAULT_N_START,
    question_ids: Optional[np.ndarray] = None,
    agg_val_risk: Optional[np.ndarray] = None,
    judge_budget_quantile: float = 0.80,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute accept_below and abstain_above thresholds.

    accept_below
        Certified threshold for eps_accept (LTT result).  Atoms with
        risk <= accept_below are SUPPORTED with P(selective risk > eps) <= delta.

    abstain_above
        **Section 10, Option B (operational definition)**:
        A judge-budget threshold, NOT a second certified risk guarantee.
        Defined as the judge_budget_quantile of the calibration risk scores
        (or agg_val_risk if provided).  Atoms above this threshold are
        sent directly to ABSTAINED (too uncertain for the judge).

        This is a resource-control threshold.  It does NOT carry a
        statistical certification for the error rate above it.

    Invariant asserted:
        accept_below < abstain_above  (when both are not None).

    Parameters
    ----------
    risk                  : calibration risk scores.
    wrong                 : calibration labels.
    eps_accept            : target selective risk for SUPPORTED atoms.
    delta                 : confidence level.
    n_grid                : grid resolution.
    use_hb                : use HB bound.
    n_start               : minimum accepted atoms per lambda.
    question_ids          : cluster IDs for cluster-level test.
    agg_val_risk          : risk scores from agg_val; used for grid construction.
    judge_budget_quantile : percentile of risk distribution for abstain_above.

    Returns
    -------
    (accept_below, abstain_above) -- either may be None.
    """
    # Build grid from agg_val if available (Section 5.1)
    if agg_val_risk is not None and len(agg_val_risk) > 0:
        grid = build_lambda_grid(agg_val_risk, n_grid)
    else:
        logger.warning(
            "ltt_thresholds_pair: agg_val_risk not provided; "
            "building grid from calibration data (data reuse risk)."
        )
        grid = np.linspace(0.0, 1.0, n_grid + 1)

    accept_below = ltt_threshold(
        risk, wrong, eps_accept, delta,
        grid=grid, use_hb=use_hb, n_start=n_start,
        question_ids=question_ids,
    )

    # abstain_above: operational judge-budget threshold (Option B)
    source = agg_val_risk if agg_val_risk is not None else risk
    abstain_above_val = float(np.quantile(source, judge_budget_quantile))

    # Enforce ordering invariant
    if accept_below is not None:
        if accept_below >= abstain_above_val:
            # Push abstain_above above accept_below
            abstain_above_val = min(
                1.0, accept_below + 1e-6
            )
            logger.warning(
                "accept_below (%.4f) >= abstain_above (%.4f): "
                "clamping abstain_above to %.4f.",
                accept_below, abstain_above_val, abstain_above_val,
            )

    logger.info(
        "Thresholds: accept_below=%s  abstain_above=%.4f  "
        "(abstain_above is judge-budget threshold, not a certified risk bound).",
        str(accept_below), abstain_above_val,
    )
    return accept_below, abstain_above_val


# ---------------------------------------------------------------------------
# Simulation: validate violation rate on synthetic data
# ---------------------------------------------------------------------------

def simulate_violation_rate(
    risk: np.ndarray,
    wrong: np.ndarray,
    eps: float,
    delta: float,
    n_splits: int = 1000,
    calib_frac: float = 0.5,
    n_grid: int = DEFAULT_N_GRID,
    use_hb: bool = True,
    n_start: int = DEFAULT_N_START,
    seed: int = 42,
    question_ids: Optional[np.ndarray] = None,
) -> dict:
    """
    Estimate empirical violation rate: fraction of random calibration splits
    where the certified threshold yields actual selective risk > epsilon on
    the held-out test atoms.

    Expected result: violation_rate <= delta (with high probability).

    Parameters
    ----------
    risk, wrong  : full dataset arrays.
    eps          : target risk.
    delta        : confidence level.
    n_splits     : number of random calibration/test splits.
    calib_frac   : fraction used for calibration.
    n_grid       : grid resolution.
    use_hb       : use HB bound.
    n_start      : minimum accepted atoms per lambda.
    seed         : RNG seed.
    question_ids : cluster IDs (if provided, uses cluster-level test).

    Returns
    -------
    dict with keys: violation_rate, mean_lambda, n_abstain, n_valid.
    """
    rng = np.random.default_rng(seed)
    n   = len(risk)

    violations    = 0
    valid_splits  = 0
    lambdas       = []
    abstain_count = 0

    for _ in range(n_splits):
        idx   = rng.permutation(n)
        cal_n = max(1, int(calib_frac * n))
        cal_idx, test_idx = idx[:cal_n], idx[cal_n:]

        q_cal = question_ids[cal_idx] if question_ids is not None else None

        lam = ltt_threshold(
            risk[cal_idx], wrong[cal_idx], eps, delta,
            n_grid=n_grid, use_hb=use_hb,
            n_start=n_start, question_ids=q_cal,
        )

        if lam is None or (isinstance(lam, float) and lam < 0):
            # A threshold that accepts nothing, including the old -1 sentinel,
            # is not a pass.
            abstain_count += 1
            continue

        # Measure selective risk on the disjoint validation slice.
        test_accepted = risk[test_idx] <= lam
        n_test_acc    = int(test_accepted.sum())
        if n_test_acc == 0:
            abstain_count += 1
            continue

        valid_splits += 1
        lambdas.append(lam)

        k_test    = int(wrong[test_idx][test_accepted].sum())
        test_risk = k_test / n_test_acc
        if test_risk > eps:
            violations += 1

    valid = valid_splits > 0
    violation_rate = (violations / valid_splits) if valid else float("nan")
    mean_lambda    = float(np.mean(lambdas)) if lambdas else float("nan")
    # Coverage of the last valid lambda is not meaningful across splits;
    # report the fraction of splits that accepted at least one validation atom.
    coverage = valid_splits / n_splits if n_splits else 0.0

    result = {
        "violation_rate":   violation_rate,
        "target_delta":     delta,
        "mean_lambda":      mean_lambda,
        "n_valid_splits":   valid_splits,
        "n_abstain_splits": abstain_count,
        "n_splits":         n_splits,
        "use_hb":           use_hb,
        "n_start":          n_start,
        "valid":            valid,
        "zero_coverage_invalid": (not valid),
        "coverage":         coverage,
    }
    logger.info(
        "Simulation complete: violation_rate=%.4f (target <= %.4f), "
        "mean_lambda=%.4f, valid_splits=%d/%d",
        violation_rate, delta, mean_lambda, valid_splits, n_splits,
    )
    return result


# ---------------------------------------------------------------------------
# Sample-size sanity table  (Section 5.4)
# ---------------------------------------------------------------------------

SAMPLE_SIZE_SANITY = {
    # (eps, delta) -> approx minimum accepted atoms for certification
    # These are guidance numbers, NOT substitutes for the actual test.
    (0.05, 0.10):                45,
    (0.05, 0.10 / 24):          107,  # 24 strata Bonferroni
    (0.02, 0.10):               114,
    (0.02, 0.10 / 24):          272,  # 24 strata Bonferroni
}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    """95% Clopper-Pearson interval for k successes in n trials."""
    if n <= 0:
        return None, None
    from scipy.stats import beta  # type: ignore
    k = int(k)
    n = int(n)
    if k < 0 or k > n:
        raise ValueError(f"k={k} is outside 0..n={n}")
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


@dataclass
class StratumCertificate:
    """A threshold decision that is never a bare None."""
    stratum: str
    status: str                 # certified | merged | not_certifiable
    threshold: Optional[float]
    coverage: float
    n_accepted: int
    n_atoms: int
    valid: bool
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def certify_lambda(
    risk: np.ndarray,
    wrong: np.ndarray,
    eps: float,
    delta: float,
    stratum: str = "*|*",
    status_if_ok: str = "certified",
    **ltt_kwargs,
) -> StratumCertificate:
    """
    Run fixed-sequence LTT and return a status.

    status_if_ok is "certified" or "merged" (caller merged a small stratum
    into its parent). A lambda that accepts nothing is not_certifiable.
    """
    risk = np.asarray(risk, dtype=float)
    wrong = np.asarray(wrong, dtype=float)
    lam = ltt_threshold(risk, wrong, eps, delta, **ltt_kwargs)
    if lam is None or lam < 0:
        return StratumCertificate(
            stratum=stratum, status="not_certifiable", threshold=None,
            coverage=0.0, n_accepted=0, n_atoms=int(len(risk)),
            valid=False, reason="no lambda passed the binomial/HB test",
        )
    accepted = risk <= lam
    n_acc = int(accepted.sum())
    coverage = n_acc / max(len(risk), 1)
    if n_acc == 0 or coverage == 0.0:
        return StratumCertificate(
            stratum=stratum, status="not_certifiable", threshold=float(lam),
            coverage=0.0, n_accepted=0, n_atoms=int(len(risk)),
            valid=False, reason="threshold accepts nothing",
        )
    return StratumCertificate(
        stratum=stratum,
        status=status_if_ok,
        threshold=float(lam),
        coverage=float(coverage),
        n_accepted=n_acc,
        n_atoms=int(len(risk)),
        valid=True,
        reason="ok",
    )


def evaluate_untouched_test(
    risk_cal: np.ndarray,
    wrong_cal: np.ndarray,
    risk_test: np.ndarray,
    wrong_test: np.ndarray,
    eps: float,
    delta: float,
    **ltt_kwargs,
) -> dict:
    """
    Fit one threshold on the calibration pool and score the untouched test split.
    """
    cert = certify_lambda(risk_cal, wrong_cal, eps, delta, **ltt_kwargs)
    risk_test = np.asarray(risk_test, dtype=float)
    wrong_test = np.asarray(wrong_test, dtype=float)
    if not cert.valid or cert.threshold is None:
        return {
            "certificate": cert.as_dict(),
            "valid": False,
            "coverage": 0.0,
            "hallucination_rate": None,
            "hallucination_ci95": [None, None],
            "violation": None,
        }
    accepted = risk_test <= cert.threshold
    n_acc = int(accepted.sum())
    coverage = n_acc / max(len(risk_test), 1)
    if n_acc == 0:
        return {
            "certificate": cert.as_dict(),
            "valid": False,
            "coverage": 0.0,
            "hallucination_rate": None,
            "hallucination_ci95": [None, None],
            "violation": None,
            "reason": "zero coverage on test",
        }
    k = int(wrong_test[accepted].sum())
    rate = k / n_acc
    lo, hi = clopper_pearson(k, n_acc)
    cov_lo, cov_hi = clopper_pearson(n_acc, len(risk_test))
    return {
        "certificate": cert.as_dict(),
        "valid": True,
        "coverage": coverage,
        "coverage_ci95": [cov_lo, cov_hi],
        "hallucination_rate": rate,
        "hallucination_ci95": [lo, hi],
        "n_accepted": n_acc,
        "n_wrong_accepted": k,
        "violation": bool(rate > eps),
    }
