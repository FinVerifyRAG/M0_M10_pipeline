"""
guarantee/drift.py
-------------------
Drift detectors and drift-aware calibration strategies for M7.

The key insight: calibrating once on static historical data is fragile when
regulations change (new rates, superseded sections, renumbered clauses).
This module provides:

  1. **Drift detectors** — signal when the risk score distribution or the
     error rate has shifted.
  2. **Calibration strategies** — how to recalibrate after drift is detected.

Drift types modelled (from the implementation plan)
----------------------------------------------------
- Value drift      : a rate, threshold, or amount changed.
- Identifier drift : a section was renumbered.
- Supersession drift: an entire regulation was superseded.
- Addition drift   : a new regulation was added.

Detectors
---------
- KS test         : compare risk-score distributions (calibration vs. recent).
- MMD (squared)   : Maximum Mean Discrepancy on risk scores.
- Version-graph   : structural signal — fires when any amendment/supersession
                    event affects sections relevant to the stratum.

Calibration strategies
----------------------
| Strategy        | Description                                           |
|-----------------|-------------------------------------------------------|
| Static          | Calibrate once; never recalibrate (baseline).        |
| Sliding window  | Use only the most recent N months of calibration data.|
| Recency-weighted| Weight atoms by recency (exponential decay).         |
| Triggered       | Recalibrate when a drift detector fires.              |

Public API
----------
    from guarantee.drift import (
        ks_drift_score,
        mmd_drift_score,
        version_graph_trigger,
        sliding_window_filter,
        recency_weights,
        DriftMonitor,
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import ks_2samp  # type: ignore

logger = logging.getLogger("guarantee.drift")


# ---------------------------------------------------------------------------
# Drift detectors
# ---------------------------------------------------------------------------

def ks_drift_score(
    reference: np.ndarray,
    recent:    np.ndarray,
) -> Tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test on risk score distributions.

    Compares ``reference`` (calibration window) against ``recent`` (new traffic).

    Returns
    -------
    (statistic, p_value) — large statistic and small p-value indicate drift.
    """
    if len(reference) < 5 or len(recent) < 5:
        logger.warning("KS test: too few samples (ref=%d, rec=%d)", len(reference), len(recent))
        return 0.0, 1.0

    stat, pval = ks_2samp(reference, recent)
    logger.debug("KS drift: stat=%.4f, p=%.4f", stat, pval)
    return float(stat), float(pval)


def mmd_drift_score(
    reference: np.ndarray,
    recent:    np.ndarray,
    gamma:     float = 1.0,
) -> float:
    """
    Squared Maximum Mean Discrepancy (MMD) with an RBF kernel.

    MMD^2 = E[k(X,X')] - 2*E[k(X,Y)] + E[k(Y,Y')]
    where k(x,y) = exp(-gamma * (x - y)^2).

    For scalar risk scores this reduces to a 1-D kernel comparison.
    A larger MMD^2 indicates greater distributional shift.

    Parameters
    ----------
    reference : risk scores from the calibration window.
    recent    : risk scores from recent traffic.
    gamma     : RBF kernel bandwidth (default 1.0).

    Returns
    -------
    float >= 0.  Positive values indicate drift.
    """
    if len(reference) == 0 or len(recent) == 0:
        return 0.0

    ref = reference.reshape(-1, 1).astype(float)
    rec = recent.reshape(-1, 1).astype(float)

    def rbf_mean(a: np.ndarray, b: np.ndarray) -> float:
        diff = a[:, None, :] - b[None, :, :]          # (n, m, 1)
        return float(np.exp(-gamma * (diff ** 2)).mean())

    mmd2 = rbf_mean(ref, ref) - 2 * rbf_mean(ref, rec) + rbf_mean(rec, rec)
    mmd2 = max(0.0, mmd2)
    logger.debug("MMD^2 drift: %.6f", mmd2)
    return mmd2


def version_graph_trigger(
    stratum:          str,
    amendment_events: List[Dict],
    since_date:       str,
) -> bool:
    """
    Returns True if any amendment/supersession event in ``amendment_events``
    affects the given stratum (matching regulator or atom type) after
    ``since_date``.

    Parameters
    ----------
    stratum          : e.g. "SEBI|RATE".
    amendment_events : list of dicts with keys: date, regulator, sections, type.
                       type is one of: amends, supersedes, renumbers, adds.
    since_date       : ISO date string "YYYY-MM-DD".

    Returns
    -------
    bool.  True = trigger recalibration.
    """
    reg, atype = (stratum.split("|", 1) + ["*"])[:2]
    try:
        since_dt = datetime.strptime(since_date, "%Y-%m-%d")
    except ValueError:
        since_dt = datetime.min

    for evt in amendment_events:
        evt_date = evt.get("date", "")
        try:
            evt_dt = datetime.strptime(evt_date, "%Y-%m-%d")
        except ValueError:
            continue

        if evt_dt < since_dt:
            continue

        evt_reg = evt.get("regulator", "*").upper()
        if reg != "*" and evt_reg != reg:
            continue

        logger.info(
            "Version-graph trigger: stratum=%s, event_type=%s, event_date=%s",
            stratum, evt.get("type"), evt_date,
        )
        return True

    return False


# ---------------------------------------------------------------------------
# Sliding-window filter
# ---------------------------------------------------------------------------

def sliding_window_filter(
    dates:         List[str],
    risk:          np.ndarray,
    wrong:         np.ndarray,
    window_months: int,
    reference_date: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep only calibration atoms from the most recent ``window_months`` months.

    Parameters
    ----------
    dates          : ISO date string per atom ("YYYY-MM-DD" or "YYYY-MM").
    risk, wrong    : aligned arrays.
    window_months  : number of months to keep.
    reference_date : cutoff date (defaults to today).

    Returns
    -------
    (risk_filtered, wrong_filtered) — arrays for atoms within the window.
    """
    if reference_date is None:
        ref = datetime.now().replace(tzinfo=None)
    else:
        ref = datetime.strptime(reference_date[:10], "%Y-%m-%d")

    cutoff = ref - timedelta(days=window_months * 30)

    mask = []
    for d in dates:
        try:
            dt = datetime.strptime(d[:10], "%Y-%m-%d")
            mask.append(dt >= cutoff)
        except ValueError:
            mask.append(True)  # Keep atoms with unparseable dates

    mask_arr = np.array(mask, dtype=bool)
    return np.asarray(risk)[mask_arr], np.asarray(wrong)[mask_arr]


# ---------------------------------------------------------------------------
# Recency-weighted calibration
# ---------------------------------------------------------------------------

def recency_weights(
    dates:          List[str],
    half_life_days: float = 180.0,
    reference_date: Optional[str] = None,
) -> np.ndarray:
    """
    Compute exponential recency weights for calibration atoms.

    Weight = exp(-ln(2) * age_days / half_life_days)

    Newer atoms receive higher weight; atoms older than ``half_life_days``
    receive weight ~0.5.

    Parameters
    ----------
    dates         : ISO date strings per atom.
    half_life_days: half-life for exponential decay (default 6 months).
    reference_date: reference date (defaults to today).

    Returns
    -------
    np.ndarray of weights, shape (N,), all positive.
    """
    if reference_date is None:
        ref = datetime.now().replace(tzinfo=None)
    else:
        ref = datetime.strptime(reference_date[:10], "%Y-%m-%d")

    weights = []
    ln2 = np.log(2.0)
    for d in dates:
        try:
            dt  = datetime.strptime(d[:10], "%Y-%m-%d")
            age = max(0.0, (ref - dt).days)
        except ValueError:
            age = 0.0
        w = np.exp(-ln2 * age / max(half_life_days, 1.0))
        weights.append(float(w))

    return np.array(weights, dtype=float)


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Kish's effective sample size: n_eff = (sum w)^2 / sum(w^2).
    Used to replace ``n`` in the LTT bound when using weighted calibration.
    """
    w  = np.asarray(weights, dtype=float)
    sw = w.sum()
    sw2 = (w ** 2).sum()
    if sw2 == 0:
        return 0.0
    return float(sw ** 2 / sw2)


# ---------------------------------------------------------------------------
# Drift monitor
# ---------------------------------------------------------------------------

@dataclass
class DriftMonitor:
    """
    Stateful drift monitor that tracks recent risk scores and fires
    recalibration requests when drift is detected.

    Attributes
    ----------
    ks_threshold    : KS p-value below which drift is declared (default 0.05).
    mmd_threshold   : MMD^2 above which drift is declared (default 0.01).
    min_recent      : minimum recent atoms before running a test (default 30).
    """
    ks_threshold:  float = 0.05
    mmd_threshold: float = 0.01
    min_recent:    int   = 30

    _reference:  np.ndarray = field(default_factory=lambda: np.array([]))
    _recent:     np.ndarray = field(default_factory=lambda: np.array([]))
    _recalib_requested: bool = False

    def set_reference(self, risk: np.ndarray) -> None:
        """Set (or replace) the reference distribution (calibration window)."""
        self._reference = np.asarray(risk, dtype=float)

    def push(self, risk_scores: np.ndarray) -> None:
        """Append new risk scores from recent traffic."""
        self._recent = np.concatenate([self._recent, np.asarray(risk_scores, dtype=float)])

    def check(self, amendment_events: Optional[List[Dict]] = None,
              stratum: Optional[str] = None,
              since_date: Optional[str] = None) -> bool:
        """
        Run drift detectors and return True if recalibration is needed.

        Also fires if a version-graph trigger is detected (if arguments given).
        """
        if len(self._recent) < self.min_recent:
            return False

        drift_detected = False

        # KS test
        _, ks_p = ks_drift_score(self._reference, self._recent)
        if ks_p < self.ks_threshold:
            logger.warning("KS drift detected: p=%.4f < threshold=%.4f", ks_p, self.ks_threshold)
            drift_detected = True

        # MMD
        mmd2 = mmd_drift_score(self._reference, self._recent)
        if mmd2 > self.mmd_threshold:
            logger.warning("MMD drift detected: MMD^2=%.6f > threshold=%.6f", mmd2, self.mmd_threshold)
            drift_detected = True

        # Version-graph trigger
        if amendment_events and stratum and since_date:
            if version_graph_trigger(stratum, amendment_events, since_date):
                logger.warning("Version-graph drift trigger for stratum=%s", stratum)
                drift_detected = True

        if drift_detected:
            self._recalib_requested = True
        return drift_detected

    def reset_recent(self) -> None:
        """Clear recent data after recalibration."""
        self._recent = np.array([])
        self._recalib_requested = False

    @property
    def recalibration_requested(self) -> bool:
        return self._recalib_requested
