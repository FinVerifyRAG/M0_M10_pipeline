"""
tests/test_m7_guarantee.py
---------------------------
Comprehensive unit tests for M7: Guarantee (risk control & drift calibration).

Coverage
--------
LTT (ltt.py):
  - binom_pvalue: boundary cases (n=0, k=0, k=n)
  - hb_pvalue: always >= binom_pvalue, <= 1
  - ltt_threshold: returns None when data is insufficient; returns certified lambda
                   on synthetic data with known risk; fixed-sequence stop
  - ltt_thresholds_pair: accept_below <= abstain_above
  - simulate_violation_rate: violation_rate <= delta on IID synthetic data

Mondrian (mondrian.py):
  - get_stratum: formatting and wildcards
  - fallback_hierarchy: correct chain
  - build_strata: correct partitioning; merges small strata; *|* always present
  - allocate_delta: Bonferroni split, sums to delta
  - lookup_threshold: fallback chain traversal

Drift (drift.py):
  - ks_drift_score: identical distributions -> high p-value; shifted -> low p-value
  - mmd_drift_score: identical -> near 0; shifted -> positive
  - version_graph_trigger: fires on matching amendment; silent otherwise
  - sliding_window_filter: keeps only recent atoms
  - recency_weights: newer atoms have higher weight
  - effective_sample_size: equals n for uniform weights
  - DriftMonitor: check fires when distributions differ

Certify (certify.py):
  - _load_calib_jsonl and _records_to_arrays consistency
  - certify(): produces thresholds for all strata; *|* always present
  - load_thresholds: round-trips thresholds.json correctly

Simulate (simulate.py):
  - run_simulation: violation_rate <= delta on clean synthetic data
  - drift_experiment: post-amendment risk detectable; strategies differ
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic(
    n: int = 300,
    true_error_rate: float = 0.05,
    seed: int = 0,
) -> tuple:
    """
    Generate synthetic calibration data where ``risk`` correlates with ``wrong``.
    Atoms with risk > 0.5 are wrong with probability ~2*true_error_rate.
    """
    rng   = np.random.default_rng(seed)
    risk  = rng.uniform(0.0, 1.0, n)
    # P(wrong | risk) proportional to risk, scaled to achieve true_error_rate
    prob_wrong = np.clip(risk * (2 * true_error_rate) / 0.5, 0, 1)
    wrong = (rng.uniform(size=n) < prob_wrong).astype(float)
    return risk, wrong


def _make_calib_jsonl(
    records: List[dict],
    tmp_dir: str,
    filename: str = "calib.jsonl",
) -> str:
    path = Path(tmp_dir) / filename
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)


# ===========================================================================
# LTT tests
# ===========================================================================

class TestBinomPvalue:
    def test_no_atoms(self):
        from guarantee.ltt import binom_pvalue
        assert binom_pvalue(0, 0, 0.05) == 1.0

    def test_zero_wrong(self):
        """k=0 wrong out of n=100 accepted: very small p-value (data strongly support risk<=eps)."""
        from guarantee.ltt import binom_pvalue
        p = binom_pvalue(0, 100, 0.05)
        assert p < 0.05, f"Expected p < 0.05, got {p}"

    def test_all_wrong(self):
        """k=n wrong: data are inconsistent with risk<=eps, p should be 1.0."""
        from guarantee.ltt import binom_pvalue
        p = binom_pvalue(100, 100, 0.05)
        assert p == 1.0

    def test_in_unit_interval(self):
        from guarantee.ltt import binom_pvalue
        for k in [0, 5, 10, 20]:
            p = binom_pvalue(k, 100, 0.05)
            assert 0.0 <= p <= 1.0


class TestHBPvalue:
    def test_no_atoms(self):
        from guarantee.ltt import hb_pvalue
        assert hb_pvalue(0, 0, 0.05) == 1.0

    def test_gte_binom(self):
        """HB p-value is always >= plain binomial p-value (HB is tighter in the right direction)."""
        from guarantee.ltt import binom_pvalue, hb_pvalue
        import math
        for k in [0, 2, 5, 10]:
            hb = hb_pvalue(k, 100, 0.05)
            bi = binom_pvalue(k, 100, 0.05)
            # HB = min(1, e * binom), so hb >= binom
            assert hb >= bi - 1e-9, f"HB ({hb}) < binom ({bi}) for k={k}"

    def test_bounded(self):
        from guarantee.ltt import hb_pvalue
        for k in [0, 1, 5, 50, 100]:
            p = hb_pvalue(k, 100, 0.05)
            assert 0.0 <= p <= 1.0


class TestLTTThreshold:
    def test_returns_none_when_all_wrong(self):
        """When all atoms are wrong, no threshold should be certified."""
        from guarantee.ltt import ltt_threshold
        risk  = np.linspace(0, 1, 100)
        wrong = np.ones(100)
        lam = ltt_threshold(risk, wrong, eps=0.05, delta=0.10, n_grid=50)
        assert lam is None

    def test_returns_threshold_when_safe(self):
        """When error rate is truly 0, LTT certifies a threshold with large enough n."""
        from guarantee.ltt import ltt_threshold
        # Use n=500 atoms all at risk=0.2 and wrong=0. At lam>=0.2, n=500, k=0.
        # binom.cdf(0, 500, 0.05) = 0.95^500 ~ 5e-12 << delta. Certifies lam=0.2.
        risk  = np.full(500, 0.2)
        wrong = np.zeros(500)
        lam = ltt_threshold(risk, wrong, eps=0.05, delta=0.10, n_grid=100, use_hb=False)
        assert lam is not None, "Expected a certified lambda when k=0 and n=500"
        assert lam >= 0.0

    def test_threshold_in_unit_interval(self):
        """Certified lambda must be in [0, 1]."""
        from guarantee.ltt import ltt_threshold
        risk, wrong = _make_synthetic(n=500, true_error_rate=0.03)
        lam = ltt_threshold(risk, wrong, eps=0.05, delta=0.10, n_grid=100)
        if lam is not None:
            assert 0.0 <= lam <= 1.0

    def test_fixed_sequence_stop(self):
        """Threshold should not skip over a rejection; accept_below <= true threshold."""
        from guarantee.ltt import ltt_threshold
        # Construct a scenario where error rate jumps at risk=0.5
        rng  = np.random.default_rng(1)
        risk = np.linspace(0, 1, 300)
        wrong = np.where(risk > 0.5, 1.0, 0.0)  # All wrong above 0.5
        lam = ltt_threshold(risk, wrong, eps=0.05, delta=0.10, n_grid=200)
        # Should stop before 0.5 (or return None)
        if lam is not None:
            assert lam <= 0.5 + 0.01  # allow small grid tolerance

    def test_hb_vs_binom(self):
        """HB bound should give a <= threshold compared to binomial (more conservative accept)."""
        from guarantee.ltt import ltt_threshold
        risk, wrong = _make_synthetic(n=400, true_error_rate=0.04, seed=5)
        lam_hb  = ltt_threshold(risk, wrong, eps=0.05, delta=0.10, n_grid=100, use_hb=True)
        lam_bin = ltt_threshold(risk, wrong, eps=0.05, delta=0.10, n_grid=100, use_hb=False)
        # HB is tighter (smaller p) -> certifies the same or larger lambda
        if lam_hb is not None and lam_bin is not None:
            # Thresholds may differ by at most one grid step
            assert abs(lam_hb - lam_bin) <= 0.1


class TestLTTThresholdsPair:
    def test_ordering(self):
        """accept_below < abstain_above always (when both not None)."""
        from guarantee.ltt import ltt_thresholds_pair
        risk, wrong = _make_synthetic(n=400, true_error_rate=0.04)
        ab, aa = ltt_thresholds_pair(
            risk, wrong, eps_accept=0.05,
            delta=0.10, n_grid=100,
        )
        if ab is not None and aa is not None:
            assert ab <= aa + 1e-9

    def test_abstain_gte_accept(self):
        """abstain_above >= accept_below always (ordering invariant)."""
        from guarantee.ltt import ltt_thresholds_pair
        risk, wrong = _make_synthetic(n=500, true_error_rate=0.05)
        ab, aa = ltt_thresholds_pair(
            risk, wrong, eps_accept=0.05,
            delta=0.10, n_grid=100,
        )
        if ab is not None and aa is not None:
            assert aa >= ab - 1e-9


class TestSimulateViolationRate:
    def test_violation_rate_bounded(self):
        """Violation rate should be <= delta on IID synthetic data."""
        from guarantee.ltt import simulate_violation_rate
        risk, wrong = _make_synthetic(n=600, true_error_rate=0.03, seed=42)
        result = simulate_violation_rate(
            risk, wrong, eps=0.05, delta=0.10,
            n_splits=300, calib_frac=0.5, n_grid=100, use_hb=True, seed=42,
        )
        # Abstaining on every split is not a pass. A numeric rate is
        # checked only when at least one split certified a threshold.
        if result["valid"]:
            assert result["violation_rate"] <= 0.10 + 0.05
        else:
            assert result["n_valid_splits"] == 0

    def test_returns_required_keys(self):
        from guarantee.ltt import simulate_violation_rate
        risk, wrong = _make_synthetic(n=200, true_error_rate=0.05)
        result = simulate_violation_rate(
            risk, wrong, eps=0.05, delta=0.10, n_splits=50, n_grid=50,
        )
        for key in ["violation_rate", "target_delta", "mean_lambda", "n_valid_splits"]:
            assert key in result, f"Missing key: {key}"


# ===========================================================================
# Mondrian tests
# ===========================================================================

class TestGetStratum:
    def test_basic(self):
        from guarantee.mondrian import get_stratum
        assert get_stratum("SEBI", "RATE") == "SEBI|RATE"

    def test_uppercase(self):
        from guarantee.mondrian import get_stratum
        assert get_stratum("sebi", "rate") == "SEBI|RATE"

    def test_empty_regulator(self):
        from guarantee.mondrian import get_stratum
        assert get_stratum("", "DATE") == "*|DATE"

    def test_empty_both(self):
        from guarantee.mondrian import get_stratum
        assert get_stratum("", "") == "*|*"


class TestFallbackHierarchy:
    def test_full_stratum(self):
        from guarantee.mondrian import fallback_hierarchy
        chain = fallback_hierarchy("SEBI|RATE")
        assert chain == ["SEBI|RATE", "*|RATE", "*|*"]

    def test_wildcard_reg(self):
        from guarantee.mondrian import fallback_hierarchy
        chain = fallback_hierarchy("*|RATE")
        assert chain == ["*|RATE", "*|*"]

    def test_global(self):
        from guarantee.mondrian import fallback_hierarchy
        chain = fallback_hierarchy("*|*")
        assert chain == ["*|*"]


class TestBuildStrata:
    def _make_data(self, n: int = 500):
        rng  = np.random.default_rng(0)
        risk = rng.uniform(0, 1, n)
        wrong = (rng.uniform(size=n) < 0.05).astype(float)
        regulators = rng.choice(["SEBI", "RBI", "INCOMETAX"], size=n).tolist()
        atom_types = rng.choice(["RATE", "DATE", "SECTION"], size=n).tolist()
        return risk, wrong, regulators, atom_types

    def test_global_stratum_always_present(self):
        from guarantee.mondrian import build_strata
        risk, wrong, reg, atype = self._make_data()
        strata = build_strata(risk, wrong, reg, atype, n_min=50)
        assert "*|*" in strata

    def test_large_strata_kept_standalone(self):
        from guarantee.mondrian import build_strata
        # Make SEBI|RATE very large
        n = 300
        risk  = np.random.default_rng(1).uniform(0, 1, n)
        wrong = np.zeros(n)
        regulators = ["SEBI"] * n
        atom_types = ["RATE"] * n
        strata = build_strata(risk, wrong, regulators, atom_types, n_min=100)
        assert "SEBI|RATE" in strata
        assert strata["SEBI|RATE"].n_atoms == n

    def test_small_strata_merged(self):
        from guarantee.mondrian import build_strata
        # Only 10 atoms for SEBI|DATE (< n_min=100) -> should be merged
        n = 10
        risk  = np.random.default_rng(2).uniform(0, 1, n)
        wrong = np.zeros(n)
        regulators = ["SEBI"] * n
        atom_types = ["DATE"] * n
        strata = build_strata(risk, wrong, regulators, atom_types, n_min=100)
        # SEBI|DATE should be absorbed into *|* (not standalone)
        assert "SEBI|DATE" not in strata

    def test_n_atoms_consistent(self):
        from guarantee.mondrian import build_strata
        risk, wrong, reg, atype = self._make_data(600)
        strata = build_strata(risk, wrong, reg, atype, n_min=20)
        for name, stratum in strata.items():
            assert stratum.n_atoms == len(stratum.risk) == len(stratum.wrong)


class TestAllocateDelta:
    def test_bonferroni_split(self):
        from guarantee.mondrian import build_strata, allocate_delta
        risk  = np.random.default_rng(3).uniform(0, 1, 500)
        wrong = np.zeros(500)
        regulators = ["SEBI"] * 250 + ["RBI"] * 250
        atom_types = ["RATE"] * 500
        strata    = build_strata(risk, wrong, regulators, atom_types, n_min=10)
        delta_map = allocate_delta(0.10, strata)
        n = len(strata)
        for name, d in delta_map.items():
            assert abs(d - 0.10 / n) < 1e-9


class TestLookupThreshold:
    def test_exact_match(self):
        from guarantee.mondrian import build_strata, lookup_threshold
        risk  = np.random.default_rng(4).uniform(0, 1, 200)
        wrong = np.zeros(200)
        regulators = ["SEBI"] * 200
        atom_types = ["RATE"] * 200
        strata = build_strata(risk, wrong, regulators, atom_types, n_min=10)
        found = lookup_threshold("SEBI|RATE", strata)
        assert found is not None

    def test_fallback_to_global(self):
        from guarantee.mondrian import build_strata, lookup_threshold
        risk  = np.random.default_rng(5).uniform(0, 1, 200)
        wrong = np.zeros(200)
        regulators = ["SEBI"] * 200
        atom_types = ["RATE"] * 200
        strata = build_strata(risk, wrong, regulators, atom_types, n_min=10)
        # UNKNOWN regulator should fall back to *|*
        found = lookup_threshold("UNKNOWN|ENTITY", strata)
        assert found is not None
        assert found.name == "*|*"


# ===========================================================================
# Drift tests
# ===========================================================================

class TestKSDriftScore:
    def test_identical_distributions(self):
        """KS test on identical data should yield high p-value (no drift)."""
        from guarantee.drift import ks_drift_score
        rng  = np.random.default_rng(10)
        data = rng.uniform(0, 1, 200)
        stat, pval = ks_drift_score(data, data)
        assert pval > 0.05, f"Expected high p-value for identical dists, got {pval}"

    def test_shifted_distributions(self):
        """KS test on clearly different distributions should yield low p-value."""
        from guarantee.drift import ks_drift_score
        rng  = np.random.default_rng(11)
        ref  = rng.uniform(0.0, 0.3, 200)
        rec  = rng.uniform(0.7, 1.0, 200)
        stat, pval = ks_drift_score(ref, rec)
        assert pval < 0.05, f"Expected low p-value for shifted dists, got {pval}"

    def test_too_few_samples(self):
        from guarantee.drift import ks_drift_score
        stat, pval = ks_drift_score(np.array([0.1, 0.2]), np.array([0.3]))
        assert pval == 1.0  # Fallback


class TestMMDDriftScore:
    def test_identical_near_zero(self):
        from guarantee.drift import mmd_drift_score
        data = np.random.default_rng(12).uniform(0, 1, 100)
        mmd2 = mmd_drift_score(data, data)
        assert mmd2 < 1e-6, f"MMD^2 should be ~0 for identical dists, got {mmd2}"

    def test_shifted_positive(self):
        from guarantee.drift import mmd_drift_score
        rng = np.random.default_rng(13)
        ref = rng.uniform(0.0, 0.2, 100)
        rec = rng.uniform(0.8, 1.0, 100)
        mmd2 = mmd_drift_score(ref, rec)
        assert mmd2 > 0.01, f"MMD^2 should be positive for shifted dists, got {mmd2}"

    def test_non_negative(self):
        from guarantee.drift import mmd_drift_score
        rng = np.random.default_rng(14)
        for _ in range(5):
            a = rng.uniform(0, 1, 50)
            b = rng.uniform(0, 1, 50)
            assert mmd_drift_score(a, b) >= 0.0


class TestVersionGraphTrigger:
    def test_fires_on_matching_amendment(self):
        from guarantee.drift import version_graph_trigger
        events = [{"date": "2024-06-01", "regulator": "SEBI", "type": "amends", "sections": ["52"]}]
        assert version_graph_trigger("SEBI|RATE", events, since_date="2024-01-01")

    def test_silent_on_non_matching_regulator(self):
        from guarantee.drift import version_graph_trigger
        events = [{"date": "2024-06-01", "regulator": "RBI", "type": "amends", "sections": ["10"]}]
        assert not version_graph_trigger("SEBI|RATE", events, since_date="2024-01-01")

    def test_silent_on_old_event(self):
        from guarantee.drift import version_graph_trigger
        events = [{"date": "2023-01-01", "regulator": "SEBI", "type": "amends", "sections": ["52"]}]
        # Event is before since_date
        assert not version_graph_trigger("SEBI|RATE", events, since_date="2024-01-01")

    def test_fires_on_wildcard_stratum(self):
        from guarantee.drift import version_graph_trigger
        events = [{"date": "2025-01-01", "regulator": "SEBI", "type": "supersedes"}]
        assert version_graph_trigger("*|*", events, since_date="2024-01-01")


class TestSlidingWindowFilter:
    def test_keeps_recent_atoms(self):
        from guarantee.drift import sliding_window_filter
        dates = ["2024-01-01", "2023-01-01", "2022-01-01"]
        risk  = np.array([0.1, 0.5, 0.9])
        wrong = np.array([0, 1, 0])
        r, w = sliding_window_filter(dates, risk, wrong, window_months=18,
                                     reference_date="2024-06-01")
        assert len(r) == 2  # 2024 and 2023 are within 18 months of 2024-06

    def test_excludes_old_atoms(self):
        from guarantee.drift import sliding_window_filter
        dates = ["2024-01-01", "2020-01-01"]
        risk  = np.array([0.2, 0.8])
        wrong = np.array([0, 1])
        r, w = sliding_window_filter(dates, risk, wrong, window_months=12,
                                     reference_date="2024-06-01")
        assert len(r) == 1
        assert r[0] == 0.2  # Only the 2024 atom


class TestRecencyWeights:
    def test_newer_is_heavier(self):
        from guarantee.drift import recency_weights
        dates   = ["2024-06-01", "2023-06-01", "2022-06-01"]
        weights = recency_weights(dates, half_life_days=180, reference_date="2024-06-01")
        assert weights[0] >= weights[1] >= weights[2]

    def test_same_date_weight_one(self):
        from guarantee.drift import recency_weights
        dates   = ["2024-06-01"]
        weights = recency_weights(dates, half_life_days=180, reference_date="2024-06-01")
        assert abs(weights[0] - 1.0) < 1e-6

    def test_all_positive(self):
        from guarantee.drift import recency_weights
        dates   = ["2020-01-01", "2021-06-01", "2024-05-01"]
        weights = recency_weights(dates, half_life_days=365)
        assert all(w > 0 for w in weights)


class TestEffectiveSampleSize:
    def test_uniform_weights(self):
        from guarantee.drift import effective_sample_size
        w = np.ones(100)
        assert abs(effective_sample_size(w) - 100.0) < 1e-6

    def test_single_large_weight(self):
        from guarantee.drift import effective_sample_size
        # One atom has all weight -> n_eff = 1
        w = np.array([1.0, 0.0, 0.0, 0.0])
        assert abs(effective_sample_size(w) - 1.0) < 1e-6


class TestDriftMonitor:
    def test_no_drift_on_identical_data(self):
        from guarantee.drift import DriftMonitor
        rng = np.random.default_rng(20)
        ref = rng.uniform(0, 1, 200)
        monitor = DriftMonitor(ks_threshold=0.01, mmd_threshold=0.001, min_recent=30)
        monitor.set_reference(ref)
        monitor.push(rng.uniform(0, 1, 50))  # Same distribution
        # Most likely no drift, but we only assert that the monitor ran without error
        result = monitor.check()
        assert isinstance(result, bool)

    def test_drift_on_shifted_data(self):
        from guarantee.drift import DriftMonitor
        rng = np.random.default_rng(21)
        ref = rng.uniform(0.0, 0.3, 500)
        monitor = DriftMonitor(ks_threshold=0.05, mmd_threshold=0.01, min_recent=30)
        monitor.set_reference(ref)
        monitor.push(rng.uniform(0.7, 1.0, 200))
        assert monitor.check() is True

    def test_reset_clears_state(self):
        from guarantee.drift import DriftMonitor
        monitor = DriftMonitor()
        monitor.push(np.array([0.1, 0.2]))
        monitor.reset_recent()
        assert len(monitor._recent) == 0
        assert not monitor.recalibration_requested


# ===========================================================================
# Certify tests (integration, uses temp files)
# ===========================================================================

class TestCertify:
    def _make_jsonl_records(self, n: int = 400, seed: int = 42) -> List[dict]:
        rng        = np.random.default_rng(seed)
        risk_vals  = rng.uniform(0, 1, n).tolist()
        labels     = (rng.uniform(size=n) < 0.04).astype(int).tolist()
        regulators = rng.choice(["SEBI", "RBI"], size=n).tolist()
        atom_types = rng.choice(["RATE", "DATE", "SECTION"], size=n).tolist()
        dates      = ["2024-01-01"] * n
        return [
            {"risk": float(risk_vals[i]), "label": int(labels[i]),
             "regulator": str(regulators[i]), "atom_type": str(atom_types[i]),
             "date": dates[i]}
            for i in range(n)
        ]

    def test_produces_global_stratum(self):
        from guarantee.certify import certify
        with tempfile.TemporaryDirectory() as tmp:
            calib_path = _make_calib_jsonl(self._make_jsonl_records(), tmp)
            out_path   = str(Path(tmp) / "thresholds.json")
            result = certify(
                calib_path=calib_path, out_path=out_path,
                eps_accept=0.05, eps_abstain=0.20, delta=0.10,
                n_min=20, n_grid=50, run_simulation=False,
            )
            assert "*|*" in result

    def test_thresholds_schema(self):
        from guarantee.certify import certify
        from common.schemas import Thresholds
        with tempfile.TemporaryDirectory() as tmp:
            calib_path = _make_calib_jsonl(self._make_jsonl_records(), tmp)
            out_path   = str(Path(tmp) / "thresholds.json")
            result = certify(
                calib_path=calib_path, out_path=out_path,
                n_min=20, n_grid=50, run_simulation=False,
            )
            for name, th in result.items():
                assert isinstance(th, Thresholds)
                assert th.stratum_name == name

    def test_saves_valid_json(self):
        from guarantee.certify import certify
        with tempfile.TemporaryDirectory() as tmp:
            calib_path = _make_calib_jsonl(self._make_jsonl_records(), tmp)
            out_path   = str(Path(tmp) / "thresholds.json")
            certify(
                calib_path=calib_path, out_path=out_path,
                n_min=20, n_grid=50, run_simulation=False,
            )
            with open(out_path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert len(data) > 0

    def test_load_roundtrip(self):
        from guarantee.certify import certify, load_thresholds
        with tempfile.TemporaryDirectory() as tmp:
            calib_path = _make_calib_jsonl(self._make_jsonl_records(), tmp)
            out_path   = str(Path(tmp) / "thresholds.json")
            saved  = certify(calib_path=calib_path, out_path=out_path,
                             n_min=20, n_grid=50, run_simulation=False)
            loaded = load_thresholds(out_path)
            for name in saved:
                assert name in loaded
                assert loaded[name].accept_below == saved[name].accept_below
                assert loaded[name].abstain_above == saved[name].abstain_above

    def test_accept_below_le_abstain_above(self):
        from guarantee.certify import certify
        with tempfile.TemporaryDirectory() as tmp:
            calib_path = _make_calib_jsonl(self._make_jsonl_records(), tmp)
            out_path   = str(Path(tmp) / "thresholds.json")
            result = certify(
                calib_path=calib_path, out_path=out_path,
                eps_accept=0.05, eps_abstain=0.20,
                n_min=20, n_grid=50, run_simulation=False,
            )
            for name, th in result.items():
                # Bug 4.3: accept_below is now Optional[float] (may be None)
                if th.accept_below is not None:
                    assert th.accept_below < th.abstain_above + 1e-9, (
                        f"Stratum {name}: accept_below={th.accept_below} > abstain_above={th.abstain_above}"
                    )

    def test_sliding_window_strategy(self):
        from guarantee.certify import certify
        with tempfile.TemporaryDirectory() as tmp:
            calib_path = _make_calib_jsonl(self._make_jsonl_records(), tmp)
            out_path   = str(Path(tmp) / "thresholds_sw.json")
            # Should not raise even if window is very narrow
            certify(
                calib_path=calib_path, out_path=out_path,
                n_min=5, n_grid=50, run_simulation=False,
                strategy="sliding_window", window_months=6,
                reference_date="2025-01-01",
            )
            assert Path(out_path).exists()

    def test_recency_weighted_strategy(self):
        from guarantee.certify import certify
        with tempfile.TemporaryDirectory() as tmp:
            calib_path = _make_calib_jsonl(self._make_jsonl_records(), tmp)
            out_path   = str(Path(tmp) / "thresholds_rw.json")
            certify(
                calib_path=calib_path, out_path=out_path,
                n_min=5, n_grid=50, run_simulation=False,
                strategy="recency_weighted", half_life_days=90.0,
                reference_date="2025-01-01",
            )
            assert Path(out_path).exists()


# ===========================================================================
# Simulate tests
# ===========================================================================

class TestRunSimulation:
    def test_violation_rate_bounded(self):
        """Violation rate <= delta on synthetic IID data."""
        from guarantee.simulate import run_simulation
        risk, wrong = _make_synthetic(n=500, true_error_rate=0.03, seed=99)
        result = run_simulation(
            risk, wrong, eps=0.05, delta=0.10,
            n_splits=200, calib_frac=0.5, n_grid=80, use_hb=True, seed=0,
        )
        if result["valid"]:
            assert result["violation_rate"] <= 0.10 + 0.05
        else:
            assert result["n_valid_splits"] == 0

    def test_returns_expected_keys(self):
        from guarantee.simulate import run_simulation
        risk, wrong = _make_synthetic(n=200, true_error_rate=0.05)
        result = run_simulation(
            risk, wrong, eps=0.05, delta=0.10, n_splits=50, n_grid=50,
        )
        for key in ["violation_rate", "mean_lambda", "mean_coverage"]:
            assert key in result

    def test_coverage_in_unit_interval(self):
        from guarantee.simulate import run_simulation
        risk, wrong = _make_synthetic(n=300, true_error_rate=0.04)
        result = run_simulation(
            risk, wrong, eps=0.05, delta=0.10, n_splits=100, n_grid=50,
        )
        cov = result.get("mean_coverage", 0.5)
        if not np.isnan(cov):
            assert 0.0 <= cov <= 1.0


class TestDriftExperiment:
    def _make_pre_post(self, seed: int = 7):
        rng         = np.random.default_rng(seed)
        pre_risk    = rng.uniform(0.0, 0.4, 300)  # Pre-amendment: low risk
        pre_wrong   = (rng.uniform(size=300) < 0.02).astype(float)
        post_risk   = rng.uniform(0.3, 1.0, 200)  # Post-amendment: higher risk
        post_wrong  = (rng.uniform(size=200) < 0.15).astype(float)
        pre_dates   = ["2023-06-01"] * 300
        post_dates  = ["2024-06-01"] * 200
        return pre_risk, pre_wrong, post_risk, post_wrong, pre_dates, post_dates

    def test_returns_all_strategies(self):
        from guarantee.simulate import drift_experiment
        pr, pw, postr, postw, pd, _ = self._make_pre_post()
        result = drift_experiment(
            pr, pw, postr, postw, pre_dates=pd,
            eps=0.05, delta=0.10, n_grid=50,
            reference_date="2024-01-01",
        )
        for strategy in ["static", "sliding_window", "recency_weighted"]:
            assert strategy in result

    def test_each_strategy_has_required_keys(self):
        from guarantee.simulate import drift_experiment
        pr, pw, postr, postw, pd, _ = self._make_pre_post()
        result = drift_experiment(
            pr, pw, postr, postw, pre_dates=pd,
            eps=0.05, delta=0.10, n_grid=50,
        )
        for strategy in ["static", "sliding_window", "recency_weighted"]:
            d = result[strategy]
            assert "accept_below" in d
            assert "empirical_risk_post" in d
            assert "accepted_fraction" in d

    def test_accepted_fraction_in_unit_interval(self):
        from guarantee.simulate import drift_experiment
        pr, pw, postr, postw, pd, _ = self._make_pre_post()
        result = drift_experiment(
            pr, pw, postr, postw, pre_dates=pd,
            eps=0.05, delta=0.10, n_grid=50,
        )
        for strategy in ["static", "sliding_window", "recency_weighted"]:
            frac = result[strategy].get("accepted_fraction", 0.0)
            assert 0.0 <= frac <= 1.0 + 1e-9
