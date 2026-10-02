"""
guarantee/
----------
M7: Risk control and drift-aware calibration.

Public API summary
------------------
    from guarantee.ltt     import ltt_threshold, ltt_thresholds_pair, simulate_violation_rate
    from guarantee.mondrian import get_stratum, build_strata, allocate_delta, lookup_threshold
    from guarantee.drift    import ks_drift_score, mmd_drift_score, DriftMonitor
    from guarantee.certify  import certify, load_thresholds
    from guarantee.simulate import run_simulation, drift_experiment

Module layout
-------------
  ltt.py       - Learn-then-Test p-values and fixed-sequence threshold search.
  mondrian.py  - Mondrian stratification by (regulator x atom_type).
  drift.py     - Drift detectors (KS, MMD, version-graph) + calibration strategies.
  certify.py   - End-to-end orchestration: strata -> LTT -> thresholds.json.
  simulate.py  - Repeated-split experiments to validate the violation rate.
"""
from guarantee.ltt import (
    binom_pvalue,
    hb_pvalue,
    ltt_threshold,
    ltt_thresholds_pair,
    simulate_violation_rate,
)
from guarantee.mondrian import (
    get_stratum,
    fallback_hierarchy,
    build_strata,
    allocate_delta,
    get_delta_for_stratum,
    lookup_threshold,
    Stratum,
)
from guarantee.drift import (
    ks_drift_score,
    mmd_drift_score,
    version_graph_trigger,
    sliding_window_filter,
    recency_weights,
    effective_sample_size,
    DriftMonitor,
)
from guarantee.certify import certify, load_thresholds
from guarantee.simulate import run_simulation, drift_experiment

__all__ = [
    # ltt
    "binom_pvalue",
    "hb_pvalue",
    "ltt_threshold",
    "ltt_thresholds_pair",
    "simulate_violation_rate",
    # mondrian
    "get_stratum",
    "fallback_hierarchy",
    "build_strata",
    "allocate_delta",
    "get_delta_for_stratum",
    "lookup_threshold",
    "Stratum",
    # drift
    "ks_drift_score",
    "mmd_drift_score",
    "version_graph_trigger",
    "sliding_window_filter",
    "recency_weights",
    "effective_sample_size",
    "DriftMonitor",
    # certify
    "certify",
    "load_thresholds",
    # simulate
    "run_simulation",
    "drift_experiment",
]
