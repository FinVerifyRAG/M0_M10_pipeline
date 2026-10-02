"""
signals/s_ver.py  —  V1 Verification Status Signal
----------------------------------------------------
s_ver encodes the deterministic V1 outcome as a numeric risk score.

Encoding (per the plan):
    MATCH      →  0.0   (evidence explicitly confirms the value → low risk)
    NOT_FOUND  →  0.5   (evidence is silent → uncertain)
    NA         →  0.5   (atom type not handled by V1 → uncertain)
    MISMATCH   →  1.0   (evidence explicitly contradicts the value → high risk)

Additionally, a one-hot vector is returned for LightGBM to use as
categorical features:
    [is_match, is_mismatch, is_not_found, is_na]

Public functions
----------------
    compute_s_ver(verified_atom) → float      scalar for inspection
    s_ver_features(verified_atom) → dict      full feature dict for aggregator
"""
from __future__ import annotations

from common.schemas import VerifiedAtom

# V1 status string constants (mirrors verify/v1_deterministic.py)
MATCH = "MATCH"
MISMATCH = "MISMATCH"
NOT_FOUND = "NOT_FOUND"
NA = "NA"

_SCORE_MAP = {
    MATCH:     0.0,
    NOT_FOUND: 0.5,
    NA:        0.5,
    MISMATCH:  1.0,
}


def compute_s_ver(verified_atom: VerifiedAtom) -> float:
    """
    Return a scalar risk score in [0, 1] based on the V1 outcome.
    Unknown status strings default to 0.5 (uncertain).
    """
    return _SCORE_MAP.get(verified_atom.v1_status, 0.5)


def s_ver_features(verified_atom: VerifiedAtom) -> dict:
    """
    Return a feature dict with both the scalar and one-hot encoding:
        {
          "s_ver":          float,
          "v1_match":       0/1,
          "v1_mismatch":    0/1,
          "v1_not_found":   0/1,
          "v1_na":          0/1,
        }
    """
    status = verified_atom.v1_status
    return {
        "s_ver":        _SCORE_MAP.get(status, 0.5),
        "v1_match":     int(status == MATCH),
        "v1_mismatch":  int(status == MISMATCH),
        "v1_not_found": int(status == NOT_FOUND),
        "v1_na":        int(status == NA),
    }
