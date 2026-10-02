"""
aggregate/calibrate_probs.py
-----------------------------
Post-hoc probability calibration so `risk` behaves like a true probability.

After LightGBM training, the raw predict_proba scores are often not
well-calibrated (over-confident or under-confident). We apply:

    Option A: Isotonic Regression (non-parametric, preferred for ≥1000 samples)
    Option B: Platt Scaling (logistic fit, better for small calibration sets)

IMPORTANT: Calibration uses the VALIDATION split of the agg_train data —
NOT the M7 calibration set (which is reserved for LTT threshold certification).

Usage
-----
    python -m aggregate.calibrate_probs \\
        --model    models/aggregator/aggregator_v1.pkl \\
        --features data/processed/feature_table.parquet \\
        --out      models/aggregator/calibrator_v1.pkl \\
        --method   isotonic
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

from aggregate.dataset import FEATURE_COLS

logger = logging.getLogger("aggregate.calibrate_probs")
logging.basicConfig(level=logging.INFO)


def calibrate(
    model_path: str,
    features_path: str,
    out_path: str,
    method: str = "isotonic",
    val_frac: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Load the trained aggregator model, score the val split, and fit a
    calibrator. Save calibrator to out_path.
    """
    import pandas as pd  # type: ignore
    from sklearn.isotonic import IsotonicRegression  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load features
    df = pd.read_parquet(features_path)

    # Hold out val split (same seed as training)
    rng = np.random.default_rng(seed)
    val_mask = rng.random(len(df)) < val_frac
    val_df = df[val_mask]

    for col in FEATURE_COLS:
        if col not in val_df.columns:
            val_df[col] = 0.5

    X_val = val_df[FEATURE_COLS].fillna(0.5).values
    y_val = val_df["label"].values

    raw_probs = model.predict_proba(X_val)[:, 1]

    # Fit calibrator
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(raw_probs, y_val)
        logger.info("Fitted Isotonic Regression calibrator on %d samples.", len(y_val))
    elif method == "platt":
        lr = LogisticRegression()
        lr.fit(raw_probs.reshape(-1, 1), y_val)
        cal = lr
        logger.info("Fitted Platt (LogReg) calibrator on %d samples.", len(y_val))
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"method": method, "calibrator": cal}, f)
    logger.info("Saved calibrator to %s", out)


def apply_calibration(raw_prob: float, calibrator_path: str) -> float:
    """
    Apply a saved calibrator to a single raw probability.
    Used at inference time in aggregate/model.py.
    """
    with open(calibrator_path, "rb") as f:
        cal_obj = pickle.load(f)

    method = cal_obj["method"]
    cal    = cal_obj["calibrator"]

    if method == "isotonic":
        return float(cal.predict([raw_prob])[0])
    elif method == "platt":
        return float(cal.predict_proba([[raw_prob]])[0, 1])
    return raw_prob


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate aggregator probabilities.")
    parser.add_argument("--model",    required=True, help="Path to aggregator_v1.pkl")
    parser.add_argument("--features", required=True, help="Path to feature_table.parquet")
    parser.add_argument("--out",      required=True, help="Output path for calibrator pkl")
    parser.add_argument("--method",   default="isotonic",
                        choices=["isotonic", "platt"])
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    calibrate(
        model_path=args.model,
        features_path=args.features,
        out_path=args.out,
        method=args.method,
        val_frac=args.val_frac,
        seed=args.seed,
    )
