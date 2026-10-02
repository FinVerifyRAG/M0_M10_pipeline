"""Release gate. Exits non-zero unless every publishable condition holds.

    python -m eval.check_publishable --results results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def check(results_dir: str = "results", splits_dir: str = "data/splits") -> list:
    """Return a list of failure strings. Empty means the gate passes."""
    failures = []
    root = Path(results_dir)
    summary = _load(root / "summary.json")
    main = _load(root / "main_table.json")
    ablation = _load(root / "ablation_table.json")
    seeds = _load(root / "seeds_summary.json")

    if not summary:
        return ["summary.json is missing"]
    if summary.get("simulated"):
        failures.append("summary is marked simulated")
    if summary.get("git_commit") in (None, "", "unknown"):
        failures.append("git commit was not recorded")
    if not summary.get("config_hash"):
        failures.append("config hash was not recorded")

    model = Path("models/aggregator/aggregator_v1.pkl")
    if not model.exists():
        failures.append("aggregator model is missing")

    n_atoms = summary.get("n_atoms") or 0
    if n_atoms < 150:
        failures.append(f"test size {n_atoms} is below 150")

    leakage = Path(splits_dir) / "leakage_report.json"
    if not leakage.exists():
        failures.append("document-level split leakage report is missing")
    else:
        report = json.loads(leakage.read_text(encoding="utf-8"))
        if report.get("leaked"):
            failures.append("a document id appears in more than one split")

    if isinstance(main, dict):
        rows = main.get("rows") or []
        if main.get("status") == "not_available":
            failures.append("main table was not computed from real systems")
        aurcs = [r.get("aurc") for r in rows if r.get("aurc") is not None]
        if len(aurcs) < 2 or len({round(a, 6) for a in aurcs}) < len(aurcs):
            failures.append("per-system AURC values are not distinct")
        for row in rows:
            if row.get("threshold") is None and row.get("system"):
                failures.append(f"threshold is None for {row.get('system')}")
            if row.get("coverage") == 0:
                failures.append(f"zero coverage for {row.get('system')}")
    else:
        failures.append("main_table.json is missing")

    if isinstance(ablation, list):
        failures.append("ablation table is not the retrained results object")
    elif not isinstance(ablation, dict) or not ablation:
        failures.append("ablation_table.json is missing")
    else:
        full = ablation.get("full_regguard") or {}
        main_full = summary.get("full_regguard") or {}
        for key in ("hallucination_rate", "coverage", "aurc"):
            if full.get(key) != main_full.get(key):
                failures.append(f"ablation Full RegGuard {key} disagrees with the main table")
        if ablation.get("status") != "computed":
            failures.append("ablation table was not retrained")

    vr = summary.get("violation_rate")
    delta = summary.get("delta", 0.10)
    if vr is None or vr != vr or vr > delta:
        failures.append("held-out violation rate is missing or above delta")

    if not seeds or not seeds.get("intervals"):
        failures.append("five-seed intervals are missing")
    elif len(seeds.get("seeds") or []) < 5:
        failures.append("fewer than 5 seeds")
    else:
        for key, interval in (seeds.get("intervals") or {}).items():
            if not interval:
                failures.append(f"no interval for {key}")

    drift = _load(root / "drift_table.json")
    if not drift:
        failures.append("drift table is missing")
    else:
        if drift.get("n_pre", 0) < 150 or drift.get("n_post", 0) < 150:
            failures.append("drift split has fewer than 150 atoms on a side")
        strategies = drift.get("strategies") or []
        thresholds = {s.get("threshold") for s in strategies}
        if len(thresholds) < 2:
            failures.append("drift strategies did not produce different thresholds")

    ret = _load(root / "retrieval_metrics.json") or {}
    ext = _load(root / "extraction_metrics.json") or {}
    if ret.get("status") != "computed" or ret.get("simulated"):
        failures.append("retrieval metrics are not from gold chunks")
    if ext.get("status") != "computed" or ext.get("simulated"):
        failures.append("extraction metrics are not from gold atoms")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="RegGuard publishable release gate")
    parser.add_argument("--results", default="results")
    parser.add_argument("--splits", default="data/splits")
    args = parser.parse_args()
    failures = check(args.results, args.splits)
    if failures:
        print("NOT PUBLISHABLE")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("PUBLISHABLE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
