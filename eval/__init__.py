"""
eval/__init__.py — M10 Evaluation package.
"""
from eval.metrics import (
    hallucination_rate, coverage, risk_coverage_curve, aurc,
    empirical_violation_rate, recall_at_k, mrr, precision_at_k,
    extraction_recall, extraction_precision, extraction_f1,
    judge_call_rate, mean_latency, compute_all_metrics,
)

__all__ = [
    "hallucination_rate", "coverage", "risk_coverage_curve", "aurc",
    "empirical_violation_rate", "recall_at_k", "mrr", "precision_at_k",
    "extraction_recall", "extraction_precision", "extraction_f1",
    "judge_call_rate", "mean_latency", "compute_all_metrics",
]
