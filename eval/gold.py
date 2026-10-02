"""Retrieval and extraction metrics from gold annotation files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from eval.metrics import (
    extraction_f1,
    extraction_precision,
    extraction_recall,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def retrieval_from_gold(path: str, ks: Optional[List[int]] = None) -> dict:
    """
    Each line: {"retrieved_ids": [...], "gold_ids": [...]}
    Returns not_available when the file is missing or empty.
    """
    ks = ks or [6, 10]
    rows = _read_jsonl(Path(path))
    if not rows:
        return {
            "status": "not_available",
            "reason": f"no gold chunk annotations at {path}",
            "simulated": False,
        }
    retrieved = [r["retrieved_ids"] for r in rows]
    gold = [r["gold_ids"] for r in rows]
    out = {"status": "computed", "simulated": False, "n_queries": len(rows)}
    for k in ks:
        out[f"recall_at_{k}"] = round(
            sum(recall_at_k(r, g, k) for r, g in zip(retrieved, gold)) / len(rows), 6
        )
        out[f"precision_at_{k}"] = round(
            sum(precision_at_k(r, g, k) for r, g in zip(retrieved, gold)) / len(rows), 6
        )
        out[f"ndcg_at_{k}"] = round(
            sum(ndcg_at_k(r, g, k) for r, g in zip(retrieved, gold)) / len(rows), 6
        )
    out["mrr"] = mrr(retrieved, gold)
    return out


def extraction_from_gold(path: str) -> dict:
    """
    Each line: {"atom_type": "RATE", "predicted": ["..."], "gold": ["..."]}
    """
    rows = _read_jsonl(Path(path))
    if not rows:
        return {
            "status": "not_available",
            "reason": f"no gold atom annotations at {path}",
            "simulated": False,
        }
    by_type: Dict[str, List[dict]] = {}
    for row in rows:
        by_type.setdefault(row["atom_type"], []).append(row)
    table = {"status": "computed", "simulated": False}
    for atom_type, group in sorted(by_type.items()):
        preds: List[str] = []
        golds: List[str] = []
        for row in group:
            preds.extend(row.get("predicted") or [])
            golds.extend(row.get("gold") or [])
        rec = extraction_recall(preds, golds)
        prec = extraction_precision(preds, golds)
        table[atom_type] = {
            "recall": rec,
            "precision": prec,
            "f1": extraction_f1(rec, prec),
            "n": len(group),
        }
    return table
