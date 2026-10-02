"""Append-only JSONL checkpoints so a failed run can resume."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()


def load_jsonl(path: str | Path) -> list:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def completed_ids(path: str | Path, key: str = "question_id") -> Set[str]:
    return {str(row[key]) for row in load_jsonl(path) if key in row and row.get("status") != "ERROR"}


def resume_pending(records: Iterable[Dict[str, Any]], checkpoint: str | Path, key: str = "question_id") -> list:
    done = completed_ids(checkpoint, key=key)
    return [r for r in records if str(r.get(key)) not in done]
