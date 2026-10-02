"""
bench/annotation/annotator.py
------------------------------
M10 Benchmark: Atom-level annotation schema and pre-labelling tools.

Labels (per the implementation plan)
-------------------------------------
  supported   — atom is factually correct and grounded in the evidence
  unsupported — atom is not found in or contradicted by the evidence
  outdated    — atom was once correct but is superseded by a later amendment

Pre-labelling pipeline (step 1)
--------------------------------
  1. Run the full pipeline on each benchmark question.
  2. Collect per-atom (question_id, atom_id, atom_text, v1_status,
     v2_entail_prob, risk, regulator, atom_type).
  3. Derive a heuristic label:
       v1 MATCH + risk < 0.3  -> "supported" (high confidence)
       v1 MISMATCH            -> "unsupported"
       risk >= 0.7            -> "unsupported"
       else                   -> "uncertain" (needs human review)
  4. Write to annotation JSONL; annotators review and fix "uncertain" rows.

Public API
----------
    from bench.annotation.annotator import AnnotationRecord, prelabel, save_annotations, load_annotations
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("bench.annotation.annotator")

# ── Valid labels ──────────────────────────────────────────────────────────────
VALID_LABELS = {"supported", "unsupported", "outdated", "uncertain"}

# ── Heuristic thresholds ──────────────────────────────────────────────────────
_CONFIDENT_SUPPORT_RISK = 0.30   # risk <= this with V1=MATCH -> "supported"
_CONFIDENT_REJECT_RISK  = 0.70   # risk >= this -> "unsupported"


@dataclass
class AnnotationRecord:
    """One atom annotation record."""
    question_id:    str
    atom_id:        str
    atom_type:      str
    atom_text:      str
    atom_claim:     str
    regulator:      str
    v1_status:      str                 # MATCH | MISMATCH | NOT_FOUND | NA
    v2_entail_prob: Optional[float]
    risk:           float
    prelabel:       str                 # heuristic label: supported|unsupported|uncertain
    label:          Optional[str]       # human-verified final label (or None if pending)
    annotator_id:   Optional[str] = None
    notes:          str = ""
    schema_version: str = "1.0"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AnnotationRecord":
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def is_verified(self) -> bool:
        return self.label is not None and self.label in VALID_LABELS - {"uncertain"}

    @property
    def is_wrong(self) -> bool:
        """True if the atom is wrong (for use as the aggregator's y label)."""
        return self.label in ("unsupported", "outdated")


def prelabel(
    v1_status:      str,
    v2_entail_prob: Optional[float],
    risk:           float,
) -> str:
    """
    Heuristic pre-label for an atom.

    Rules (in order):
      1. V1 MISMATCH -> unsupported (deterministic contradiction)
      2. risk >= _CONFIDENT_REJECT_RISK -> unsupported
      3. V1 MATCH and risk <= _CONFIDENT_SUPPORT_RISK -> supported
      4. V2 entail_prob >= 0.9 and risk <= _CONFIDENT_SUPPORT_RISK -> supported
      5. else -> uncertain (needs human review)
    """
    if v1_status == "MISMATCH":
        return "unsupported"
    if risk >= _CONFIDENT_REJECT_RISK:
        return "unsupported"
    if v1_status == "MATCH" and risk <= _CONFIDENT_SUPPORT_RISK:
        return "supported"
    if v2_entail_prob is not None and v2_entail_prob >= 0.90 and risk <= _CONFIDENT_SUPPORT_RISK:
        return "supported"
    return "uncertain"


def save_annotations(records: List[AnnotationRecord], path: str) -> None:
    """Write annotation records to JSONL."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
    logger.info("Saved %d annotation records to %s", len(records), path)


def load_annotations(path: str) -> List[AnnotationRecord]:
    """Load annotation records from JSONL."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(AnnotationRecord.from_dict(json.loads(line)))
    return records


def annotation_stats(records: List[AnnotationRecord]) -> dict:
    """Compute annotation statistics."""
    total     = len(records)
    verified  = sum(1 for r in records if r.is_verified)
    supported = sum(1 for r in records if r.label == "supported")
    unsup     = sum(1 for r in records if r.label == "unsupported")
    outdated  = sum(1 for r in records if r.label == "outdated")
    uncertain = sum(1 for r in records if r.label == "uncertain" or r.label is None)

    return {
        "total":      total,
        "verified":   verified,
        "supported":  supported,
        "unsupported": unsup,
        "outdated":   outdated,
        "uncertain":  uncertain,
        "pct_verified": round(verified / max(total, 1) * 100, 1),
        "pct_wrong":    round((unsup + outdated) / max(total, 1) * 100, 1),
    }
