"""
bench/splits.py
----------------
M10 Benchmark: Dataset splitting into agg_train / agg_val / calibration / test.

Critical rules (from the implementation plan)
---------------------------------------------
  • Split by **question family or document**, NEVER by atom.
    Atoms from the same question are correlated; leaking them across splits
    invalidates the guarantee calibration.
  • `agg_train`    — used to train the M6 LightGBM aggregator.
  • `agg_val`      — carved from agg_train (20%); used for aggregator
                     hyperparameter tuning only.  Never touches calibration.
  • `calibration`  — used to calibrate M7 LTT thresholds (agg_val must not
                     appear here; they come from different questions).
  • `test`         — held-out evaluation (never seen during training or calibration).
  • NLI training data lives in a completely disjoint document set
    (use disjoint_document_split to enforce this).
  • Temporal / drift splits are ordered by issue_date (not shuffled).

Phase 3: Statistical certification split
----------------------------------------
  `split_questions` now also returns `agg_val` in its dict:
    splits = split_questions(questions, config)
    agg_train   = splits["agg_train"]
    agg_val     = splits["agg_val"]
    calibration = splits["calibration"]
    test        = splits["test"]

Phase 4: Leakage prevention
----------------------------
  `disjoint_document_split(all_chunks, questions)` produces a document set
  for NLI training that is provably disjoint from calibration/test documents.

Public API
----------
    from bench.splits import split_questions, SplitConfig, load_split, disjoint_document_split
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("bench.splits")


@dataclass
class SplitConfig:
    """Configuration for question splitting."""
    agg_train_frac:  float = 0.50    # Fraction for aggregator training
    calibration_frac: float = 0.25   # Fraction for LTT calibration
    test_frac:        float = 0.25   # Fraction for held-out test
    seed:            int   = 42
    split_by:        str   = "family"   # "family" | "document" | "random"
    min_per_split:   int   = 20         # Minimum questions per split
    # Phase 3: agg_val is carved from agg_train.
    # agg_val_frac is a fraction OF agg_train, NOT of the total.
    agg_val_frac:    float = 0.20       # 20% of agg_train → agg_val

    def __post_init__(self):
        total = self.agg_train_frac + self.calibration_frac + self.test_frac
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Fractions must sum to 1.0, got {total:.4f}")
        if not 0.0 < self.agg_val_frac < 1.0:
            raise ValueError(f"agg_val_frac must be in (0, 1), got {self.agg_val_frac}")


def split_questions(
    questions: list,           # List[QuestionItem] (or dicts with same fields)
    config:    Optional[SplitConfig] = None,
) -> Dict[str, list]:
    """
    Split benchmark questions into agg_train / agg_val / calibration / test.

    Phase 3: ``agg_val`` is carved from ``agg_train`` after the primary split.
    It is used for aggregator hyperparameter tuning and MUST NOT overlap with
    ``calibration`` or ``test``.

    Splitting strategy
    ------------------
    ``split_by="family"``:
        Group questions by family, then deterministically assign whole groups
        to splits so the fractions are approximately met.  This prevents a
        RATE question from the same chunk appearing in both train and test.

    ``split_by="document"``:
        Group by source_chunk_id's document prefix.  Stronger isolation.

    ``split_by="random"``:
        Plain random shuffle + fraction cut.  Only for quick experiments.

    Returns
    -------
    Dict with keys ``"agg_train"``, ``"agg_val"``, ``"calibration"``, ``"test"``.
    """
    if config is None:
        config = SplitConfig()
    if not questions:
        return {"agg_train": [], "agg_val": [], "calibration": [], "test": []}

    rng = random.Random(config.seed)

    if config.split_by == "family":
        base = _split_by_key(questions, key_fn=lambda q: _get_field(q, "family"), config=config, rng=rng)
    elif config.split_by == "document":
        base = _split_by_key(questions, key_fn=lambda q: _get_field(q, "source_chunk_id", "").split("_")[0], config=config, rng=rng)
    else:  # random
        base = _split_random(questions, config, rng)

    # Phase 3: carve agg_val from agg_train
    base["agg_train"], base["agg_val"] = _carve_agg_val(
        base["agg_train"], config.agg_val_frac, rng
    )
    _log_split_sizes(base)
    return base


def _get_field(q, field: str, default="") -> str:
    if hasattr(q, field):
        return getattr(q, field, default)
    if isinstance(q, dict):
        return q.get(field, default)
    return default


def _split_by_key(questions, key_fn, config: SplitConfig, rng: random.Random) -> Dict[str, list]:
    """Assign whole key-groups to splits proportionally."""
    # Group questions by key
    groups: Dict[str, list] = {}
    for q in questions:
        k = key_fn(q)
        groups.setdefault(k, []).append(q)

    keys = list(groups.keys())
    rng.shuffle(keys)
    n = len(keys)

    n_agg  = max(1, round(n * config.agg_train_frac))
    n_calib = max(1, round(n * config.calibration_frac))
    # Remainder goes to test
    n_agg  = min(n_agg,  n - 2)
    n_calib = min(n_calib, n - n_agg - 1)

    agg_keys   = keys[:n_agg]
    calib_keys = keys[n_agg : n_agg + n_calib]
    test_keys  = keys[n_agg + n_calib :]

    # NOTE: agg_val is NOT carved here; split_questions() handles that
    splits = {
        "agg_train":   [q for k in agg_keys   for q in groups[k]],
        "agg_val":     [],   # placeholder; carved in split_questions()
        "calibration": [q for k in calib_keys for q in groups[k]],
        "test":        [q for k in test_keys  for q in groups[k]],
    }
    return splits


def _split_random(questions, config: SplitConfig, rng: random.Random) -> Dict[str, list]:
    """Plain random split."""
    qs = list(questions)
    rng.shuffle(qs)
    n = len(qs)
    n_agg   = round(n * config.agg_train_frac)
    n_calib = round(n * config.calibration_frac)
    splits = {
        "agg_train":   qs[:n_agg],
        "agg_val":     [],   # placeholder; carved in split_questions()
        "calibration": qs[n_agg : n_agg + n_calib],
        "test":        qs[n_agg + n_calib:],
    }
    return splits


def _carve_agg_val(
    agg_train: list,
    val_frac: float,
    rng: random.Random,
) -> Tuple[list, list]:
    """
    Phase 3: Carve an agg_val subset from agg_train.

    The carved subset is used ONLY for aggregator validation / hyperparameter
    tuning and MUST NOT be mixed into calibration or test data.

    Returns (remaining_agg_train, agg_val).
    """
    if not agg_train:
        return [], []
    shuffled = list(agg_train)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_frac))
    agg_val   = shuffled[:n_val]
    remaining = shuffled[n_val:]
    logger.info("Phase 3: agg_val carved: %d items  agg_train remaining: %d", len(agg_val), len(remaining))
    return remaining, agg_val


def _log_split_sizes(splits: Dict[str, list]) -> None:
    for name, items in splits.items():
        logger.info("Split %s: %d questions", name, len(items))


# ── Temporal (drift) split ────────────────────────────────────────────────────

def temporal_split(
    questions: list,
    cutoff_date: str,       # ISO date "YYYY-MM-DD"
) -> Tuple[list, list]:
    """
    Split questions into pre-cutoff (training/calibration) and
    post-cutoff (drift test) based on ``issue_date``.

    Returns (pre, post) lists.
    """
    pre, post = [], []
    for q in questions:
        issue_date = _get_field(q, "issue_date", "")
        if issue_date and issue_date >= cutoff_date:
            post.append(q)
        else:
            pre.append(q)
    logger.info("Temporal split (cutoff=%s): pre=%d, post=%d", cutoff_date, len(pre), len(post))
    return pre, post


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_split(split: Dict[str, list], out_dir: str) -> None:
    """Save each split as a JSONL file in out_dir."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    for name, items in split.items():
        outpath = p / f"{name}.jsonl"
        with open(outpath, "w", encoding="utf-8") as f:
            for item in items:
                d = item.to_dict() if hasattr(item, "to_dict") else item
                f.write(json.dumps(d) + "\n")
        logger.info("Saved %s: %d items -> %s", name, len(items), outpath)


def load_split(split_dir: str) -> Dict[str, list]:
    """Load all split JSONL files from split_dir."""
    from bench.question_gen.generator import QuestionItem
    p = Path(split_dir)
    result = {}
    for name in ("agg_train", "agg_val", "calibration", "test"):
        path = p / f"{name}.jsonl"
        items = []
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(QuestionItem.from_dict(json.loads(line)))
        result[name] = items
    return result


# ── Phase 4: Leakage prevention ───────────────────────────────────────────────

def disjoint_document_split(
    all_chunks: list,
    questions:  list,
    config:     Optional[SplitConfig] = None,
) -> Dict[str, list]:
    """
    Phase 4: Produce a set of document chunks whose *source documents* are
    provably disjoint from the calibration and test question documents.

    This is mandatory for NLI training data: the NLI model must not train on
    evidence chunks from the same documents that will appear in calibration or
    test, otherwise the V2 NLI score leaks into M7 thresholds.

    Algorithm
    ---------
    1. Split questions with split_questions().
    2. Collect the set of ``doc_id`` prefixes (e.g. chunk_id.split('_')[0])
       from calibration + test questions.
    3. Return chunks whose document prefix is NOT in that banned set as
       ``nli_train_chunks``.
    4. The banned chunks are returned as ``held_out_chunks``.

    Parameters
    ----------
    all_chunks : List[Chunk] — the full corpus.
    questions  : List[QuestionItem] — used to compute the banned document set.
    config     : SplitConfig (uses split_by="document" internally).

    Returns
    -------
    Dict with keys:
      ``nli_train_chunks``  — safe for NLI training.
      ``held_out_chunks``   — must NOT be used for NLI training.
      ``banned_doc_ids``    — set of document prefixes that are banned.
    """
    if config is None:
        config = SplitConfig(split_by="document")

    splits = split_questions(questions, config)

    # Collect banned doc prefixes from calibration + test
    banned_docs: set = set()
    for split_name in ("calibration", "test"):
        for q in splits.get(split_name, []):
            src = _get_field(q, "source_chunk_id", "")
            if src:
                # Document prefix = everything before the first '_'
                banned_docs.add(src.split("_")[0])

    nli_train: list = []
    held_out:  list = []
    for chunk in all_chunks:
        chunk_id = getattr(chunk, "chunk_id", getattr(chunk, "id", ""))
        doc_prefix = chunk_id.split("_")[0] if chunk_id else ""
        if doc_prefix in banned_docs:
            held_out.append(chunk)
        else:
            nli_train.append(chunk)

    logger.info(
        "Phase 4 disjoint split: banned_docs=%d  nli_train=%d  held_out=%d",
        len(banned_docs), len(nli_train), len(held_out),
    )
    return {
        "nli_train_chunks": nli_train,
        "held_out_chunks":  held_out,
        "banned_doc_ids":   list(banned_docs),
    }


class LeakageError(ValueError):
    """A document id appears in more than one split."""


def document_leakage(splits: Dict[str, list], doc_field: str = "doc_id") -> dict:
    """Return leaked document ids. A document may appear in only one split."""
    owner = {}
    leaked = []
    for name, rows in splits.items():
        if name == "agg_val":
            # agg_val is carved from train and is not a document-level holdout.
            continue
        for row in rows:
            doc = _get_field(row, doc_field, "") or _get_field(row, "document_id", "")
            if not doc:
                continue
            if doc in owner and owner[doc] != name:
                leaked.append({"doc_id": doc, "splits": sorted({owner[doc], name})})
            else:
                owner[doc] = name
    return {"leaked": leaked, "n_documents": len(owner)}


def assert_no_document_leakage(splits: Dict[str, list], doc_field: str = "doc_id") -> dict:
    report = document_leakage(splits, doc_field=doc_field)
    if report["leaked"]:
        raise LeakageError(f"Document leakage: {report['leaked'][:5]}")
    return report


def split_labeled_atoms(
    rows: list,
    out_dir: str = "data/splits",
    config: Optional[SplitConfig] = None,
) -> Dict[str, list]:
    """
    Document-level train / calibration / test split.
    Refuses unlabeled rows. Does not invent labels.
    """
    unlabeled = [r for r in rows if r.get("label") not in ("supported", "unsupported", "outdated")]
    if unlabeled:
        raise ValueError(
            f"Refusing to split {len(unlabeled)} unlabeled rows. "
            "Label them before creating data/splits."
        )
    prepared = []
    for row in rows:
        item = dict(row)
        doc = item.get("doc_id") or item.get("document_id") or ""
        item["doc_id"] = doc
        # split_questions(document) groups on the prefix of source_chunk_id.
        item["source_chunk_id"] = str(doc).replace("_", "-")
        prepared.append(item)
    cfg = config or SplitConfig(split_by="document", min_per_split=1)
    splits = split_questions(prepared, cfg)
    report = assert_no_document_leakage(splits)
    dest = Path(out_dir)
    save_split(splits, str(dest))
    (dest / "leakage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return splits


def ensure_min_calibration(
    splits: Dict[str, list],
    min_calib: int = 100,
) -> bool:
    """
    Phase 3/4 guard: returns True if the calibration split has at least
    ``min_calib`` questions.  Warns and returns False otherwise.

    M7 LTT requires a minimum calibration set to produce valid p-values.
    Fewer than ~100 calibration atoms per stratum causes everything to be
    ABSTAINED (accept_below=None).
    """
    n = len(splits.get("calibration", []))
    if n < min_calib:
        logger.warning(
            "Calibration split has only %d questions (minimum %d). "
            "M7 certification will produce mostly ABSTAINED decisions.",
            n, min_calib,
        )
        return False
    logger.info("Calibration split size OK: %d >= %d", n, min_calib)
    return True
