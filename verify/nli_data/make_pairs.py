"""
verify/nli_data/make_pairs.py
------------------------------
Build NLI training pairs: (premise=chunk_text, hypothesis=atom_claim, label).

Labels:
    entailment   – atom is a correct statement supported by the chunk
    contradiction – atom has a perturbed/wrong value vs. the chunk
    neutral      – atom is about a different topic, unrelated to chunk

Usage (standalone):
    python -m verify.nli_data.make_pairs \\
        --atoms  data/processed/atoms_labeled.jsonl \\
        --chunks data/processed/chunks.jsonl \\
        --out    data/processed/nli_pairs.jsonl \\
        --n_neutral 2

The script emits one JSONL line per NLI pair with fields:
    {premise, hypothesis, label, atom_id, chunk_id, atom_type, regulator}
"""
from __future__ import annotations

import argparse
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any

from verify.nli_data.perturb import perturb

logger = logging.getLogger("verify.nli_data.make_pairs")
logging.basicConfig(level=logging.INFO)

_rng = random.Random(42)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d NLI pairs to %s", len(rows), path)


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def make_entailment(atom: Dict, chunk_text: str) -> Dict:
    """True atom ← chunk: ENTAILMENT."""
    return {
        "premise": chunk_text,
        "hypothesis": atom.get("claim", atom.get("text", "")),
        "label": "entailment",
        "atom_id": atom.get("atom_id", ""),
        "chunk_id": atom.get("cited_chunk", ""),
        "atom_type": atom.get("type", ""),
        "regulator": atom.get("regulator", ""),
    }


def make_contradiction(atom: Dict, chunk_text: str) -> Dict | None:
    """Perturbed atom ← same chunk: CONTRADICTION."""
    atom_type = atom.get("type", "")
    claim = atom.get("claim", atom.get("text", ""))
    perturbed = perturb(atom_type, claim)
    if perturbed is None:
        return None
    return {
        "premise": chunk_text,
        "hypothesis": perturbed,
        "label": "contradiction",
        "atom_id": atom.get("atom_id", ""),
        "chunk_id": atom.get("cited_chunk", ""),
        "atom_type": atom_type,
        "regulator": atom.get("regulator", ""),
    }


def make_neutral(atom: Dict, other_chunk_text: str) -> Dict:
    """True atom ← unrelated chunk: NEUTRAL."""
    return {
        "premise": other_chunk_text,
        "hypothesis": atom.get("claim", atom.get("text", "")),
        "label": "neutral",
        "atom_id": atom.get("atom_id", ""),
        "chunk_id": "",
        "atom_type": atom.get("type", ""),
        "regulator": atom.get("regulator", ""),
    }


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_pairs(
    atoms_path: str,
    chunks_path: str,
    out_path: str,
    n_neutral: int = 2,
) -> List[Dict]:
    """
    Build NLI training pairs.

    For each atom:
    - 1 entailment pair (atom claim ← its cited chunk)
    - 1 contradiction pair (perturbed claim ← same chunk)
    - n_neutral neutral pairs (atom claim ← random unrelated chunks)

    Atoms without a cited_chunk or whose chunk is not found are skipped.
    """
    atoms = _load_jsonl(atoms_path)
    chunks_list = _load_jsonl(chunks_path)
    chunk_map: Dict[str, str] = {c["chunk_id"]: c["text"] for c in chunks_list}

    all_texts = list(chunk_map.values())
    pairs: List[Dict] = []

    n_entail = 0
    n_contra = 0
    n_neut = 0
    n_skipped = 0

    for atom in atoms:
        cited = atom.get("cited_chunk")
        if not cited or cited not in chunk_map:
            n_skipped += 1
            continue

        chunk_text = chunk_map[cited]

        # Entailment
        pairs.append(make_entailment(atom, chunk_text))
        n_entail += 1

        # Contradiction
        contra = make_contradiction(atom, chunk_text)
        if contra:
            pairs.append(contra)
            n_contra += 1

        # Neutral (random chunks from OTHER regulators if possible)
        atom_reg = atom.get("regulator", "")
        candidates = [
            t for cid, t in chunk_map.items()
            if cid != cited
        ]
        if not candidates:
            candidates = all_texts

        chosen = _rng.sample(candidates, min(n_neutral, len(candidates)))
        for c_text in chosen:
            pairs.append(make_neutral(atom, c_text))
            n_neut += 1

    logger.info(
        "Pairs built: entailment=%d, contradiction=%d, neutral=%d | skipped=%d",
        n_entail, n_contra, n_neut, n_skipped,
    )

    _rng.shuffle(pairs)
    _write_jsonl(pairs, out_path)
    return pairs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build NLI training pairs for RegGuard.")
    parser.add_argument("--atoms",  required=True, help="Path to labeled atoms JSONL")
    parser.add_argument("--chunks", required=True, help="Path to corpus chunks JSONL")
    parser.add_argument("--out",    required=True, help="Output JSONL path")
    parser.add_argument("--n_neutral", type=int, default=2,
                        help="Number of neutral pairs per atom (default 2)")
    args = parser.parse_args()

    build_pairs(
        atoms_path=args.atoms,
        chunks_path=args.chunks,
        out_path=args.out,
        n_neutral=args.n_neutral,
    )
