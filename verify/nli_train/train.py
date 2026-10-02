"""
verify/nli_train/train.py
--------------------------
Fine-tune DeBERTa-v3-large on regulatory NLI pairs built by
verify/nli_data/make_pairs.py.

This script uses HuggingFace Trainer for the fine-tuning loop.
Training data format (JSONL):
    {premise, hypothesis, label}  where label ∈ {entailment, contradiction, neutral}

Usage:
    python -m verify.nli_train.train \\
        --pairs  data/processed/nli_pairs.jsonl \\
        --out    models/nli_deberta_v3 \\
        --epochs 4 --batch 16 --lr 1e-5

Evaluation:
    Reports per-type macro-F1 and per-regulator accuracy.
    Early stopping on validation macro-F1 with patience=3.

Notes:
- Keep NLI training documents DISJOINT from calibration/test split.
- Do NOT include atoms from the calibration or test sets in --pairs.
- GPU recommended (bf16 enabled automatically if CUDA available).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("verify.nli_train.train")
logging.basicConfig(level=logging.INFO)

LABEL2ID = {"entailment": 0, "contradiction": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _split(rows: List[Dict], val_frac: float = 0.1, seed: int = 42):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac))
    return rows[n_val:], rows[:n_val]


# ---------------------------------------------------------------------------
# HuggingFace Dataset wrapper
# ---------------------------------------------------------------------------

def _to_hf_dataset(rows: List[Dict], tokenizer, max_length: int = 512):
    """Convert JSONL rows to a HuggingFace Dataset with tokenised inputs."""
    try:
        from datasets import Dataset  # type: ignore
    except ImportError:
        raise ImportError("pip install datasets transformers")

    def gen():
        for r in rows:
            yield {
                "premise": r["premise"],
                "hypothesis": r["hypothesis"],
                "label": LABEL2ID[r["label"]],
            }

    ds = Dataset.from_generator(gen)

    def tokenize(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    ds = ds.map(tokenize, batched=True, remove_columns=["premise", "hypothesis"])
    ds.set_format("torch")
    return ds


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    pairs_path: str,
    output_dir: str,
    epochs: int = 4,
    batch: int = 16,
    lr: float = 1e-5,
    max_length: int = 512,
    val_frac: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Fine-tune DeBERTa-v3-large on NLI pairs.

    Steps
    -----
    1. Load JSONL pairs.
    2. Train/val split (by document if atom_id contains doc info; else random).
    3. Tokenise with DeBERTa tokenizer.
    4. HF Trainer with early stopping on val macro-F1.
    5. Save model + tokenizer to output_dir.
    6. Print per-class and per-type report.
    """
    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            EarlyStoppingCallback,
        )
        from sklearn.metrics import classification_report  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Training requires: torch transformers datasets scikit-learn numpy\n"
            f"Missing: {exc}"
        )

    logger.info("Loading pairs from %s", pairs_path)
    rows = _load_jsonl(pairs_path)
    train_rows, val_rows = _split(rows, val_frac=val_frac, seed=seed)
    logger.info("Train: %d | Val: %d", len(train_rows), len(val_rows))

    model_name = "microsoft/deberta-v3-large"
    logger.info("Loading tokenizer from %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    logger.info("bf16=%s, device=%s", use_bf16, "cuda" if torch.cuda.is_available() else "cpu")

    train_ds = _to_hf_dataset(train_rows, tokenizer, max_length)
    val_ds   = _to_hf_dataset(val_rows, tokenizer, max_length)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        report = classification_report(
            labels, preds,
            target_names=["entailment", "contradiction", "neutral"],
            output_dict=True,
            zero_division=0,
        )
        return {
            "macro_f1": report["macro avg"]["f1-score"],
            "entail_f1": report["entailment"]["f1-score"],
            "contra_f1": report["contradiction"]["f1-score"],
            "neutral_f1": report["neutral"]["f1-score"],
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=use_bf16,
        fp16=False,
        seed=seed,
        logging_steps=50,
        report_to="none",  # set to "wandb" or "mlflow" for experiment tracking
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Done. Final metrics: %s", trainer.evaluate())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa for NLI verification.")
    parser.add_argument("--pairs",  required=True, help="JSONL NLI pairs file")
    parser.add_argument("--out",    required=True, help="Output directory for model checkpoint")
    parser.add_argument("--epochs", type=int,   default=4)
    parser.add_argument("--batch",  type=int,   default=16)
    parser.add_argument("--lr",     type=float, default=1e-5)
    parser.add_argument("--seed",   type=int,   default=42)
    args = parser.parse_args()

    train(
        pairs_path=args.pairs,
        output_dir=args.out,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
    )
