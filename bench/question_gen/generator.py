"""
bench/question_gen/generator.py
--------------------------------
M10 Benchmark: Generates diverse regulatory questions from document chunks.

Question families (as per the implementation plan):
  1. RATE       — "What is the rate/percentage for X?"
  2. THRESHOLD  — "What is the threshold/limit for Y?"
  3. SECTION    — "What does Section N of [Act] state about Z?"
  4. DATE       — "When did regulation X come into effect?"
  5. APPLICABILITY — "To whom does rule X apply?"
  6. ENTITY     — "Who is responsible for X under [Act]?"
  7. FOLLOWUP   — Second-turn follow-up questions referencing a prior answer.
  8. SPECIFIC   — Document-specific questions requiring exact text.
  9. TEMPORAL   — Questions whose answer changed after an amendment.

Public API
----------
    from bench.question_gen.generator import generate_questions, QuestionItem
    questions = generate_questions(chunks, n=500, seed=42)
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("bench.question_gen.generator")

# ── Supported question families ───────────────────────────────────────────────
QUESTION_FAMILIES = [
    "RATE", "THRESHOLD", "SECTION", "DATE",
    "APPLICABILITY", "ENTITY", "FOLLOWUP", "SPECIFIC", "TEMPORAL",
]

# ── Template banks (per family) ───────────────────────────────────────────────
_TEMPLATES: dict[str, list[str]] = {
    "RATE": [
        "What is the {rate_name} rate applicable to {entity} under {regulator} regulations?",
        "What percentage is required for {rate_name} as per {regulator} guidelines?",
        "What is the current {rate_name} requirement for {entity}?",
    ],
    "THRESHOLD": [
        "What is the minimum/maximum threshold for {threshold_name} under {regulator} regulations?",
        "What is the permissible limit for {threshold_name} as per {regulator}?",
        "What threshold applies to {threshold_name} for {entity}?",
    ],
    "SECTION": [
        "What does {section_ref} of {doc_title} state?",
        "What are the provisions of {section_ref} under {regulator} regulations?",
        "Explain the requirements under {section_ref} of {doc_title}.",
    ],
    "DATE": [
        "When did {doc_title} come into effect?",
        "What is the effective date of the {regulator} circular on {topic}?",
        "When were the {topic} regulations last amended by {regulator}?",
    ],
    "APPLICABILITY": [
        "To whom does {doc_title} apply?",
        "Which entities are covered under {topic} regulations of {regulator}?",
        "Who must comply with {section_ref} of {doc_title}?",
    ],
    "ENTITY": [
        "Who is responsible for {topic} under {regulator} regulations?",
        "Which authority oversees {topic} as per {doc_title}?",
        "What is the role of {entity} in {topic} according to {regulator}?",
    ],
    "SPECIFIC": [
        "What are the specific requirements for {topic} as per {regulator}?",
        "Describe the obligations of {entity} regarding {topic} under {doc_title}.",
        "What must {entity} do regarding {topic} according to {regulator} guidelines?",
    ],
    "TEMPORAL": [
        "What were the {topic} requirements under {regulator} as of {year}?",
        "How did the {topic} rules change after the {regulator} amendment of {year}?",
        "What was the {rate_name} rate before the {year} revision by {regulator}?",
    ],
}

# ── Regex for extracting hints from chunk text ────────────────────────────────
_PCT_RE  = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|per\s?cent|percent)", re.I)
_INR_RE  = re.compile(r"(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?\s*(?:lakh|crore|thousand)?", re.I)
_SEC_RE  = re.compile(r"\b(?:Section|Regulation|Clause|Rule)\s+\d+[A-Z]?(?:\(\w+\))*", re.I)
_DATE_RE = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|"
                      r"October|November|December)\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b", re.I)


def _extract_hints(text: str) -> dict:
    """Extract hints from a chunk for template slot filling."""
    pcts   = _PCT_RE.findall(text)
    secs   = _SEC_RE.findall(text)
    dates  = _DATE_RE.findall(text)
    return {
        "has_rate":    len(pcts) > 0,
        "has_section": len(secs) > 0,
        "has_date":    len(dates) > 0,
        "first_pct":   pcts[0] if pcts else "N/A",
        "first_sec":   secs[0] if secs else "N/A",
        "first_date":  dates[0] if dates else "N/A",
    }


def _chunk_topic(text: str, max_words: int = 6) -> str:
    """Extract a short topic phrase from chunk text."""
    # Take the first meaningful phrase (skip boilerplate preambles)
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text[:300])
    words = clean.split()
    return " ".join(words[:max_words]) if words else "regulatory compliance"


# ── QuestionItem dataclass ────────────────────────────────────────────────────

@dataclass
class QuestionItem:
    """One benchmark question with metadata."""
    question_id:    str
    question:       str
    family:         str         # RATE / THRESHOLD / SECTION / DATE / ...
    regulator:      str         # RBI / SEBI / INCOMETAX / ...
    source_chunk_id: str        # Chunk used to generate the question
    issue_date:     str         # From chunk metadata
    gold_chunk_ids: List[str]   # Chunks that should appear in retrieval
    expected_atoms: List[str]   # Atom types expected in the answer
    temporal_tag:   bool = False  # True if answer changes across amendments
    amendment_date: Optional[str] = None  # Date of relevant amendment
    label:          Optional[str] = None  # "supported"|"unsupported"|"outdated" (post-annotation)
    schema_version: str = "1.0"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QuestionItem":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Core generator ────────────────────────────────────────────────────────────

def _make_qid(chunk_id: str, family: str, idx: int) -> str:
    h = hashlib.sha1(f"{chunk_id}:{family}:{idx}".encode()).hexdigest()[:8]
    return f"q_{family.lower()}_{h}"


def _fill_template(template: str, chunk, hints: dict, rng: random.Random) -> str:
    """Fill a template with chunk metadata and extracted hints."""
    meta    = getattr(chunk, "metadata", {}) or {}
    regulator = getattr(chunk, "regulator", "regulator")
    doc_title = meta.get("doc_title", meta.get("source", regulator + " circular"))
    section   = hints.get("first_sec", meta.get("section", "Section 1"))
    issue_date= getattr(chunk, "issue_date", "")
    year      = issue_date[:4] if issue_date else "recent"
    topic     = _chunk_topic(chunk.text)
    entity    = rng.choice(["banks", "NBFCs", "listed entities", "market intermediaries",
                             "regulated entities", "mutual funds"])
    rate_name = rng.choice(["CRR", "SLR", "repo rate", "risk weight", "capital adequacy",
                             "provisioning"])
    threshold_name = rng.choice(["exposure limit", "NPA threshold", "investment cap",
                                  "minimum capital", "disclosure trigger"])

    return template.format(
        rate_name=rate_name,
        threshold_name=threshold_name,
        entity=entity,
        regulator=regulator,
        doc_title=doc_title[:60],
        section_ref=section,
        topic=topic,
        year=year,
    )


def generate_questions(
    chunks: list,
    n: int = 500,
    seed: int = 42,
    family_weights: Optional[dict] = None,
) -> List[QuestionItem]:
    """
    Generate n benchmark questions from a list of Chunk objects.

    Parameters
    ----------
    chunks         : List of Chunk objects (from common.schemas).
    n              : Target number of questions (de-duplicated by text).
    seed           : RNG seed.
    family_weights : Dict[family -> relative weight]. Defaults to uniform.

    Returns
    -------
    List of QuestionItem objects.
    """
    if not chunks:
        logger.warning("No chunks provided; returning empty question list.")
        return []

    rng = random.Random(seed)

    weights = family_weights or {f: 1 for f in QUESTION_FAMILIES}
    # Remove FOLLOWUP/TEMPORAL from initial generation (needs special handling)
    base_families = [f for f in QUESTION_FAMILIES if f not in ("FOLLOWUP", "TEMPORAL")]
    base_weights  = [weights.get(f, 1) for f in base_families]

    questions: dict[str, QuestionItem] = {}   # dedup by text
    attempts = 0
    max_attempts = n * 10

    while len(questions) < n and attempts < max_attempts:
        attempts += 1
        chunk = rng.choice(chunks)
        hints = _extract_hints(chunk.text)

        # Choose family; bias toward chunk content
        if hints["has_rate"]:
            family = rng.choices(
                base_families,
                weights=[w * (3 if f == "RATE" else 1) for f, w in zip(base_families, base_weights)],
            )[0]
        elif hints["has_section"]:
            family = rng.choices(
                base_families,
                weights=[w * (3 if f == "SECTION" else 1) for f, w in zip(base_families, base_weights)],
            )[0]
        else:
            family = rng.choices(base_families, weights=base_weights)[0]

        templates = _TEMPLATES.get(family, _TEMPLATES["SPECIFIC"])
        template  = rng.choice(templates)

        try:
            question_text = _fill_template(template, chunk, hints, rng)
        except KeyError:
            continue

        if question_text in questions:
            continue

        qid = _make_qid(chunk.chunk_id, family, len(questions))
        qi = QuestionItem(
            question_id=qid,
            question=question_text,
            family=family,
            regulator=getattr(chunk, "regulator", "UNKNOWN"),
            source_chunk_id=chunk.chunk_id,
            issue_date=getattr(chunk, "issue_date", ""),
            gold_chunk_ids=[chunk.chunk_id],
            expected_atoms=[_FAMILY_TO_ATOM.get(family, "ENTITY")],
        )
        questions[question_text] = qi

    result = list(questions.values())[:n]
    logger.info("Generated %d/%d questions (seed=%d)", len(result), n, seed)
    return result


_FAMILY_TO_ATOM = {
    "RATE":          "RATE",
    "THRESHOLD":     "THRESHOLD",
    "SECTION":       "SECTION",
    "DATE":          "DATE",
    "APPLICABILITY": "APPLICABILITY",
    "ENTITY":        "ENTITY",
    "FOLLOWUP":      "ENTITY",
    "SPECIFIC":      "ENTITY",
    "TEMPORAL":      "DATE",
}


# ── Temporal question generator ───────────────────────────────────────────────

def generate_temporal_questions(
    chunks_before: list,
    chunks_after: list,
    n: int = 50,
    seed: int = 42,
) -> List[QuestionItem]:
    """
    Generate TEMPORAL questions using before/after amendment chunk pairs.

    Parameters
    ----------
    chunks_before : Chunks from the superseded version of a regulation.
    chunks_after  : Chunks from the effective version after amendment.
    n             : Number of temporal questions.
    seed          : RNG seed.

    Returns
    -------
    List of QuestionItem objects with temporal_tag=True.
    """
    rng = random.Random(seed)
    questions: list[QuestionItem] = []

    pairs = list(zip(chunks_before, chunks_after))
    if not pairs:
        return []

    for i in range(min(n, len(pairs))):
        cb, ca = rng.choice(pairs)
        hints  = _extract_hints(cb.text)
        template = rng.choice(_TEMPLATES["TEMPORAL"])
        try:
            text = _fill_template(template, cb, hints, rng)
        except KeyError:
            continue

        qi = QuestionItem(
            question_id=_make_qid(cb.chunk_id, "TEMPORAL", i),
            question=text,
            family="TEMPORAL",
            regulator=getattr(cb, "regulator", "UNKNOWN"),
            source_chunk_id=cb.chunk_id,
            issue_date=getattr(cb, "issue_date", ""),
            gold_chunk_ids=[cb.chunk_id, ca.chunk_id],
            expected_atoms=["DATE", "RATE"],
            temporal_tag=True,
            amendment_date=getattr(ca, "effective_from", None),
        )
        questions.append(qi)

    return questions


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_questions(questions: List[QuestionItem], path: str) -> None:
    """Save questions as JSONL."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q.to_dict()) + "\n")
    logger.info("Saved %d questions to %s", len(questions), path)


def load_questions(path: str) -> List[QuestionItem]:
    """Load questions from JSONL."""
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(QuestionItem.from_dict(json.loads(line)))
    return questions
