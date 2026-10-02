"""
atoms/splitter.py
------------------
Two-pass atom extraction pipeline.

Pass 1: Fast Regex — deterministic extraction of RATE, THRESHOLD, SECTION,
        DATE patterns with character spans.
Pass 2: LLM — semantic extraction with full structured fields, claims,
        ENTITY/APPLICABILITY atoms, and COMPUTED atom evidence blocks.

Merge logic
-----------
- LLM atoms take priority (they carry structured fields and claims).
- Regex atoms fill gaps where the LLM missed a literal pattern.
- Regex atoms synthesise a minimal Atom with claim derived from text.

Phase 2: Regex rescue atoms now receive the structured operator/value/unit
fields extracted by the regex pass so they are not structurally empty.
"""
from typing import List
from common.schemas import Atom, GeneratedAnswer, AtomOperator
from atoms.llm_extract import LLMAtomExtractor
from atoms.regex_extract import regex_extract, find_span
from common.llm_client import LLMClient
import uuid
import logging

logger = logging.getLogger("atoms.splitter")

# Regex-rescued atoms: infer operator from atom type
_TYPE_DEFAULT_OPERATOR = {
    "RATE":      AtomOperator.EQ,       # '5%' means '= 5%'
    "THRESHOLD": AtomOperator.EQ,       # '₹5 lakh' means '= ₹5 lakh'
    "DATE":      AtomOperator.EQ,       # specific date
    "SECTION":   AtomOperator.EQ,       # section reference
}


class AtomSplitter:
    """
    Two-pass atom extraction pipeline:
    1. Fast Regex pass: catches RATE, THRESHOLD, SECTION, DATE patterns.
    2. LLM pass: semantic extraction with full claims, structured fields,
       ENTITY/APPLICABILITY atoms, and COMPUTED atom evidence.

    The two result sets are merged and deduplicated by span/text.
    """

    def __init__(self, llm: LLMClient, prompt_path: str = "atoms/prompts/extract_v1.txt"):
        self.llm_extractor = LLMAtomExtractor(llm=llm, prompt_path=prompt_path)

    def split(self, answer: GeneratedAnswer) -> List[Atom]:
        answer_text = answer.answer_text

        # Short-circuit: "Not found in evidence" yields no atoms
        if "not found in evidence" in answer_text.lower():
            return []

        # --- Pass 1: Fast Regex ---
        regex_hits = regex_extract(answer_text)
        regex_texts = {h["text"] for h in regex_hits}

        # --- Pass 2: LLM Extraction (structured fields + COMPUTED support) ---
        llm_atoms = self.llm_extractor.extract(answer_text)

        # --- Merge: LLM takes priority; regex fills gaps ---
        llm_texts = {a.text for a in llm_atoms}
        merged_atoms = list(llm_atoms)

        for hit in regex_hits:
            if hit["text"] not in llm_texts:
                # LLM missed this regex hit; create a minimal structured atom
                atom_type = hit["type"]
                operator  = _TYPE_DEFAULT_OPERATOR.get(atom_type, AtomOperator.UNKNOWN)

                merged_atoms.append(Atom(
                    atom_id=f"atom_{uuid.uuid4().hex[:8]}",
                    type=atom_type,
                    text=hit["text"],
                    # Phase 2: propagate operator inferred from type
                    operator=operator,
                    claim=f"The answer states '{hit['text']}'.",
                    cited_chunk=None,
                    span=hit["span"],
                ))
                logger.debug(f"Regex rescued atom: {hit['text']} ({atom_type}) op={operator}")

        # Sort by position in text for readability
        merged_atoms.sort(key=lambda a: a.span[0] if a.span else 999999)

        return merged_atoms
