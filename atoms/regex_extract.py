import re
from typing import List, Optional, Tuple
from common.schemas import Atom, ATOM_TYPES

# Pre-compiled patterns for each atom type for fast regex fallback
_RATE_PATTERN = re.compile(
    r'\d+(?:\.\d+)?\s*(?:per\s*cent|percent|%(?:\s*per\s*annum|\s*p\.a\.)?)|'
    r'\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?',
    re.IGNORECASE
)
_THRESHOLD_PATTERN = re.compile(
    r'(₹\s*\d[\d,]*(?:\.\d+)?\s*(?:crore|lakh|thousand|cr\.?)?|'
    r'\b\d+\s*(?:days?|months?|years?|weeks?))\b',
    re.IGNORECASE
)
_SECTION_PATTERN = re.compile(
    r'\b(?:regulation|section|clause|rule|article|para(?:graph)?|'
    r'circular|master\s+direction|schedule)\s*\d+(?:\(\w+\))*(?:/\w+)*\b',
    re.IGNORECASE
)
_DATE_PATTERN = re.compile(
    r'\b(?:FY\s*\d{4}[-–]\d{2,4}|'
    r'\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}|'
    r'\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|'
    r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
    r'Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b',
    re.IGNORECASE
)

_PATTERN_MAP = {
    "RATE": _RATE_PATTERN,
    "THRESHOLD": _THRESHOLD_PATTERN,
    "SECTION": _SECTION_PATTERN,
    "DATE": _DATE_PATTERN,
}

def regex_extract(answer_text: str) -> List[dict]:
    """
    Fast regex-based pass for RATE, THRESHOLD, SECTION, DATE atoms.
    Returns dicts of {type, text, span} for enrichment by the LLM extractor.
    """
    found = []
    seen_spans = set()

    for atom_type, pattern in _PATTERN_MAP.items():
        for match in pattern.finditer(answer_text):
            span = (match.start(), match.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)
            found.append({
                "type": atom_type,
                "text": match.group(0).strip(),
                "span": span
            })

    return found


def find_span(answer_text: str, atom_text: str) -> Optional[Tuple[int, int]]:
    """Find the exact character span of atom_text inside answer_text."""
    idx = answer_text.find(atom_text)
    if idx == -1:
        return None
    return (idx, idx + len(atom_text))
