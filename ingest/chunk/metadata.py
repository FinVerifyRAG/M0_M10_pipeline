import re
from typing import Dict, Any, Optional
from common.dates import parse_indian_date

def extract_metadata(text: str) -> Dict[str, Any]:
    """
    Extracts dates and supersession phrases from raw text chunks using regex.
    """
    metadata = {
        "extracted_issue_date": None,
        "extracted_effective_date": None,
        "supersedes": [],
        "amends": []
    }
    
    # Attempt to find dates near "dated" or "issue date"
    date_pattern = r"(?:dated|date of issue|issued on)[\s:]*([\d]{1,2}[/-][\d]{1,2}[/-][\d]{2,4}|[\d]{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+[\d]{4})"
    issue_match = re.search(date_pattern, text, re.IGNORECASE)
    if issue_match:
        parsed = parse_indian_date(issue_match.group(1))
        if parsed:
            metadata["extracted_issue_date"] = parsed.strftime("%Y-%m-%d")

    # Attempt to find effective dates
    effective_pattern = r"(?:effective from|with effect from|w\.e\.f\.?)[\s:]*([\d]{1,2}[/-][\d]{1,2}[/-][\d]{2,4}|[\d]{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+[\d]{4})"
    effective_match = re.search(effective_pattern, text, re.IGNORECASE)
    if effective_match:
        parsed = parse_indian_date(effective_match.group(1))
        if parsed:
            metadata["extracted_effective_date"] = parsed.strftime("%Y-%m-%d")

    # Extract supersession text
    # "in supersession of circular <id>"
    super_pattern = r"(?:in supersession of|supersedes)(?:\s+the)?\s+(?:master\s+)?(?:circular|direction|guideline)[s]?\s+([A-Za-z0-9\.\-\/]+)"
    for match in re.finditer(super_pattern, text, re.IGNORECASE):
        metadata["supersedes"].append(match.group(1).rstrip('.').strip())
        
    # Extract amendment text
    amend_pattern = r"amends(?:\s+the)?\s+(?:master\s+)?(?:circular|direction|guideline)[s]?\s+([A-Za-z0-9\.\-\/]+)"
    for match in re.finditer(amend_pattern, text, re.IGNORECASE):
        metadata["amends"].append(match.group(1).rstrip('.').strip())

    return metadata
