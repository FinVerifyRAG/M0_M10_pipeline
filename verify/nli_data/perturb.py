"""
verify/nli_data/perturb.py
--------------------------
Perturbation functions that generate CONTRADICTION hypotheses from true
regulatory statements. Used by make_pairs.py to build NLI training data.

Each function takes a *raw* atom text (or claim sentence) and returns a
*perturbed* version where the value has been changed so the hypothesis
contradicts the premise.

Design rationale:
- Perturbations are type-specific: a percentage perturber only touches
  numbers that look like percentages.
- We perturb by ±(10–50%) of the original value to stay plausible.
- For sections, we increment the section number so the reference is wrong.
- For dates, we shift by 1–3 years.
"""
from __future__ import annotations

import re
import random
from datetime import date, timedelta
from typing import Optional

from verify.normalizers import norm_pct, norm_amount, norm_date, norm_section

_rng = random.Random(42)

# ---------------------------------------------------------------------------
# Percentage perturbation
# ---------------------------------------------------------------------------

_PCT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(per\s*cent|percent|%)",
    re.IGNORECASE,
)


def perturb_rate(text: str) -> Optional[str]:
    """
    Replace the first percentage in text with a wrong value.
    '5%' → '7%' or '3%'  (random ±10–50% delta)
    Returns None if no percentage found.
    """
    m = _PCT_RE.search(text)
    if not m:
        return None
    original = float(m.group(1))
    delta = _rng.uniform(0.1, 0.5) * original
    direction = _rng.choice([1, -1])
    new_val = max(0.0, original + direction * delta)
    new_val = round(new_val, 2)
    perturbed = text[: m.start(1)] + str(new_val) + text[m.end(1):]
    return perturbed


# ---------------------------------------------------------------------------
# Rupee / threshold perturbation
# ---------------------------------------------------------------------------

_INR_RE = re.compile(
    r"(₹|Rs\.?|INR|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*"
    r"(lakh|lakhs?|lac|lacs?|crore|crores?|cr\.?|thousand|thousands?)?",
    re.IGNORECASE,
)


def perturb_threshold(text: str) -> Optional[str]:
    """
    Replace the first monetary amount in text with a wrong value.
    '₹5 lakh' → '₹3 lakh' or '₹8 lakh'.
    Returns None if no amount found.
    """
    m = _INR_RE.search(text)
    if not m or not m.group(2):
        return None
    num_str = m.group(2).replace(",", "")
    try:
        original = float(num_str)
    except ValueError:
        return None
    delta = _rng.uniform(0.1, 0.5) * original
    direction = _rng.choice([1, -1])
    new_val = max(0.0, original + direction * delta)
    new_val = round(new_val, 2)
    new_num = f"{new_val:g}"
    perturbed = (
        text[: m.start(2)] + new_num + text[m.end(2):]
    )
    return perturbed


# ---------------------------------------------------------------------------
# Section reference perturbation
# ---------------------------------------------------------------------------

_SEC_RE = re.compile(
    r"(\b(?:section|regulation|clause|rule|article|para(?:graph)?|schedule)"
    r"\s*)(\d+)([A-Z]?)(\(\w+\))*",
    re.IGNORECASE,
)


def perturb_section(text: str) -> Optional[str]:
    """
    Increment the main section number to produce a wrong reference.
    'Section 52(4)' → 'Section 53(4)' or 'Section 51(4)'.
    Returns None if no section reference found.
    """
    m = _SEC_RE.search(text)
    if not m:
        return None
    original_num = int(m.group(2))
    delta = _rng.choice([1, 2, -1, -2])
    new_num = max(1, original_num + delta)
    perturbed = (
        text[: m.start(2)] + str(new_num) + text[m.end(2):]
    )
    return perturbed


# ---------------------------------------------------------------------------
# Date perturbation
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(
    r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b|"
    r"\b(\d{4})-(\d{2})-(\d{2})\b",
)


def perturb_date(text: str) -> Optional[str]:
    """
    Shift the first recognisable date by ±365–1095 days.
    Returns None if no date found.
    """
    m = _DATE_RE.search(text)
    if not m:
        return None
    try:
        if m.group(4):  # ISO
            d = date(int(m.group(4)), int(m.group(5)), int(m.group(6)))
        else:
            d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None
    shift = _rng.randint(365, 1095) * _rng.choice([1, -1])
    new_date = d + timedelta(days=shift)
    new_str = new_date.strftime("%d/%m/%Y")
    perturbed = text[: m.start()] + new_str + text[m.end():]
    return perturbed


# ---------------------------------------------------------------------------
# Generic dispatcher
# ---------------------------------------------------------------------------

def perturb(atom_type: str, text: str) -> Optional[str]:
    """
    Dispatch to the correct perturbation function based on atom type.

    Returns the perturbed string, or None if no perturbation possible.
    """
    fn_map = {
        "RATE": perturb_rate,
        "THRESHOLD": perturb_threshold,
        "SECTION": perturb_section,
        "DATE": perturb_date,
    }
    fn = fn_map.get(atom_type)
    if fn is None:
        return None
    return fn(text)
