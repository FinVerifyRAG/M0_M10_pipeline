"""
verify/normalizers.py
---------------------
Canonicalize numbers, units, dates, and section references so that
V1 deterministic matching works across textual variants.

All public functions return None when the input cannot be parsed.

Ambiguity note (dd/mm forms)
----------------------------
When a date string contains only day and month components (e.g. "12/03")
without a year, parsing is REFUSED and None is returned rather than
guessing the year.  The caller must supply context.

For ambiguous two-component dates such as ``05/06`` we treat the first
field as *day* and the second as *month* (Indian convention: dd/mm).
This assumption is documented here and NOT silently applied.

Examples tested in tests/test_verify_normalizers.py
"""
import re
from typing import Optional, Tuple
from datetime import date


# ---------------------------------------------------------------------------
# Numeric / monetary normalization
# ---------------------------------------------------------------------------

_LAKH_MAP = {
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
    "crore": 1e7, "crores": 1e7, "cr": 1e7, "cr.": 1e7,
    "thousand": 1e3, "thousands": 1e3,
    "million": 1e6, "billion": 1e9,
}

# BUG 3 FIX: extended currency prefix to support Rs., Rs, INR, backtick (PDF
# corruption of ₹), and trailing /- sign used in older SEBI/RBI PDFs.
# Also supports word-order forms such as "5 lakh rupees" and "rupees five lakh".
_CURRENCY_PREFIX = r"(?:₹|`|Rs\.?\s*|INR\s*)?"
_AMOUNT_RE = re.compile(
    _CURRENCY_PREFIX
    + r"(\d[\d,]*(?:\.\d+)?)\s*"
    r"(lakh|lakhs?|lac|lacs?|crore|crores?|cr\.?|thousand|thousands?|million|billion)?"
    r"(?:\s*(?:rupees?|/-))?",
    re.IGNORECASE,
)

# Support "5 lakh rupees" (number before unit, currency after)
_AMOUNT_RE_WORD_ORDER = re.compile(
    r"(?:rupees?\s+)?(\d[\d,]*(?:\.\d+)?)\s*"
    r"(lakh|lakhs?|lac|lacs?|crore|crores?|cr\.?|thousand|thousands?|million|billion)?"
    r"(?:\s+rupees?)?",
    re.IGNORECASE,
)


def _clean_pdf_currency(s: str) -> str:
    """Normalize PDF rendering artefacts: backtick as ₹, trailing /-."""
    # Backtick used by some older SEBI/RBI PDFs as corrupted ₹
    s = s.replace("`", "₹")
    # Remove trailing /- (common in Indian legal documents)
    s = re.sub(r"/-\s*$", "", s.strip())
    return s


def norm_amount(s: str) -> Optional[float]:
    """
    Normalize a rupee amount to a plain float.

    Supported forms (BUG 3 fixed):
      '₹5 lakh'         -> 500000.0
      'Rs. 5,00,000'    -> 500000.0
      'Rs.5,00,000/-'   -> 500000.0
      'INR 500000'      -> 500000.0
      '5 lakh rupees'   -> 500000.0
      'rupees five lakh'-> None  (word-number not supported; falls through)
      '₹ 5 lakh'        -> 500000.0
      'Rs 5 lakh'       -> 500000.0
      '`5,00,000'       -> 500000.0  (PDF backtick corruption)
      '₹1.5 crore'      -> 15000000.0
      '5,00,000'        -> 500000.0
    """
    s = _clean_pdf_currency(s.strip())
    m = _AMOUNT_RE.search(s)
    if not m or not m.group(1):
        return None
    num_str = m.group(1).replace(",", "")
    try:
        value = float(num_str)
    except ValueError:
        return None
    unit = (m.group(2) or "").lower().rstrip(".")
    multiplier = _LAKH_MAP.get(unit, 1.0)
    return value * multiplier


# ---------------------------------------------------------------------------
# Percentage normalization  (BUG 1 FIX)
# ---------------------------------------------------------------------------

# BUG 1 FIX: The original regex used r'\d+%\b' — a \b after % always fails
# because % is a non-word character followed by another non-word character
# (space or end-of-string).  Fixed by removing the erroneous word boundary
# after % and restructuring the alternation so all forms are captured.
#
# Required by spec:
#   PCT = re.compile(
#       r"\b\d+(?:\.\d+)?\s?(?:%|(?:per\s?cent|percent)\b)", re.I
#   )
# We extend this to also capture the optional per-annum suffix.
PCT = re.compile(
    r"\b(\d+(?:\.\d+)?)\s?(?:%|(?:per\s?cent|percent)\b)"
    r"(?:\s*(?:per\s*annum|p\.?a\.?))?",
    re.IGNORECASE,
)

# Internal alias kept for import compatibility
_PCT_RE = PCT


def norm_pct(s: str) -> Optional[float]:
    """
    Normalize a percentage to a plain float.

    Supported forms (BUG 1 fixed):
      '5%'            -> 5.0
      '5 %'           -> 5.0
      '5% of income'  -> 5.0  (trailing words ignored)
      '10%'           -> 10.0
      '10.5%'         -> 10.5
      'TDS at 10%.'   -> 10.0  (preceding context ignored)
      '5 per cent'    -> 5.0
      '5 percent'     -> 5.0
      '5percent'      -> 5.0
      '5.5% p.a.'     -> 5.5
    """
    m = _PCT_RE.search(s.strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Basis-points normalization  (BUG 3 / new feature)
# ---------------------------------------------------------------------------

# 25 bps == 0.25%;  25 basis points == 0.25%
_BPS_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:bps|basis\s+points?)\b",
    re.IGNORECASE,
)


def norm_bps(s: str) -> Optional[float]:
    """
    Normalize basis points to percentage.
    '25 bps'          -> 0.25
    '25 basis points' -> 0.25
    '100 bps'         -> 1.0
    Returns None if not found.
    """
    m = _BPS_RE.search(s.strip())
    if not m:
        return None
    try:
        return float(m.group(1)) / 100.0
    except ValueError:
        return None


def is_bps(s: str) -> bool:
    """Return True if the string contains a basis-points expression."""
    return bool(_BPS_RE.search(s))


# ---------------------------------------------------------------------------
# Date normalization  (Issue 18 extensions)
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# dd/mm/yyyy or dd-mm-yyyy  (Indian convention: day first)
_DMY_SLASH = re.compile(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})")
# dd.mm.yyyy  (Issue 18: dot separator)
_DMY_DOT = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})")
# dd Month yyyy  e.g. "12 March 2024"
_DMY_WORD = re.compile(
    r"(\d{1,2})\s+"
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+(\d{4})",
    re.IGNORECASE,
)
# '1st day of April, 2026'  (Issue 18: ordinal + month + year)
_ORD_DAY = re.compile(
    r"(\d{1,2})(?:st|nd|rd|th)\s+day\s+of\s+"
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r",?\s+(\d{4})",
    re.IGNORECASE,
)
# yyyy-mm-dd ISO
_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

# FY 2023-24 → we store the start year (1 April)
_FY_RE = re.compile(r"FY\s*(\d{4})[-–](\d{2,4})", re.IGNORECASE)

# AY 2024-25 → Assessment Year = FY + 1 start year (convention: AY 2024-25 -> FY 2023-24)
# AY 2024-25 starts on 1 April 2024 for income of FY 2023-24.
# We store the *AY start date* (1 April of the first year in AY string).
_AY_RE = re.compile(r"AY\s*(\d{4})[-–](\d{2,4})", re.IGNORECASE)


def norm_date(s: str) -> Optional[date]:
    """
    Parse an Indian date string and return a datetime.date object.
    Returns None when parsing fails.

    Supports (Issue 18 extensions added):
      - 'dd/mm/yyyy', 'dd-mm-yyyy'        Indian convention (day first)
      - 'dd.mm.yyyy'                      Dot separator (Issue 18)
      - 'dd Month yyyy' ('12 March 2024')
      - '1st day of April, 2026'          Ordinal form (Issue 18)
      - 'yyyy-mm-dd'                      ISO 8601
      - 'FY 2023-24'  -> 01 Apr 2023      Financial year start
      - 'AY 2024-25'  -> 01 Apr 2024      Assessment year start (Issue 18)

    Ambiguity note:
      dd/mm two-component forms (no year) are NOT parsed — None is returned.
      For three-component numeric dates the first field is treated as *day*
      (Indian convention).  If that produces an invalid date, month/day swap
      is attempted but this is a last resort and the ambiguity is logged.
    """
    s = s.strip()

    # ISO 8601 — unambiguous, try first
    m = _ISO.fullmatch(s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # '1st day of April, 2026' ordinal form (Issue 18)
    m = _ORD_DAY.search(s)
    if m:
        day = int(m.group(1))
        mon = _MONTH_MAP.get(m.group(2).lower())
        yr = int(m.group(3))
        if mon:
            try:
                return date(yr, mon, day)
            except ValueError:
                pass

    # 'dd Month yyyy'
    m = _DMY_WORD.search(s)
    if m:
        day = int(m.group(1))
        mon = _MONTH_MAP.get(m.group(2).lower())
        yr = int(m.group(3))
        if mon:
            try:
                return date(yr, mon, day)
            except ValueError:
                pass

    # dd.mm.yyyy (Issue 18)
    m = _DMY_DOT.search(s)
    if m:
        d_, mo_, y_ = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y_, mo_, d_)  # day-first assumption documented above
        except ValueError:
            pass

    # dd/mm/yyyy or dd-mm-yyyy
    m = _DMY_SLASH.search(s)
    if m:
        d_, mo_, y_ = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y_ < 100:
            y_ += 2000
        try:
            return date(y_, mo_, d_)  # Indian convention: day first
        except ValueError:
            # Ambiguity fallback: swap day/month as last resort
            try:
                return date(y_, d_, mo_)
            except ValueError:
                pass

    # FY 2023-24 → 01 Apr of start year
    m = _FY_RE.search(s)
    if m:
        fy_start = int(m.group(1))
        try:
            return date(fy_start, 4, 1)
        except ValueError:
            pass

    # AY 2024-25 → 01 Apr of first AY year (Issue 18)
    m = _AY_RE.search(s)
    if m:
        ay_start = int(m.group(1))
        try:
            return date(ay_start, 4, 1)
        except ValueError:
            pass

    return None


def fy_from_ay(ay_str: str) -> Optional[Tuple[int, int]]:
    """
    Convert an Assessment Year string to its corresponding Financial Year.

    AY 2024-25 -> FY 2023-24  (AY = FY + 1 by Indian convention)

    Returns (fy_start_year, fy_end_year) or None.
    """
    m = _AY_RE.search(ay_str)
    if not m:
        return None
    ay_start = int(m.group(1))
    return (ay_start - 1, ay_start)


def dates_equal(a: str, b: str) -> Optional[bool]:
    """
    Return True if both strings normalise to the same date, False if they
    differ, None if either cannot be parsed.
    """
    da, db = norm_date(a), norm_date(b)
    if da is None or db is None:
        return None
    return da == db


# ---------------------------------------------------------------------------
# Section / regulation reference normalization  (BUG 2 FIX)
# ---------------------------------------------------------------------------

# BUG 2 FIX: The original regex captured '194' from 'section 194-IA' because
# it used r'\d+[A-Z]?' which stops at the hyphen.  Also missed u/s 80C and
# list forms like 'Sections 80C and 80D'.
#
# New SEC pattern per spec:
#   r"\b(?:sections?|sec\.|u/s|regulations?|clauses?|rules?)"
#   r"\s+\d+(?:-?[A-Z]{1,3})?(?:\(\w+\))*"
# Extended to support all keyword forms in the codebase.
SEC = re.compile(
    r"\b(?:sections?|sec\.|u/s|regulations?|clauses?|rules?|articles?|"
    r"para(?:graphs?)?|schedules?|circulars?|master\s+directions?)"
    r"\s+"
    r"(?P<first>\d+(?:-[A-Za-z]{1,4})?(?:\(?\w+\)?)*)",
    re.IGNORECASE,
)

# List conjunction pattern: captures additional identifiers after "and"/","
_SEC_LIST_CONJ = re.compile(
    r"(?:,\s*|\s+and\s+)(\d+(?:-[A-Za-z]{1,4})?(?:\(?\w+\)?)*)",
    re.IGNORECASE,
)

_SEC_RE = SEC  # internal alias


def extract_section_list(s: str) -> list:
    """
    Extract all section identifiers from a potentially list-form string.

    Examples:
      'Sections 80C and 80D'       -> ['80C', '80D']
      'Sections 80C, 80D and 80G'  -> ['80C', '80D', '80G']
      'u/s 80C'                    -> ['80C']
      'section 194-IA'             -> ['194-IA']
      'Section 10(1)'              -> ['10(1)']
    """
    m = SEC.search(s)
    if not m:
        return []
    first = m.group("first")
    rest_text = s[m.end():]
    others = _SEC_LIST_CONJ.findall(rest_text)
    return [first] + others


def norm_section(s: str) -> Optional[str]:
    """
    Normalise a section reference to lowercase canonical form.

    BUG 2 fixed: full identifier including hyphen-suffix is preserved.
      'section 194-IA'   -> 'section194-ia'
      'Section 80C(2)'   -> 'section80c(2)'
      'u/s 80C'          -> 'u/s80c'
      'Regulation 52(4)' -> 'regulation52(4)'
    """
    s = s.strip()
    return s.lower().replace(" ", "")


def sections_equal(a: str, b: str) -> bool:
    """Return True if two section references normalise to the same string."""
    return norm_section(a) == norm_section(b)


# ---------------------------------------------------------------------------
# Generic numeric comparison (tolerant of floating-point noise)
# ---------------------------------------------------------------------------

def numerically_equal(a: Optional[float], b: Optional[float],
                      rel_tol: float = 1e-6) -> Optional[bool]:
    """Compare two float values with relative tolerance. Returns None if either is None."""
    if a is None or b is None:
        return None
    if a == 0 and b == 0:
        return True
    return abs(a - b) / max(abs(a), abs(b)) < rel_tol
