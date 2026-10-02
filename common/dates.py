import re
from datetime import datetime
from typing import Optional, Tuple

def parse_indian_date(date_str: str) -> Optional[datetime]:
    """
    Parses common Indian date formats into a datetime object.
    Supports: '12 March 2024', '12/03/2024', '12-03-2024'.
    Returns None if parsing fails.
    """
    date_str = date_str.strip()
    
    # Try DD/MM/YYYY or DD-MM-YYYY
    match = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", date_str)
    if match:
        day, month, year = map(int, match.groups())
        try:
            return datetime(year, month, day)
        except ValueError:
            return None
            
    # Try DD Month YYYY (e.g., 12 March 2024)
    try:
        return datetime.strptime(date_str, "%d %B %Y")
    except ValueError:
        pass
        
    # Try DD Mon YYYY (e.g., 12 Mar 2024)
    try:
        return datetime.strptime(date_str, "%d %b %Y")
    except ValueError:
        pass
        
    return None

def parse_financial_year(fy_str: str) -> Optional[Tuple[int, int]]:
    """
    Parses financial year strings like 'FY 2023-24' or '2023-24'.
    Returns a tuple of (start_year, end_year).
    """
    match = re.search(r"(\d{4})-(\d{2,4})", fy_str)
    if match:
        start_year = int(match.group(1))
        end_year_raw = match.group(2)
        if len(end_year_raw) == 2:
            end_year = (start_year // 100) * 100 + int(end_year_raw)
        else:
            end_year = int(end_year_raw)
        return (start_year, end_year)
    return None

def compare_dates(date1: str, date2: str) -> int:
    """
    Compares two date strings.
    Returns -1 if date1 < date2, 0 if date1 == date2, 1 if date1 > date2.
    Raises ValueError if dates cannot be parsed.
    """
    d1 = parse_indian_date(date1)
    d2 = parse_indian_date(date2)
    
    if not d1 or not d2:
        raise ValueError(f"Could not parse one of the dates: '{date1}', '{date2}'")
        
    if d1 < d2:
        return -1
    elif d1 > d2:
        return 1
    return 0
