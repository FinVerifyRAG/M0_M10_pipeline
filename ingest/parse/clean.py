import re
import unicodedata

def clean_text(text: str) -> str:
    """
    Cleans raw PDF text by:
    1. Normalizing unicode characters
    2. Fixing broken hyphenations at the end of lines
    3. Normalizing rupee symbols and formats
    4. Removing extraneous whitespaces
    """
    # 1. Normalize unicode (NFKC standardizes characters like ﬁ to fi)
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Fix hyphenation across lines (e.g., "regu-\nlation" -> "regulation")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # 3. Normalize Rupee symbols to a standard "Rs." or "₹" (we'll standardize to ₹)
    text = re.sub(r'(?:Rs\.?|INR|Rupees)\s*', '₹ ', text, flags=re.IGNORECASE)
    
    # 4. Standardize quotes
    text = text.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
    
    # 5. Remove extra whitespace but preserve newlines for paragraphs
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()
