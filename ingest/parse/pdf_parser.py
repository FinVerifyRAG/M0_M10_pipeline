import re
from typing import List, Dict, Any

try:
    import pymupdf  # PyMuPDF
except ImportError:
    try:
        import fitz as pymupdf  # Fallback to fitz legacy name
    except ImportError:
        pymupdf = None

from ingest.parse.clean import clean_text

class RegulatoryPDFParser:
    """Parses regulatory PDF documents into page-level structured text blocks with headers."""

    def __init__(self):
        pass

    def parse_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extracts text, page numbers, and structural headings from a PDF file."""
        if pymupdf is None:
            raise ImportError("PyMuPDF is required to parse PDFs. Please run: pip install pymupdf")
        pages = []

        try:
            doc = pymupdf.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")

                # Clean header/footer artifacts like page numbers at bottom
                cleaned_text = self._clean_page_text(text)
                
                # Apply normalization (hyphens, rupees, unicode)
                cleaned_text = clean_text(cleaned_text)

                if cleaned_text.strip():
                    pages.append({
                        "page_number": page_num + 1,
                        "text": cleaned_text,
                        "raw_text": text
                    })

            doc.close()
        except Exception as e:
            print(f"[Error] Failed to parse PDF {file_path}: {e}")

        return pages

    def _clean_page_text(self, text: str) -> str:
        """Removes repeated header/footer noise while preserving section structure."""
        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            if re.match(r'^(page\s+\d+(\s+of\s+\d+)?|\d+)$', stripped, re.IGNORECASE):
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)
