import re
from typing import List, Dict, Any
from common.schemas import Chunk

class LegalChunker:
    """Structure-aware chunker tailored for RBI & SEBI regulatory guidelines."""

    def __init__(self):
        pass

    def chunk_document(self, doc_metadata: Dict[str, Any], pages: List[Dict[str, Any]]) -> List[Chunk]:
        """Splits page text into structure-aware chunks based ONLY on legal boundaries (not token windows)."""
        chunks: List[Chunk] = []
        if not pages:
            return chunks

        doc_title = doc_metadata.get("doc_title", "Regulatory Document")
        doc_id = doc_metadata.get("doc_id", "doc")
        issuer = doc_metadata.get("issuer", "Regulatory Authority")
        year = doc_metadata.get("year", "Unknown")
        issue_date = doc_metadata.get("issue_date", "Unknown")
        source_url = doc_metadata.get("source_url", "Unknown")

        current_chapter = "General"
        current_section = "General"
        
        # We accumulate text for a specific section path
        current_text = []

        chapter_pattern = re.compile(r'^(chapter|part)\s+([IVXLCDM\d]+[:\.\-]?.*)', re.IGNORECASE)
        section_pattern = re.compile(r'^(\d+[\.\d]*\s+[A-Z].*|section\s+\d+.*|paragraph\s+\d+.*|regulation\s+\d+.*)', re.IGNORECASE)

        chunk_counter = 0

        def flush_chunk(chap, sec, lines):
            nonlocal chunk_counter
            if not lines:
                return
            chunk_text = "\n".join(lines).strip()
            if not chunk_text:
                return
                
            chunk_counter += 1
            meta = {
                "doc_title": doc_title,
                "doc_id": doc_id,
                "chapter": chap,
                "section": sec,
                "section_id": f"{issuer}_{chap}_{sec}".replace(" ", "_"),
                "chunk_index": chunk_counter
            }

            chunks.append(Chunk(
                chunk_id=f"{doc_id}_{chunk_counter}",
                text=chunk_text,
                regulator=issuer,
                issue_date=issue_date,
                source_url=source_url,
                metadata=meta
            ))

        for page in pages:
            text = page["text"]
            lines = text.split("\n")

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                chap_match = chapter_pattern.match(stripped)
                sec_match = section_pattern.match(stripped)

                # If we hit a new structural boundary, flush the previous section
                if chap_match or sec_match:
                    flush_chunk(current_chapter, current_section, current_text)
                    current_text = []
                    
                    if chap_match:
                        current_chapter = stripped[:80]
                    if sec_match:
                        current_section = stripped[:100]

                current_text.append(stripped)

        # Flush final trailing text
        flush_chunk(current_chapter, current_section, current_text)

        return chunks
