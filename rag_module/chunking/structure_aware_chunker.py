import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class RegulatoryChunk:
    chunk_id: str
    doc_id: str
    text: str
    breadcrumb_text: str
    metadata: Dict[str, Any]

class RegulatoryStructureChunker:
    """Structure-aware chunker tailored for RBI & SEBI regulatory guidelines."""

    def __init__(self, target_chunk_size: int = 600, overlap: int = 100):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def chunk_document(self, doc_metadata: Dict[str, Any], pages: List[Dict[str, Any]]) -> List[RegulatoryChunk]:
        """Splits page text into structure-aware chunks with contextual breadcrumbs."""
        chunks: List[RegulatoryChunk] = []
        if not pages:
            return chunks

        doc_title = doc_metadata.get("doc_title", "Regulatory Document")
        issuer = doc_metadata.get("issuer", "Regulatory Authority")
        category = doc_metadata.get("category", "Guideline")
        year = doc_metadata.get("year", "Unknown")

        current_chapter = "General"
        current_section = "Main Provisions"

        # Regex patterns for regulatory headings
        chapter_pattern = re.compile(r'^(chapter|part)\s+([IVXLCDM\d]+[:\.\-]?.*)', re.IGNORECASE)
        section_pattern = re.compile(r'^(\d+[\.\d]*\s+[A-Z].*|section\s+\d+.*|paragraph\s+\d+.*)', re.IGNORECASE)

        chunk_counter = 0

        for page in pages:
            page_num = page["page_number"]
            text = page["text"]
            lines = text.split("\n")

            buffer_lines = []
            buffer_word_count = 0

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                # Check for chapter update
                chap_match = chapter_pattern.match(stripped)
                if chap_match:
                    current_chapter = stripped[:80]

                # Check for section update
                sec_match = section_pattern.match(stripped)
                if sec_match:
                    current_section = stripped[:100]

                buffer_lines.append(stripped)
                buffer_word_count += len(stripped.split())

                # Flush buffer when target chunk size is reached
                if buffer_word_count >= self.target_chunk_size:
                    chunk_text = "\n".join(buffer_lines)
                    chunk_counter += 1

                    breadcrumb = f"[Issuer: {issuer} | Category: {category} | Year: {year} | Doc: {doc_title} | Chapter: {current_chapter} | Section: {current_section} | Page: {page_num}]"
                    full_text_with_breadcrumb = f"{breadcrumb}\n\n{chunk_text}"

                    meta = {
                        "doc_id": doc_metadata["doc_id"],
                        "file_path": doc_metadata["file_path"],
                        "issuer": issuer,
                        "category": category,
                        "year": year,
                        "doc_title": doc_title,
                        "chapter": current_chapter,
                        "section": current_section,
                        "page_number": page_num,
                        "chunk_index": chunk_counter
                    }

                    chunks.append(RegulatoryChunk(
                        chunk_id=f"{doc_metadata['doc_id']}_chunk_{chunk_counter}",
                        doc_id=doc_metadata["doc_id"],
                        text=chunk_text,
                        breadcrumb_text=full_text_with_breadcrumb,
                        metadata=meta
                    ))

                    # Retain last few lines for overlap
                    overlap_lines = buffer_lines[-3:] if len(buffer_lines) > 3 else []
                    buffer_lines = list(overlap_lines)
                    buffer_word_count = sum(len(l.split()) for l in buffer_lines)

            # Flush remaining page buffer
            if buffer_lines:
                chunk_text = "\n".join(buffer_lines)
                chunk_counter += 1

                breadcrumb = f"[Issuer: {issuer} | Category: {category} | Year: {year} | Doc: {doc_title} | Chapter: {current_chapter} | Section: {current_section} | Page: {page_num}]"
                full_text_with_breadcrumb = f"{breadcrumb}\n\n{chunk_text}"

                meta = {
                    "doc_id": doc_metadata["doc_id"],
                    "file_path": doc_metadata["file_path"],
                    "issuer": issuer,
                    "category": category,
                    "year": year,
                    "doc_title": doc_title,
                    "chapter": current_chapter,
                    "section": current_section,
                    "page_number": page_num,
                    "chunk_index": chunk_counter
                }

                chunks.append(RegulatoryChunk(
                    chunk_id=f"{doc_metadata['doc_id']}_chunk_{chunk_counter}",
                    doc_id=doc_metadata["doc_id"],
                    text=chunk_text,
                    breadcrumb_text=full_text_with_breadcrumb,
                    metadata=meta
                ))

        return chunks
