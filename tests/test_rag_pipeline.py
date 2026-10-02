import os
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None

from config import BASE_DIR, DATASET_DIR
from rag_module.ingestion import DatasetScanner, RegulatoryPDFParser
from rag_module.chunking import RegulatoryStructureChunker, RegulatoryChunk
from rag_module.indexing import ChromaVectorStore
from rag_module.retrieval import BM25Retriever, HybridRerankRetriever
from rag_module.generation import RegulatoryGenerator

def test_dataset_scanner():
    scanner = DatasetScanner(root_dir=DATASET_DIR)
    docs = scanner.scan_documents(max_docs=5)
    assert len(docs) > 0
    first = docs[0]
    assert "doc_id" in first
    assert first["issuer"] in ["RBI", "SEBI"]
    assert "file_path" in first

def test_pdf_parser_and_chunker(tmp_path):
    scanner = DatasetScanner(root_dir=DATASET_DIR)
    docs = scanner.scan_documents(max_docs=2)
    assert len(docs) > 0

    parser = RegulatoryPDFParser()
    chunker = RegulatoryStructureChunker(target_chunk_size=300)

    pages = parser.parse_pdf(docs[0]["file_path"])
    assert len(pages) > 0

    chunks = chunker.chunk_document(docs[0], pages)
    assert len(chunks) > 0
    first_chunk = chunks[0]
    assert "[Issuer:" in first_chunk.breadcrumb_text
    assert first_chunk.metadata["doc_id"] == docs[0]["doc_id"]

def test_bm25_retriever():
    chunk1 = RegulatoryChunk(
        chunk_id="c1",
        doc_id="doc1",
        text="The Cash Reserve Ratio (CRR) requirement is 4 percent.",
        breadcrumb_text="[Issuer: RBI | Doc: CRR] The Cash Reserve Ratio (CRR) requirement is 4 percent.",
        metadata={"doc_title": "CRR Guidelines", "section": "Section 2"}
    )
    chunk2 = RegulatoryChunk(
        chunk_id="c2",
        doc_id="doc2",
        text="Alternative Investment Funds (AIF) are registered under SEBI.",
        breadcrumb_text="[Issuer: SEBI | Doc: AIF] Alternative Investment Funds (AIF) are registered under SEBI.",
        metadata={"doc_title": "AIF Regulations", "section": "Section 3"}
    )

    retriever = BM25Retriever()
    retriever.build_index([chunk1, chunk2])

    results = retriever.search("Cash Reserve Ratio CRR", top_k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "c1"

def test_generator_claims_extraction():
    generator = RegulatoryGenerator()
    dummy_chunks = [{
        "text": "[Issuer: RBI | Doc: CRR] Banks must maintain CRR at 4 percent.",
        "metadata": {"doc_title": "CRR 2009", "section": "Section 2.1", "year": "2009"}
    }]

    result = generator.generate_answer("What is the CRR percentage?", dummy_chunks)
    assert "answer" in result
    assert "citations" in result
    assert "claims" in result
    assert len(result["citations"]) > 0
    assert len(result["claims"]) > 0
