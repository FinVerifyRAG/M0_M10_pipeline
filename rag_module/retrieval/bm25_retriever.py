import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from ..chunking.structure_aware_chunker import RegulatoryChunk

def regulatory_tokenize(text: str) -> List[str]:
    """Custom tokenizer preserving section IDs, circular refs, percentages, and acronyms."""
    if not text:
        return []
    # Tokenize words, numbers, hyphenated terms, and slash-separated circular IDs
    tokens = re.findall(r'\b[A-Za-z0-9\-\.\/\%]+\b', text.lower())
    return [t for t in tokens if len(t) > 1]

class BM25Retriever:
    """Sparse BM25 retriever for exact clause, term, and number matching."""

    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = Path(index_path) if index_path else None
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[RegulatoryChunk] = []
        self.tokenized_corpus: List[List[str]] = []

    def build_index(self, chunks: List[RegulatoryChunk]):
        """Builds BM25 index over regulatory chunks."""
        print(f"[BM25] Tokenizing {len(chunks)} chunks for sparse index...")
        self.chunks = chunks
        self.tokenized_corpus = [regulatory_tokenize(c.breadcrumb_text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        if self.index_path:
            self.save_index(self.index_path)

    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """Searches BM25 index and returns top-k ranked chunks with scores."""
        if not self.bm25 or not self.chunks:
            print("[Warning] BM25 index is empty.")
            return []

        query_tokens = regulatory_tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        # Sort indices by BM25 score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.breadcrumb_text,
                "metadata": chunk.metadata,
                "score": score
            })
        return results

    def save_index(self, file_path: Path):
        """Saves BM25 index to pickle file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "tokenized_corpus": self.tokenized_corpus}, f)
        print(f"[BM25] Saved sparse index to {file_path}")

    def load_index(self, file_path: Path) -> bool:
        """Loads BM25 index from pickle file if exists."""
        file_path = Path(file_path)
        if not file_path.exists():
            return False

        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.tokenized_corpus = data["tokenized_corpus"]
                self.bm25 = BM25Okapi(self.tokenized_corpus)
            print(f"[BM25] Loaded index with {len(self.chunks)} chunks from {file_path}")
            return True
        except Exception as e:
            print(f"[Error] Failed to load BM25 index from {file_path}: {e}")
            return False
