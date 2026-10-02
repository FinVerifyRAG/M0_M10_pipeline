import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from common.schemas import Chunk

def regulatory_tokenize(text: str) -> List[str]:
    if not text:
        return []
    # Tokenize preserving sections like 52(4), hyphens, dots, and slashes
    raw_tokens = re.findall(r'[a-z0-9\-\.\/\%\(\)]+', text.lower())
    tokens = [t.rstrip('.,;:') for t in raw_tokens]
    return [t for t in tokens if len(t) > 1]

class BM25Index:
    def __init__(self, index_path: Optional[str] = None):
        self.index_path = Path(index_path) if index_path else None
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: Dict[str, Chunk] = {}
        self.tokenized_corpus: List[List[str]] = []
        self.chunk_ids: List[str] = []

    def build_index(self, chunks: List[Chunk]):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.tokenized_corpus = [regulatory_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        if self.index_path:
            self.save_index(self.index_path)

    def search(self, query: str, top_k: int = 50) -> List[str]:
        if not self.bm25 or not self.chunk_ids:
            return []

        query_tokens = regulatory_tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.chunk_ids[i] for i in top_indices]

    def save_index(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "chunk_ids": self.chunk_ids, "tokenized_corpus": self.tokenized_corpus}, f)

    def load_index(self, file_path: Path) -> bool:
        if not file_path.exists():
            return False
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.chunk_ids = data["chunk_ids"]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        return True
