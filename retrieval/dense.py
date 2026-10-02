import os
from pathlib import Path
from typing import List, Optional
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from common.schemas import Chunk

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DenseIndex:
    def __init__(self, persist_dir: str, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.persist_dir = Path(persist_dir)
        self.model_name = model_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = SentenceTransformer(model_name, device=_DEVICE)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name="rag_corpus",
            metadata={"hnsw:space": "cosine"}
        )

    def build_index(self, chunks: List[Chunk], batch_size: int = 100):
        if not chunks:
            return

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = [c.chunk_id for c in batch]
            texts = [c.text for c in batch]
            
            embeddings = self.embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
            
            # Chroma metadata must be strictly str, int, float, bool
            metadatas = []
            for c in batch:
                meta = {
                    "regulator": c.regulator,
                    "issue_date": c.issue_date,
                    "source_url": c.source_url
                }
                metadatas.append(meta)

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )

    def search(self, query: str, top_k: int = 50) -> List[str]:
        if self.collection.count() == 0:
            return []
            
        query_embedding = self.embedder.encode([f"Represent this sentence for searching relevant passages: {query}"], convert_to_numpy=True).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count())
        )

        if results and results.get("ids") and results["ids"][0]:
            return results["ids"][0]
        return []
