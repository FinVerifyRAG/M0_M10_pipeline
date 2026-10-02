import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from ..chunking.structure_aware_chunker import RegulatoryChunk

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ChromaVectorStore:
    """Persistent ChromaDB vector store backed by BAAI/bge-small-en-v1.5 embeddings."""

    def __init__(self, persist_dir: Path, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.persist_dir = Path(persist_dir)
        self.model_name = model_name

        # Ensure directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        print(f"[VectorStore] Loading embedding model: {model_name} on {_DEVICE}...")
        self.embedder = SentenceTransformer(model_name, device=_DEVICE)

        print(f"[VectorStore] Initializing ChromaDB at {self.persist_dir}...")
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name="rbi_sebi_guidelines",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[RegulatoryChunk], batch_size: int = 100):
        """Indexes regulatory chunks into ChromaDB with embeddings and metadata."""
        if not chunks:
            return

        print(f"[VectorStore] Indexing {len(chunks)} chunks into ChromaDB...")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = [c.chunk_id for c in batch]
            texts = [c.breadcrumb_text for c in batch]

            # BGE model prefix recommendation for query/passage
            embeddings = self.embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
            metadatas = [c.metadata for c in batch]

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )

        print(f"[VectorStore] Total documents in index: {self.collection.count()}")

    def similarity_search(self, query: str, top_k: int = 15, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Performs dense vector similarity search."""
        query_embedding = self.embedder.encode([f"Represent this sentence for searching relevant passages: {query}"], convert_to_numpy=True).tolist()

        kwargs = {
            "query_embeddings": query_embedding,
            "n_results": min(top_k, max(1, self.collection.count()))
        }
        if filter_dict:
            kwargs["where"] = filter_dict

        results = self.collection.query(**kwargs)

        search_hits = []
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)

            for cid, doc, meta, dist in zip(ids, docs, metadatas, distances):
                # Convert distance to similarity score
                similarity = 1.0 - float(dist)
                search_hits.append({
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": meta,
                    "score": similarity
                })

        return search_hits

    def count(self) -> int:
        return self.collection.count()
