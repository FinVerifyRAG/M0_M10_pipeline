from typing import List, Dict, Any, Optional
import torch
from sentence_transformers import CrossEncoder
from ..indexing.vector_store import ChromaVectorStore
from .bm25_retriever import BM25Retriever

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class HybridRerankRetriever:
    """Hybrid Retriever combining Dense Vector + Sparse BM25 search with Cross-Encoder reranking."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_retriever: BM25Retriever,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.reranker_model_name = reranker_model_name

        print(f"[Reranker] Loading Cross-Encoder reranker: {reranker_model_name} on {_DEVICE}...")
        self.cross_encoder = CrossEncoder(reranker_model_name, device=_DEVICE)

    def retrieve(
        self,
        query: str,
        top_k_dense: int = 15,
        top_k_sparse: int = 15,
        top_k_final: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Executes Hybrid Dense + Sparse RRF search followed by Cross-Encoder reranking."""
        # 1. Retrieve Dense Hits
        dense_hits = self.vector_store.similarity_search(query, top_k=top_k_dense, filter_dict=filter_dict)

        # 2. Retrieve Sparse Hits
        sparse_hits = self.bm25_retriever.search(query, top_k=top_k_sparse)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits):
            cid = hit["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_map[cid] = hit

        for rank, hit in enumerate(sparse_hits):
            cid = hit["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = hit

        if not rrf_scores:
            print("[Warning] No candidates returned from Dense or Sparse retrieval.")
            return []

        # Sort candidate pool by RRF score
        sorted_candidates = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        candidate_chunks = [chunk_map[cid] for cid in sorted_candidates[:top_k_dense + top_k_sparse]]

        # 4. Cross-Encoder Reranking
        pairs = [[query, chunk["text"]] for chunk in candidate_chunks]
        cross_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)

        # Attach rerank score and sort
        for chunk, c_score in zip(candidate_chunks, cross_scores):
            chunk["rerank_score"] = float(c_score)

        reranked_chunks = sorted(candidate_chunks, key=lambda c: c["rerank_score"], reverse=True)
        final_top_k = reranked_chunks[:top_k_final]

        print(f"[HybridRetrieval] Retrieved {len(final_top_k)} reranked chunks for query: '{query[:40]}...'")
        return final_top_k
