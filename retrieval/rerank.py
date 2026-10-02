from typing import List, Tuple
import torch
from sentence_transformers import CrossEncoder
from common.schemas import Chunk

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.cross_encoder = CrossEncoder(model_name, device=_DEVICE)

    def rerank(self, query: str, candidates: List[Chunk], top_k: int = 6) -> List[Chunk]:
        if not candidates:
            return []
            
        pairs = [[query, chunk.text] for chunk in candidates]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        
        # Attach scores and sort
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, score in scored_candidates[:top_k]]
