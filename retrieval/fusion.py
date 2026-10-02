from typing import List, Dict

def rrf(rank_lists: List[List[str]], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    
    for rank_list in rank_lists:
        for rank, chunk_id in enumerate(rank_list, 1):
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
            scores[chunk_id] += 1.0 / (k + rank)
            
    sorted_chunks = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return sorted_chunks
