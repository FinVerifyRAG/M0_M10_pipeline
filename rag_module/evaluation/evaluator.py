import json
from pathlib import Path
from typing import List, Dict, Any
from ..retrieval.hybrid_reranker import HybridRerankRetriever
from ..generation.generator import RegulatoryGenerator

class RAGEvaluator:
    """Evaluation framework for measuring RAG retrieval accuracy and answer generation quality."""

    def __init__(self, retriever: HybridRerankRetriever, generator: RegulatoryGenerator):
        self.retriever = retriever
        self.generator = generator

    def evaluate_dataset(self, eval_dataset_path: Path, top_k: int = 5) -> Dict[str, Any]:
        """Runs evaluation over ground-truth benchmark dataset."""
        eval_path = Path(eval_dataset_path)
        if not eval_path.exists():
            print(f"[Error] Evaluation dataset file not found: {eval_path}")
            return {}

        with open(eval_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        print(f"[Evaluator] Starting evaluation on {len(test_cases)} test cases (Top-K={top_k})...\n")

        total_reciprocal_rank = 0.0
        hits = 0
        total_precision = 0.0
        total_claims_generated = 0
        citation_count = 0

        eval_results = []

        for item in test_cases:
            qid = item["query_id"]
            query = item["query"]
            target_kw = item["target_doc_keyword"].lower()
            expected_kws = [k.lower() for k in item.get("expected_keywords", [])]

            # Run Retrieval
            retrieved_chunks = self.retriever.retrieve(query, top_k_final=top_k)

            first_rank = 0
            relevant_retrieved = 0

            for rank, chunk in enumerate(retrieved_chunks, start=1):
                text_content = chunk.get("text", "").lower()
                doc_title = chunk.get("metadata", {}).get("doc_title", "").lower()

                # Check relevance matching
                is_relevant = (target_kw in doc_title or target_kw in text_content or
                              any(ek in text_content for ek in expected_kws))

                if is_relevant:
                    relevant_retrieved += 1
                    if first_rank == 0:
                        first_rank = rank

            prec = relevant_retrieved / top_k if top_k > 0 else 0.0
            rr = 1.0 / first_rank if first_rank > 0 else 0.0

            if first_rank > 0:
                hits += 1

            total_precision += prec
            total_reciprocal_rank += rr

            # Run Answer Generation
            gen_output = self.generator.generate_answer(query, retrieved_chunks)
            claims = gen_output.get("claims", [])
            citations = gen_output.get("citations", [])

            total_claims_generated += len(claims)
            if citations:
                citation_count += 1

            eval_results.append({
                "query_id": qid,
                "query": query,
                "first_hit_rank": first_rank,
                "precision_at_k": round(prec, 4),
                "mrr": round(rr, 4),
                "model_used": gen_output.get("model_used"),
                "citations_found": len(citations),
                "claims_extracted": len(claims)
            })

        num_samples = len(test_cases)
        mrr_score = total_reciprocal_rank / num_samples if num_samples > 0 else 0.0
        mean_precision = total_precision / num_samples if num_samples > 0 else 0.0
        hit_rate = hits / num_samples if num_samples > 0 else 0.0
        citation_rate = citation_count / num_samples if num_samples > 0 else 0.0

        summary = {
            "num_test_cases": num_samples,
            "top_k": top_k,
            "mean_precision_at_k": round(mean_precision, 4),
            "mrr": round(mrr_score, 4),
            "hit_rate": round(hit_rate, 4),
            "citation_coverage_rate": round(citation_rate, 4),
            "avg_claims_per_query": round(total_claims_generated / num_samples, 2) if num_samples > 0 else 0,
            "detailed_results": eval_results
        }

        print("=== EVALUATION SUMMARY ===")
        print(f"Hit Rate@{top_k}: {summary['hit_rate'] * 100:.1f}%")
        print(f"Precision@{top_k}: {summary['mean_precision_at_k']:.4f}")
        print(f"MRR@{top_k}: {summary['mrr']:.4f}")
        print(f"Citation Coverage: {summary['citation_coverage_rate'] * 100:.1f}%")
        print(f"Avg Claims/Query: {summary['avg_claims_per_query']}")
        print("==========================\n")

        return summary
