import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config import (
    DATASET_DIR, PERSIST_DIRECTORY, SPARSE_INDEX_PATH,
    EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, OLLAMA_MODEL_NAME,
    OLLAMA_HOST, TOP_K_DENSE, TOP_K_SPARSE, TOP_K_RERANKED
)
from rag_module.ingestion import DatasetScanner, RegulatoryPDFParser
from rag_module.chunking import RegulatoryStructureChunker
from rag_module.indexing import ChromaVectorStore
from rag_module.retrieval import BM25Retriever, HybridRerankRetriever
from rag_module.generation import RegulatoryGenerator
from rag_module.evaluation import RAGEvaluator


def build_pipeline():
    """Initializes vector store, BM25 retriever, hybrid reranker, and generator."""
    vector_store = ChromaVectorStore(persist_dir=PERSIST_DIRECTORY, model_name=EMBEDDING_MODEL_NAME)

    bm25_retriever = BM25Retriever(index_path=SPARSE_INDEX_PATH)
    if not bm25_retriever.load_index(SPARSE_INDEX_PATH):
        print("[Warning] BM25 index file not found. Run 'python main.py ingest' first.")

    hybrid_retriever = HybridRerankRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        reranker_model_name=RERANKER_MODEL_NAME
    )

    generator = RegulatoryGenerator(model_name=OLLAMA_MODEL_NAME, ollama_host=OLLAMA_HOST)

    return vector_store, bm25_retriever, hybrid_retriever, generator


def run_ingest(
    max_docs: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Parse PDFs, structure-chunk, and dual-index. Shared by CLI and Streamlit."""
    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    _progress("=== RBI & SEBI REGULATORY INGESTION PIPELINE ===")
    scanner = DatasetScanner(root_dir=DATASET_DIR)
    doc_descriptors = scanner.scan_documents(max_docs=max_docs)

    if not doc_descriptors:
        msg = f"No PDF files found in {DATASET_DIR}"
        _progress(f"[Error] {msg}")
        return {"success": False, "message": msg, "num_docs": 0, "num_chunks": 0}

    parser = RegulatoryPDFParser()
    chunker = RegulatoryStructureChunker()

    all_chunks = []
    _progress(f"[Ingestion] Parsing {len(doc_descriptors)} PDF documents...")

    for idx, doc_meta in enumerate(doc_descriptors, start=1):
        _progress(f"[{idx}/{len(doc_descriptors)}] Parsing: {doc_meta['filename']}")
        pages = parser.parse_pdf(doc_meta["file_path"])
        chunks = chunker.chunk_document(doc_meta, pages)
        all_chunks.extend(chunks)

    _progress(f"[Ingestion] Total structure-aware chunks generated: {len(all_chunks)}")

    vector_store = ChromaVectorStore(persist_dir=PERSIST_DIRECTORY, model_name=EMBEDDING_MODEL_NAME)
    vector_store.add_chunks(all_chunks)

    bm25_retriever = BM25Retriever(index_path=SPARSE_INDEX_PATH)
    bm25_retriever.build_index(all_chunks)

    _progress("[Success] Ingestion & dual indexing complete!")
    return {
        "success": True,
        "message": "Ingestion & dual indexing complete",
        "num_docs": len(doc_descriptors),
        "num_chunks": len(all_chunks),
    }


def run_query(
    query: str,
    hybrid_retriever: HybridRerankRetriever,
    generator: RegulatoryGenerator,
    filter_dict: Optional[Dict[str, Any]] = None,
    top_k_dense: int = TOP_K_DENSE,
    top_k_sparse: int = TOP_K_SPARSE,
    top_k_final: int = TOP_K_RERANKED,
) -> Dict[str, Any]:
    """Hybrid retrieve + generate. Returns answer, citations, claims, chunks, model_used."""
    context_chunks = hybrid_retriever.retrieve(
        query=query,
        top_k_dense=top_k_dense,
        top_k_sparse=top_k_sparse,
        top_k_final=top_k_final,
        filter_dict=filter_dict,
    )
    result = generator.generate_answer(query, context_chunks)
    return {
        "query": query,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "claims": result.get("claims", []),
        "model_used": result.get("model_used", ""),
        "chunks": context_chunks,
    }


def inspect_retrieval(
    query: str,
    vector_store: ChromaVectorStore,
    bm25_retriever: BM25Retriever,
    hybrid_retriever: HybridRerankRetriever,
    filter_dict: Optional[Dict[str, Any]] = None,
    top_k_dense: int = TOP_K_DENSE,
    top_k_sparse: int = TOP_K_SPARSE,
    top_k_final: int = TOP_K_RERANKED,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return dense, sparse, and reranked hits for UI inspection (no generation)."""
    dense_hits = vector_store.similarity_search(query, top_k=top_k_dense, filter_dict=filter_dict)
    sparse_hits = bm25_retriever.search(query, top_k=top_k_sparse)
    reranked = hybrid_retriever.retrieve(
        query=query,
        top_k_dense=top_k_dense,
        top_k_sparse=top_k_sparse,
        top_k_final=top_k_final,
        filter_dict=filter_dict,
    )
    return {
        "dense": dense_hits,
        "sparse": sparse_hits,
        "reranked": reranked,
    }


def run_evaluation(
    hybrid_retriever: HybridRerankRetriever,
    generator: RegulatoryGenerator,
    eval_dataset_path: Optional[Path] = None,
    top_k: int = TOP_K_RERANKED,
    save_report: bool = True,
) -> Dict[str, Any]:
    """Run benchmark evaluation; optionally write evaluation_report.json."""
    evaluator = RAGEvaluator(retriever=hybrid_retriever, generator=generator)
    if eval_dataset_path is None:
        eval_dataset_path = Path(__file__).parent / "data" / "eval_dataset.json"

    summary = evaluator.evaluate_dataset(eval_dataset_path=eval_dataset_path, top_k=top_k)

    if save_report and summary:
        output_report_path = Path(__file__).parent / "evaluation_report.json"
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Evaluation] Full report saved to {output_report_path}")

    return summary


def handle_ingest(args):
    """CLI wrapper for ingestion."""
    run_ingest(max_docs=args.max_docs)


def handle_query(args):
    """CLI wrapper for query."""
    print(f"\n=== REGULATORY QUERY: '{args.query}' ===")
    _vector_store, _bm25, hybrid_retriever, generator = build_pipeline()
    result = run_query(args.query, hybrid_retriever, generator)

    print("\n--- GROUNDED REGULATORY ANSWER ---")
    print(result["answer"])

    print("\n--- SOURCE CITATIONS ---")
    for c in result["citations"]:
        print(f" - {c}")

    print("\n--- DISCRETE CLAIMS (FOR DOWNSTREAM SLM VERIFIER) ---")
    for idx, claim in enumerate(result["claims"], start=1):
        print(f" [{idx}] {claim}")

    print(f"\n[Engine Used: {result['model_used']}]")


def handle_eval(args):
    """CLI wrapper for evaluation."""
    print("=== RUNNING RAG EVALUATION BENCHMARK ===")
    _vector_store, _bm25, hybrid_retriever, generator = build_pipeline()
    run_evaluation(hybrid_retriever, generator)


def main():
    parser = argparse.ArgumentParser(description="RBI & SEBI Financial Regulatory RAG Module")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Parse dataset PDFs and build vector + BM25 index")
    ingest_parser.add_argument("--max-docs", type=int, default=None, help="Maximum number of PDF documents to index (optional)")

    query_parser = subparsers.add_parser("query", help="Query the regulatory RAG pipeline")
    query_parser.add_argument("query", type=str, help="Regulatory query string")

    subparsers.add_parser("eval", help="Run benchmark evaluation suite")

    args = parser.parse_args()

    if args.command == "ingest":
        handle_ingest(args)
    elif args.command == "query":
        handle_query(args)
    elif args.command == "eval":
        handle_eval(args)


if __name__ == "__main__":
    main()
