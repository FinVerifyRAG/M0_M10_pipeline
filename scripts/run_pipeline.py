#!/usr/bin/env python3
"""
scripts/run_pipeline.py
Chains module execution from M1 to M9 end-to-end for CLI queries and evaluation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.logging import setup_logger
from app.pipeline import run_pipeline

logger = setup_logger("pipeline", log_level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Run the RegGuard End-to-End Pipeline")
    parser.add_argument("--query", type=str, help="Run a specific query end-to-end")
    parser.add_argument("--date", type=str, default=None, help="Query effective date (YYYY-MM-DD)")
    parser.add_argument("--skip-judge", action="store_true", help="Skip M8 LLM judge call")
    parser.add_argument("--rewrite-mode", type=str, default="flag", choices=["flag", "trim", "stub_on_abstain", "none"], help="Answer rewrite mode")
    parser.add_argument("--json", action="store_true", help="Output full result as JSON")
    args = parser.parse_args()

    if not args.query:
        print("Please provide a query with --query 'your question'")
        parser.print_help()
        return

    logger.info("Executing RegGuard pipeline for query: %s", args.query)
    result = run_pipeline(
        question=args.query,
        query_date=args.date,
        skip_judge=args.skip_judge,
        rewrite_mode=args.rewrite_mode,
    )

    if args.json:
        out = {
            "query": result.query,
            "query_date": result.query_date,
            "answer": result.answer_text,
            "rewritten_answer": result.rewritten_answer,
            "status": result.verification.status,
            "coverage": result.verification.coverage,
            "atoms": [
                {
                    "id": a.atom_id,
                    "type": a.atom_type,
                    "text": a.text,
                    "status": a.status,
                    "risk": a.risk,
                    "risk_label": a.risk_label,
                }
                for a in result.atoms
            ],
            "sources": [{"chunk_id": s.chunk_id, "regulator": s.regulator, "section": s.section} for s in result.sources],
            "latency": result.latency,
        }
        print(json.dumps(out, indent=2))
        return

    print("\n" + "="*70)
    print(f"QUERY: {result.query} (Date: {result.query_date})")
    print("="*70)
    print(f"\n[VERIFICATION STATUS: {result.verification.status_label}] (Coverage: {result.verification.coverage:.1%})")
    print(f"Explanation: {result.verification.explanation}\n")
    
    print("--- REWRITTEN / VERIFIED ANSWER ---")
    if result.disclaimer:
        print(f"[{result.disclaimer}]")
    print(result.answer)

    print("\n--- ATOMIC VERIFICATION BREAKDOWN ---")
    if not result.atoms:
        print("  (No discrete atomic facts detected)")
    for a in result.atoms:
        print(f"  [{a.status}] ({a.atom_type}) '{a.text}' -> Risk: {a.risk:.2f} ({a.risk_label})")
        if a.evidence_quote:
            print(f"      Evidence: \"{a.evidence_quote[:100]}...\"")

    print("\n--- CITED REGULATORY SOURCES ---")
    for s in result.sources:
        print(f"  - [{s.chunk_id}] {s.regulator} | Section: {s.section}")

    print("\n--- LATENCY BREAKDOWN (s) ---")
    for stage, sec in result.latency.items():
        print(f"  {stage:12}: {sec:.3f}s")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

