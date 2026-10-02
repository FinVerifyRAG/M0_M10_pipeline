#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_full_eval.py  --  RegGuard M0->M10 End-to-End Evaluation
Fixes: (1) atoms KeyError, (2) Groq rate-limit sleep, (3) default thresholds
"""
from __future__ import annotations
import json, logging, pickle, sys, time, os
from datetime import date
from pathlib import Path
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(name)-28s %(levelname)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("run_full_eval")

QUERIES = [
    {"question": "What is the minimum Capital Adequacy Ratio required for scheduled commercial banks under RBI guidelines?",        "query_date": "2024-01-01", "family": "rate"},
    {"question": "What are the KYC norms for opening a savings account as per RBI master circular?",                                "query_date": "2024-01-01", "family": "applicability"},
    {"question": "What is the timeline for submission of Suspicious Transaction Report to FIU-IND?",                                "query_date": "2024-01-01", "family": "threshold"},
    {"question": "What are the disclosure requirements for listed companies under SEBI LODR regulations?",                          "query_date": "2024-01-01", "family": "section"},
    {"question": "What is the penalty for non-compliance with SEBI Insider Trading regulations?",                                   "query_date": "2024-01-01", "family": "entity"},
]

# ── Default thresholds when thresholds.json not present ──────────────────────
# accept_below=0.35 -> SUPPORTED, 0.35-0.70 -> UNCERTAIN (judge), >0.70 -> ABSTAINED
def _make_default_thresholds():
    from common.schemas import Thresholds
    strata = [
        "*|*", "RBI|RATE", "RBI|THRESHOLD", "RBI|SECTION", "RBI|DATE",
        "RBI|ENTITY", "RBI|APPLICABILITY", "SEBI|RATE", "SEBI|THRESHOLD",
        "SEBI|SECTION", "SEBI|DATE", "SEBI|ENTITY", "SEBI|APPLICABILITY",
        "*|RATE", "*|THRESHOLD", "*|SECTION", "*|DATE", "*|ENTITY", "*|APPLICABILITY",
    ]
    return {s: Thresholds(stratum_name=s, accept_below=0.35, abstain_above=0.70) for s in strata}


def _old_chunk_to_new(oc):
    from common.schemas import Chunk
    meta = oc.metadata or {}
    issuer = meta.get("issuer", "").upper()
    reg = "RBI" if "RBI" in issuer else ("SEBI" if "SEBI" in issuer else issuer or "UNKNOWN")
    yr = str(meta.get("year", ""))
    iss = f"{yr}-01-01" if yr.isdigit() else "1970-01-01"
    return Chunk(chunk_id=oc.chunk_id, text=oc.text, regulator=reg,
                 issue_date=iss, source_url=meta.get("file_path", ""), metadata=meta)


def build_retriever():
    logger.info("M2: Loading BM25 index (42,986 chunks)...")
    with open(ROOT / ".bm25_index.pkl", "rb") as f:
        raw = pickle.load(f)
    old_chunks = raw["chunks"]
    tokenized  = raw["tokenized_corpus"]
    logger.info("  Converting %d chunks...", len(old_chunks))

    from retrieval.bm25 import BM25Index
    from retrieval.dense import DenseIndex
    from retrieval.rerank import Reranker
    from retrieval.retriever import Retriever
    from ingest.version_graph.build import VersionGraph

    new_chunks = [_old_chunk_to_new(c) for c in old_chunks]
    chunk_dict = {c.chunk_id: c for c in new_chunks}

    bm25_obj = BM25Index()
    bm25_obj.chunks = chunk_dict
    bm25_obj.chunk_ids = list(chunk_dict.keys())
    bm25_obj.tokenized_corpus = tokenized
    bm25_obj.bm25 = BM25Okapi(tokenized)
    logger.info("  BM25: %d chunks ready", len(chunk_dict))

    dense = DenseIndex(persist_dir=str(ROOT / ".chroma_db"))
    logger.info("  ChromaDB: %d vectors", dense.collection.count())

    reranker = Reranker()
    vg = VersionGraph()
    return Retriever(bm25_index=bm25_obj, dense_index=dense, reranker=reranker, version_graph=vg)


def build_generator():
    logger.info("M3: Generator model=%s", os.environ.get("LLM_MODEL"))
    from common.llm_client import LLMClient
    from generation.generator import Generator
    llm = LLMClient(base_url=os.environ.get("LLM_BASE_URL"),
                    api_key=os.environ.get("LLM_API_KEY"),
                    default_model=os.environ.get("LLM_MODEL"))
    return Generator(llm_client=llm)


def build_judge():
    logger.info("M8: Initialising judge LLM...")
    try:
        from judge.judge_llm import JudgeLLM
        j = JudgeLLM(); logger.info("  JudgeLLM ready"); return j
    except Exception as e:
        logger.warning("  JudgeLLM unavailable (%s) -- skip_judge=True", e)
        return None


def run_one_query(question, query_date, retriever, generator, thresholds, judge, strict=True):
    from atoms.extractor import extract
    from verify.cascade import verify
    from aggregate.model import score
    from decision.combine import decide as m8_decide
    from decision.answer_rewrite import rewrite_answer

    lat = {}
    t = time.perf_counter(); rr = retriever.retrieve(query=question, query_date=query_date); lat["retrieve"] = round(time.perf_counter()-t,3)
    logger.info("  M2: %d chunks (%.3fs)", len(rr.chunks), lat["retrieve"])

    t = time.perf_counter(); answer = generator.generate(rr); lat["generate"] = round(time.perf_counter()-t,3)
    logger.info("  M3: %d chars (%.3fs)", len(answer.answer_text), lat["generate"])

    t = time.perf_counter(); atoms = extract(answer); lat["extract"] = round(time.perf_counter()-t,3)
    logger.info("  M4: %d atoms (%.3fs)", len(atoms), lat["extract"])

    t = time.perf_counter(); v_atoms = verify(atoms, rr); lat["verify"] = round(time.perf_counter()-t,3)
    logger.info("  M5: %d verified (%.3fs)", len(v_atoms), lat["verify"])

    t = time.perf_counter(); s_atoms = score(v_atoms, answer, rr, strict=strict); lat["score"] = round(time.perf_counter()-t,3)
    logger.info("  M6: %d scored (%.3fs)", len(s_atoms), lat["score"])

    t = time.perf_counter()
    ad = m8_decide(scored_atoms=s_atoms, thresholds=thresholds, judge=judge, rr=rr, answer=answer, skip_judge=(judge is None))
    lat["decide"] = round(time.perf_counter()-t,3)
    logger.info("  M8: %s (%.3fs)", ad.summary(), lat["decide"])

    t = time.perf_counter(); rw = rewrite_answer(answer=answer, answer_decision=ad, mode="flag"); lat["rewrite"] = round(time.perf_counter()-t,3)

    atom_rows = [{"risk": d.risk,
                  "label": "supported" if d.final_status in ("SUPPORTED","VERIFIED") else "unsupported",
                  "decision": d.final_status, "atom_type": d.atom_type, "stratum": d.stratum}
                 for d in ad.atom_decisions]

    return {
        "question": question, "query_date": query_date,
        "answer": rw.text, "original_answer": answer.answer_text,
        "answer_status": ad.answer_status, "coverage": ad.coverage,
        "n_atoms": len(ad.atom_decisions), "n_supported": ad.n_supported,
        "n_verified": ad.n_verified, "n_uncertain": ad.n_uncertain,
        "n_abstained": ad.n_abstained, "n_not_verified": ad.n_not_verified,
        "judge_calls": ad.judge_calls, "latency": lat,
        "atoms": atom_rows,  # kept -- callers must pop themselves
        "sources": [{"chunk_id": c.chunk_id, "regulator": c.regulator,
                     "section": str((c.metadata or {}).get("section",""))} for c in rr.chunks[:6]],
    }


def run_m7_ltt(all_atoms, strict: bool = True):
    import numpy as np
    from common.errors import StrictFailure
    from guarantee.ltt import certify_lambda, simulate_violation_rate
    logger.info("M7: LTT on %d atoms...", len(all_atoms))
    if not all_atoms:
        if strict:
            raise StrictFailure("No atoms to certify. Refusing to simulate a calibration pool.")
        return {"threshold": None, "status": "not_certifiable", "simulate": {}, "n_atoms": 0}
    risk_arr  = np.array([a["risk"] for a in all_atoms])
    wrong_arr = np.array([0.0 if a["label"]=="supported" else 1.0 for a in all_atoms])
    cert = certify_lambda(risk_arr, wrong_arr, eps=0.05, delta=0.10, use_hb=False)
    sim = simulate_violation_rate(risk_arr, wrong_arr, eps=0.05, delta=0.10, n_splits=500, seed=42, use_hb=False)
    if strict and not cert.valid:
        raise StrictFailure(f"LTT status={cert.status}: {cert.reason}")
    if strict and not sim.get("valid"):
        raise StrictFailure("LTT held-out check has zero valid coverage.")
    logger.info(
        "  LTT status=%s threshold=%s coverage=%.3f",
        cert.status, cert.threshold, cert.coverage,
    )
    return {
        "threshold": cert.threshold,
        "status": cert.status,
        "coverage": cert.coverage,
        "simulate": sim,
        "n_atoms": int(len(risk_arr)),
    }


def run_m10(all_atoms, strict: bool = True):
    from eval.run_all import run_all
    rd = ROOT/"results"; rd.mkdir(exist_ok=True)
    jpath = rd/"real_test_atoms.jsonl"
    with open(jpath,"w",encoding="utf-8") as f:
        for a in all_atoms: f.write(json.dumps(a)+"\n")
    logger.info("M10: %d real atoms -> %s", len(all_atoms), jpath)
    return run_all(
        results_dir=str(rd), seed=42, n_splits=500, eps=0.05, delta=0.10,
        test_data=str(jpath), strict=strict,
    )


def print_query_result(i, total, q, r, atom_rows):
    SEP="="*72
    print(f"\n{SEP}")
    print(f"  QUERY {i}/{total}  [{q['family'].upper()}]")
    print(f"  {r['question']}")
    print(SEP)
    print(f"  Status  : {r['answer_status']}  | Coverage: {r['coverage']:.1%}  | Atoms: {r['n_atoms']}  | Judge calls: {r['judge_calls']}")
    print(f"  Breakdown: supported={r['n_supported']} verified={r['n_verified']} uncertain={r['n_uncertain']} abstained={r['n_abstained']} not_verified={r['n_not_verified']}")
    print()
    ans = r["answer"][:700]+("..." if len(r["answer"])>700 else "")
    print(f"ANSWER:\n{ans}\n")
    print(f"ATOMS ({len(atom_rows)}):")
    for a in atom_rows:
        bar="#"*int(a["risk"]*10)+"."*(10-int(a["risk"]*10))
        print(f"  [{a['decision']:16s}] ({a['atom_type']:15s}) risk={a['risk']:.3f} [{bar}]")
    print("\nSOURCES:")
    for s in r.get("sources",[])[:4]:
        print(f"  [{s['chunk_id'][:42]}] {s['regulator']} | {s['section'][:50]}")
    print("\nLATENCY:")
    for stage,sec in r["latency"].items():
        print(f"  {stage:12s}: {sec:.3f}s")


def print_final_report(qr, m7, m10, all_atoms):
    SEP="="*72; DIV="-"*72
    print(f"\n\n{SEP}")
    print("  RegGuard  M0 -> M10  FINAL METRIC REPORT")
    print(f"  Generated : {date.today().isoformat()}")
    print(f"  Dataset   : pre-ingested RBI/SEBI PDFs (42,986 chunks)")
    print(f"  Atoms     : {len(all_atoms)} real atoms collected from pipeline")
    print(SEP)

    # Per-query table
    print(f"\n{'#':<4} {'Family':<16} {'Status':<24} {'Cover':>7} {'Atoms':>6} {'n_sup':>6} {'n_unc':>6} {'Time':>8}")
    print(DIV)
    for i,r in enumerate(qr,1):
        ts = sum(r.get("latency",{}).values())
        err = " [ERR]" if r["answer_status"]=="ERROR" else ""
        print(f"{i:<4} {r.get('family','?'):<16} {r['answer_status']:<24} {r['coverage']:>6.1%} "
              f"{r['n_atoms']:>6} {r.get('n_supported',0)+r.get('n_verified',0):>6} "
              f"{r.get('n_uncertain',0):>6} {ts:>7.1f}s{err}")

    # M10 metrics
    s = m10.get("summary",{})
    print(f"\n{DIV}")
    print("M10 EVALUATION METRICS  (real pipeline atoms):")
    print(f"  Atoms evaluated      : {s.get('n_atoms','N/A')}")
    hr = s.get("hallucination_rate", float("nan"))
    if hr == hr:  # not NaN
        print(f"  Hallucination rate   : {hr:.4f}   (target <= {s.get('eps',0.05):.2f})")
    else:
        print(f"  Hallucination rate   : N/A  (no accepted atoms -- all abstained or 0 atoms)")
    print(f"  Coverage (answered)  : {s.get('coverage',0):.4f}")
    print(f"  AURC                 : {s.get('aurc',0):.4f}  (lower = better; 0=perfect)")
    vr = s.get("violation_rate",1.0); ok = s.get("violation_rate_ok",False)
    print(f"  Violation rate       : {vr:.4f}   {'PASSES' if ok else 'FAILS'} (delta=0.10)")
    print(f"  Retrieval Recall@6   : {s.get('retrieval_recall_at_6',0):.4f}")
    print(f"  Retrieval MRR        : {s.get('retrieval_mrr',0):.4f}")

    # M7 Guarantee
    thr = m7.get("threshold"); sim = m7.get("simulate",{})
    print(f"\n{DIV}")
    print("M7 GUARANTEE (Learn-then-Test, Mondrian stratification):")
    print(f"  LTT threshold        : {thr:.4f}" if thr is not None else "  LTT threshold        : None (insufficient calibration data to certify)")
    print(f"  Simulated viol rate  : {sim.get('violation_rate',0):.4f}  (target <= 0.10)")
    print(f"  Statistical bound    : {'HOLDS' if sim.get('violation_rate',1)<0.10 else 'VIOLATED'}  (delta=0.10)")

    # Ablation
    abl = m10.get("ablation_table",[])
    if abl:
        print(f"\n{DIV}")
        print("ABLATION TABLE  (AURC lower = better, Cov higher = better):")
        print(f"  {'Condition':<44} {'AURC':>7} {'Hall':>7} {'Cov':>7}")
        for row in abl:
            print(f"  {row['condition']:<44} {row['aurc']:>7.4f} {row.get('hallucination_rate',0):>7.4f} {row['coverage']:>7.4f}")

    # Retrieval
    ret = m10.get("retrieval_metrics",{})
    if ret:
        print(f"\n{DIV}")
        print("RETRIEVAL METRICS (M2):")
        print(f"  Recall@6     = {ret.get('recall_at_6',0):.4f}")
        print(f"  Recall@10    = {ret.get('recall_at_10',0):.4f}")
        print(f"  MRR          = {ret.get('mrr',0):.4f}")
        print(f"  Precision@6  = {ret.get('precision_at_6',0):.4f}")

    # Extraction
    ext = m10.get("extraction_metrics",{})
    if ext:
        print(f"\n{DIV}")
        print("EXTRACTION METRICS (M4):")
        print(f"  {'Type':<16} {'Recall':>8} {'Precision':>10} {'F1':>8}")
        for t,v in ext.items():
            print(f"  {t:<16} {v.get('recall',0):>8.4f} {v.get('precision',0):>10.4f} {v.get('f1',0):>8.4f}")

    # Drift
    dr = m10.get("drift_table",{})
    if dr and dr.get("strategies"):
        print(f"\n{DIV}")
        print(f"DRIFT EXPERIMENT (M7, cutoff={dr.get('cutoff_date','?')}, n_pre={dr.get('n_pre')}, n_post={dr.get('n_post')}):")
        print(f"  {'Strategy':<24} {'Viol Rate':>10} {'Threshold':>10} {'Coverage':>10}")
        for st in dr.get("strategies",[]):
            print(f"  {st.get('strategy','?'):<24} {st.get('violation_rate',0):>10.4f} {st.get('threshold',0):>10.4f} {st.get('coverage',0):>10.4f}")
        print(f"  Best strategy: {dr.get('best_strategy','?')}")

    # Atom risk distribution
    if all_atoms:
        print(f"\n{DIV}")
        print("ATOM RISK DISTRIBUTION (real pipeline atoms):")
        buckets = {"[0.00-0.35) SUPPORTED": 0, "[0.35-0.70) UNCERTAIN": 0, "[0.70-1.00] ABSTAINED": 0}
        for a in all_atoms:
            r = a["risk"]
            if r < 0.35: buckets["[0.00-0.35) SUPPORTED"] += 1
            elif r < 0.70: buckets["[0.35-0.70) UNCERTAIN"] += 1
            else: buckets["[0.70-1.00] ABSTAINED"] += 1
        for label, cnt in buckets.items():
            pct = cnt/len(all_atoms)*100
            bar = "#"*int(pct/5)
            print(f"  {label:<30} {cnt:>3} ({pct:5.1f}%)  {bar}")

    # Output files
    print(f"\n{DIV}")
    print("OUTPUT FILES (results/):")
    for fn in ["pipeline_report.json","summary.json","main_table.json","ablation_table.json",
               "violation_rate.json","drift_table.json","retrieval_metrics.json","extraction_metrics.json",
               "latency_breakdown.json","rc_curve.json","real_test_atoms.jsonl"]:
        p = ROOT/"results"/fn
        sz = f" ({p.stat().st_size//1024}KB)" if p.exists() else ""
        print(f"  {'OK' if p.exists() else 'MISSING':7s} {fn}{sz}")
    print(f"\n{SEP}\n")


def main():
    import argparse
    from common.checkpoint import append_jsonl
    from common.errors import StrictFailure
    from common.experiment import load_experiment, provenance
    from common.rate_limit import configure_bucket

    parser = argparse.ArgumentParser(description="RegGuard end-to-end evaluation")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--questions", default=None, help="JSONL with question, query_date, family")
    args = parser.parse_args()

    exp = load_experiment()
    rate = exp.get("rate_limit") or {}
    configure_bucket(float(rate.get("requests_per_second", 0.0)), float(rate.get("capacity", 1)))
    stamp = provenance()

    questions = QUERIES
    if args.questions:
        questions = []
        with open(args.questions, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
    for q in questions:
        if not q.get("query_date"):
            raise StrictFailure("Every question needs its own query_date")

    print("\n"+"="*72)
    print("  RegGuard: End-to-End M0 -> M10 Evaluation")
    print("  Dataset: pre-ingested RBI/SEBI (42,986 chunks | 743MB ChromaDB | 275MB BM25)")
    print("="*72+"\n")
    t_start = time.perf_counter()

    # Build module singletons
    retriever = build_retriever()
    generator = build_generator()
    judge     = build_judge()

    # M7 Thresholds -- load or use calibrated defaults
    thresholds = {}
    tp = ROOT/"models"/"guarantee"/"thresholds.json"
    if tp.exists():
        from guarantee.certify import load_thresholds
        thresholds = load_thresholds(str(tp))
        logger.info("M7: Thresholds loaded from %s", tp)
    else:
        if args.strict:
            raise StrictFailure(
                "No certified thresholds at models/guarantee/thresholds.json. "
                "Refusing heuristic accept<0.35 / abstain>0.70."
            )
        thresholds = _make_default_thresholds()
        logger.warning("Non-strict run is using heuristic thresholds.")

    all_atoms, query_results = [], []

    checkpoint = ROOT / "results" / "checkpoints" / "queries.jsonl"
    for i, q in enumerate(questions, 1):
        logger.info("="*50)
        logger.info("QUERY %d/%d  [%s]: %s", i, len(questions), q.get("family", "").upper(), q["question"][:70])
        logger.info("="*50)

        atom_rows = []
        try:
            r = run_one_query(q["question"], q["query_date"], retriever, generator, thresholds, judge, strict=args.strict)
            r["family"] = q.get("family", "")
            atom_rows = r.pop("atoms")   # pop atoms for all_atoms, keep separately for printing
            all_atoms.extend(atom_rows)
            query_results.append(r)
            append_jsonl(checkpoint, {"question": q["question"], "status": "OK", "n_atoms": r["n_atoms"], **stamp})
            print_query_result(i, len(questions), q, r, atom_rows)
        except Exception as exc:
            append_jsonl(checkpoint, {"question": q["question"], "status": "ERROR", "error": str(exc), **stamp})
            if args.strict:
                raise StrictFailure(f"Query {i} ended in ERROR: {exc}") from exc
            logger.error("Query %d ERROR: %s", i, exc, exc_info=True)
            query_results.append({
                "question": q["question"], "family": q["family"],
                "answer_status": "ERROR", "coverage": 0.0, "n_atoms": 0,
                "n_supported": 0, "n_verified": 0, "n_uncertain": 0,
                "n_abstained": 0, "n_not_verified": 0, "judge_calls": 0,
                "latency": {}, "answer": "", "sources": [], "error": str(exc),
            })
            # Still print the error row
            print(f"\n{'='*72}")
            print(f"  QUERY {i}/{len(questions)}  [{q.get('family','').upper()}]  -- ERROR")
            print(f"  {str(exc)[:200]}")

    logger.info("Atoms collected: %d from %d queries", len(all_atoms), len(questions))

    # M7: LTT
    m7 = run_m7_ltt(all_atoms, strict=args.strict)

    # M10: Full eval suite
    m10 = run_m10(all_atoms, strict=args.strict)

    # Print final report
    print_final_report(query_results, m7, m10, all_atoms)

    # Save pipeline report
    rd = ROOT/"results"; rp = rd/"pipeline_report.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump({
            "generated": date.today().isoformat(),
            **stamp,
            "dataset": {"n_chunks": 42986, "n_queries": len(QUERIES), "n_atoms_total": len(all_atoms)},
            "queries": query_results,
            "m7": {
                "threshold": m7.get("threshold"),
                "violation_rate": m7.get("simulate",{}).get("violation_rate"),
                "n_atoms": m7.get("n_atoms"),
            },
            "m10_summary": m10.get("summary",{}),
            "ablation_table": m10.get("ablation_table",[]),
            "retrieval_metrics": m10.get("retrieval_metrics",{}),
            "extraction_metrics": m10.get("extraction_metrics",{}),
            "drift_table": m10.get("drift_table",{}),
        }, f, indent=2, default=str)
    logger.info("Report saved: %s  |  Total time: %.1fs", rp, time.perf_counter()-t_start)


if __name__ == "__main__":
    main()