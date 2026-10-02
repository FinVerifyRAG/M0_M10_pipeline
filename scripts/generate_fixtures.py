import os
from common.schemas import Chunk, RetrievalResult, GeneratedAnswer, Atom, VerifiedAtom, ScoredAtom, Thresholds, Decision
from common.io import write_jsonl

def generate_fixtures():
    fixture_dir = "tests/fixtures/common"
    os.makedirs(fixture_dir, exist_ok=True)
    
    chunk = Chunk(
        chunk_id="chunk_001",
        text="The minimum CRR is 4.5% of NDTL.",
        regulator="RBI",
        issue_date="2024-01-01",
        source_url="https://rbi.org.in"
    )
    
    rr = RetrievalResult(
        query="What is the CRR?",
        query_date="2024-02-01",
        chunks=[chunk]
    )
    
    answer = GeneratedAnswer(
        query="What is the CRR?",
        answer_text="The CRR is 4.5%.",
        citations=["chunk_001"]
    )
    
    atom = Atom(
        atom_id="atom_1",
        text="CRR is 4.5%.",
        type="RATE",
        span=(4, 16),
        cited_chunk_id="chunk_001"
    )
    
    verified = VerifiedAtom(
        atom=atom,
        v1_status="MATCH",
        v2_entail_prob=0.99
    )
    
    scored = ScoredAtom(
        verified=verified,
        risk=0.01
    )
    
    thresholds = Thresholds(
        stratum_name="RBI|RATE",
        accept_below=0.1,
        abstain_above=0.8
    )
    
    decision = Decision(
        atom_id="atom_1",
        status="SUPPORTED",
        risk=0.01
    )
    
    write_jsonl(f"{fixture_dir}/chunk.jsonl", [chunk])
    write_jsonl(f"{fixture_dir}/retrieval_result.jsonl", [rr])
    write_jsonl(f"{fixture_dir}/generated_answer.jsonl", [answer])
    write_jsonl(f"{fixture_dir}/atom.jsonl", [atom])
    write_jsonl(f"{fixture_dir}/verified_atom.jsonl", [verified])
    write_jsonl(f"{fixture_dir}/scored_atom.jsonl", [scored])
    write_jsonl(f"{fixture_dir}/thresholds.jsonl", [thresholds])
    write_jsonl(f"{fixture_dir}/decision.jsonl", [decision])

if __name__ == "__main__":
    generate_fixtures()
    print("Fixtures generated successfully.")
