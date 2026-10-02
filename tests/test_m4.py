import pytest
from atoms.regex_extract import regex_extract, find_span

SAMPLE_ANSWER = (
    "Under Regulation 52(4) [RBI_1], equity schemes must hold a minimum of 65% in equities. "
    "The maximum investment in a single stock is capped at ₹500 crore [RBI_2]. "
    "This rule is effective from 12 March 2024 [RBI_3]. "
    "For open-ended schemes only, the lock-in period is 3 years. "
    "Not found in evidence for hybrid scheme details."
)

NO_EVIDENCE_ANSWER = "Not found in evidence."

def test_regex_rate_detection():
    hits = regex_extract(SAMPLE_ANSWER)
    texts = [h["text"] for h in hits]
    # Should catch 65%
    assert any("65%" in t for t in texts), f"65% not found in: {texts}"

def test_regex_threshold_detection():
    hits = regex_extract(SAMPLE_ANSWER)
    texts = [h["text"] for h in hits]
    # Should catch ₹500 crore
    assert any("500" in t and "crore" in t.lower() for t in texts), f"₹500 crore not found in: {texts}"

def test_regex_section_detection():
    hits = regex_extract(SAMPLE_ANSWER)
    texts = [h["text"] for h in hits]
    # Should catch Regulation 52(4)
    assert any("52" in t for t in texts), f"Regulation 52 not found in: {texts}"

def test_regex_date_detection():
    hits = regex_extract(SAMPLE_ANSWER)
    texts = [h["text"] for h in hits]
    assert any("March" in t or "2024" in t for t in texts), f"Date not found in: {texts}"

def test_find_span_exact():
    span = find_span(SAMPLE_ANSWER, "₹500 crore")
    assert span is not None
    assert SAMPLE_ANSWER[span[0]:span[1]] == "₹500 crore"

def test_find_span_missing():
    span = find_span(SAMPLE_ANSWER, "₹999 crore")
    assert span is None

def test_splitter_not_found_returns_empty():
    from atoms.splitter import AtomSplitter
    from common.schemas import GeneratedAnswer

    class DummyLLM:
        def chat(self, **kwargs):
            return {"text": "[]", "model": "dummy"}

    splitter = AtomSplitter(llm=DummyLLM())
    ga = GeneratedAnswer(
        query="What is the limit?",
        answer_text=NO_EVIDENCE_ANSWER,
        citations=[]
    )
    atoms = splitter.split(ga)
    assert atoms == []

def test_llm_extractor_deduplication():
    from atoms.llm_extract import LLMAtomExtractor
    import json

    class DummyLLM:
        def chat(self, user=None, system=None, temperature=0.0, logprobs=False, **kwargs):
            payload = json.dumps([
                {"type": "RATE", "text": "65%", "claim": "Equity must be 65%.", "cited_chunk": "RBI_1"},
                {"type": "RATE", "text": "65%", "claim": "Equity must be 65% duplicated.", "cited_chunk": None}
            ])
            return {"text": payload, "model": "dummy"}

    # Must NOT contain 'not found in evidence' so short-circuit doesn't fire
    clean_answer = "Under Regulation 52(4), equity schemes must hold 65% in equities."
    extractor = LLMAtomExtractor(llm=DummyLLM())
    atoms = extractor.extract(clean_answer)
    rate_atoms = [a for a in atoms if a.type == "RATE" and a.text == "65%"]
    assert len(rate_atoms) == 1, f"Duplicate atoms not deduplicated, got {len(rate_atoms)}"

def test_atom_schema_roundtrip():
    from common.schemas import Atom
    import json
    atom = Atom(
        atom_id="atom_abc123",
        type="RATE",
        text="4.5%",
        claim="The CRR is 4.5%.",
        cited_chunk="RBI_1",
        span=(10, 14)
    )
    json_str = atom.model_dump_json()
    loaded = Atom.model_validate_json(json_str)
    assert loaded.type == "RATE"
    assert loaded.text == "4.5%"
    assert loaded.claim == "The CRR is 4.5%."
    assert loaded.cited_chunk == "RBI_1"
    assert loaded.span == (10, 14)
