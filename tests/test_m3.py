import pytest
from generation.generator import Generator

class DummyLLM:
    def chat(self, user, system, temperature, logprobs=False, **kwargs):
        # Mock response based on the prompt
        if "Evidence:" in user:
            return {"text": "According to the rules [RBI_1] and [RBI_2], the CRR is 4.5%.", "model": "dummy", "token_logprobs": [-0.1, -0.2]}
        else:
            return {"text": "I don't know the answer.", "model": "dummy"}

def test_citation_parsing():
    gen = Generator(llm_client=DummyLLM())
    cites = gen.parse_citations("The rate is 5% [chunk_123] and [chunk_456]. Also [chunk_123] again.")
    assert "chunk_123" in cites
    assert "chunk_456" in cites
    assert len(cites) == 2
    
def test_generation_pipeline():
    from common.schemas import RetrievalResult, Chunk
    rr = RetrievalResult(
        query="What is CRR?",
        query_date="2024-01-01",
        chunks=[Chunk(chunk_id="RBI_1", text="CRR is 4.5%", regulator="RBI", issue_date="2024", source_url="")]
    )
    
    gen = Generator(llm_client=DummyLLM())
    answer = gen.generate(rr)
    
    assert "RBI_1" in answer.citations
    assert "RBI_2" in answer.citations
    assert answer.metadata["no_context_answer"] == "I don't know the answer."
    assert answer.token_logprobs == [-0.1, -0.2]
