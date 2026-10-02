from retrieval.fusion import rrf

def test_rrf():
    list1 = ["chunk_A", "chunk_B", "chunk_C"]
    list2 = ["chunk_C", "chunk_A", "chunk_D"]
    
    # RRF(k=60)
    # chunk_A: 1/61 + 1/62 = 0.0325
    # chunk_B: 1/62 = 0.0161
    # chunk_C: 1/63 + 1/61 = 0.0322
    # chunk_D: 1/63 = 0.0158
    
    fused = rrf([list1, list2], k=60)
    assert fused[0] == "chunk_A"
    assert fused[1] == "chunk_C"
    assert fused[2] == "chunk_B"
    assert fused[3] == "chunk_D"

def test_regulatory_tokenize():
    from retrieval.bm25 import regulatory_tokenize
    tokens = regulatory_tokenize("Under Regulation 52(4), the rate is 5.50%.")
    assert "52(4)" in tokens
    assert "5.50%" in tokens

def test_temporal_filter():
    from retrieval.temporal import temporal_filter
    from common.schemas import Chunk
    from ingest.version_graph.build import VersionGraph
    
    # Mock chunks
    c1 = Chunk(chunk_id="c1", text="old", regulator="RBI", issue_date="2022-01-01", source_url="", effective_from="2022-01-01", effective_to="2023-01-01")
    c2 = Chunk(chunk_id="c2", text="new", regulator="RBI", issue_date="2023-01-01", source_url="", effective_from="2023-01-01", effective_to=None)
    c3 = Chunk(chunk_id="c3", text="future", regulator="RBI", issue_date="2025-01-01", source_url="", effective_from="2025-01-01", effective_to=None)
    
    graph = VersionGraph() # Empty graph, testing hard bounds
    
    # Query date 2022-06-01: c1 should pass
    filtered = temporal_filter([c1, c2, c3], graph, query_date="2022-06-01")
    assert len(filtered) == 1
    assert filtered[0].chunk_id == "c1"
    
    # Query date 2024-01-01: c2 should pass
    filtered = temporal_filter([c1, c2, c3], graph, query_date="2024-01-01")
    assert len(filtered) == 1
    assert filtered[0].chunk_id == "c2"
