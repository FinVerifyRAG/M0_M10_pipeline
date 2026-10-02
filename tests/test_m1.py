import pytest
from ingest.version_graph.build import VersionGraph, VersionNode
from ingest.version_graph.query import in_force, history
from ingest.chunk.metadata import extract_metadata

def test_version_graph_amendments():
    graph = VersionGraph()
    
    # Base Document (Issue: Jan 1, 2023)
    n1 = VersionNode(doc_id="doc_base", section_id="sec_42", effective_from="2023-01-01", effective_to="2023-06-01")
    # Amendment 1 (Issue: Jun 1, 2023)
    n2 = VersionNode(doc_id="doc_amend_1", section_id="sec_42", effective_from="2023-06-01", effective_to="2024-01-01")
    # Amendment 2 (Issue: Jan 1, 2024)
    n3 = VersionNode(doc_id="doc_amend_2", section_id="sec_42", effective_from="2024-01-01", effective_to=None)
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    
    # Queries
    assert in_force(graph, "sec_42", "2023-03-01") == "doc_base"
    assert in_force(graph, "sec_42", "2023-07-01") == "doc_amend_1"
    assert in_force(graph, "sec_42", "2024-05-01") == "doc_amend_2"
    
def test_metadata_extraction():
    text = "Issued on 12 March 2024. This circular is effective from 15/03/2024. In supersession of circular RBI/2023-24/11."
    meta = extract_metadata(text)
    
    assert meta["extracted_issue_date"] == "2024-03-12"
    assert meta["extracted_effective_date"] == "2024-03-15"
    assert "RBI/2023-24/11" in meta["supersedes"]

def test_text_cleaner():
    from ingest.parse.clean import clean_text
    raw_text = "The minimum balance is Rs. 1000\nwhich is regu-\nlation 5."
    clean = clean_text(raw_text)
    assert "₹ 1000" in clean
    assert "regulation" in clean
