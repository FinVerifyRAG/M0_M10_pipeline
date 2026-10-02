import os
import pytest
from common.schemas import RetrievalResult, Chunk
from common.io import write_jsonl, read_jsonl
from common.dates import parse_indian_date, parse_financial_year, compare_dates
from datetime import datetime

def test_io_roundtrip(tmp_path):
    chunk = Chunk(chunk_id="c1", text="hello", regulator="RBI", issue_date="2024-01-01", source_url="url")
    rr = RetrievalResult(query="q", query_date="2024-01-01", chunks=[chunk])
    
    file_path = tmp_path / "rr.jsonl"
    write_jsonl(str(file_path), [rr])
    
    loaded = read_jsonl(str(file_path), RetrievalResult)
    assert len(loaded) == 1
    assert loaded[0].query == "q"
    assert loaded[0].chunks[0].chunk_id == "c1"
    assert loaded[0].schema_version == "1.0"

def test_dates_parsing():
    assert parse_indian_date("12 March 2024") == datetime(2024, 3, 12)
    assert parse_indian_date("12/03/2024") == datetime(2024, 3, 12)
    assert parse_indian_date("12-03-2024") == datetime(2024, 3, 12)
    
    assert parse_financial_year("FY 2023-24") == (2023, 2024)
    assert parse_financial_year("2021-22") == (2021, 2022)
    
    assert compare_dates("12/03/2024", "13 March 2024") == -1
    assert compare_dates("12/03/2024", "12-03-2024") == 0
    assert compare_dates("15/03/2024", "12 March 2024") == 1
