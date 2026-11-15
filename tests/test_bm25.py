"""Unit tests for retrieval/bm25.py."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from retrieval.bm25 import (
    build_bm25_index,
    load_bm25_index,
    retrieve_bm25,
    tokenize,
)


class DummyPoint:
    def __init__(self, point_id: str, child_id: str, text: str, parent_id: str = "p1"):
        self.id = point_id
        self.payload = {
            "child_id": child_id,
            "parent_id": parent_id,
            "paper_id": "paper1",
            "title": "Sample Paper",
            "year": 2024,
            "section": "1 Introduction",
            "section_type": "body",
            "page_start": 1,
            "page_end": 1,
            "page": 1,
            "is_table": False,
            "text": text,
        }


def test_tokenize_preserves_numbers_and_hyphens():
    text = "BAAI/bge-reranker-v2-m3 achieves state-of-the-art results on GPT-4 in 2024!"
    tokens = tokenize(text)
    assert "baai" in tokens
    assert "bge-reranker-v2-m3" in tokens
    assert "state-of-the-art" in tokens
    assert "gpt-4" in tokens
    assert "2024" in tokens


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_build_and_load_bm25_index(tmp_path: Path):
    index_path = tmp_path / "bm25_index.pkl"

    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "scholar_papers_v3"
    mock_client.get_collections.return_value.collections = [mock_col]

    points = [
        DummyPoint("uuid-1", "c1", "ChainPoll detects hallucinations in large language models."),
        DummyPoint("uuid-2", "c2", "Docling parses structured tables and sections from academic PDFs."),
    ]
    mock_client.scroll.return_value = (points, None)

    col_info = MagicMock()
    col_info.points_count = 2
    mock_client.get_collection.return_value = col_info

    # Build index
    saved_path = build_bm25_index(
        output_path=index_path,
        collection_name="scholar_papers_v3",
        client=mock_client,
    )
    assert saved_path.exists()

    # Load index (with verification)
    loaded = load_bm25_index(
        index_path=index_path,
        collection_name="scholar_papers_v3",
        client=mock_client,
        verify=True,
    )
    assert loaded["child_count"] == 2
    assert loaded["child_ids"] == ["c1", "c2"]

    # Query index
    hits = retrieve_bm25(
        query="ChainPoll hallucination",
        top_k=2,
        index_path=index_path,
        collection_name="scholar_papers_v3",
        client=mock_client,
        verify=False,
    )
    assert len(hits) == 2
    assert hits[0]["child_id"] == "c1"
    assert hits[0]["score"] > hits[1]["score"]
    assert "bm25_score" in hits[0]


def test_load_bm25_count_mismatch_fails(tmp_path: Path):
    index_path = tmp_path / "bm25_index.pkl"

    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "scholar_papers_v3"
    mock_client.get_collections.return_value.collections = [mock_col]

    points = [DummyPoint("uuid-1", "c1", "Sample child text")]
    mock_client.scroll.return_value = (points, None)

    # Build with 1 point
    build_bm25_index(
        output_path=index_path,
        collection_name="scholar_papers_v3",
        client=mock_client,
    )

    # Mock Qdrant collection reporting 5 points now
    col_info = MagicMock()
    col_info.points_count = 5
    mock_client.get_collection.return_value = col_info

    # Clear memory cache by passing fresh path
    with pytest.raises(ValueError) as exc_info:
        load_bm25_index(
            index_path=index_path,
            collection_name="scholar_papers_v3",
            client=mock_client,
            verify=True,
        )

    assert "BM25 index mismatch with Qdrant collection" in str(exc_info.value)
    assert "python retrieval/bm25.py --build" in str(exc_info.value)


def test_load_bm25_missing_file_fails(tmp_path: Path):
    missing_path = tmp_path / "non_existent.pkl"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_bm25_index(index_path=missing_path, verify=False)
    assert "BM25 index file not found" in str(exc_info.value)
