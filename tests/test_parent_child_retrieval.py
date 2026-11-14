"""Unit tests for retrieval/parent_child.py."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ingestion.chunk_structured import save_parents_to_db
from retrieval.parent_child import retrieve_parent_child


class DummyHit:
    def __init__(self, parent_id: str, score: float, text: str = "child text"):
        self.payload = {"parent_id": parent_id, "text": text}
        self.score = score


@pytest.fixture
def mock_parents_db(tmp_path: Path):
    db_path = tmp_path / "parents.db"
    parents = [
        {
            "parent_id": "p1_sec_0",
            "paper_id": "p1",
            "title": "Paper One",
            "year": 2024,
            "section": "1 Introduction",
            "section_type": "body",
            "page_start": 1,
            "page_end": 2,
            "text": "Full intro of paper 1 spanning pages 1-2.",
            "token_count": 100,
            "is_table": False,
        },
        {
            "parent_id": "p1_sec_1",
            "paper_id": "p1",
            "title": "Paper One",
            "year": 2024,
            "section": "2 Methodology",
            "section_type": "body",
            "page_start": 3,
            "page_end": 3,
            "text": "Methodology section of paper 1.",
            "token_count": 80,
            "is_table": False,
        },
        {
            "parent_id": "p2_tbl_0",
            "paper_id": "p2",
            "title": "Paper Two",
            "year": 2023,
            "section": "Table 1: Accuracies",
            "section_type": "table",
            "page_start": 4,
            "page_end": 4,
            "text": "| Model | Accuracy |\n|---|---|\n| Baseline | 70% |",
            "token_count": 30,
            "is_table": True,
        },
    ]
    save_parents_to_db(parents, db_path=db_path)
    return db_path


def test_retrieve_parent_child_deduplication_and_ranking(mock_parents_db):
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "scholar_papers_v2"
    mock_client.get_collections.return_value.collections = [mock_col]

    # 4 child hits:
    # 2 hits for parent "p1_sec_0" with scores 0.75 and 0.88
    # 1 hit for parent "p2_tbl_0" with score 0.95
    # 1 hit for parent "p1_sec_1" with score 0.60
    hits = [
        DummyHit(parent_id="p2_tbl_0", score=0.95),
        DummyHit(parent_id="p1_sec_0", score=0.75),
        DummyHit(parent_id="p1_sec_0", score=0.88),  # Higher score for p1_sec_0
        DummyHit(parent_id="p1_sec_1", score=0.60),
    ]
    mock_client.search.return_value = hits

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(1024)

    results = retrieve_parent_child(
        query="test query",
        top_k=2,
        parents_db=mock_parents_db,
        collection_name="scholar_papers_v2",
        client=mock_client,
        model=mock_model,
    )

    # Top-K = 2: should return top 2 deduplicated parents
    assert len(results) == 2

    # First result should be p2_tbl_0 (score 0.95)
    assert results[0]["parent_id"] == "p2_tbl_0"
    assert results[0]["score"] == 0.95
    assert results[0]["is_table"] is True
    assert results[0]["page_start"] == 4
    assert results[0]["page_end"] == 4

    # Second result should be p1_sec_0 with the MAX score of 0.88 (not 0.75)
    assert results[1]["parent_id"] == "p1_sec_0"
    assert results[1]["score"] == 0.88
    assert results[1]["page_start"] == 1
    assert results[1]["page_end"] == 2
    assert "Full intro of paper 1" in results[1]["text"]


def test_retrieve_parent_child_empty_query():
    results = retrieve_parent_child(query="   ")
    assert results == []


def test_retrieve_parent_child_missing_collection(tmp_path: Path):
    db_path = tmp_path / "parents.db"
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(1024)

    results = retrieve_parent_child(
        query="query",
        collection_name="nonexistent_collection",
        parents_db=db_path,
        client=mock_client,
        model=mock_model,
    )
    assert results == []
