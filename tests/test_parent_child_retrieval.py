"""Unit tests for retrieval/parent_child.py."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ingestion.chunk_structured import save_parents_to_db
from retrieval.parent_child import retrieve_parent_child


class DummyHit:
    def __init__(
        self,
        parent_id: str,
        score: float,
        text: str = "child text",
        child_id: str = "c0",
    ):
        self.payload = {
            "parent_id": parent_id,
            "text": text,
            "child_id": child_id,
        }
        self.score = score
        self.id = child_id


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
        {
            "parent_id": "p1_app_0",
            "paper_id": "p1",
            "title": "Paper One",
            "year": 2024,
            "section": "A Overview of Appendix",
            "section_type": "appendix",
            "page_start": 5,
            "page_end": 5,
            "text": "Detailed appendix text.",
            "token_count": 60,
            "is_table": False,
        },
        {
            "parent_id": "p1_ref_0",
            "paper_id": "p1",
            "title": "Paper One",
            "year": 2024,
            "section": "References",
            "section_type": "references",
            "page_start": 4,
            "page_end": 4,
            "text": "[1] First citation.",
            "token_count": 40,
            "is_table": False,
        },
    ]
    save_parents_to_db(parents, db_path=db_path)
    return db_path


def test_retrieve_parent_child_deduplication_and_ranking(mock_parents_db):
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "scholar_papers_v2"
    mock_client.get_collections.return_value.collections = [mock_col]

    hits = [
        DummyHit(parent_id="p2_tbl_0", score=0.95),
        DummyHit(parent_id="p1_sec_0", score=0.75),
        DummyHit(parent_id="p1_sec_0", score=0.88),
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

    assert len(results) == 2
    assert results[0]["parent_id"] == "p2_tbl_0"
    assert results[0]["score"] == 0.95
    assert results[0]["is_table"] is True
    assert results[0]["page_start"] == 4
    assert results[0]["page_end"] == 4

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


def test_retrieve_parent_child_appendix_filtering(mock_parents_db):
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "scholar_papers_v2"
    mock_client.get_collections.return_value.collections = [mock_col]

    hits = [
        DummyHit(parent_id="p1_app_0", score=0.99),
        DummyHit(parent_id="p1_sec_0", score=0.85),
    ]
    mock_client.search.return_value = hits
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(1024)

    # 1. include_appendix=False -> should drop p1_app_0
    results_no_app = retrieve_parent_child(
        query="test query",
        top_k=2,
        parents_db=mock_parents_db,
        collection_name="scholar_papers_v2",
        client=mock_client,
        model=mock_model,
        include_appendix=False,
    )
    assert len(results_no_app) == 1
    assert results_no_app[0]["parent_id"] == "p1_sec_0"

    # 2. include_appendix=True -> should include p1_app_0 with highest score
    results_with_app = retrieve_parent_child(
        query="test query",
        top_k=2,
        parents_db=mock_parents_db,
        collection_name="scholar_papers_v2",
        client=mock_client,
        model=mock_model,
        include_appendix=True,
    )
    assert len(results_with_app) == 2
    assert results_with_app[0]["parent_id"] == "p1_app_0"
    assert results_with_app[0]["section_type"] == "appendix"


def test_retrieve_parent_child_references_filtering(mock_parents_db):
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "scholar_papers_v2"
    mock_client.get_collections.return_value.collections = [mock_col]

    hits = [
        DummyHit(parent_id="p1_ref_0", score=0.98),
        DummyHit(parent_id="p1_sec_0", score=0.80),
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
        include_references=False,
    )
    assert len(results) == 1
    assert results[0]["parent_id"] == "p1_sec_0"


def test_retrieve_parent_child_candidate_expansion(mock_parents_db):
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "scholar_papers_v2"
    mock_client.get_collections.return_value.collections = [mock_col]

    # In first call (limit 8, top_k 2 * 4), returns hits where all are appendix
    # In second call (expanded limit), returns body hits as well
    def search_side_effect(collection_name, query_vector, limit, with_payload):
        if limit <= 8:
            return [DummyHit(parent_id="p1_app_0", score=0.99, child_id=f"c_app_{i}") for i in range(8)]
        else:
            return (
                [DummyHit(parent_id="p1_app_0", score=0.99, child_id=f"c_app_{i}") for i in range(8)]
                + [
                    DummyHit(parent_id="p1_sec_0", score=0.88, child_id="c_sec_0"),
                    DummyHit(parent_id="p1_sec_1", score=0.82, child_id="c_sec_1"),
                ]
            )

    mock_client.search.side_effect = search_side_effect
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(1024)

    results = retrieve_parent_child(
        query="test query",
        top_k=2,
        parents_db=mock_parents_db,
        collection_name="scholar_papers_v2",
        client=mock_client,
        model=mock_model,
        include_appendix=False,
        include_references=False,
    )
    # Should have expanded search and retrieved the 2 body parents
    assert len(results) == 2
    assert results[0]["parent_id"] == "p1_sec_0"
    assert results[1]["parent_id"] == "p1_sec_1"


def test_retrieve_parent_child_debug_output(mock_parents_db, capsys):
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "scholar_papers_v2"
    mock_client.get_collections.return_value.collections = [mock_col]

    hits = [
        DummyHit(parent_id="p1_sec_0", score=0.8812345678, text="Child snippet text", child_id="c_001"),
        DummyHit(parent_id="p1_app_0", score=0.7712345678, text="Appendix child snippet", child_id="c_002"),
    ]
    mock_client.search.return_value = hits
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(1024)

    retrieve_parent_child(
        query="test query",
        top_k=2,
        parents_db=mock_parents_db,
        collection_name="scholar_papers_v2",
        client=mock_client,
        model=mock_model,
        debug=True,
    )

    captured = capsys.readouterr().out
    assert "DEBUG: Top 2 Child Hits" in captured
    assert "Child ID: c_001 | Parent ID: p1_sec_0" in captured
    assert "Parent Heading: 1 Introduction" in captured
    assert "Section Type: body" in captured
    assert "Raw Score: 0.8812345678" in captured
    # Assert section_type resolved from parents.db, not Qdrant payload:
    assert "Parent Heading: A Overview of Appendix" in captured
    assert "Section Type: appendix" in captured
    assert "Raw Score: 0.7712345678" in captured
