"""Unit tests for retrieval/hybrid.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.chunk_structured import save_parents_to_db
from retrieval.hybrid import (
    compute_rrf,
    merge_children_to_parents,
    retrieve_hybrid,
)


@pytest.fixture
def mock_parents_db(tmp_path: Path):
    db_path = tmp_path / "parents.db"
    parents = [
        {
            "parent_id": "p1_sec_0",
            "paper_id": "paper1",
            "title": "Paper One",
            "year": 2024,
            "section": "1 Introduction",
            "section_type": "body",
            "page_start": 1,
            "page_end": 2,
            "text": "Full intro of paper 1.",
            "token_count": 100,
            "is_table": False,
        },
        {
            "parent_id": "p1_sec_1",
            "paper_id": "paper1",
            "title": "Paper One",
            "year": 2024,
            "section": "2 Methodology",
            "section_type": "body",
            "page_start": 3,
            "page_end": 3,
            "text": "Methodology section.",
            "token_count": 80,
            "is_table": False,
        },
        {
            "parent_id": "p2_sec_0",
            "paper_id": "paper2",
            "title": "Paper Two",
            "year": 2023,
            "section": "1 Background",
            "section_type": "body",
            "page_start": 1,
            "page_end": 1,
            "text": "Paper two background.",
            "token_count": 50,
            "is_table": False,
        },
        {
            "parent_id": "p1_app_0",
            "paper_id": "paper1",
            "title": "Paper One",
            "year": 2024,
            "section": "Appendix A",
            "section_type": "appendix",
            "page_start": 8,
            "page_end": 9,
            "text": "Appendix details.",
            "token_count": 60,
            "is_table": False,
        },
    ]
    save_parents_to_db(parents, db_path=db_path)
    return db_path


def test_compute_rrf_scoring():
    dense_hits = [
        {"child_id": "c1", "parent_id": "p1", "score": 0.95},
        {"child_id": "c2", "parent_id": "p2", "score": 0.85},
    ]
    bm25_hits = [
        {"child_id": "c2", "parent_id": "p2", "score": 12.0},
        {"child_id": "c3", "parent_id": "p3", "score": 10.0},
    ]

    # k=60
    # c1: rank 1 in dense (1/61) = 0.016393
    # c2: rank 2 in dense (1/62) + rank 1 in bm25 (1/61) = 0.016129 + 0.016393 = 0.032522
    # c3: rank 2 in bm25 (1/62) = 0.016129
    fused = compute_rrf(dense_hits, bm25_hits, rrf_k=60, top_n_children=10)
    assert len(fused) == 3

    # c2 should be rank 1 due to appearing in both lists
    assert fused[0]["child_id"] == "c2"
    expected_c2_score = (1.0 / 62.0) + (1.0 / 61.0)
    assert pytest.approx(fused[0]["rrf_score"], rel=1e-4) == expected_c2_score
    assert fused[0]["dense_rank"] == 2
    assert fused[0]["bm25_rank"] == 1

    # c1 should be rank 2
    assert fused[1]["child_id"] == "c1"


def test_merge_children_to_parents_best_score_and_filter(mock_parents_db: Path):
    children = [
        {"child_id": "c1_0", "parent_id": "p1_sec_0", "rrf_score": 0.030},
        {"child_id": "c1_1", "parent_id": "p1_sec_0", "rrf_score": 0.045},  # higher child score
        {"child_id": "c1_app", "parent_id": "p1_app_0", "rrf_score": 0.050},  # appendix
        {"child_id": "c2_0", "parent_id": "p2_sec_0", "rrf_score": 0.020},
    ]

    # Without appendix
    parents = merge_children_to_parents(
        children=children,
        parents_db=mock_parents_db,
        include_appendix=False,
        top_k=5,
    )
    # p1_app_0 should be filtered out
    p_ids = [p["parent_id"] for p in parents]
    assert "p1_app_0" not in p_ids
    assert "p1_sec_0" in p_ids
    assert "p2_sec_0" in p_ids

    # p1_sec_0 score should be the max child score (0.045)
    p1 = next(p for p in parents if p["parent_id"] == "p1_sec_0")
    assert pytest.approx(p1["score"]) == 0.045


def test_merge_children_max_parents_per_paper(mock_parents_db: Path):
    children = [
        {"child_id": "c1_0", "parent_id": "p1_sec_0", "rrf_score": 0.040},
        {"child_id": "c1_1", "parent_id": "p1_sec_1", "rrf_score": 0.035},
        {"child_id": "c2_0", "parent_id": "p2_sec_0", "rrf_score": 0.025},
    ]

    # Limit to 1 parent per paper
    parents = merge_children_to_parents(
        children=children,
        parents_db=mock_parents_db,
        max_parents_per_paper=1,
        top_k=5,
    )
    assert len(parents) == 2
    paper_ids = [p["paper_id"] for p in parents]
    assert paper_ids == ["paper1", "paper2"]
    assert parents[0]["parent_id"] == "p1_sec_0"


@patch("retrieval.hybrid.retrieve_bm25")
def test_retrieve_hybrid_pipeline(mock_bm25, mock_parents_db: Path):
    mock_bm25.return_value = [
        {"child_id": "c1", "parent_id": "p1_sec_0", "score": 8.0, "text": "BM25 snippet", "section": "Intro"}
    ]

    mock_client = MagicMock()
    hit_dense = MagicMock()
    hit_dense.id = "uuid-1"
    hit_dense.score = 0.90
    hit_dense.payload = {
        "child_id": "c1",
        "parent_id": "p1_sec_0",
        "paper_id": "paper1",
        "title": "Paper One",
        "year": 2024,
        "section": "1 Introduction",
        "section_type": "body",
        "page_start": 1,
        "page_end": 2,
        "page": 1,
        "is_table": False,
        "text": "Dense snippet",
    }
    mock_client.search.return_value = [hit_dense]

    mock_model = MagicMock()
    mock_model.encode.return_value = [0.1] * 1024

    results, debug_info = retrieve_hybrid(
        query="Test query",
        top_k=5,
        parents_db=mock_parents_db,
        client=mock_client,
        model=mock_model,
        verify_bm25=False,
        debug=True,
    )

    assert len(results) == 1
    assert results[0]["parent_id"] == "p1_sec_0"
    assert "dense_hits" in debug_info
    assert "bm25_hits" in debug_info
    assert "fused_children" in debug_info
