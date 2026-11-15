"""Unit tests for retrieval/rerank.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.chunk_structured import save_parents_to_db
from retrieval.rerank import (
    rerank_children,
    retrieve_hybrid_rerank,
    truncate_text_to_tokens,
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
            "text": "Intro text.",
            "token_count": 50,
            "is_table": False,
        },
        {
            "parent_id": "p2_sec_0",
            "paper_id": "paper2",
            "title": "Paper Two",
            "year": 2023,
            "section": "2 Details",
            "section_type": "body",
            "page_start": 4,
            "page_end": 4,
            "text": "Details text.",
            "token_count": 60,
            "is_table": False,
        },
    ]
    save_parents_to_db(parents, db_path=db_path)
    return db_path


def test_truncate_text_to_tokens():
    long_text = "word " * 600
    truncated = truncate_text_to_tokens(long_text, max_tokens=512)
    assert len(truncated.split()) == 512


def test_rerank_children_with_disk_cache(tmp_path: Path):
    cache_file = tmp_path / "rerank_cache.json"

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.85, 0.42]

    children = [
        {"child_id": "c1", "parent_id": "p1", "text": "Child one text"},
        {"child_id": "c2", "parent_id": "p2", "text": "Child two text"},
    ]

    # First call: uncached, invokes mock_model.predict
    results = rerank_children(
        query="What is RAG?",
        children=children,
        model=mock_model,
        cache_path=cache_file,
        top_n=2,
    )

    assert mock_model.predict.called
    assert len(results) == 2
    assert results[0]["child_id"] == "c1"
    assert results[0]["rerank_score"] == 0.85
    assert not results[0]["cached"]

    # Verify cache file was written
    assert cache_file.exists()
    with open(cache_file, "r") as f:
        cache_content = json.load(f)
    assert "What is RAG?:::c1" in cache_content

    # Second call: cached, should NOT call mock_model.predict again
    mock_model.predict.reset_mock()
    results_cached = rerank_children(
        query="What is RAG?",
        children=children,
        model=mock_model,
        cache_path=cache_file,
        top_n=2,
    )

    assert not mock_model.predict.called
    assert results_cached[0]["cached"]
    assert results_cached[0]["rerank_score"] == 0.85


@patch("retrieval.rerank.retrieve_hybrid")
def test_retrieve_hybrid_rerank_pipeline(mock_hybrid, mock_parents_db: Path, tmp_path: Path):
    mock_hybrid.return_value = (
        [],
        {
            "dense_hits": [],
            "bm25_hits": [],
            "fused_children": [
                {
                    "child_id": "c1",
                    "parent_id": "p1_sec_0",
                    "paper_id": "paper1",
                    "title": "Paper One",
                    "text": "Intro text.",
                    "score": 0.03,
                },
                {
                    "child_id": "c2",
                    "parent_id": "p2_sec_0",
                    "paper_id": "paper2",
                    "title": "Paper Two",
                    "text": "Details text.",
                    "score": 0.02,
                },
            ],
            "parent_results": [],
        },
    )

    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.91, 0.45]

    cache_path = tmp_path / "rerank_cache.json"

    results, debug_info = retrieve_hybrid_rerank(
        query="How does hallucination mitigation work?",
        top_k=5,
        parents_db=mock_parents_db,
        reranker_model=mock_reranker,
        rerank_cache_path=cache_path,
        debug=True,
    )

    assert len(results) == 2
    assert results[0]["parent_id"] == "p1_sec_0"
    assert pytest.approx(results[0]["score"]) == 0.91
    assert "reranked_children" in debug_info
