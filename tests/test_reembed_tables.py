"""Unit tests for ingestion/reembed_tables.py."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ingestion.chunk_structured import save_parents_to_db
from ingestion.reembed_tables import load_table_parents, reembed_tables


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
            "page_end": 1,
            "text": "Intro text.",
            "token_count": 2,
            "is_table": False,
        },
        {
            "parent_id": "p1_tbl_0",
            "paper_id": "p1",
            "title": "Paper One",
            "year": 2024,
            "section": "Table 1",
            "section_type": "body",
            "page_start": 2,
            "page_end": 2,
            "text": "Table 1: Caption\n\n| Col1 | Col2 |\n|---|---|\n| 1 | 2 |",
            "token_count": 10,
            "is_table": True,
        },
        {
            "parent_id": "p2_tbl_0",
            "paper_id": "p2",
            "title": "Paper Two",
            "year": 2023,
            "section": "Table II",
            "section_type": "appendix",
            "page_start": 5,
            "page_end": 5,
            "text": "Table II: Appendix Caption\n\n| Acc |\n|---|\n| 95% |",
            "token_count": 8,
            "is_table": True,
        },
    ]
    save_parents_to_db(parents, db_path=db_path)
    return db_path


def test_load_table_parents(mock_parents_db: Path):
    # Loads only records where is_table == 1
    tables = load_table_parents(parents_db=mock_parents_db)
    assert len(tables) == 2
    assert tables[0]["parent_id"] == "p1_tbl_0"
    assert tables[0]["section"] == "Table 1"
    assert tables[1]["parent_id"] == "p2_tbl_0"
    assert tables[1]["section"] == "Table II"
    assert tables[1]["section_type"] == "appendix"

    # With limit=1 (paper)
    tables_limited = load_table_parents(parents_db=mock_parents_db, limit_papers=1)
    assert len(tables_limited) == 1
    assert tables_limited[0]["paper_id"] == "p1"


def test_reembed_tables_updates_payload_and_vectors(mock_parents_db: Path):
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "scholar_papers_v2"
    mock_client.get_collections.return_value.collections = [mock_col]

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((2, 1024))

    count = reembed_tables(
        parents_db=mock_parents_db,
        collection_name="scholar_papers_v2",
        client=mock_client,
        embed_model=mock_model,
    )
    assert count == 2
    assert mock_client.upsert.called

    # Inspect points passed to upsert
    upsert_call = mock_client.upsert.call_args
    points = upsert_call.kwargs.get("points")
    assert len(points) == 2

    # Check payload of table 1
    p1 = points[0]
    assert p1.payload["section"] == "Table 1"
    assert p1.payload["section_type"] == "body"
    assert "Table 1: Caption" in p1.payload["text"]
    assert p1.payload["is_table"] is True

    # Check payload of table 2
    p2 = points[1]
    assert p2.payload["section"] == "Table II"
    assert p2.payload["section_type"] == "appendix"
    assert "Table II: Appendix Caption" in p2.payload["text"]
    assert p2.payload["is_table"] is True
