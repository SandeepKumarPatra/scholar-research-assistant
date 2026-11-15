"""Unit tests for ingestion/build_index_v2.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ingestion.build_index_v2 import build_index_v2, parse_args


@pytest.fixture
def mock_parsed_data(tmp_path: Path):
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    parents_db = tmp_path / "parents.db"

    paper1 = {
        "paper_id": "p1",
        "title": "Paper 1",
        "year": 2024,
        "sections": [
            {
                "section_id": "p1_sec_0",
                "heading": "Introduction",
                "section_type": "body",
                "page_start": 1,
                "page_end": 1,
                "text": "Intro text of paper 1.",
            }
        ],
        "tables": [
            {
                "table_id": "p1_tbl_0",
                "page": 2,
                "page_start": 2,
                "page_end": 2,
                "caption": "Table 1",
                "markdown": "| Col1 | Col2 |\n|---|---|\n| 1 | 2 |",
            }
        ],
    }
    paper2 = {
        "paper_id": "p2",
        "title": "Paper 2",
        "year": 2023,
        "sections": [
            {
                "section_id": "p2_sec_0",
                "heading": "Abstract",
                "section_type": "body",
                "page_start": 1,
                "page_end": 1,
                "text": "Abstract text of paper 2.",
            }
        ],
        "tables": [],
    }

    (parsed_dir / "p1.json").write_text(json.dumps(paper1), encoding="utf-8")
    (parsed_dir / "p2.json").write_text(json.dumps(paper2), encoding="utf-8")

    return parsed_dir, parents_db


def test_build_index_v2_success(mock_parsed_data):
    parsed_dir, parents_db = mock_parsed_data

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    mock_embed = MagicMock()
    mock_embed.encode.side_effect = lambda texts, **kwargs: np.zeros((len(texts), 1024))

    with patch("ingestion.chunk_structured.get_tokenizer", return_value=None):
        num_indexed = build_index_v2(
            parsed_dir=parsed_dir,
            parents_db=parents_db,
            collection_name="scholar_papers_v2",
            client=mock_client,
            embed_model=mock_embed,
            recreate_index=False,
        )

    # Paper 1 has 1 sec + 1 table = 2 children. Paper 2 has 1 sec = 1 child. Total = 3
    assert num_indexed == 3
    assert mock_client.create_collection.called
    assert mock_client.upsert.called
    assert parents_db.exists()

    # Check payload structure on upsert call
    upsert_args, upsert_kwargs = mock_client.upsert.call_args
    points = upsert_kwargs.get("points") or upsert_args[1]
    assert len(points) > 0
    first_payload = points[0].payload
    assert "child_id" in first_payload
    assert "parent_id" in first_payload
    assert "paper_id" in first_payload
    assert "page_start" in first_payload
    assert "page_end" in first_payload


def test_build_index_v2_limit(mock_parsed_data):
    parsed_dir, parents_db = mock_parsed_data

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    mock_embed = MagicMock()
    mock_embed.encode.side_effect = lambda texts, **kwargs: np.zeros((len(texts), 1024))

    with patch("ingestion.chunk_structured.get_tokenizer", return_value=None):
        num_indexed = build_index_v2(
            parsed_dir=parsed_dir,
            parents_db=parents_db,
            collection_name="scholar_papers_v2",
            client=mock_client,
            embed_model=mock_embed,
            limit=1,
            recreate_index=False,
        )

    # Only 1 paper indexed (paper 1 has 2 children: sec + table)
    assert num_indexed == 2


def test_build_index_v2_recreate(mock_parsed_data):
    parsed_dir, parents_db = mock_parsed_data

    mock_col = MagicMock()
    mock_col.name = "scholar_papers_v2"
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = [mock_col]

    mock_embed = MagicMock()
    mock_embed.encode.side_effect = lambda texts, **kwargs: np.zeros((len(texts), 1024))

    with patch("ingestion.chunk_structured.get_tokenizer", return_value=None):
        num_indexed = build_index_v2(
            parsed_dir=parsed_dir,
            parents_db=parents_db,
            collection_name="scholar_papers_v2",
            client=mock_client,
            embed_model=mock_embed,
            recreate_index=True,
        )

    assert num_indexed == 3
    assert mock_client.delete_collection.called
    assert mock_client.create_collection.called


def test_build_index_v2_empty_dir(tmp_path: Path):
    empty_parsed = tmp_path / "empty_parsed"
    empty_parsed.mkdir()
    parents_db = tmp_path / "parents.db"

    mock_client = MagicMock()
    mock_embed = MagicMock()

    with patch("ingestion.chunk_structured.get_tokenizer", return_value=None):
        num_indexed = build_index_v2(
            parsed_dir=empty_parsed,
            parents_db=parents_db,
            client=mock_client,
            embed_model=mock_embed,
        )

    assert num_indexed == 0


def test_build_index_v2_collection_v3_argument(mock_parsed_data):
    parsed_dir, parents_db = mock_parsed_data

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    mock_embed = MagicMock()
    mock_embed.encode.side_effect = lambda texts, **kwargs: np.zeros((len(texts), 1024))

    with patch("ingestion.chunk_structured.get_tokenizer", return_value=None):
        num_indexed = build_index_v2(
            parsed_dir=parsed_dir,
            parents_db=parents_db,
            collection_name="scholar_papers_v3",
            client=mock_client,
            embed_model=mock_embed,
            recreate_index=False,
        )

    assert num_indexed == 3
    # Check that collection scholar_papers_v3 was created and points were upserted to it
    mock_client.create_collection.assert_called_once()
    create_args, create_kwargs = mock_client.create_collection.call_args
    assert create_kwargs.get("collection_name") == "scholar_papers_v3"

    upsert_args, upsert_kwargs = mock_client.upsert.call_args
    assert upsert_kwargs.get("collection_name") == "scholar_papers_v3"


def test_parse_args_collection_flag(monkeypatch):
    test_args = ["prog", "--collection", "scholar_papers_v3", "--limit", "10"]
    monkeypatch.setattr("sys.argv", test_args)
    args = parse_args()
    assert args.collection_name == "scholar_papers_v3"
    assert args.limit == 10
