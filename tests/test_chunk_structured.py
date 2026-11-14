"""Unit tests for ingestion/chunk_structured.py."""

import json
from pathlib import Path

import pytest

from unittest.mock import patch

from ingestion.chunk_structured import (
    chunk_all_parsed_papers,
    chunk_parent_into_children,
    get_parents_by_ids,
    init_parents_db,
    process_paper_chunks,
    save_parents_to_db,
    split_section_if_too_large,
)


def test_split_section_if_too_large_normal():
    section = {
        "section_id": "p1_sec_0",
        "heading": "Introduction",
        "section_type": "body",
        "page_start": 1,
        "page_end": 2,
        "text": "This is a brief section text with few words.",
    }
    parents = split_section_if_too_large(
        section=section,
        paper_id="p1",
        title="Test Paper",
        year=2024,
        max_parent_tokens=1500,
    )
    assert len(parents) == 1
    assert parents[0]["parent_id"] == "p1_sec_0"
    assert parents[0]["section"] == "Introduction"
    assert parents[0]["page_start"] == 1
    assert parents[0]["page_end"] == 2
    assert parents[0]["is_table"] is False


def test_split_section_if_too_large_overflow():
    # Create section text with ~600 tokens
    long_text = " ".join([f"word{i}" for i in range(600)])
    section = {
        "section_id": "p1_sec_1",
        "heading": "Very Long Section",
        "section_type": "body",
        "page_start": 3,
        "page_end": 5,
        "text": long_text,
    }
    # Set max_parent_tokens to 250
    parents = split_section_if_too_large(
        section=section,
        paper_id="p1",
        title="Test Paper",
        year=2024,
        max_parent_tokens=250,
    )
    # 600 tokens split by 250 -> 3 parents
    assert len(parents) == 3
    assert parents[0]["parent_id"] == "p1_sec_1_part0"
    assert "Part 1" in parents[0]["section"]
    assert parents[1]["parent_id"] == "p1_sec_1_part1"
    assert "Part 2" in parents[1]["section"]
    assert parents[2]["parent_id"] == "p1_sec_1_part2"
    assert "Part 3" in parents[2]["section"]


def test_chunk_parent_into_children_and_tables():
    parent = {
        "parent_id": "p1_sec_0",
        "paper_id": "p1",
        "title": "Title",
        "year": 2024,
        "section": "Intro",
        "section_type": "body",
        "page_start": 1,
        "page_end": 2,
        "text": " ".join([f"token_{i}" for i in range(700)]),
        "is_table": False,
    }
    children = chunk_parent_into_children(
        parent=parent,
        child_chunk_size=300,
        child_chunk_overlap=30,
    )
    assert len(children) >= 2
    for idx, c in enumerate(children):
        assert c["parent_id"] == "p1_sec_0"
        assert c["paper_id"] == "p1"
        assert c["page_start"] == 1
        assert c["page_end"] == 2
        assert c["is_table"] is False
        assert c["chunk_index"] == idx

    # Test table as parent
    tbl_parent = {
        "parent_id": "p1_tbl_0",
        "paper_id": "p1",
        "title": "Title",
        "year": 2024,
        "section": "Table 1",
        "section_type": "table",
        "page_start": 3,
        "page_end": 3,
        "text": "| A | B |\n|---|---|\n| 1 | 2 |",
        "is_table": True,
    }
    tbl_children = chunk_parent_into_children(tbl_parent)
    assert len(tbl_children) == 1
    assert tbl_children[0]["parent_id"] == "p1_tbl_0"
    assert tbl_children[0]["is_table"] is True
    assert tbl_children[0]["text"] == tbl_parent["text"]


def test_process_paper_chunks_references_toggle():
    parsed_paper = {
        "paper_id": "p_toggle",
        "title": "Toggle Paper",
        "year": 2023,
        "sections": [
            {
                "section_id": "p_sec_body",
                "heading": "Introduction",
                "section_type": "body",
                "page_start": 1,
                "page_end": 1,
                "text": "Body content here.",
            },
            {
                "section_id": "p_sec_refs",
                "heading": "References",
                "section_type": "references",
                "page_start": 2,
                "page_end": 2,
                "text": "[1] Sample reference item.",
            },
        ],
        "tables": [],
    }

    # 1. include_references = False
    parents_no_ref, children_no_ref = process_paper_chunks(
        parsed_paper=parsed_paper,
        include_references=False,
    )
    # Both sections should be saved as parents
    assert len(parents_no_ref) == 2
    # Only body should be indexed as child
    assert len(children_no_ref) == 1
    assert children_no_ref[0]["parent_id"] == "p_sec_body"

    # 2. include_references = True
    parents_with_ref, children_with_ref = process_paper_chunks(
        parsed_paper=parsed_paper,
        include_references=True,
    )
    assert len(parents_with_ref) == 2
    assert len(children_with_ref) == 2


def test_sqlite_parents_store(tmp_path: Path):
    db_path = tmp_path / "test_parents.db"

    parents = [
        {
            "parent_id": "p1_sec_0",
            "paper_id": "p1",
            "title": "Paper 1",
            "year": 2024,
            "section": "Intro",
            "section_type": "body",
            "page_start": 1,
            "page_end": 2,
            "text": "Intro text.",
            "token_count": 2,
            "is_table": False,
        },
        {
            "parent_id": "p1_tbl_0",
            "paper_id": "p1",
            "title": "Paper 1",
            "year": 2024,
            "section": "Table 1",
            "section_type": "table",
            "page_start": 3,
            "page_end": 3,
            "text": "| Col |",
            "token_count": 2,
            "is_table": True,
        },
    ]

    saved = save_parents_to_db(parents, db_path=db_path)
    assert saved == 2

    # Query back
    fetched = get_parents_by_ids(["p1_sec_0", "p1_tbl_0", "nonexistent"], db_path=db_path)
    assert len(fetched) == 2
    assert "p1_sec_0" in fetched
    assert fetched["p1_sec_0"]["title"] == "Paper 1"
    assert fetched["p1_sec_0"]["page_start"] == 1
    assert fetched["p1_sec_0"]["page_end"] == 2
    assert fetched["p1_tbl_0"]["is_table"] is True


@patch("ingestion.chunk_structured.get_tokenizer", return_value=None)
def test_chunk_all_parsed_papers_end_to_end(mock_tok, tmp_path: Path):
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    db_path = tmp_path / "parents.db"

    paper_data = {
        "paper_id": "paper_end2end",
        "title": "End to End Paper",
        "year": 2024,
        "sections": [
            {
                "section_id": "sec_0",
                "heading": "Abstract",
                "section_type": "body",
                "page_start": 1,
                "page_end": 1,
                "text": "This is the abstract.",
            }
        ],
        "tables": [],
    }
    (parsed_dir / "paper_end2end.json").write_text(json.dumps(paper_data), encoding="utf-8")

    parents, children = chunk_all_parsed_papers(
        parsed_dir=parsed_dir,
        parents_db=db_path,
        limit=None,
        save_to_db=True,
    )
    assert len(parents) == 1
    assert len(children) == 1
    assert db_path.exists()
