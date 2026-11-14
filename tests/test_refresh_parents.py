"""Unit tests for ingestion/refresh_parents.py."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ingestion.chunk_structured import save_parents_to_db
from ingestion.refresh_parents import (
    backup_database,
    get_parents_summary,
    refresh_parents_metadata,
    restore_database,
    update_parsed_json_files,
)


@pytest.fixture
def sample_parsed_dir(tmp_path: Path):
    p_dir = tmp_path / "parsed"
    p_dir.mkdir()

    paper = {
        "paper_id": "p_test_refresh",
        "title": "Refresh Test Paper",
        "year": 2024,
        "sections": [
            {
                "section_id": "p_test_refresh_sec_0",
                "heading": "1 Introduction",
                "section_type": "body",
                "page_start": 1,
                "page_end": 1,
                "text": "Introduction text with several words here.",
            },
            {
                "section_id": "p_test_refresh_sec_1",
                "heading": "References",
                "section_type": "references",
                "page_start": 2,
                "page_end": 2,
                "text": "[1] First citation.",
            },
            {
                "section_id": "p_test_refresh_sec_2",
                "heading": "A Overview of System",
                "section_type": "body",  # Initially tagged body before refresh
                "page_start": 3,
                "page_end": 3,
                "text": "Long appendix text with many tokens for testing warning thresholds.",
            },
        ],
        "tables": [
            {
                "table_id": "p_test_refresh_tbl_0",
                "page": 3,
                "page_start": 3,
                "page_end": 3,
                "caption": "Table 1: Appendix table",
                "markdown": "| Col1 | Col2 |\n|---|---|\n| A | B |",
            }
        ],
    }
    (p_dir / "p_test_refresh.json").write_text(json.dumps(paper), encoding="utf-8")
    return p_dir


def test_backup_and_restore_database(tmp_path: Path):
    db_path = tmp_path / "parents.db"
    db_path.write_text("dummy database content", encoding="utf-8")

    backup = backup_database(db_path)
    assert backup is not None
    assert backup.exists()
    assert "parents_" in backup.name
    assert backup.name.endswith(".db.bak")
    assert backup.read_text(encoding="utf-8") == "dummy database content"

    # Modify original
    db_path.write_text("corrupted content", encoding="utf-8")
    assert db_path.read_text(encoding="utf-8") == "corrupted content"

    # Restore
    restore_database(backup, db_path)
    assert db_path.read_text(encoding="utf-8") == "dummy database content"


def test_update_parsed_json_files(sample_parsed_dir: Path):
    has_refs = update_parsed_json_files(sample_parsed_dir)
    assert has_refs.get("p_test_refresh") is True

    # Read back JSON
    updated = json.loads((sample_parsed_dir / "p_test_refresh.json").read_text(encoding="utf-8"))
    sections = updated["sections"]
    assert sections[0]["section_type"] == "body"
    assert sections[1]["section_type"] == "references"
    # Section after references became appendix:
    assert sections[2]["section_type"] == "appendix"

    # Table on page 3 inherited appendix section_type:
    tables = updated["tables"]
    assert tables[0]["section_type"] == "appendix"


@patch("ingestion.chunk_structured.get_tokenizer", return_value=None)
def test_refresh_parents_metadata_success_and_id_invariance(mock_tok, sample_parsed_dir: Path, tmp_path: Path):
    db_path = tmp_path / "parents.db"

    # Create initial parents.db with identical parent_ids
    initial_parents = [
        {
            "parent_id": "p_test_refresh_sec_0",
            "paper_id": "p_test_refresh",
            "title": "Refresh Test Paper",
            "year": 2024,
            "section": "1 Introduction",
            "section_type": "body",
            "page_start": 1,
            "page_end": 1,
            "text": "Introduction text with several words here.",
            "token_count": 6,
            "is_table": False,
        },
        {
            "parent_id": "p_test_refresh_sec_1",
            "paper_id": "p_test_refresh",
            "title": "Refresh Test Paper",
            "year": 2024,
            "section": "References",
            "section_type": "references",
            "page_start": 2,
            "page_end": 2,
            "text": "[1] First citation.",
            "token_count": 3,
            "is_table": False,
        },
        {
            "parent_id": "p_test_refresh_sec_2",
            "paper_id": "p_test_refresh",
            "title": "Refresh Test Paper",
            "year": 2024,
            "section": "A Overview of System",
            "section_type": "body",  # Initially body in DB
            "page_start": 3,
            "page_end": 3,
            "text": "Long appendix text with many tokens for testing warning thresholds.",
            "token_count": 10,
            "is_table": False,
        },
        {
            "parent_id": "p_test_refresh_tbl_0",
            "paper_id": "p_test_refresh",
            "title": "Refresh Test Paper",
            "year": 2024,
            "section": "Table 1: Appendix table",
            "section_type": "table",
            "page_start": 3,
            "page_end": 3,
            "text": "| Col1 | Col2 |\n|---|---|\n| A | B |",
            "token_count": 5,
            "is_table": True,
        },
    ]
    save_parents_to_db(initial_parents, db_path=db_path)

    before_ids, before_summary = get_parents_summary(db_path)
    assert len(before_ids) == 4

    # Run refresh
    success = refresh_parents_metadata(parsed_dir=sample_parsed_dir, parents_db=db_path)
    assert success is True

    # Assert ID invariance: all parent_ids must match exactly
    after_ids, after_summary = get_parents_summary(db_path)
    assert after_ids == before_ids

    # Assert section_type updated in SQLite
    p_summary = after_summary["p_test_refresh"]
    assert p_summary["appendix"] == 2  # 1 section + 1 table
    assert p_summary["references"] == 1
    assert p_summary["body"] == 1


@patch("ingestion.chunk_structured.get_tokenizer", return_value=None)
def test_refresh_parents_metadata_aborts_and_restores_on_mismatch(mock_tok, sample_parsed_dir: Path, tmp_path: Path):
    db_path = tmp_path / "parents.db"

    # Pre-populate DB with an EXTRA parent that won't exist in parsed_dir
    initial_parents = [
        {
            "parent_id": "unmatched_parent_id_123",
            "paper_id": "p_test_refresh",
            "title": "Refresh Test Paper",
            "year": 2024,
            "section": "Ghost Section",
            "section_type": "body",
            "page_start": 1,
            "page_end": 1,
            "text": "Ghost.",
            "token_count": 1,
            "is_table": False,
        },
    ]
    save_parents_to_db(initial_parents, db_path=db_path)

    # Refresh should fail due to ID set mismatch, restore backup, and raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        refresh_parents_metadata(parsed_dir=sample_parsed_dir, parents_db=db_path)

    assert "Parent ID set changed during metadata refresh" in str(exc_info.value)

    # Verify backup was restored
    restored_ids, _ = get_parents_summary(db_path)
    assert "unmatched_parent_id_123" in restored_ids
