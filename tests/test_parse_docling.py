"""Unit tests for ingestion/parse_docling.py."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.parse_docling import (
    classify_section_type,
    extract_page_from_item,
    parse_docling_document,
    print_parsing_stats,
    process_all_pdfs,
    sanitize_filename,
)


class DummyProv:
    def __init__(self, page_no: int):
        self.page_no = page_no


class DummyItem:
    def __init__(self, text: str, label: str, page_no: int = 1):
        self.text = text
        self.label = label
        self.prov = [DummyProv(page_no)]


class DummyTable:
    def __init__(self, md: str, page_no: int = 1, caption: str = ""):
        self._md = md
        self.prov = [DummyProv(page_no)]
        self._caption = caption

    def export_to_markdown(self) -> str:
        return self._md

    def caption_text(self, doc=None) -> str:
        return self._caption


class DummyDoc:
    def __init__(self, items, tables=None):
        self._items = items
        self.tables = tables or []

    def iterate_items(self):
        for item in self._items:
            yield (item, 0)


def test_classify_section_type():
    assert classify_section_type("1. Introduction") == "body"
    assert classify_section_type("Background and Related Work") == "body"
    assert classify_section_type("References") == "references"
    assert classify_section_type("8. References") == "references"
    assert classify_section_type("Bibliography") == "references"
    assert classify_section_type("Works Cited") == "references"
    assert classify_section_type("Literature Cited") == "references"
    # Exact match only: prefix match should NOT match references
    assert classify_section_type("References to Prior Work") == "body"
    assert classify_section_type("Works Cited in Section 4") == "body"
    assert classify_section_type("Appendix A") == "appendix"
    assert classify_section_type("Supplementary Material") == "appendix"
    # Past references flag returns appendix
    assert classify_section_type("A Overview of HalluLens", past_references=True) == "appendix"
    assert classify_section_type("Discussion", past_references=True) == "appendix"


def test_parse_docling_document_appendix_after_references():
    items = [
        DummyItem("1 Introduction", "section_header", page_no=1),
        DummyItem("Intro text.", "paragraph", page_no=1),
        DummyItem("References", "section_header", page_no=2),
        DummyItem("[1] Some Citation.", "paragraph", page_no=2),
        DummyItem("A Overview of HalluLens", "section_header", page_no=3),
        DummyItem("Appendix lettered text.", "paragraph", page_no=3),
        DummyItem("B Additional Proofs", "section_header", page_no=4),
        DummyItem("Proof details.", "paragraph", page_no=4),
    ]
    doc = DummyDoc(items=items, tables=[])
    parsed = parse_docling_document(
        doc=doc,
        paper_id="paper_with_appendix",
        title="Paper With Appendix",
        year=2024,
    )
    sections = parsed["sections"]
    assert len(sections) == 4
    assert sections[0]["heading"] == "1 Introduction"
    assert sections[0]["section_type"] == "body"
    assert sections[1]["heading"] == "References"
    assert sections[1]["section_type"] == "references"
    # Sections after references must be appendix
    assert sections[2]["heading"] == "A Overview of HalluLens"
    assert sections[2]["section_type"] == "appendix"
    assert sections[3]["heading"] == "B Additional Proofs"
    assert sections[3]["section_type"] == "appendix"


def test_sanitize_filename():
    assert sanitize_filename("2411.18583v1") == "2411.18583v1"
    assert sanitize_filename("paper/with:special?chars") == "paper_with_special_chars"


def test_parse_docling_document_structured():
    items = [
        DummyItem("1 Introduction", "section_header", page_no=1),
        DummyItem("This is the introduction paragraph.", "paragraph", page_no=1),
        DummyItem("Continuing on page 2.", "paragraph", page_no=2),
        DummyItem("2 Methodology", "section_header", page_no=2),
        DummyItem("Method details here.", "paragraph", page_no=2),
        DummyItem("References", "section_header", page_no=3),
        DummyItem("[1] Author, A paper.", "paragraph", page_no=3),
    ]
    tables = [
        DummyTable("| Col1 | Col2 |\n|---|---|\n| A | B |", page_no=2, caption="Table 1: Test")
    ]
    doc = DummyDoc(items=items, tables=tables)

    parsed = parse_docling_document(
        doc=doc,
        paper_id="test_paper_1",
        title="Sample Paper Title",
        year=2024,
    )

    assert parsed["paper_id"] == "test_paper_1"
    assert parsed["title"] == "Sample Paper Title"
    assert parsed["year"] == 2024
    assert parsed["has_heading_fallback"] is False

    sections = parsed["sections"]
    assert len(sections) == 3

    # Section 1
    assert sections[0]["heading"] == "1 Introduction"
    assert sections[0]["section_type"] == "body"
    assert sections[0]["page_start"] == 1
    assert sections[0]["page_end"] == 2
    assert "introduction paragraph" in sections[0]["text"]

    # Section 2
    assert sections[1]["heading"] == "2 Methodology"
    assert sections[1]["section_type"] == "body"
    assert sections[1]["page_start"] == 2
    assert sections[1]["page_end"] == 2

    # Section 3: References
    assert sections[2]["heading"] == "References"
    assert sections[2]["section_type"] == "references"

    # Tables
    assert len(parsed["tables"]) == 1
    assert parsed["tables"][0]["page"] == 2
    assert "| Col1 | Col2 |" in parsed["tables"][0]["markdown"]
    assert parsed["tables"][0]["caption"] == "Table 1: Test"


def test_parse_docling_document_no_headings_fallback(caplog):
    # Document with NO headings
    items = [
        DummyItem("First page raw content.", "paragraph", page_no=1),
        DummyItem("More text on first page.", "paragraph", page_no=1),
        DummyItem("Second page content without headings.", "paragraph", page_no=2),
    ]
    doc = DummyDoc(items=items, tables=[])

    with caplog.at_level(logging.WARNING):
        parsed = parse_docling_document(
            doc=doc,
            paper_id="unheaded_paper",
            title="Unheaded Paper",
            year=2023,
        )

    assert parsed["has_heading_fallback"] is True
    assert "No section headings detected in paper 'unheaded_paper'" in caplog.text

    sections = parsed["sections"]
    assert len(sections) == 2
    assert sections[0]["heading"] == "Page 1"
    assert sections[0]["page_start"] == 1
    assert sections[0]["page_end"] == 1
    assert "First page raw content" in sections[0]["text"]

    assert sections[1]["heading"] == "Page 2"
    assert sections[1]["page_start"] == 2
    assert sections[1]["page_end"] == 2


def test_process_all_pdfs_caching_and_limit(tmp_path: Path):
    data_dir = tmp_path / "data"
    pdfs_dir = data_dir / "pdfs"
    parsed_dir = data_dir / "parsed"
    metadata_file = data_dir / "metadata.json"

    pdfs_dir.mkdir(parents=True)
    metadata = [
        {"id": "paper_1", "title": "Paper 1", "year": 2021},
        {"id": "paper_2", "title": "Paper 2", "year": 2022},
        {"id": "paper_3", "title": "Paper 3", "year": 2023},
    ]
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")

    # Create dummy pdf files
    for m in metadata:
        (pdfs_dir / f"{m['id']}.pdf").write_bytes(b"%PDF-dummy")

    mock_converter = MagicMock()
    mock_doc = DummyDoc(items=[DummyItem("Intro", "section_header", 1), DummyItem("Body", "paragraph", 1)])
    mock_res = MagicMock()
    mock_res.document = mock_doc
    mock_converter.convert.return_value = mock_res

    # 1. Run with limit=2
    count = process_all_pdfs(
        data_dir=data_dir,
        pdfs_dir=pdfs_dir,
        parsed_dir=parsed_dir,
        metadata_file=metadata_file,
        limit=2,
        force=False,
        converter=mock_converter,
    )
    assert count == 2
    assert (parsed_dir / "paper_1.json").exists()
    assert (parsed_dir / "paper_2.json").exists()
    assert not (parsed_dir / "paper_3.json").exists()

    # 2. Run again without force: should skip existing files
    count_second = process_all_pdfs(
        data_dir=data_dir,
        pdfs_dir=pdfs_dir,
        parsed_dir=parsed_dir,
        metadata_file=metadata_file,
        limit=None,
        force=False,
        converter=mock_converter,
    )
    # Only paper_3 needed parsing
    assert count_second == 1
    assert (parsed_dir / "paper_3.json").exists()


def test_print_parsing_stats(tmp_path: Path, capsys):
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()

    sample_json = {
        "paper_id": "test_p1",
        "title": "A Very Long Paper Title For Testing Statistics Printout",
        "year": 2024,
        "has_heading_fallback": False,
        "sections": [
            {"heading": "Intro", "section_type": "body", "text": "word " * 100, "token_count": 100},
            {"heading": "References", "section_type": "references", "text": "ref " * 50, "token_count": 50},
        ],
        "tables": [{"markdown": "| a | b |"}],
    }
    (parsed_dir / "test_p1.json").write_text(json.dumps(sample_json), encoding="utf-8")

    print_parsing_stats(parsed_dir=parsed_dir)
    captured = capsys.readouterr().out

    assert "PARSING STATS SUMMARY" in captured
    assert "test_p1" in captured
    assert "Detected" in captured
    assert "TOTALS: 1 papers" in captured
