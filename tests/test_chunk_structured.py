"""Unit tests for ingestion/chunk_structured.py."""

import json
from pathlib import Path

import pytest

from unittest.mock import patch

from ingestion.chunk_structured import (
    chunk_all_parsed_papers,
    chunk_parent_into_children,
    extract_table_label,
    find_enclosing_section,
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


def test_extract_table_label():
    # Numbers
    assert extract_table_label("Table 1: Detailed comparison", "Fallback") == "Table 1"
    assert extract_table_label("Table 12 - Ablation", "Fallback") == "Table 12"
    assert extract_table_label("Table 3. Metrics", "Fallback") == "Table 3"
    # Roman numerals
    assert extract_table_label("Table I: ROUGE scores", "Fallback") == "Table I"
    assert extract_table_label("Table IV: Model settings", "Fallback") == "Table IV"
    assert extract_table_label("Table XII.", "Fallback") == "Table XII"
    # Strips trailing punctuation
    assert extract_table_label("Table 2:   ", "Fallback") == "Table 2"
    # Non-matches fall back
    assert extract_table_label("Table on accuracy", "Fallback") == "Fallback"
    assert extract_table_label("Overview", "Fallback") == "Fallback"
    assert extract_table_label("", "Fallback") == "Fallback"


def test_table_parent_inherits_section_type_and_keeps_caption():
    parsed_paper = {
        "paper_id": "p_table_test",
        "title": "Table Test Paper",
        "year": 2024,
        "sections": [
            {
                "section_id": "sec_intro",
                "heading": "Introduction",
                "section_type": "body",
                "page_start": 1,
                "page_end": 2,
                "text": "Intro text.",
            },
            {
                "section_id": "sec_refs",
                "heading": "References",
                "section_type": "references",
                "page_start": 3,
                "page_end": 3,
                "text": "[1] Citations.",
            },
            {
                "section_id": "sec_app",
                "heading": "A Additional Results",
                "section_type": "appendix",
                "page_start": 4,
                "page_end": 5,
                "text": "Appendix text.",
            },
        ],
        "tables": [
            {
                "table_id": "p_table_test_tbl_0",
                "page": 1,
                "page_start": 1,
                "page_end": 1,
                "caption": "Table 1: Baseline Results across benchmarks.",
                "markdown": "| Col1 | Col2 |\n|---|---|\n| 1 | 2 |",
            },
            {
                "table_id": "p_table_test_tbl_1",
                "page": 4,
                "page_start": 4,
                "page_end": 4,
                "caption": "Table II: Appendix Detailed Ablations.",
                "markdown": "| Metric | Val |\n|---|---|\n| Acc | 99 |",
            },
        ],
    }

    parents, children = process_paper_chunks(parsed_paper=parsed_paper)

    # Find table parents
    tbl_parents = [p for p in parents if p.get("is_table")]
    assert len(tbl_parents) == 2

    # Table 1 in Introduction -> section_type "body"
    assert tbl_parents[0]["parent_id"] == "p_table_test_tbl_0"
    assert tbl_parents[0]["section"] == "Table 1"  # short label only
    assert tbl_parents[0]["section_type"] == "body"  # inherited from body section
    assert tbl_parents[0]["text"].startswith("Table 1: Baseline Results across benchmarks.")
    assert "| Col1 | Col2 |" in tbl_parents[0]["text"]

    # Table II in Appendix -> section_type "appendix"
    assert tbl_parents[1]["parent_id"] == "p_table_test_tbl_1"
    assert tbl_parents[1]["section"] == "Table II"  # short label only
    assert tbl_parents[1]["section_type"] == "appendix"  # inherited from appendix section
    assert tbl_parents[1]["text"].startswith("Table II: Appendix Detailed Ablations.")
    assert "| Metric | Val |" in tbl_parents[1]["text"]


def test_children_no_mid_word_and_contiguous_spans():
    """Verify child chunking never starts or ends mid-word and is always a contiguous span."""
    parent = {
        "parent_id": "paper_test_sec_0",
        "paper_id": "paper_test",
        "title": "Title of Test",
        "year": 2024,
        "section": "1 Introduction",
        "section_type": "body",
        "page_start": 1,
        "page_end": 2,
        "text": (
            "Large language models often suffer from hallucinations in retrieval augmented generation. "
            "Recent studies by Smith et al. suggest dense vector search mitigates factuality errors. "
            "For example, soft errors in GPU memory can induce hallucinations as noted in Fig. 2. "
            "Our experiments on benchmark datasets demonstrate a 15% reduction in hallucination rates. "
            "Furthermore, structure-aware chunking preserves table context and heading hierarchies. "
            "In conclusion, parent-child retrieval combines granular search precision with broad LLM context."
        ),
        "is_table": False,
    }

    # Set child_chunk_size small enough to trigger multi-chunk sentence packing
    children = chunk_parent_into_children(
        parent=parent,
        child_chunk_size=30,
        tokenizer=None,
    )
    assert len(children) >= 2

    p_text = parent["text"]
    for c in children:
        c_text = c["text"]
        # 1. Contiguous span
        assert c_text in p_text, f"Child text not found in parent text: {c_text}"

        idx = p_text.find(c_text)
        assert idx != -1

        # 2. No start mid-word
        if idx > 0:
            char_before = p_text[idx - 1]
            assert not (char_before.isalnum() and c_text[0].isalnum()), (
                f"Child starts mid-word! Preceding char: {char_before!r}, Child start: {c_text[:10]!r}"
            )

        # 3. No end mid-word
        end_idx = idx + len(c_text)
        if end_idx < len(p_text):
            char_after = p_text[end_idx]
            assert not (c_text[-1].isalnum() and char_after.isalnum()), (
                f"Child ends mid-word! Child end: {c_text[-10:]!r}, Following char: {char_after!r}"
            )


def test_one_sentence_overlap_and_parent_ids_unchanged():
    """Verify 1-sentence overlap between consecutive sentence-packed child chunks and parent ID invariance."""
    parsed_paper = {
        "paper_id": "paper_overlap",
        "title": "Overlap Paper",
        "year": 2024,
        "sections": [
            {
                "section_id": "paper_overlap_sec_0",
                "heading": "Introduction",
                "section_type": "body",
                "page_start": 1,
                "page_end": 1,
                "text": (
                    "Sentence zero is introductory. "
                    "Sentence one describes the problem. "
                    "Sentence two introduces the methodology. "
                    "Sentence three evaluates the results. "
                    "Sentence four concludes the work."
                ),
            }
        ],
        "tables": [
            {
                "table_id": "paper_overlap_tbl_0",
                "page": 2,
                "page_start": 2,
                "page_end": 2,
                "caption": "Table 1: Benchmark Comparisons",
                "markdown": "| System | Accuracy |\n|---|---|\n| Scholar | 98% |",
            }
        ],
    }

    parents, children = process_paper_chunks(
        parsed_paper=parsed_paper,
        child_chunk_size=16,  # Force ~2 sentences per chunk
        tokenizer=None,
    )

    # 1. Verify parent IDs remain unchanged
    parent_ids = [p["parent_id"] for p in parents]
    assert parent_ids == ["paper_overlap_sec_0", "paper_overlap_tbl_0"]

    # 2. Verify section children have one-sentence overlap
    sec_children = [c for c in children if not c["is_table"]]
    assert len(sec_children) >= 2

    for k in range(len(sec_children) - 1):
        curr_text = sec_children[k]["text"]
        next_text = sec_children[k + 1]["text"]
        curr_sentences = [s.strip() for s in curr_text.split(".") if s.strip()]
        next_sentences = [s.strip() for s in next_text.split(".") if s.strip()]
        last_s_curr = curr_sentences[-1]
        first_s_next = next_sentences[0]
        assert last_s_curr == first_s_next, (
            f"Expected 1-sentence overlap: last of chunk {k} was '{last_s_curr}', "
            f"first of chunk {k+1} was '{first_s_next}'"
        )

    # 3. Verify table child is standalone and embeds caption + markdown
    tbl_children = [c for c in children if c["is_table"]]
    assert len(tbl_children) == 1
    assert tbl_children[0]["parent_id"] == "paper_overlap_tbl_0"
    assert tbl_children[0]["child_id"] == "paper_overlap_tbl_0_c0"
    assert "Table 1: Benchmark Comparisons" in tbl_children[0]["text"]
    assert "| System | Accuracy |" in tbl_children[0]["text"]
    assert tbl_children[0]["text"] == parents[1]["text"]
    assert tbl_children[0]["text"] in parents[1]["text"]


def test_overlong_sentence_split_word_boundaries():
    """Verify an over-long sentence is split strictly at word boundaries without breaking words."""
    long_sentence = " ".join([f"word{i:03d}" for i in range(300)])
    parent = {
        "parent_id": "p_long_sec_0",
        "paper_id": "p_long",
        "title": "Long Sentence Paper",
        "year": 2024,
        "section": "Long Section",
        "section_type": "body",
        "page_start": 1,
        "page_end": 1,
        "text": long_sentence,
        "is_table": False,
    }

    children = chunk_parent_into_children(
        parent=parent,
        child_chunk_size=100,
        tokenizer=None,
    )
    assert len(children) >= 3

    p_text = parent["text"]
    for c in children:
        c_text = c["text"]
        assert c_text in p_text
        words = c_text.split()
        for w in words:
            assert w.startswith("word") and len(w) == 7

        idx = p_text.find(c_text)
        if idx > 0:
            assert p_text[idx - 1] == " "
        end_idx = idx + len(c_text)
        if end_idx < len(p_text):
            assert p_text[end_idx] == " "
