"""Unit tests for scripts/compare_modes.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.compare_modes import (
    format_source_label,
    generate_comparison_report,
    run_retrieval_for_mode,
)


def test_format_source_label():
    # Naive mode
    naive_item = {"title": "Paper A", "page": 3}
    assert format_source_label(naive_item, "naive") == "Paper A, Page 3"

    # Single-page parent
    p1 = {"title": "Paper B", "page_start": 4, "page_end": 4, "is_table": False}
    assert format_source_label(p1, "parent_child") == "Paper B, Page 4"

    # Multi-page parent table
    p2 = {"title": "Paper C", "page_start": 5, "page_end": 7, "is_table": True}
    assert format_source_label(p2, "hybrid") == "Paper C, Pages 5-7 [Table]"


def test_generate_comparison_report(tmp_path: Path):
    out_file = tmp_path / "test_report.md"
    eval_data = [
        {
            "question": "What is ChainPoll?",
            "modes": {
                "naive": {
                    "results": [{"title": "Paper 1", "page": 1, "score": 0.8}],
                    "elapsed_ms": 12.5,
                },
                "parent_child": {
                    "results": [{"title": "Paper 1", "page_start": 1, "page_end": 2, "section": "Intro", "score": 0.9}],
                    "elapsed_ms": 15.0,
                },
                "hybrid": {
                    "results": [{"title": "Paper 1", "page_start": 1, "page_end": 2, "section": "Intro", "score": 0.03}],
                    "elapsed_ms": 18.2,
                },
                "hybrid_rerank": {
                    "results": [{"title": "Paper 1", "page_start": 1, "page_end": 2, "section": "Intro", "score": 0.95}],
                    "elapsed_ms": 45.1,
                },
            },
        }
    ]

    content = generate_comparison_report(eval_data, output_path=out_file)
    assert out_file.exists()
    assert "Retrieval Modes Comparison Report" in content
    assert "Latency Benchmark Summary" in content
    assert "What is ChainPoll?" in content
    assert "Hybrid + Rerank" in content


@patch("scripts.compare_modes.retrieve")
@patch("scripts.compare_modes.retrieve_parent_child")
@patch("scripts.compare_modes.retrieve_hybrid")
@patch("scripts.compare_modes.retrieve_hybrid_rerank")
def test_run_retrieval_for_mode(mock_rerank, mock_hybrid, mock_pc, mock_naive):
    mock_naive.return_value = [{"title": "Paper 1", "page": 1, "score": 0.8}]
    mock_pc.return_value = [{"title": "Paper 1", "page_start": 1, "page_end": 2, "score": 0.9}]
    mock_hybrid.return_value = ([{"title": "Paper 1", "page_start": 1, "page_end": 2, "score": 0.03}], {})
    mock_rerank.return_value = ([{"title": "Paper 1", "page_start": 1, "page_end": 2, "score": 0.95}], {})

    # Naive
    res, t = run_retrieval_for_mode("naive", "Test question")
    assert len(res) == 1
    assert t >= 0.0
    assert mock_naive.called

    # Parent-child
    res, t = run_retrieval_for_mode("parent_child", "Test question")
    assert len(res) == 1
    assert mock_pc.called

    # Hybrid
    res, t = run_retrieval_for_mode("hybrid", "Test question")
    assert len(res) == 1
    assert mock_hybrid.called

    # Hybrid Rerank
    res, t = run_retrieval_for_mode("hybrid_rerank", "Test question")
    assert len(res) == 1
    assert mock_rerank.called
