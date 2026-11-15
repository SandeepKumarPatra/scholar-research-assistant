#!/usr/bin/env python3
"""Compare retrieval modes across benchmark questions.

Runs retrieval ONLY (zero Gemini calls) across:
1. naive (Phase 1)
2. parent_child (Phase 2)
3. hybrid (Phase 3: Dense + BM25 + RRF)
4. hybrid_rerank (Phase 3: Hybrid + Cross-Encoder)

Outputs a comprehensive comparison markdown table and timing analysis to compare_modes.md.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ingestion.chunk_structured import load_config
from retrieval.hybrid import retrieve_hybrid
from retrieval.naive import retrieve
from retrieval.parent_child import retrieve_parent_child
from retrieval.rerank import retrieve_hybrid_rerank

DEFAULT_QUESTIONS = [
    "What methods reduce hallucination in LLMs?",
    "What is ChainPoll?",
    "How is hallucination measured in benchmarks?",
    "What is the HalluLens benchmark designed to evaluate?",
    "Which papers discuss hallucination in multilingual settings?",
]

MODES = ["naive", "parent_child", "hybrid", "hybrid_rerank"]
DEFAULT_OUTPUT_FILE = Path("compare_modes.md")


def format_source_label(item: Dict[str, Any], mode: str) -> str:
    """Format paper title and page/page range for display."""
    title = item.get("title", "Untitled")
    if mode == "naive":
        page = item.get("page", 1)
        return f"{title}, Page {page}"

    p_start = item.get("page_start", 1)
    p_end = item.get("page_end", p_start)
    if p_start == p_end:
        p_str = f"Page {p_start}"
    else:
        p_str = f"Pages {p_start}-{p_end}"

    tbl = " [Table]" if item.get("is_table") else ""
    return f"{title}, {p_str}{tbl}"


def run_retrieval_for_mode(
    mode: str,
    question: str,
    top_k: int = 5,
    qdrant_path: str = "./qdrant_data",
    parents_db: Path = Path("data/parents.db"),
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Execute retrieval for a specific mode and measure execution time in seconds."""
    if config is None:
        config = load_config()

    retrieval_cfg = config.get("retrieval", {})
    col_v3 = retrieval_cfg.get("collection_v3", "scholar_papers_v3")
    col_naive = retrieval_cfg.get("collection_naive", "scholar_papers")

    start_time = time.perf_counter()

    if mode == "naive":
        results = retrieve(
            query=question,
            top_k=top_k,
            qdrant_path=qdrant_path,
            collection_name=col_naive,
        )
    elif mode == "parent_child":
        results = retrieve_parent_child(
            query=question,
            top_k=top_k,
            qdrant_path=qdrant_path,
            collection_name=col_v3,
            parents_db=parents_db,
            debug=False,
        )
    elif mode == "hybrid":
        results, _ = retrieve_hybrid(
            query=question,
            top_k=top_k,
            qdrant_path=qdrant_path,
            collection_name=col_v3,
            parents_db=parents_db,
            debug=False,
        )
    elif mode == "hybrid_rerank":
        results, _ = retrieve_hybrid_rerank(
            query=question,
            top_k=top_k,
            qdrant_path=qdrant_path,
            collection_name=col_v3,
            parents_db=parents_db,
            debug=False,
        )
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    elapsed_s = time.perf_counter() - start_time
    return results, elapsed_s


def generate_comparison_report(
    eval_data: List[Dict[str, Any]],
    output_path: Path = DEFAULT_OUTPUT_FILE,
) -> str:
    """Generate Markdown report comparing all retrieval modes."""
    lines: List[str] = []
    lines.append("# Retrieval Modes Comparison Report (Phase 1 vs 2 vs 3)\n")
    lines.append(
        "Evaluation benchmark running **retrieval ONLY** (no Gemini calls) across "
        "`naive`, `parent_child`, `hybrid` (Dense + BM25 + RRF), and `hybrid_rerank` "
        "(Cross-Encoder).\n"
    )

    # 1. Timing Summary Table
    lines.append("## 1. Latency Benchmark Summary\n")
    lines.append("| Question | Naive (ms) | Parent-Child (ms) | Hybrid (ms) | Hybrid+Rerank (ms) |")
    lines.append("|:---|---:|---:|---:|---:|")

    mode_keys = ["naive", "parent_child", "hybrid", "hybrid_rerank"]
    mode_totals: Dict[str, float] = {m: 0.0 for m in mode_keys}

    for item in eval_data:
        q_text = item["question"]
        t_naive = item["modes"]["naive"]["elapsed_ms"]
        t_pc = item["modes"]["parent_child"]["elapsed_ms"]
        t_hy = item["modes"]["hybrid"]["elapsed_ms"]
        t_rr = item["modes"]["hybrid_rerank"]["elapsed_ms"]

        for m in mode_keys:
            mode_totals[m] += item["modes"][m]["elapsed_ms"]

        lines.append(f"| {q_text} | {t_naive:.1f} | {t_pc:.1f} | {t_hy:.1f} | {t_rr:.1f} |")

    n_q = len(eval_data) if eval_data else 1
    avg_row = [f"{mode_totals[m] / n_q:.1f}" for m in mode_keys]
    lines.append(f"| **Average** | **{avg_row[0]}** | **{avg_row[1]}** | **{avg_row[2]}** | **{avg_row[3]}** |\n")

    # 2. Detailed Per-Question Results Table
    lines.append("## 2. Top-5 Retrieved Sources per Mode\n")

    for idx, item in enumerate(eval_data, 1):
        q_text = item["question"]
        lines.append(f"### Question {idx}: {q_text}\n")

        lines.append("| Rank | Naive (Phase 1) | Parent-Child (Phase 2) | Hybrid (Phase 3) | Hybrid + Rerank (Phase 3) |")
        lines.append("|:---:|:---|:---|:---|:---|")

        # Gather top 5 per mode
        top_k_rows = 5
        for r in range(top_k_rows):
            row_cols = [str(r + 1)]
            for m in mode_keys:
                m_hits = item["modes"][m]["results"]
                if r < len(m_hits):
                    hit = m_hits[r]
                    label = format_source_label(hit, m)
                    sec = hit.get("section", "")
                    sec_str = f" (*{sec}*)" if sec and sec != "Untitled" else ""
                    score = hit.get("score", 0.0)
                    row_cols.append(f"[{label}]{sec_str} `({score:.4f})`")
                else:
                    row_cols.append("-")
            lines.append("| " + " | ".join(row_cols) + " |")
        lines.append("\n---\n")

    content = "\n".join(lines)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return content


def main() -> None:
    """CLI runner for comparing retrieval modes."""
    parser = argparse.ArgumentParser(description="Compare retrieval modes across questions.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Destination markdown file (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top contexts per mode (default: 5)",
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default="./qdrant_data",
        help="Local Qdrant database directory",
    )
    parser.add_argument(
        "--parents-db",
        type=Path,
        default=Path("data/parents.db"),
        help="Path to SQLite parents database",
    )
    args = parser.parse_args()

    config = load_config()
    questions = DEFAULT_QUESTIONS

    print(f"Comparing {len(MODES)} retrieval modes across {len(questions)} test questions...")
    eval_data: List[Dict[str, Any]] = []

    for q_idx, q in enumerate(questions, 1):
        print(f"\n[{q_idx}/{len(questions)}] Question: '{q}'")
        q_record = {"question": q, "modes": {}}

        for mode in MODES:
            print(f"  -> Running mode: '{mode}'...", end="", flush=True)
            try:
                results, elapsed = run_retrieval_for_mode(
                    mode=mode,
                    question=q,
                    top_k=args.top_k,
                    qdrant_path=args.qdrant_path,
                    parents_db=args.parents_db,
                    config=config,
                )
                elapsed_ms = elapsed * 1000.0
                print(f" done ({len(results)} contexts, {elapsed_ms:.1f} ms)")
                q_record["modes"][mode] = {
                    "results": results,
                    "elapsed_ms": elapsed_ms,
                }
            except Exception as e:
                print(f" FAILED: {e}")
                q_record["modes"][mode] = {
                    "results": [],
                    "elapsed_ms": 0.0,
                    "error": str(e),
                }

        eval_data.append(q_record)

    report_content = generate_comparison_report(eval_data, output_path=args.output)
    print(f"\nComparison report generated and saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
