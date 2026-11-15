#!/usr/bin/env python3
"""CLI interface for Scholar RAG.

Usage:
    python cli.py "your question"
    python cli.py "your question" --mode naive
    python cli.py "your question" --mode parent_child
    python cli.py "your question" --mode hybrid
    python cli.py "your question" --mode hybrid_rerank
    python cli.py "your question" --mode hybrid_rerank --debug
"""

import argparse
import sys
from pathlib import Path

from generation.answer import answer_question
from ingestion.chunk_structured import load_config
from retrieval.hybrid import retrieve_hybrid
from retrieval.naive import retrieve
from retrieval.parent_child import retrieve_parent_child
from retrieval.rerank import retrieve_hybrid_rerank


def main() -> None:
    config = load_config()
    cfg_retrieval = config.get("retrieval", {})
    default_mode = cfg_retrieval.get("mode", "parent_child")
    default_qdrant_path = config.get("storage", {}).get("qdrant_path", "./qdrant_data")
    default_parents_db = config.get("storage", {}).get("parents_db", "data/parents.db")

    parser = argparse.ArgumentParser(
        description="Scholar CLI: Ask questions grounded in indexed arXiv research papers."
    )
    parser.add_argument(
        "question",
        type=str,
        help="The question to ask about the indexed research papers.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["naive", "parent_child", "hybrid", "hybrid_rerank"],
        default=default_mode,
        help=f"Retrieval mode: 'naive', 'parent_child', 'hybrid', or 'hybrid_rerank' (default: {default_mode}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=cfg_retrieval.get("top_k", 5),
        help="Number of context chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default=default_qdrant_path,
        help=f"Path to local Qdrant database (default: {default_qdrant_path}).",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Qdrant collection name (default: scholar_papers for naive, scholar_papers_v3 for others).",
    )
    parser.add_argument(
        "--parents-db",
        type=Path,
        default=Path(default_parents_db),
        help=f"Path to SQLite database for parent records (default: {default_parents_db}).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass response disk cache.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-stage diagnostic rankings (child hits, BM25, RRF, reranking, and parents).",
    )
    args = parser.parse_args()

    # Determine collection name
    if args.collection_name:
        collection_name = args.collection_name
    elif args.mode in ["parent_child", "hybrid", "hybrid_rerank"]:
        collection_name = cfg_retrieval.get("collection_v3", cfg_retrieval.get("collection_v2", "scholar_papers_v3"))
    else:
        collection_name = cfg_retrieval.get("collection_naive", "scholar_papers")

    # Step 1: Retrieval
    try:
        if args.mode == "hybrid_rerank":
            contexts, _ = retrieve_hybrid_rerank(
                query=args.question,
                top_k=args.top_k,
                qdrant_path=args.qdrant_path,
                collection_name=collection_name,
                parents_db=args.parents_db,
                debug=args.debug,
            )
        elif args.mode == "hybrid":
            contexts, _ = retrieve_hybrid(
                query=args.question,
                top_k=args.top_k,
                qdrant_path=args.qdrant_path,
                collection_name=collection_name,
                parents_db=args.parents_db,
                debug=args.debug,
            )
        elif args.mode == "parent_child":
            contexts = retrieve_parent_child(
                query=args.question,
                top_k=args.top_k,
                qdrant_path=args.qdrant_path,
                collection_name=collection_name,
                parents_db=args.parents_db,
                debug=args.debug,
            )
        else:
            contexts = retrieve(
                query=args.question,
                top_k=args.top_k,
                qdrant_path=args.qdrant_path,
                collection_name=collection_name,
            )
    except Exception as e:
        if args.mode in ["hybrid", "hybrid_rerank"]:
            build_cmd = "python ingestion/build_index_v2.py && python retrieval/bm25.py --build"
        elif args.mode == "parent_child":
            build_cmd = "python ingestion/build_index_v2.py"
        else:
            build_cmd = "python ingestion/build_index.py"

        print(f"\n[Error during retrieval ({args.mode})]: {e}", file=sys.stderr)
        print(
            f"Please ensure you have built the index first using:\n    {build_cmd}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if not contexts:
        if args.mode in ["hybrid", "hybrid_rerank"]:
            build_cmd = "python ingestion/build_index_v2.py && python retrieval/bm25.py --build"
        elif args.mode == "parent_child":
            build_cmd = "python ingestion/build_index_v2.py"
        else:
            build_cmd = "python ingestion/build_index.py"

        print(f"\nNo relevant contexts found for question: '{args.question}' (mode: {args.mode})")
        print(f"Please check that the index is populated with `{build_cmd}`.")
        sys.exit(0)

    # Step 2: Generation with Grounding and Citations
    try:
        answer, sources = answer_question(
            question=args.question,
            contexts=contexts,
            use_cache=not args.no_cache,
        )
    except Exception as e:
        print(f"\n[Error during generation]: {e}", file=sys.stderr)
        print(
            "Please ensure GEMINI_API_KEY is set in your .env file or environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 3: Output Formatting
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)
    print("\n" + "=" * 60)
    print("SOURCES:")
    print("=" * 60)
    if sources:
        for src in sources:
            print(f"- {src}")
    else:
        print("No sources cited.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
