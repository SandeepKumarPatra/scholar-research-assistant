#!/usr/bin/env python3
"""CLI interface for Scholar Naive RAG.

Usage:
    python cli.py "your question"
"""

import argparse
import sys
from pathlib import Path

from generation.answer import answer_question
from retrieval.naive import retrieve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scholar CLI: Ask questions grounded in indexed arXiv research papers."
    )
    parser.add_argument(
        "question",
        type=str,
        help="The question to ask about the indexed research papers.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of context chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default="./qdrant_data",
        help="Path to local Qdrant database (default: ./qdrant_data).",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="scholar_papers",
        help="Qdrant collection name (default: scholar_papers).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass response disk cache.",
    )
    args = parser.parse_args()

    # Step 1: Dense Retrieval
    try:
        contexts = retrieve(
            query=args.question,
            top_k=args.top_k,
            qdrant_path=args.qdrant_path,
            collection_name=args.collection_name,
        )
    except Exception as e:
        print(f"\n[Error during retrieval]: {e}", file=sys.stderr)
        print(
            "Please ensure you have built the index first using:\n"
            "    python ingestion/build_index.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if not contexts:
        print(f"\nNo relevant contexts found for question: '{args.question}'")
        print("Please check that the index is populated with `python ingestion/build_index.py`.")
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
