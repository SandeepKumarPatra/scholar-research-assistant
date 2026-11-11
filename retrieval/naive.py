#!/usr/bin/env python3
"""Naive dense retrieval module for scholar papers.

Performs top-K dense vector search using BAAI/bge-m3 embeddings against
the local file-backed Qdrant collection at ./qdrant_data.
"""

import argparse
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("scholar.retrieval")

DEFAULT_COLLECTION_NAME = "scholar_papers"
DEFAULT_QDRANT_PATH = "./qdrant_data"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_TOP_K = 5


def get_qdrant_client(qdrant_path: str = DEFAULT_QDRANT_PATH):
    """Instantiate a local QdrantClient pointing to the file-mode database."""
    from qdrant_client import QdrantClient

    return QdrantClient(path=qdrant_path)


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    """Load the SentenceTransformer model (preferring local cache)."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    qdrant_path: str = DEFAULT_QDRANT_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    client: Optional[Any] = None,
    model: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top-K relevant chunks for a user query using dense vector similarity.

    Args:
        query: User question or search query string.
        top_k: Number of chunks to retrieve (default: 5).
        qdrant_path: Directory path for local Qdrant database.
        collection_name: Qdrant collection name.
        model_name: SentenceTransformer model name.
        client: Optional pre-initialized QdrantClient.
        model: Optional pre-initialized SentenceTransformer model.

    Returns:
        List of dicts containing chunk metadata, text, and similarity score.
    """
    if not query or not query.strip():
        return []

    if client is None:
        client = get_qdrant_client(qdrant_path=qdrant_path)

    if model is None:
        model = get_embedding_model(model_name=model_name)

    # Generate query embedding
    query_vector = model.encode(query, normalize_embeddings=True)
    if hasattr(query_vector, "tolist"):
        query_vector = query_vector.tolist()

    # Search Qdrant collection
    results: List[Dict[str, Any]] = []
    try:
        # Check if collection exists
        existing_collections = [c.name for c in client.get_collections().collections]
        if collection_name not in existing_collections:
            logger.warning(
                f"Collection '{collection_name}' not found in Qdrant at {qdrant_path}. "
                "Please run ingestion/build_index.py first."
            )
            return []

        # Use client.search or client.query_points
        if hasattr(client, "search"):
            hits = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )
        else:
            response = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            hits = response.points

        for hit in hits:
            payload = hit.payload or {}
            paper_id = payload.get("paper_id") or payload.get("id", "Unknown")
            title = payload.get("title", "Untitled")
            year = payload.get("year", 0)
            page = payload.get("page", 1)
            text = payload.get("text", "")
            score = float(hit.score) if hasattr(hit, "score") and hit.score is not None else 0.0

            results.append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "year": year,
                    "page": page,
                    "text": text,
                    "score": score,
                }
            )
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise

    return results


def main() -> None:
    """CLI testing helper for naive retrieval."""
    parser = argparse.ArgumentParser(description="Dense top-K retrieval test.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top K results (default: 5)")
    parser.add_argument("--qdrant-path", type=str, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--collection-name", type=str, default=DEFAULT_COLLECTION_NAME)
    args = parser.parse_args()

    results = retrieve(
        query=args.query,
        top_k=args.top_k,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection_name,
    )

    print(f"\n--- Retrieved {len(results)} chunks for: '{args.query}' ---")
    for idx, r in enumerate(results, 1):
        print(f"\n[{idx}] Score: {r['score']:.4f} | Source: {r['title']} (p. {r['page']})")
        print(f"    ID: {r['paper_id']} | Year: {r['year']}")
        print(f"    Snippet: {r['text'][:150]}...")


if __name__ == "__main__":
    main()
