#!/usr/bin/env python3
"""Parent-child dense retrieval module for scholar papers.

Performs vector search over child chunks (~300 tokens) in 'scholar_papers_v2',
resolves and deduplicates matching parent sections/tables from data/parents.db,
and returns rich parent context with page ranges and retrieval scores.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestion.chunk_structured import (
    DEFAULT_PARENTS_DB,
    get_parents_by_ids,
    load_config,
)

logger = logging.getLogger("scholar.retrieval.parent_child")

DEFAULT_COLLECTION_NAME = "scholar_papers_v2"
DEFAULT_QDRANT_PATH = "./qdrant_data"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_TOP_K = 5

# Process-level model cache to avoid re-loading per question
_CACHED_EMBED_MODEL: Optional[Any] = None


def get_qdrant_client(qdrant_path: str = DEFAULT_QDRANT_PATH) -> Any:
    """Instantiate a local QdrantClient pointing to file-mode database."""
    from qdrant_client import QdrantClient

    return QdrantClient(path=qdrant_path)


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Load and cache SentenceTransformer model once per process."""
    global _CACHED_EMBED_MODEL
    if _CACHED_EMBED_MODEL is not None:
        return _CACHED_EMBED_MODEL

    from sentence_transformers import SentenceTransformer

    try:
        _CACHED_EMBED_MODEL = SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        _CACHED_EMBED_MODEL = SentenceTransformer(model_name)
    return _CACHED_EMBED_MODEL


def retrieve_parent_child(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_multiplier: int = 3,
    qdrant_path: str = DEFAULT_QDRANT_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    parents_db: Path = DEFAULT_PARENTS_DB,
    model_name: str = DEFAULT_MODEL_NAME,
    client: Optional[Any] = None,
    model: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top-K parent sections by searching child chunks.

    Args:
        query: User question string.
        top_k: Number of deduplicated parent contexts to return (default: 5).
        candidate_multiplier: Ratio of child candidates to fetch (default: 3x top_k).
        qdrant_path: Local Qdrant path.
        collection_name: Qdrant collection name (default: scholar_papers_v2).
        parents_db: Path to SQLite database containing parent records.
        model_name: SentenceTransformers model identifier.
        client: Optional pre-initialized QdrantClient.
        model: Optional pre-initialized SentenceTransformer model.

    Returns:
        List of dicts containing parent metadata, full text, score, and page range.
    """
    if not query or not query.strip():
        return []

    if client is None:
        client = get_qdrant_client(qdrant_path=qdrant_path)

    if model is None:
        model = get_embedding_model(model_name=model_name)

    # 1. Embed query
    query_vector = model.encode(query, normalize_embeddings=True)
    if hasattr(query_vector, "tolist"):
        query_vector = query_vector.tolist()

    # 2. Check Qdrant collection
    try:
        existing_collections = [c.name for c in client.get_collections().collections]
        if collection_name not in existing_collections:
            logger.warning(
                f"Collection '{collection_name}' not found in Qdrant at {qdrant_path}. "
                "Please run ingestion/build_index_v2.py first."
            )
            return []

        candidate_limit = max(top_k * candidate_multiplier, 15)

        if hasattr(client, "search"):
            hits = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=candidate_limit,
                with_payload=True,
            )
        else:
            response = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=candidate_limit,
                with_payload=True,
            )
            hits = response.points

    except Exception as e:
        logger.error(f"Error querying Qdrant collection '{collection_name}': {e}")
        raise

    if not hits:
        logger.info(f"No child hits returned for query: '{query}'")
        return []

    # 3. Collect parent IDs and track highest child score per parent
    parent_best_score: Dict[str, float] = {}
    parent_first_seen: Dict[str, int] = {}

    for idx, hit in enumerate(hits):
        payload = hit.payload or {}
        parent_id = payload.get("parent_id")
        if not parent_id:
            continue

        score = float(hit.score) if hasattr(hit, "score") and hit.score is not None else 0.0

        if parent_id not in parent_best_score:
            parent_best_score[parent_id] = score
            parent_first_seen[parent_id] = idx
        else:
            if score > parent_best_score[parent_id]:
                parent_best_score[parent_id] = score

    # 4. Lookup parent records from SQLite
    parents_map = get_parents_by_ids(
        parent_ids=list(parent_best_score.keys()),
        db_path=Path(parents_db),
    )

    # 5. Build and rank deduplicated parent contexts
    ranked_parents: List[Dict[str, Any]] = []
    for parent_id, best_score in parent_best_score.items():
        p_record = parents_map.get(parent_id)
        if not p_record:
            logger.warning(f"Parent record '{parent_id}' not found in {parents_db}")
            continue

        p_start = p_record.get("page_start", 1)
        p_end = p_record.get("page_end", p_start)

        ranked_parents.append(
            {
                "parent_id": p_record["parent_id"],
                "paper_id": p_record["paper_id"],
                "title": p_record["title"],
                "year": p_record["year"],
                "section": p_record["section"],
                "section_type": p_record.get("section_type", "body"),
                "page_start": p_start,
                "page_end": p_end,
                "page": p_start,  # For compatibility with downstream naive format
                "text": p_record["text"],
                "token_count": p_record.get("token_count", 0),
                "is_table": p_record.get("is_table", False),
                "score": best_score,
            }
        )

    # Sort primarily by score descending
    ranked_parents.sort(key=lambda x: x["score"], reverse=True)
    results = ranked_parents[:top_k]

    # Log retrieval results with scores (per project rules)
    logger.info(f"Retrieved {len(results)} parent contexts for: '{query}'")
    for idx, r in enumerate(results, 1):
        if r["page_start"] == r["page_end"]:
            p_range = f"Page {r['page_start']}"
        else:
            p_range = f"Pages {r['page_start']}-{r['page_end']}"

        logger.info(
            f"[{idx}] Score: {r['score']:.4f} | Source: {r['title']} ({p_range}) | "
            f"Sec: {r['section']} (ID: {r['paper_id']})"
        )

    return results


def main() -> None:
    """CLI testing helper for parent-child retrieval."""
    parser = argparse.ArgumentParser(description="Parent-child dense retrieval test.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top K results (default: 5)")
    parser.add_argument("--qdrant-path", type=str, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--collection-name", type=str, default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--parents-db", type=Path, default=DEFAULT_PARENTS_DB)
    args = parser.parse_args()

    results = retrieve_parent_child(
        query=args.query,
        top_k=args.top_k,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection_name,
        parents_db=args.parents_db,
    )

    print(f"\n--- Retrieved {len(results)} parent contexts for: '{args.query}' ---")
    for idx, r in enumerate(results, 1):
        if r["page_start"] == r["page_end"]:
            p_str = f"Page {r['page_start']}"
        else:
            p_str = f"Pages {r['page_start']}-{r['page_end']}"

        tbl_tag = " [TABLE]" if r.get("is_table") else ""
        print(f"\n[{idx}] Score: {r['score']:.4f} | Source: {r['title']} ({p_str}){tbl_tag}")
        print(f"    Section: {r['section']} | Year: {r['year']} | ID: {r['paper_id']}")
        snippet = r["text"][:200].replace("\n", " ")
        print(f"    Snippet: {snippet}...")


if __name__ == "__main__":
    main()
