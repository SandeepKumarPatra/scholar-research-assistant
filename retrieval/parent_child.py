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

DEFAULT_COLLECTION_NAME_V2 = "scholar_papers_v2"
DEFAULT_COLLECTION_NAME_V3 = "scholar_papers_v3"
DEFAULT_COLLECTION_NAME = "scholar_papers_v3"
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
    candidate_multiplier: int = 4,
    max_candidate_multiplier: int = 10,
    qdrant_path: str = DEFAULT_QDRANT_PATH,
    collection_name: Optional[str] = None,
    parents_db: Path = DEFAULT_PARENTS_DB,
    model_name: str = DEFAULT_MODEL_NAME,
    client: Optional[Any] = None,
    model: Optional[Any] = None,
    include_appendix: Optional[bool] = None,
    include_references: Optional[bool] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Retrieve top-K parent sections by searching child chunks.

    Applies include_appendix and include_references filtering at query time.
    Expands the candidate pool iteratively until top_k unique non-filtered
    parents remain (capped at top_k * 10).

    Args:
        query: User question string.
        top_k: Number of deduplicated parent contexts to return (default: 5).
        candidate_multiplier: Initial child candidates multiplier (default: 4x top_k).
        max_candidate_multiplier: Maximum candidate multiplier cap (default: 10x top_k).
        qdrant_path: Local Qdrant path.
        collection_name: Qdrant collection name (default: scholar_papers_v2).
        parents_db: Path to SQLite database containing parent records.
        model_name: SentenceTransformers model identifier.
        client: Optional pre-initialized QdrantClient.
        model: Optional pre-initialized SentenceTransformer model.
        include_appendix: Whether to include appendix parents (defaults to config.yaml).
        include_references: Whether to include references parents (defaults to config.yaml).
        debug: If True, prints top 10 child hits before parent merging.

    Returns:
        List of dicts containing parent metadata, full text, score, and page range.
    """
    if not query or not query.strip():
        return []

    # Load configuration toggles if not explicitly passed
    config = load_config()
    chunk_cfg = config.get("chunking", {})
    retrieval_cfg = config.get("retrieval", {})

    if collection_name is None:
        collection_name = retrieval_cfg.get(
            "collection_v3", retrieval_cfg.get("collection_v2", DEFAULT_COLLECTION_NAME)
        )

    if include_appendix is None:
        include_appendix = retrieval_cfg.get(
            "include_appendix", chunk_cfg.get("include_appendix", False)
        )
    if include_references is None:
        include_references = retrieval_cfg.get(
            "include_references", chunk_cfg.get("include_references", False)
        )

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
    except Exception as e:
        logger.error(f"Error checking Qdrant collections: {e}")
        raise

    # 3. Iteratively expand candidate pool until top_k non-filtered parents remain
    max_candidates = top_k * max_candidate_multiplier
    current_multiplier = candidate_multiplier
    debug_printed = False

    valid_parents: List[Dict[str, Any]] = []

    while True:
        candidate_limit = min(top_k * current_multiplier, max_candidates)

        try:
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

        # 4. Debug output: print top 10 child hits before parent merging
        if debug and not debug_printed:
            top_hits = hits[:10]
            debug_pids = list(
                dict.fromkeys(
                    h.payload.get("parent_id")
                    for h in top_hits
                    if getattr(h, "payload", None) and h.payload.get("parent_id")
                )
            )
            debug_parents_map = (
                get_parents_by_ids(debug_pids, db_path=Path(parents_db))
                if debug_pids
                else {}
            )

            print(f"\n--- DEBUG: Top {len(top_hits)} Child Hits (Before Parent Merging) ---")
            for rank, hit in enumerate(top_hits, 1):
                payload = getattr(hit, "payload", {}) or {}
                child_id = payload.get("child_id", str(getattr(hit, "id", "")))
                parent_id = payload.get("parent_id", "")
                p_info = debug_parents_map.get(parent_id, {})
                sec_type = p_info.get("section_type", "unknown")
                heading = p_info.get("section", "unknown")
                raw_score = (
                    hit.score
                    if hasattr(hit, "score") and hit.score is not None
                    else 0.0
                )
                text_snip = payload.get("text", "")[:100].replace("\n", " ")
                print(f"[{rank}] Child ID: {child_id} | Parent ID: {parent_id}")
                print(
                    f"    Parent Heading: {heading} | Section Type: {sec_type} | Raw Score: {raw_score}"
                )
                print(f"    Text: {text_snip}...")
            print("-" * 60 + "\n")
            debug_printed = True

        # 5. Collect parent IDs and track highest child score per parent
        parent_best_score: Dict[str, float] = {}
        for hit in hits:
            payload = getattr(hit, "payload", {}) or {}
            parent_id = payload.get("parent_id")
            if not parent_id:
                continue

            score = (
                float(hit.score)
                if hasattr(hit, "score") and hit.score is not None
                else 0.0
            )
            if parent_id not in parent_best_score or score > parent_best_score[parent_id]:
                parent_best_score[parent_id] = score

        # 6. Lookup parent records from SQLite
        parents_map = get_parents_by_ids(
            parent_ids=list(parent_best_score.keys()),
            db_path=Path(parents_db),
        )

        # 7. Filter parents based on query-time section_type criteria
        current_valid: List[Dict[str, Any]] = []
        for parent_id, best_score in parent_best_score.items():
            p_record = parents_map.get(parent_id)
            if not p_record:
                continue

            sec_type = p_record.get("section_type", "body")
            if sec_type == "appendix" and not include_appendix:
                continue
            if sec_type == "references" and not include_references:
                continue

            p_start = p_record.get("page_start", 1)
            p_end = p_record.get("page_end", p_start)
            current_valid.append(
                {
                    "parent_id": p_record["parent_id"],
                    "paper_id": p_record["paper_id"],
                    "title": p_record["title"],
                    "year": p_record["year"],
                    "section": p_record["section"],
                    "section_type": sec_type,
                    "page_start": p_start,
                    "page_end": p_end,
                    "page": p_start,
                    "text": p_record["text"],
                    "token_count": p_record.get("token_count", 0),
                    "is_table": p_record.get("is_table", False),
                    "score": best_score,
                }
            )

        valid_parents = current_valid

        # Stop condition: enough unique parents, or reached cap, or exhausted hits
        if (
            len(valid_parents) >= top_k
            or candidate_limit >= max_candidates
            or len(hits) < candidate_limit
        ):
            break

        current_multiplier += 2

    # Sort primarily by score descending
    valid_parents.sort(key=lambda x: x["score"], reverse=True)
    results = valid_parents[:top_k]

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
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help=f"Qdrant collection name (default: from config.yaml or {DEFAULT_COLLECTION_NAME}).",
    )
    parser.add_argument("--parents-db", type=Path, default=DEFAULT_PARENTS_DB)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print top 10 child hits before parent merging.",
    )
    parser.add_argument(
        "--include-appendix",
        action="store_true",
        default=None,
        help="Force include appendix sections.",
    )
    parser.add_argument(
        "--include-references",
        action="store_true",
        default=None,
        help="Force include references sections.",
    )
    args = parser.parse_args()

    results = retrieve_parent_child(
        query=args.query,
        top_k=args.top_k,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection_name,
        parents_db=args.parents_db,
        include_appendix=args.include_appendix,
        include_references=args.include_references,
        debug=args.debug,
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
