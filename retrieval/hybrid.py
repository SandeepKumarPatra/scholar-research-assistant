#!/usr/bin/env python3
"""Hybrid retrieval module combining dense semantic search and BM25 lexical search.

Pipeline:
1. Fetches top 50 dense candidates and top 50 BM25 candidates.
2. Fuses candidate children using Reciprocal Rank Fusion (RRF, k=60).
3. Retains the top 30 fused children.
4. Merges children to parent sections/tables (taking best child score per parent).
5. Applies query-time appendix/references filtering and max_parents_per_paper cap.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ingestion.chunk_structured import (
    DEFAULT_PARENTS_DB,
    get_parents_by_ids,
    load_config,
)
from retrieval.bm25 import (
    DEFAULT_BM25_INDEX_PATH,
    retrieve_bm25,
)
from retrieval.parent_child import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_QDRANT_PATH,
    DEFAULT_TOP_K,
    get_embedding_model,
    get_qdrant_client,
)

logger = logging.getLogger("scholar.retrieval.hybrid")


def compute_rrf(
    dense_hits: List[Dict[str, Any]],
    bm25_hits: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_n_children: int = 30,
) -> List[Dict[str, Any]]:
    """Fuse dense and BM25 child rankings using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(d) = sum(1.0 / (k + rank_m(d))) for m in [dense, bm25]
        where rank is 1-based (rank=1 for the highest scoring chunk).

    Args:
        dense_hits: List of child hit dicts from dense retrieval.
        bm25_hits: List of child hit dicts from BM25 retrieval.
        rrf_k: RRF smoothing constant (default: 60).
        top_n_children: Number of fused children to retain (default: 30).

    Returns:
        List of fused child hit dicts sorted by RRF score descending.
    """
    fused_map: Dict[str, Dict[str, Any]] = {}

    # 1. Process dense hits
    for rank, hit in enumerate(dense_hits, 1):
        child_id = hit["child_id"]
        if child_id not in fused_map:
            fused_map[child_id] = {
                "child_id": child_id,
                "parent_id": hit.get("parent_id", ""),
                "paper_id": hit.get("paper_id", ""),
                "title": hit.get("title", ""),
                "year": hit.get("year", 0),
                "section": hit.get("section", ""),
                "section_type": hit.get("section_type", "body"),
                "page_start": hit.get("page_start", 1),
                "page_end": hit.get("page_end", 1),
                "page": hit.get("page", 1),
                "is_table": hit.get("is_table", False),
                "text": hit.get("text", ""),
                "dense_score": hit.get("score", 0.0),
                "dense_rank": rank,
                "bm25_score": None,
                "bm25_rank": None,
                "rrf_score": 0.0,
            }
        fused_map[child_id]["dense_score"] = hit.get("score", 0.0)
        fused_map[child_id]["dense_rank"] = rank
        fused_map[child_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    # 2. Process BM25 hits
    for rank, hit in enumerate(bm25_hits, 1):
        child_id = hit["child_id"]
        if child_id not in fused_map:
            fused_map[child_id] = {
                "child_id": child_id,
                "parent_id": hit.get("parent_id", ""),
                "paper_id": hit.get("paper_id", ""),
                "title": hit.get("title", ""),
                "year": hit.get("year", 0),
                "section": hit.get("section", ""),
                "section_type": hit.get("section_type", "body"),
                "page_start": hit.get("page_start", 1),
                "page_end": hit.get("page_end", 1),
                "page": hit.get("page", 1),
                "is_table": hit.get("is_table", False),
                "text": hit.get("text", ""),
                "dense_score": None,
                "dense_rank": None,
                "bm25_score": hit.get("score", 0.0),
                "bm25_rank": rank,
                "rrf_score": 0.0,
            }
        else:
            fused_map[child_id]["bm25_score"] = hit.get("score", 0.0)
            fused_map[child_id]["bm25_rank"] = rank

        fused_map[child_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    # 3. Sort children by RRF score descending
    sorted_children = sorted(fused_map.values(), key=lambda x: x["rrf_score"], reverse=True)

    for rank, child in enumerate(sorted_children[:top_n_children], 1):
        child["rank"] = rank
        child["score"] = child["rrf_score"]

    return sorted_children[:top_n_children]


def merge_children_to_parents(
    children: List[Dict[str, Any]],
    parents_db: Path = DEFAULT_PARENTS_DB,
    include_appendix: bool = False,
    include_references: bool = False,
    max_parents_per_paper: Optional[int] = None,
    score_key: str = "rrf_score",
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """Merge child hits into unique parent sections using the best child score per parent.

    Applies query-time section filtering (appendix/references) and diversity cap
    (max_parents_per_paper).

    Args:
        children: Ranked list of child chunks.
        parents_db: Path to SQLite database containing parent records.
        include_appendix: Whether to retain appendix sections.
        include_references: Whether to retain references sections.
        max_parents_per_paper: Maximum parents to allow from a single paper (None = no cap).
        score_key: Key in child dict to use for scoring (default: rrf_score).
        top_k: Number of parents to return.

    Returns:
        List of deduplicated, filtered parent dicts sorted by score descending.
    """
    if not children:
        return []

    # 1. Track highest child score per parent ID
    parent_best_score: Dict[str, float] = {}
    parent_best_child: Dict[str, str] = {}

    for c in children:
        p_id = c.get("parent_id")
        if not p_id:
            continue
        c_score = float(c.get(score_key, c.get("score", 0.0)))
        if p_id not in parent_best_score or c_score > parent_best_score[p_id]:
            parent_best_score[p_id] = c_score
            parent_best_child[p_id] = c.get("child_id", "")

    # 2. Fetch parent records from SQLite
    parents_map = get_parents_by_ids(
        parent_ids=list(parent_best_score.keys()),
        db_path=Path(parents_db),
    )

    # 3. Filter and assemble valid parents
    candidates: List[Dict[str, Any]] = []
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

        candidates.append(
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
                "best_child_id": parent_best_child.get(parent_id, ""),
            }
        )

    # 4. Sort parents by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # 5. Apply max_parents_per_paper cap
    results: List[Dict[str, Any]] = []
    paper_counts: Dict[str, int] = {}

    for cand in candidates:
        pid = cand["paper_id"]
        if max_parents_per_paper is not None and max_parents_per_paper > 0:
            if paper_counts.get(pid, 0) >= max_parents_per_paper:
                continue

        results.append(cand)
        paper_counts[pid] = paper_counts.get(pid, 0) + 1

        if len(results) >= top_k:
            break

    return results


def retrieve_hybrid(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    dense_candidates: Optional[int] = None,
    bm25_candidates: Optional[int] = None,
    rrf_k: Optional[int] = None,
    top_n_children: int = 30,
    max_parents_per_paper: Optional[int] = None,
    qdrant_path: str = DEFAULT_QDRANT_PATH,
    collection_name: Optional[str] = None,
    parents_db: Path = DEFAULT_PARENTS_DB,
    bm25_index_path: Path = DEFAULT_BM25_INDEX_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    client: Optional[Any] = None,
    model: Optional[Any] = None,
    include_appendix: Optional[bool] = None,
    include_references: Optional[bool] = None,
    verify_bm25: bool = True,
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Retrieve top-K parent sections via dense + BM25 Reciprocal Rank Fusion.

    Args:
        query: Query string.
        top_k: Number of parent contexts to return (default: 5).
        dense_candidates: Number of dense children to retrieve (default from config: 50).
        bm25_candidates: Number of BM25 children to retrieve (default from config: 50).
        rrf_k: RRF constant k (default from config: 60).
        top_n_children: Number of children to keep after RRF fusion (default: 30).
        max_parents_per_paper: Max parents per paper (default from config: None).
        qdrant_path: Qdrant database directory.
        collection_name: Qdrant child collection name.
        parents_db: Path to SQLite parents database.
        bm25_index_path: Path to serialized BM25 index.
        model_name: SentenceTransformers model name.
        client: Optional pre-initialized QdrantClient.
        model: Optional pre-initialized SentenceTransformer.
        include_appendix: Query-time appendix toggle.
        include_references: Query-time references toggle.
        verify_bm25: Whether to verify BM25 against Qdrant.
        debug: If True, prints per-stage top 10 diagnostic rankings.

    Returns:
        Tuple of (results_parents, debug_info_dict).
    """
    if not query or not query.strip():
        return [], {}

    # Load configuration defaults
    config = load_config()
    retrieval_cfg = config.get("retrieval", {})
    chunk_cfg = config.get("chunking", {})

    if collection_name is None:
        collection_name = retrieval_cfg.get(
            "collection_v3", retrieval_cfg.get("collection_v2", DEFAULT_COLLECTION_NAME)
        )
    if dense_candidates is None:
        dense_candidates = retrieval_cfg.get("dense_candidates", 50)
    if bm25_candidates is None:
        bm25_candidates = retrieval_cfg.get("bm25_candidates", 50)
    if rrf_k is None:
        rrf_k = retrieval_cfg.get("rrf_k", 60)
    if max_parents_per_paper is None:
        max_parents_per_paper = retrieval_cfg.get("max_parents_per_paper", None)

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

    # -------------------------------------------------------------
    # Stage 1: Dense Retrieval
    # -------------------------------------------------------------
    query_vector = model.encode(query, normalize_embeddings=True)
    if hasattr(query_vector, "tolist"):
        query_vector = query_vector.tolist()

    if hasattr(client, "search"):
        dense_points = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=dense_candidates,
            with_payload=True,
        )
    else:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=dense_candidates,
            with_payload=True,
        )
        dense_points = response.points

    dense_hits: List[Dict[str, Any]] = []
    for rank, p in enumerate(dense_points, 1):
        payload = getattr(p, "payload", {}) or {}
        score = float(p.score) if hasattr(p, "score") and p.score is not None else 0.0
        dense_hits.append(
            {
                "child_id": payload.get("child_id", str(p.id)),
                "parent_id": payload.get("parent_id", ""),
                "paper_id": payload.get("paper_id", ""),
                "title": payload.get("title", ""),
                "year": payload.get("year", 0),
                "section": payload.get("section", ""),
                "section_type": payload.get("section_type", "body"),
                "page_start": payload.get("page_start", 1),
                "page_end": payload.get("page_end", 1),
                "page": payload.get("page", 1),
                "is_table": payload.get("is_table", False),
                "text": payload.get("text", ""),
                "score": score,
                "rank": rank,
            }
        )

    # -------------------------------------------------------------
    # Stage 2: BM25 Retrieval
    # -------------------------------------------------------------
    bm25_hits = retrieve_bm25(
        query=query,
        top_k=bm25_candidates,
        index_path=bm25_index_path,
        qdrant_path=qdrant_path,
        collection_name=collection_name,
        client=client,
        verify=verify_bm25,
    )

    # -------------------------------------------------------------
    # Stage 3: Reciprocal Rank Fusion (RRF)
    # -------------------------------------------------------------
    fused_children = compute_rrf(
        dense_hits=dense_hits,
        bm25_hits=bm25_hits,
        rrf_k=rrf_k,
        top_n_children=top_n_children,
    )

    # -------------------------------------------------------------
    # Stage 4: Merge to Parents
    # -------------------------------------------------------------
    parent_results = merge_children_to_parents(
        children=fused_children,
        parents_db=Path(parents_db),
        include_appendix=include_appendix,
        include_references=include_references,
        max_parents_per_paper=max_parents_per_paper,
        score_key="rrf_score",
        top_k=top_k,
    )

    debug_info = {
        "dense_hits": dense_hits,
        "bm25_hits": bm25_hits,
        "fused_children": fused_children,
        "parent_results": parent_results,
    }

    if debug:
        print_hybrid_debug(debug_info)

    logger.info(
        f"Hybrid retrieval finished for '{query}': {len(parent_results)} parents returned "
        f"(from {len(dense_hits)} dense + {len(bm25_hits)} BM25 -> {len(fused_children)} fused children)."
    )

    return parent_results, debug_info


def print_hybrid_debug(debug_info: Dict[str, Any]) -> None:
    """Print per-stage top 10 diagnostic rankings for hybrid retrieval."""
    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL DEBUG REPORT")
    print("=" * 70)

    # Stage 1: Dense
    dense_hits = debug_info.get("dense_hits", [])[:10]
    print(f"\n--- STAGE 1: Top {len(dense_hits)} Dense Child Candidates ---")
    for r, h in enumerate(dense_hits, 1):
        print(
            f"[{r:2d}] Dense Score: {h['score']:.4f} | Child ID: {h['child_id']} | "
            f"Parent: {h['parent_id']} | Sec: {h['section']}"
        )
        print(f"     Text: {h['text'][:90]}...")

    # Stage 2: BM25
    bm25_hits = debug_info.get("bm25_hits", [])[:10]
    print(f"\n--- STAGE 2: Top {len(bm25_hits)} BM25 Child Candidates ---")
    for r, h in enumerate(bm25_hits, 1):
        print(
            f"[{r:2d}] BM25 Score: {h['score']:.4f} | Child ID: {h['child_id']} | "
            f"Parent: {h['parent_id']} | Sec: {h['section']}"
        )
        print(f"     Text: {h['text'][:90]}...")

    # Stage 3: RRF Fused Children
    fused_children = debug_info.get("fused_children", [])[:10]
    print(f"\n--- STAGE 3: Top {len(fused_children)} Fused Children (RRF) ---")
    for r, h in enumerate(fused_children, 1):
        d_rank_str = f"#{h['dense_rank']}" if h.get("dense_rank") else "None"
        b_rank_str = f"#{h['bm25_rank']}" if h.get("bm25_rank") else "None"
        score_val = h.get("rrf_score", h.get("score", 0.0))
        sec_val = h.get("section", "")
        title_val = h.get("title", "")
        print(
            f"[{r:2d}] RRF Score: {score_val:.6f} (Dense: {d_rank_str}, BM25: {b_rank_str}) | "
            f"Child: {h['child_id']} | Parent: {h['parent_id']}"
        )
        print(f"     Section: {sec_val} | Title: {title_val}")

    # Stage 4: Merged Parents
    parents = debug_info.get("parent_results", [])
    print(f"\n--- STAGE 4: Top {len(parents)} Merged Parents (Post-Filter) ---")
    for r, p in enumerate(parents, 1):
        p_range = f"Page {p['page_start']}" if p["page_start"] == p["page_end"] else f"Pages {p['page_start']}-{p['page_end']}"
        print(
            f"[{r:2d}] Score: {p['score']:.6f} | Source: {p['title']} ({p_range}) | "
            f"Sec: {p['section']} (ID: {p['paper_id']})"
        )
    print("=" * 70 + "\n")


def main() -> None:
    """CLI testing helper for hybrid retrieval."""
    parser = argparse.ArgumentParser(description="Hybrid Dense + BM25 retrieval test.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--dense-candidates", type=int, default=50)
    parser.add_argument("--bm25-candidates", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--max-parents-per-paper", type=int, default=None)
    parser.add_argument("--qdrant-path", type=str, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--collection-name", type=str, default=None)
    parser.add_argument("--parents-db", type=Path, default=DEFAULT_PARENTS_DB)
    parser.add_argument("--bm25-index-path", type=Path, default=DEFAULT_BM25_INDEX_PATH)
    parser.add_argument("--debug", action="store_true", help="Print per-stage diagnostic rankings")
    args = parser.parse_args()

    results, _ = retrieve_hybrid(
        query=args.query,
        top_k=args.top_k,
        dense_candidates=args.dense_candidates,
        bm25_candidates=args.bm25_candidates,
        rrf_k=args.rrf_k,
        max_parents_per_paper=args.max_parents_per_paper,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection_name,
        parents_db=args.parents_db,
        bm25_index_path=args.bm25_index_path,
        debug=args.debug,
    )

    if not args.debug:
        print(f"\n--- Retrieved {len(results)} parent contexts for: '{args.query}' ---")
        for idx, r in enumerate(results, 1):
            p_range = f"Page {r['page_start']}" if r["page_start"] == r["page_end"] else f"Pages {r['page_start']}-{r['page_end']}"
            tbl_tag = " [TABLE]" if r.get("is_table") else ""
            print(f"\n[{idx}] Score: {r['score']:.6f} | Source: {r['title']} ({p_range}){tbl_tag}")
            print(f"    Section: {r['section']} | Paper ID: {r['paper_id']}")
            print(f"    Snippet: {r['text'][:180].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
