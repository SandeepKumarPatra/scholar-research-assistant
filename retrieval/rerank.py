#!/usr/bin/env python3
"""Cross-encoder reranking module for Scholar child chunks.

Reranks candidate child chunks from hybrid search using a cross-encoder model
(BAAI/bge-reranker-v2-m3 or configured model), truncates child text to 512 tokens,
shows a tqdm progress bar, caches scores on disk keyed by (query, child_id), and
merges to parent sections with diversity and query-time filtering.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ingestion.chunk_structured import (
    DEFAULT_PARENTS_DB,
    load_config,
)
from retrieval.hybrid import (
    merge_children_to_parents,
    print_hybrid_debug,
    retrieve_hybrid,
)
from retrieval.parent_child import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_QDRANT_PATH,
    DEFAULT_TOP_K,
)

logger = logging.getLogger("scholar.retrieval.rerank")

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANK_CACHE_PATH = Path(".cache/rerank_cache.json")
DEFAULT_RERANK_TOP_N = 20
MAX_TOKENS = 512

# Process-level singleton cache for loaded model
_CACHED_RERANK_MODEL: Optional[Any] = None
_IN_MEMORY_RERANK_CACHE: Optional[Dict[str, float]] = None


def get_reranker_model(model_name: str = DEFAULT_RERANKER_MODEL) -> Any:
    """Load and cache CrossEncoder model once per process."""
    global _CACHED_RERANK_MODEL
    if _CACHED_RERANK_MODEL is not None:
        return _CACHED_RERANK_MODEL

    from sentence_transformers import CrossEncoder

    logger.info(f"Loading CrossEncoder reranker model: '{model_name}'...")
    try:
        _CACHED_RERANK_MODEL = CrossEncoder(model_name, max_length=MAX_TOKENS, local_files_only=True)
    except Exception:
        _CACHED_RERANK_MODEL = CrossEncoder(model_name, max_length=MAX_TOKENS)

    return _CACHED_RERANK_MODEL


def load_rerank_cache(cache_path: Path = DEFAULT_RERANK_CACHE_PATH) -> Dict[str, float]:
    """Load reranker score disk cache."""
    global _IN_MEMORY_RERANK_CACHE
    if _IN_MEMORY_RERANK_CACHE is not None:
        return _IN_MEMORY_RERANK_CACHE

    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                _IN_MEMORY_RERANK_CACHE = json.load(f)
                return _IN_MEMORY_RERANK_CACHE
        except Exception as e:
            logger.warning(f"Could not load rerank cache from '{cache_path}': {e}. Using empty cache.")

    _IN_MEMORY_RERANK_CACHE = {}
    return _IN_MEMORY_RERANK_CACHE


def save_rerank_cache(cache_data: Dict[str, float], cache_path: Path = DEFAULT_RERANK_CACHE_PATH) -> None:
    """Save reranker score cache to disk."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save rerank cache to '{cache_path}': {e}")


def make_cache_key(query: str, child_id: str) -> str:
    """Generate deterministic cache key for (query, child_id)."""
    return f"{query.strip()}:::{child_id.strip()}"


def truncate_text_to_tokens(text: str, tokenizer: Optional[Any] = None, max_tokens: int = MAX_TOKENS) -> str:
    """Truncate text to max_tokens using model tokenizer if available, or whitespace fallback."""
    if not text:
        return ""

    if tokenizer is not None:
        try:
            tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens, add_special_tokens=False)
            return tokenizer.decode(tokens)
        except Exception:
            pass

    # Word-level fallback
    words = text.split()
    if len(words) > max_tokens:
        return " ".join(words[:max_tokens])
    return text


def rerank_children(
    query: str,
    children: List[Dict[str, Any]],
    model_name: str = DEFAULT_RERANKER_MODEL,
    model: Optional[Any] = None,
    cache_path: Path = DEFAULT_RERANK_CACHE_PATH,
    top_n: int = DEFAULT_RERANK_TOP_N,
) -> List[Dict[str, Any]]:
    """Rerank candidate children by child text using CrossEncoder and score caching.

    Args:
        query: Query string.
        children: List of candidate child chunk dicts.
        model_name: CrossEncoder model identifier.
        model: Optional pre-loaded CrossEncoder instance.
        cache_path: Path to disk cache JSON.
        top_n: Number of candidate children to rerank.

    Returns:
        List of reranked children sorted by cross-encoder score descending.
    """
    if not children or not query.strip():
        return []

    candidates = children[:top_n]
    cache = load_rerank_cache(cache_path=cache_path)

    # 1. Identify which candidate pairs need inference vs cache hits
    pairs_to_predict: List[Tuple[str, str]] = []
    indices_to_predict: List[int] = []

    tokenizer = None
    if model is None:
        model = get_reranker_model(model_name=model_name)
    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer

    reranked: List[Dict[str, Any]] = []
    for idx, child in enumerate(candidates):
        c_copy = dict(child)
        child_id = c_copy["child_id"]
        ckey = make_cache_key(query, child_id)

        if ckey in cache:
            c_copy["rerank_score"] = float(cache[ckey])
            c_copy["score"] = float(cache[ckey])
            c_copy["cached"] = True
            reranked.append(c_copy)
        else:
            truncated_text = truncate_text_to_tokens(c_copy.get("text", ""), tokenizer=tokenizer, max_tokens=MAX_TOKENS)
            pairs_to_predict.append((query, truncated_text))
            indices_to_predict.append(idx)
            c_copy["cached"] = False
            reranked.append(c_copy)

    # 2. Run model predictions with tqdm progress bar if any pairs are uncached
    if pairs_to_predict:
        logger.info(f"Reranking {len(pairs_to_predict)} uncached child candidates with {model_name}...")
        try:
            # Predict scores with tqdm progress
            scores = model.predict(
                pairs_to_predict,
                show_progress_bar=True,
                batch_size=16,
            )
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                scores = list(scores)
        except Exception as e:
            logger.error(f"Error during CrossEncoder predict: {e}")
            raise

        # Update items and disk cache
        for pred_idx, cand_idx in enumerate(indices_to_predict):
            score_val = float(scores[pred_idx])
            target_child = reranked[cand_idx]
            target_child["rerank_score"] = score_val
            target_child["score"] = score_val

            ckey = make_cache_key(query, target_child["child_id"])
            cache[ckey] = score_val

        save_rerank_cache(cache, cache_path=cache_path)

    # 3. Sort children by rerank score descending
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    for rank, child in enumerate(reranked, 1):
        child["rerank_rank"] = rank

    return reranked


def retrieve_hybrid_rerank(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    dense_candidates: Optional[int] = None,
    bm25_candidates: Optional[int] = None,
    rrf_k: Optional[int] = None,
    rerank_top_n: Optional[int] = None,
    max_parents_per_paper: Optional[int] = None,
    qdrant_path: str = DEFAULT_QDRANT_PATH,
    collection_name: Optional[str] = None,
    parents_db: Path = DEFAULT_PARENTS_DB,
    model_name: Optional[str] = None,
    reranker_model_name: Optional[str] = None,
    rerank_cache_path: Path = DEFAULT_RERANK_CACHE_PATH,
    include_appendix: Optional[bool] = None,
    include_references: Optional[bool] = None,
    client: Optional[Any] = None,
    embed_model: Optional[Any] = None,
    reranker_model: Optional[Any] = None,
    verify_bm25: bool = True,
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute full Phase 3 Hybrid Rerank retrieval pipeline.

    1. Dense + BM25 Candidate Retrieval.
    2. Reciprocal Rank Fusion (k=60), keeping top 30 children.
    3. Cross-Encoder reranking of top rerank_top_n (default 20) children by child text.
    4. Merging best child rerank score per parent with section filtering & diversity cap.

    Returns:
        Tuple of (results_parents, debug_info_dict).
    """
    if not query or not query.strip():
        return [], {}

    config = load_config()
    retrieval_cfg = config.get("retrieval", {})
    models_cfg = config.get("models", {})
    chunk_cfg = config.get("chunking", {})

    if rerank_top_n is None:
        rerank_top_n = retrieval_cfg.get("rerank_top_n", DEFAULT_RERANK_TOP_N)
    if reranker_model_name is None:
        reranker_model_name = models_cfg.get("reranker", DEFAULT_RERANKER_MODEL)
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

    # 1. Run Hybrid Candidate Retrieval (Dense + BM25 + RRF)
    # We retrieve the top 30 children from RRF
    _, hybrid_debug = retrieve_hybrid(
        query=query,
        top_k=top_k,
        dense_candidates=dense_candidates,
        bm25_candidates=bm25_candidates,
        rrf_k=rrf_k,
        top_n_children=30,
        max_parents_per_paper=max_parents_per_paper,
        qdrant_path=qdrant_path,
        collection_name=collection_name,
        parents_db=parents_db,
        model_name=model_name or models_cfg.get("embedding", "BAAI/bge-m3"),
        client=client,
        model=embed_model,
        include_appendix=include_appendix,
        include_references=include_references,
        verify_bm25=verify_bm25,
        debug=False,
    )

    fused_children = hybrid_debug.get("fused_children", [])

    # 2. Cross-Encoder Rerank top N children
    reranked_children = rerank_children(
        query=query,
        children=fused_children,
        model_name=reranker_model_name,
        model=reranker_model,
        cache_path=rerank_cache_path,
        top_n=rerank_top_n,
    )

    # 3. Merge Reranked Children to Parents
    parent_results = merge_children_to_parents(
        children=reranked_children,
        parents_db=Path(parents_db),
        include_appendix=include_appendix,
        include_references=include_references,
        max_parents_per_paper=max_parents_per_paper,
        score_key="rerank_score",
        top_k=top_k,
    )

    debug_info = {
        "dense_hits": hybrid_debug.get("dense_hits", []),
        "bm25_hits": hybrid_debug.get("bm25_hits", []),
        "fused_children": fused_children,
        "reranked_children": reranked_children,
        "parent_results": parent_results,
    }

    if debug:
        print_hybrid_rerank_debug(debug_info)

    logger.info(
        f"Hybrid Rerank retrieval finished for '{query}': {len(parent_results)} parents returned "
        f"(reranked top {len(reranked_children)} children)."
    )

    return parent_results, debug_info


def print_hybrid_rerank_debug(debug_info: Dict[str, Any]) -> None:
    """Print full multi-stage diagnostic report for hybrid reranking."""
    # Print Stages 1-3
    print_hybrid_debug(debug_info)

    # Stage 4: Cross-Encoder Reranked Children
    reranked_children = debug_info.get("reranked_children", [])[:10]
    print(f"--- STAGE 4: Top {len(reranked_children)} Cross-Encoder Reranked Children ---")
    for r, h in enumerate(reranked_children, 1):
        cached_tag = " [CACHED]" if h.get("cached") else ""
        sec_str = h.get("section", "")
        title_str = h.get("title", "")
        print(
            f"[{r:2d}] Rerank Score: {h['rerank_score']:+.4f}{cached_tag} | "
            f"Child: {h['child_id']} | Parent: {h.get('parent_id', '')} | Sec: {sec_str}"
        )
        print(f"     Title: {title_str} | RRF Prev Rank: #{h.get('rank', 'N/A')}")
        print(f"     Text: {h['text'][:90]}...\n")

    # Stage 5: Final Merged Parents
    parents = debug_info.get("parent_results", [])
    print(f"--- STAGE 5: Top {len(parents)} Final Merged Parents (Cross-Encoder) ---")
    for r, p in enumerate(parents, 1):
        p_range = f"Page {p['page_start']}" if p["page_start"] == p["page_end"] else f"Pages {p['page_start']}-{p['page_end']}"
        tbl_tag = " [TABLE]" if p.get("is_table") else ""
        print(
            f"[{r:2d}] Score: {p['score']:+.4f} | Source: {p['title']} ({p_range}){tbl_tag} | "
            f"Sec: {p['section']} (ID: {p['paper_id']})"
        )
    print("=" * 70 + "\n")


def main() -> None:
    """CLI testing helper for hybrid rerank retrieval."""
    parser = argparse.ArgumentParser(description="Hybrid Rerank (Dense + BM25 + CrossEncoder) retrieval test.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rerank-top-n", type=int, default=DEFAULT_RERANK_TOP_N)
    parser.add_argument("--model-name", type=str, default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--debug", action="store_true", help="Print per-stage diagnostic rankings")
    args = parser.parse_args()

    results, _ = retrieve_hybrid_rerank(
        query=args.query,
        top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,
        reranker_model_name=args.model_name,
        debug=args.debug,
    )

    if not args.debug:
        print(f"\n--- Retrieved {len(results)} parent contexts for: '{args.query}' ---")
        for idx, r in enumerate(results, 1):
            p_range = f"Page {r['page_start']}" if r["page_start"] == r["page_end"] else f"Pages {r['page_start']}-{r['page_end']}"
            tbl_tag = " [TABLE]" if r.get("is_table") else ""
            print(f"\n[{idx}] Score: {r['score']:+.4f} | Source: {r['title']} ({p_range}){tbl_tag}")
            print(f"    Section: {r['section']} | Paper ID: {r['paper_id']}")
            print(f"    Snippet: {r['text'][:180].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
