#!/usr/bin/env python3
"""BM25 lexical retrieval module for Scholar child chunks.

Builds and queries a BM25 index over the ~300 token child chunk texts in the
scholar_papers_v3 Qdrant collection (or collection specified in config.yaml).
"""

import argparse
import logging
import math
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ingestion.chunk_structured import load_config

# Setup logging
logger = logging.getLogger("scholar.retrieval.bm25")

DEFAULT_QDRANT_PATH = "./qdrant_data"
DEFAULT_COLLECTION_NAME = "scholar_papers_v3"
DEFAULT_BM25_INDEX_PATH = Path("data/bm25_index.pkl")

# Process-level cache for loaded indexes, keyed by (index_path, collection_name)
_CACHED_BM25_INDEXES: Dict[Tuple[str, str], Any] = {}


def reset_bm25_cache() -> None:
    """Clear in-memory BM25 index cache (useful for testing or rebuilding)."""
    global _CACHED_BM25_INDEXES
    _CACHED_BM25_INDEXES.clear()


# Fallback pure-Python BM25Okapi implementation if rank_bm25 is not installed
class BM25OkapiFallback:
    """Pure-Python BM25Okapi implementation matching rank_bm25 API."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self._initialize(corpus)

    def _initialize(self, corpus: List[List[str]]) -> None:
        df: Dict[str, int] = {}
        for doc in corpus:
            frequencies: Dict[str, int] = {}
            for word in doc:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                df[word] = df.get(word, 0) + 1

        for word, freq in df.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[word] = idf

    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for q in query:
            q_idf = self.idf.get(q, 0.0)
            if q_idf <= 0.0:
                continue
            for idx, freq_dict in enumerate(self.doc_freqs):
                freq = freq_dict.get(q, 0)
                if freq > 0:
                    denom = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_len[idx] / self.avgdl))
                    scores[idx] += q_idf * (freq * (self.k1 + 1.0)) / denom
        return scores


try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = BM25OkapiFallback


def tokenize(text: str) -> List[str]:
    """Tokenize text to lowercase words, keeping numbers and hyphenated terms.

    Example:
        'BAAI/bge-reranker-v2-m3 and gpt-3.5-turbo in 2024' ->
        ['baai', 'bge-reranker-v2-m3', 'and', 'gpt-3', '5-turbo', 'in', '2024']
    """
    if not text:
        return []
    return re.findall(r"(?u)\b[a-z0-9]+(?:-[a-z0-9]+)*\b", text.lower())


def get_qdrant_client(qdrant_path: str = DEFAULT_QDRANT_PATH) -> Any:
    """Instantiate a local QdrantClient."""
    from qdrant_client import QdrantClient

    return QdrantClient(path=qdrant_path)


def fetch_all_qdrant_chunks(
    client: Any,
    collection_name: str,
) -> List[Dict[str, Any]]:
    """Scroll and fetch all points and payloads from a Qdrant collection."""
    points: List[Dict[str, Any]] = []
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=250,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for r in records:
            payload = getattr(r, "payload", {}) or {}
            child_id = payload.get("child_id", str(r.id))
            points.append(
                {
                    "point_id": str(r.id),
                    "child_id": child_id,
                    "text": payload.get("text", ""),
                    "payload": payload,
                }
            )

        if next_offset is None or not records:
            break
        offset = next_offset

    # Sort deterministically by child_id
    points.sort(key=lambda x: x["child_id"])
    return points


def build_bm25_index(
    output_path: Path = DEFAULT_BM25_INDEX_PATH,
    qdrant_path: str = DEFAULT_QDRANT_PATH,
    collection_name: Optional[str] = None,
    client: Optional[Any] = None,
) -> Path:
    """Build BM25 index over Qdrant child chunks and save to disk.

    Args:
        output_path: Destination path for the pickled BM25 index.
        qdrant_path: Path to local Qdrant directory.
        collection_name: Name of Qdrant child collection.
        client: Optional pre-initialized QdrantClient.

    Returns:
        Path to the saved index file.
    """
    config = load_config()
    if collection_name is None:
        collection_name = config.get("retrieval", {}).get(
            "collection_v3",
            config.get("retrieval", {}).get("collection_v2", DEFAULT_COLLECTION_NAME),
        )

    if client is None:
        client = get_qdrant_client(qdrant_path=qdrant_path)

    # Check collection existence
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        raise ValueError(
            f"Collection '{collection_name}' not found in Qdrant at {qdrant_path}. "
            "Please run 'python ingestion/build_index_v2.py' first."
        )

    logger.info(f"Fetching child chunks from Qdrant collection '{collection_name}'...")
    chunks = fetch_all_qdrant_chunks(client, collection_name=collection_name)
    if not chunks:
        raise ValueError(f"Collection '{collection_name}' contains 0 points. Cannot build BM25 index.")

    logger.info(f"Tokenizing {len(chunks)} child chunks with hyphen/number preservation...")
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]

    logger.info("Fitting BM25Okapi model...")
    bm25_model = BM25Okapi(tokenized_corpus)

    index_data = {
        "collection_name": collection_name,
        "child_count": len(chunks),
        "child_ids": [c["child_id"] for c in chunks],
        "point_ids": [c["point_id"] for c in chunks],
        "payloads": [c["payload"] for c in chunks],
        "bm25": bm25_model,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(
        f"BM25 index built successfully with {len(chunks)} chunks and saved to '{output_path}'."
    )
    return output_path


def load_bm25_index(
    index_path: Path = DEFAULT_BM25_INDEX_PATH,
    qdrant_path: str = DEFAULT_QDRANT_PATH,
    collection_name: Optional[str] = None,
    client: Optional[Any] = None,
    verify: bool = True,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """Load BM25 index from disk once per process and verify against Qdrant.

    At load time, verifies child count and child IDs match the target Qdrant collection.
    Fails with a clear error message if there is a mismatch.

    Args:
        index_path: Path to serialized BM25 index.
        qdrant_path: Path to local Qdrant directory.
        collection_name: Qdrant collection name.
        client: Optional pre-initialized QdrantClient.
        verify: If True, validates index against Qdrant collection.
        force_reload: If True, bypasses memory cache.

    Returns:
        Index data dictionary.
    """
    global _CACHED_BM25_INDEXES

    config = load_config()
    if collection_name is None:
        collection_name = config.get("retrieval", {}).get(
            "collection_v3",
            config.get("retrieval", {}).get("collection_v2", DEFAULT_COLLECTION_NAME),
        )

    index_path = Path(index_path)
    cache_key = (str(index_path.resolve()), str(collection_name))
    if not force_reload and cache_key in _CACHED_BM25_INDEXES:
        return _CACHED_BM25_INDEXES[cache_key]

    if not index_path.exists():
        raise FileNotFoundError(
            f"BM25 index file not found at '{index_path}'. "
            "Please build the index first using:\n"
            f"    python retrieval/bm25.py --build --collection {collection_name}"
        )

    try:
        with open(index_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load BM25 index from '{index_path}': {e}. "
            "Please rebuild with 'python retrieval/bm25.py --build'."
        )

    if verify:
        if client is None:
            client = get_qdrant_client(qdrant_path=qdrant_path)

        # Check collection presence
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            raise ValueError(
                f"Target collection '{collection_name}' does not exist in Qdrant at {qdrant_path}."
            )

        col_info = client.get_collection(collection_name)
        qdrant_count = getattr(col_info, "points_count", None)

        bm25_count = data.get("child_count", 0)
        bm25_ids = data.get("child_ids", [])

        if qdrant_count is not None and qdrant_count != bm25_count:
            raise ValueError(
                f"BM25 index mismatch with Qdrant collection '{collection_name}': "
                f"BM25 index contains {bm25_count} child chunks, but Qdrant has {qdrant_count}. "
                "Please rebuild the index using:\n"
                f"    python retrieval/bm25.py --build --collection {collection_name}"
            )

        # Verify child ID set matches
        q_chunks = fetch_all_qdrant_chunks(client, collection_name=collection_name)
        qdrant_ids = [c["child_id"] for c in q_chunks]
        if set(qdrant_ids) != set(bm25_ids):
            raise ValueError(
                f"BM25 index child IDs do not match Qdrant collection '{collection_name}'. "
                f"(BM25: {len(bm25_ids)} IDs, Qdrant: {len(qdrant_ids)} IDs). "
                "Please rebuild the index using:\n"
                f"    python retrieval/bm25.py --build --collection {collection_name}"
            )

    _CACHED_BM25_INDEXES[cache_key] = data
    return _CACHED_BM25_INDEXES[cache_key]


def retrieve_bm25(
    query: str,
    top_k: int = 50,
    index_path: Path = DEFAULT_BM25_INDEX_PATH,
    qdrant_path: str = DEFAULT_QDRANT_PATH,
    collection_name: Optional[str] = None,
    client: Optional[Any] = None,
    verify: bool = True,
) -> List[Dict[str, Any]]:
    """Retrieve top-K child chunks using BM25 lexical scoring.

    Args:
        query: Query string.
        top_k: Number of child candidates to return (default: 50).
        index_path: Path to serialized BM25 index.
        qdrant_path: Qdrant database path.
        collection_name: Qdrant child collection name.
        client: Optional pre-initialized QdrantClient.
        verify: Whether to verify against Qdrant at load time.

    Returns:
        List of child chunk dictionaries sorted by BM25 score descending.
    """
    if not query or not query.strip():
        return []

    index_data = load_bm25_index(
        index_path=index_path,
        qdrant_path=qdrant_path,
        collection_name=collection_name,
        client=client,
        verify=verify,
    )

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    bm25 = index_data["bm25"]
    scores = bm25.get_scores(query_tokens)

    payloads = index_data["payloads"]
    child_ids = index_data["child_ids"]

    # Rank documents by score descending
    scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results: List[Dict[str, Any]] = []
    for rank, idx in enumerate(scored_indices[:top_k], 1):
        score = float(scores[idx])
        payload = payloads[idx]
        child_id = child_ids[idx]
        results.append(
            {
                "child_id": child_id,
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
                "bm25_score": score,
                "rank": rank,
            }
        )

    return results


def main() -> None:
    """CLI testing and build entry point for BM25 retrieval."""
    parser = argparse.ArgumentParser(description="BM25 lexical retrieval and index builder.")
    parser.add_argument("query", nargs="?", type=str, default=None, help="Search query")
    parser.add_argument("--build", action="store_true", help="Build BM25 index from Qdrant child collection")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_BM25_INDEX_PATH,
        help=f"Path for BM25 index (default: {DEFAULT_BM25_INDEX_PATH})",
    )
    parser.add_argument("--qdrant-path", type=str, default=DEFAULT_QDRANT_PATH)
    parser.add_argument(
        "--collection",
        "--collection-name",
        dest="collection_name",
        type=str,
        default=None,
        help="Target Qdrant child collection name (default: from config.yaml)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top K child candidates (default: 10)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.build:
        saved_path = build_bm25_index(
            output_path=args.index_path,
            qdrant_path=args.qdrant_path,
            collection_name=args.collection_name,
        )
        print(f"BM25 index successfully saved to: {saved_path}")
        return

    if not args.query:
        print("Please provide a query string or use --build to create the index.")
        return

    results = retrieve_bm25(
        query=args.query,
        top_k=args.top_k,
        index_path=args.index_path,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection_name,
    )

    print(f"\n--- Top {len(results)} BM25 Child Chunks for: '{args.query}' ---")
    for r in results:
        print(
            f"[{r['rank']}] Score: {r['score']:.4f} | Child: {r['child_id']} | "
            f"Parent: {r['parent_id']} | Sec: {r['section']} ({r['section_type']})"
        )
        print(f"    Text: {r['text'][:120]}...\n")


if __name__ == "__main__":
    main()
