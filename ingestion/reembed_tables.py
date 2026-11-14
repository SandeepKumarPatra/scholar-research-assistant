#!/usr/bin/env python3
"""Re-embed table child chunks into Qdrant collection scholar_papers_v2.

Reads table parents from data/parents.db (which include the caption in text and
short label "Table N" as section name), computes dense vector embeddings using
BAAI/bge-m3, and updates vectors and payloads in Qdrant.

Usage:
    python ingestion/reembed_tables.py --limit 5
    python ingestion/reembed_tables.py
"""

import argparse
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import uuid

from tqdm import tqdm

from ingestion.chunk_structured import DEFAULT_PARENTS_DB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scholar.reembed_tables")

DEFAULT_COLLECTION_NAME = "scholar_papers_v2"
DEFAULT_QDRANT_PATH = "./qdrant_data"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Load SentenceTransformer embedding model."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def get_qdrant_client(qdrant_path: str = DEFAULT_QDRANT_PATH) -> Any:
    """Instantiate a local QdrantClient."""
    from qdrant_client import QdrantClient

    Path(qdrant_path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=qdrant_path)


def load_table_parents(
    parents_db: Path = DEFAULT_PARENTS_DB,
    limit_papers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Query table parent records from SQLite parents.db."""
    if not parents_db.exists():
        logger.warning(f"Parents DB not found at: {parents_db}")
        return []

    conn = sqlite3.connect(str(parents_db))
    conn.row_factory = sqlite3.Row

    if limit_papers is not None and limit_papers > 0:
        # Fetch distinct paper_ids up to limit
        cursor = conn.execute(
            "SELECT DISTINCT paper_id FROM parents WHERE is_table = 1 ORDER BY paper_id LIMIT ?",
            (limit_papers,),
        )
        allowed_papers = set(row[0] for row in cursor.fetchall())
        if not allowed_papers:
            conn.close()
            return []

        placeholders = ",".join(["?"] * len(allowed_papers))
        query = f"""
            SELECT parent_id, paper_id, title, year, section, section_type,
                   page_start, page_end, text, token_count, is_table
            FROM parents
            WHERE is_table = 1 AND paper_id IN ({placeholders})
            ORDER BY paper_id, page_start
        """
        cursor = conn.execute(query, list(allowed_papers))
    else:
        query = """
            SELECT parent_id, paper_id, title, year, section, section_type,
                   page_start, page_end, text, token_count, is_table
            FROM parents
            WHERE is_table = 1
            ORDER BY paper_id, page_start
        """
        cursor = conn.execute(query)

    rows = cursor.fetchall()
    conn.close()

    tables = [
        {
            "parent_id": r["parent_id"],
            "paper_id": r["paper_id"],
            "title": r["title"],
            "year": r["year"],
            "section": r["section"],
            "section_type": r["section_type"],
            "page_start": r["page_start"],
            "page_end": r["page_end"],
            "text": r["text"],
            "token_count": r["token_count"],
            "is_table": bool(r["is_table"]),
        }
        for r in rows
    ]
    return tables


def reembed_tables(
    parents_db: Path = DEFAULT_PARENTS_DB,
    qdrant_path: str = DEFAULT_QDRANT_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    limit: Optional[int] = None,
    client: Optional[Any] = None,
    embed_model: Optional[Any] = None,
) -> int:
    """Re-embed table child chunks into scholar_papers_v2 and update payloads.

    Args:
        parents_db: Path to SQLite parents database.
        qdrant_path: Local file path for Qdrant storage.
        collection_name: Qdrant collection name (default: scholar_papers_v2).
        model_name: SentenceTransformers model identifier.
        limit: Limit number of papers to re-embed.
        client: Optional pre-initialized QdrantClient.
        embed_model: Optional pre-initialized SentenceTransformer.

    Returns:
        Number of table child chunks re-embedded and upserted.
    """
    from qdrant_client.http import models

    table_parents = load_table_parents(parents_db=parents_db, limit_papers=limit)
    if not table_parents:
        logger.info("No table parent records found to re-embed.")
        return 0

    logger.info(
        f"Found {len(table_parents)} table records to re-embed "
        f"(limit={limit}, collection={collection_name})"
    )

    if embed_model is None:
        logger.info(f"Loading embedding model: {model_name}...")
        embed_model = get_embedding_model(model_name=model_name)

    texts = [t["text"] for t in table_parents]
    logger.info(f"Generating dense embeddings for {len(texts)} table chunks...")
    embeddings = embed_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    if client is None:
        client = get_qdrant_client(qdrant_path=qdrant_path)

    existing_collections = [col.name for col in client.get_collections().collections]
    if collection_name not in existing_collections:
        raise ValueError(
            f"Collection '{collection_name}' does not exist in Qdrant at {qdrant_path}. "
            "Please run ingestion/build_index_v2.py first."
        )

    # Build Qdrant points with updated payloads and vectors
    points = []
    for parent, vector in zip(table_parents, embeddings):
        vec_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        child_id = f"{parent['parent_id']}_c0"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, child_id))

        payload = {
            "text": parent["text"],
            "child_id": child_id,
            "parent_id": parent["parent_id"],
            "paper_id": parent["paper_id"],
            "title": parent["title"],
            "year": parent["year"],
            "section": parent["section"],  # Short label "Table N"
            "section_type": parent.get("section_type", "body"),
            "page_start": parent.get("page_start", 1),
            "page_end": parent.get("page_end", 1),
            "page": parent.get("page_start", 1),
            "is_table": True,
            "chunk_index": 0,
        }
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vec_list,
                payload=payload,
            )
        )

    # Upsert updated table points in batches
    batch_size = 64
    for i in tqdm(range(0, len(points), batch_size), desc="Upserting tables to Qdrant"):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
        )

    logger.info(
        f"Successfully re-embedded and updated {len(points)} table chunks in '{collection_name}'."
    )
    return len(points)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Re-embed ONLY table chunks into Qdrant collection scholar_papers_v2."
    )
    parser.add_argument(
        "--parents-db",
        type=Path,
        default=DEFAULT_PARENTS_DB,
        help=f"Path to SQLite parents database (default: {DEFAULT_PARENTS_DB}).",
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default=DEFAULT_QDRANT_PATH,
        help=f"Local path to Qdrant data (default: {DEFAULT_QDRANT_PATH}).",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION_NAME}).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model identifier (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers whose tables to re-embed.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for reembed_tables."""
    args = parse_args()
    reembed_tables(
        parents_db=args.parents_db,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection_name,
        model_name=args.model_name,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
