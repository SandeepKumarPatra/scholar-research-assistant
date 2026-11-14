#!/usr/bin/env python3
"""Build Qdrant vector index v2 (scholar_papers_v2) for child chunks.

Embeds ONLY the children with BAAI/bge-m3 and indexes them into the local Qdrant
collection 'scholar_papers_v2'. Saves parent sections/tables into SQLite (data/parents.db).
Provides tqdm progress bars, --limit, and --recreate arguments.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ingestion.chunk_structured import (
    DEFAULT_PARSED_DIR,
    DEFAULT_PARENTS_DB,
    chunk_all_parsed_papers,
    load_config,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scholar.build_index_v2")

DEFAULT_COLLECTION_NAME_V2 = "scholar_papers_v2"
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


def build_index_v2(
    parsed_dir: Path = DEFAULT_PARSED_DIR,
    parents_db: Path = DEFAULT_PARENTS_DB,
    qdrant_path: str = DEFAULT_QDRANT_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME_V2,
    model_name: str = DEFAULT_MODEL_NAME,
    limit: Optional[int] = None,
    recreate_index: bool = False,
    client: Optional[Any] = None,
    embed_model: Optional[Any] = None,
) -> int:
    """Build Qdrant collection for child chunks and populate SQLite parents database.

    Args:
        parsed_dir: Path to directory with parsed paper JSONs.
        parents_db: Path to SQLite database for parents.
        qdrant_path: Local file path for Qdrant storage.
        collection_name: Qdrant collection name (default: scholar_papers_v2).
        model_name: Model identifier for SentenceTransformers.
        limit: Limit number of papers to index.
        recreate_index: If True, deletes and recreates collection.
        client: Optional pre-initialized QdrantClient.
        embed_model: Optional pre-initialized SentenceTransformer.

    Returns:
        Number of child chunks indexed.
    """
    from qdrant_client.http import models

    config = load_config()

    # Step 1: Generate parents and children from parsed papers
    parents, children = chunk_all_parsed_papers(
        parsed_dir=parsed_dir,
        parents_db=parents_db,
        config=config,
        limit=limit,
        save_to_db=True,
    )

    if not children:
        logger.warning("No child chunks generated. Ensure parsed JSON files exist in data/parsed/.")
        return 0

    logger.info(
        f"Prepared {len(parents)} parents (saved to {parents_db}) "
        f"and {len(children)} child chunks for indexing."
    )

    # Step 2: Initialize embedding model
    if embed_model is None:
        logger.info(f"Loading embedding model: {model_name}...")
        embed_model = get_embedding_model(model_name=model_name)

    texts = [child["text"] for child in children]
    logger.info(f"Generating dense embeddings for {len(texts)} child chunks...")

    embeddings = embed_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Step 3: Initialize Qdrant Client
    if client is None:
        logger.info(f"Connecting to local Qdrant at: {qdrant_path}")
        client = get_qdrant_client(qdrant_path=qdrant_path)

    # Check collections
    existing_collections = [col.name for col in client.get_collections().collections]
    if collection_name in existing_collections:
        if recreate_index:
            logger.info(f"Recreating existing collection: {collection_name}")
            client.delete_collection(collection_name=collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
            )
        else:
            logger.info(f"Collection '{collection_name}' already exists. Upserting points...")
    else:
        logger.info(f"Creating collection: {collection_name} (dim={EMBEDDING_DIM})")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
        )

    # Step 4: Prepare Qdrant Points
    points = []
    for child, vector in zip(children, embeddings):
        vec_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        payload = {
            "text": child["text"],
            "child_id": child["child_id"],
            "parent_id": child["parent_id"],
            "paper_id": child["paper_id"],
            "title": child["title"],
            "year": child["year"],
            "section": child["section"],
            "section_type": child.get("section_type", "body"),
            "page_start": child.get("page_start", 1),
            "page_end": child.get("page_end", 1),
            "page": child.get("page", 1),
            "is_table": child.get("is_table", False),
            "chunk_index": child.get("chunk_index", 0),
        }
        points.append(
            models.PointStruct(
                id=child["point_id"],
                vector=vec_list,
                payload=payload,
            )
        )

    # Step 5: Upsert points with progress bar
    batch_size = 64
    for i in tqdm(range(0, len(points), batch_size), desc="Upserting to Qdrant"):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
        )

    logger.info(
        f"Successfully indexed {len(points)} child chunks into '{collection_name}' at {qdrant_path}."
    )
    return len(points)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for build_index_v2."""
    parser = argparse.ArgumentParser(
        description="Index child chunks into Qdrant collection scholar_papers_v2 and store parents in SQLite."
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=DEFAULT_PARSED_DIR,
        help="Directory with parsed paper JSONs (default: data/parsed).",
    )
    parser.add_argument(
        "--parents-db",
        type=Path,
        default=DEFAULT_PARENTS_DB,
        help="Path to SQLite parents database (default: data/parents.db).",
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default=DEFAULT_QDRANT_PATH,
        help="Local file path for Qdrant storage (default: ./qdrant_data).",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME_V2,
        help="Qdrant collection name (default: scholar_papers_v2).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformers model name (default: BAAI/bge-m3).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers to index.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate Qdrant collection if it exists.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for index v2."""
    args = parse_args()
    build_index_v2(
        parsed_dir=args.parsed_dir,
        parents_db=args.parents_db,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection_name,
        model_name=args.model_name,
        limit=args.limit,
        recreate_index=args.recreate,
    )


if __name__ == "__main__":
    main()
