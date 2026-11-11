#!/usr/bin/env python3
"""Build local Qdrant vector index for scholar papers.

Extracts text per page using pypdf, splits into 800-token chunks with 100-token
overlap, embeds with BAAI/bge-m3, and indexes chunks into a local file-backed
Qdrant collection with metadata (id, title, year, page).
"""

import argparse
import json
import logging
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pypdf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scholar.build_index")

DEFAULT_COLLECTION_NAME = "scholar_papers"
DEFAULT_QDRANT_PATH = "./qdrant_data"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
EMBEDDING_DIM = 1024


def sanitize_filename(name: str) -> str:
    """Sanitize paper id to match downloaded PDF filenames."""
    clean = re.sub(r'[\\/*?:"<>|]', "_", name)
    return clean.strip()


def get_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    """Attempt to load Hugging Face tokenizer for token counting, or fallback."""
    try:
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception:
            return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.warning(
            f"Could not load Hugging Face tokenizer for '{model_name}': {e}. "
            "Falling back to regex-based token approximation."
        )
        return None


def simple_tokenize(text: str) -> List[str]:
    """Fallback tokenization splitting into words and punctuation."""
    return re.findall(r"\w+|[^\w\s]", text)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    tokenizer: Optional[Any] = None,
) -> List[str]:
    """Split text into fixed-size chunks (chunk_size tokens, chunk_overlap overlap).

    If text is shorter than chunk_size, returns a single chunk.
    """
    clean = text.strip()
    if not clean:
        return []

    step = max(1, chunk_size - chunk_overlap)

    if tokenizer is not None:
        try:
            tokens = tokenizer.encode(clean, add_special_tokens=False)
            if len(tokens) <= chunk_size:
                return [clean]

            chunks = []
            for i in range(0, len(tokens), step):
                chunk_tokens = tokens[i : i + chunk_size]
                chunk_str = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
                if chunk_str:
                    chunks.append(chunk_str)
                if i + chunk_size >= len(tokens):
                    break
            return chunks
        except Exception as e:
            logger.warning(f"Tokenizer encoding failed: {e}. Using fallback tokenization.")

    # Fallback word/punctuation tokenization
    tokens = simple_tokenize(clean)
    if len(tokens) <= chunk_size:
        return [clean]

    chunks = []
    for i in range(0, len(tokens), step):
        sub_tokens = tokens[i : i + chunk_size]
        chunk_str = " ".join(sub_tokens).strip()
        # Clean spacing around punctuation
        chunk_str = re.sub(r"\s+([.,!?;:])", r"\1", chunk_str)
        if chunk_str:
            chunks.append(chunk_str)
        if i + chunk_size >= len(tokens):
            break

    return chunks


def extract_pages_from_pdf(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract text per page using pypdf.

    Returns a list of (page_number, text) tuples where page_number is 1-indexed.
    """
    pages_text: List[Tuple[int, str]] = []
    if not pdf_path.exists():
        logger.warning(f"PDF not found: {pdf_path}")
        return pages_text

    try:
        reader = pypdf.PdfReader(str(pdf_path))
        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
                pages_text.append((page_idx, text.strip()))
            except Exception as e:
                logger.warning(f"Error extracting page {page_idx} from {pdf_path.name}: {e}")
                pages_text.append((page_idx, ""))
    except Exception as e:
        logger.error(f"Failed to read PDF {pdf_path}: {e}")

    return pages_text


def load_metadata_map(metadata_file: Path) -> Dict[str, Dict[str, Any]]:
    """Load metadata.json mapping paper_id to metadata record."""
    if not metadata_file.exists():
        logger.warning(f"Metadata file not found: {metadata_file}")
        return {}

    with open(metadata_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    meta_map = {}
    for r in records:
        if "id" in r:
            meta_map[r["id"]] = r
    return meta_map


def prepare_documents(
    metadata_map: Dict[str, Dict[str, Any]],
    pdfs_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    tokenizer: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Extract and chunk documents with full metadata."""
    documents: List[Dict[str, Any]] = []

    for paper_id, meta in metadata_map.items():
        title = meta.get("title", "")
        year = meta.get("year", 0)

        # Locate PDF file
        safe_id = sanitize_filename(paper_id)
        pdf_path = pdfs_dir / f"{safe_id}.pdf"
        if not pdf_path.exists():
            # Try raw id
            pdf_path = pdfs_dir / f"{paper_id}.pdf"

        if not pdf_path.exists():
            logger.warning(f"Skipping {paper_id}: PDF file not found in {pdfs_dir}")
            continue

        pages = extract_pages_from_pdf(pdf_path)
        logger.info(f"Processing paper '{title[:45]}...' ({paper_id}): {len(pages)} pages")

        chunk_counter = 0
        for page_num, page_text in pages:
            if not page_text:
                continue

            chunks = chunk_text(
                page_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                tokenizer=tokenizer,
            )

            for chunk_str in chunks:
                if not chunk_str.strip():
                    continue

                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{paper_id}_p{page_num}_c{chunk_counter}"))
                documents.append(
                    {
                        "point_id": point_id,
                        "text": chunk_str,
                        "metadata": {
                            "id": paper_id,
                            "paper_id": paper_id,
                            "title": title,
                            "year": year,
                            "page": page_num,
                            "chunk_index": chunk_counter,
                        },
                    }
                )
                chunk_counter += 1

    return documents


def build_index(
    data_dir: Path = Path("data"),
    qdrant_path: str = DEFAULT_QDRANT_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    recreate_index: bool = False,
) -> int:
    """Build or rebuild local Qdrant collection."""
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from sentence_transformers import SentenceTransformer

    pdfs_dir = data_dir / "pdfs"
    metadata_file = data_dir / "metadata.json"

    metadata_map = load_metadata_map(metadata_file)
    if not metadata_map:
        logger.error(f"No metadata found in {metadata_file}. Run ingestion first.")
        return 0

    tokenizer = get_tokenizer(model_name)
    documents = prepare_documents(
        metadata_map=metadata_map,
        pdfs_dir=pdfs_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=tokenizer,
    )

    if not documents:
        logger.warning("No document chunks created. Check if PDFs exist in data/pdfs.")
        return 0

    logger.info(f"Extracted {len(documents)} chunks across {len(metadata_map)} papers.")

    # Initialize SentenceTransformer embedding model
    logger.info(f"Loading embedding model: {model_name}...")
    try:
        embed_model = SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        embed_model = SentenceTransformer(model_name)

    texts = [doc["text"] for doc in documents]
    logger.info(f"Generating dense embeddings for {len(texts)} chunks...")
    embeddings = embed_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Initialize local Qdrant client
    Path(qdrant_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Connecting to local Qdrant at: {qdrant_path}")
    client = QdrantClient(path=qdrant_path)

    # Create collection if needed
    collections = [col.name for col in client.get_collections().collections]
    if collection_name in collections:
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

    # Prepare Qdrant points
    points = []
    for doc, vector in zip(documents, embeddings):
        payload = {
            "text": doc["text"],
            **doc["metadata"],
        }
        points.append(
            models.PointStruct(
                id=doc["point_id"],
                vector=vector.tolist(),
                payload=payload,
            )
        )

    # Batch upsert points
    batch_size = 64
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
        )

    logger.info(f"Successfully indexed {len(points)} chunks into '{collection_name}' at {qdrant_path}.")
    return len(points)


def parse_args() -> argparse.Namespace:
    """CLI argument parser for build_index."""
    parser = argparse.ArgumentParser(
        description="Extract PDF pages, chunk text, embed with bge-m3, and store into local Qdrant."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to data directory containing pdfs/ and metadata.json (default: data)",
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default=DEFAULT_QDRANT_PATH,
        help="Local file path for Qdrant storage (default: ./qdrant_data)",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help="Qdrant collection name (default: scholar_papers)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformers model name (default: BAAI/bge-m3)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size in tokens (default: 800)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap in tokens (default: 100)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate Qdrant collection if it exists",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for indexing."""
    args = parse_args()
    build_index(
        data_dir=args.data_dir,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection_name,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        recreate_index=args.recreate,
    )


if __name__ == "__main__":
    main()
