#!/usr/bin/env python3
"""Structure-aware parent-child chunking for Scholar papers.

Transforms parsed documents into:
1. Parents: Whole sections (split if > max_parent_tokens, default 1500) and tables.
   Stored exclusively in SQLite (data/parents.db).
2. Children: ~300 token pieces of those sections (and tables as separate chunks)
   used for dense vector search. Excludes References/Bibliography by default.
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scholar.chunk_structured")

DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_PARENTS_DB = Path("data/parents.db")
DEFAULT_PARSED_DIR = Path("data/parsed")
DEFAULT_CHILD_CHUNK_SIZE = 300
DEFAULT_CHILD_CHUNK_OVERLAP = 30
DEFAULT_MAX_PARENT_TOKENS = 1500


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from config.yaml, or return default dictionary."""
    defaults: Dict[str, Any] = {
        "retrieval": {
            "mode": "parent_child",
            "top_k": 5,
            "collection_naive": "scholar_papers",
            "collection_v2": "scholar_papers_v2",
        },
        "chunking": {
            "child_chunk_size": DEFAULT_CHILD_CHUNK_SIZE,
            "child_chunk_overlap": DEFAULT_CHILD_CHUNK_OVERLAP,
            "max_parent_tokens": DEFAULT_MAX_PARENT_TOKENS,
            "include_references": False,
            "include_appendix": True,
        },
        "storage": {
            "qdrant_path": "./qdrant_data",
            "parsed_dir": "data/parsed",
            "parents_db": "data/parents.db",
            "metadata_file": "data/metadata.json",
            "pdfs_dir": "data/pdfs",
        },
        "models": {
            "embedding": "BAAI/bge-m3",
        },
    }

    if not config_path.exists():
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if isinstance(cfg, dict):
            # Deep merge defaults
            for section, vals in defaults.items():
                if section not in cfg:
                    cfg[section] = vals
                elif isinstance(vals, dict) and isinstance(cfg[section], dict):
                    for k, v in vals.items():
                        cfg[section].setdefault(k, v)
            return cfg
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}. Using defaults.")

    return defaults


def simple_tokenize(text: str) -> List[str]:
    """Fallback tokenization splitting into words and punctuation."""
    return re.findall(r"\w+|[^\w\s]", text)


def estimate_tokens(text: str) -> int:
    """Return estimated token count of text."""
    return len(simple_tokenize(text))


def get_tokenizer(model_name: str = "BAAI/bge-m3") -> Optional[Any]:
    """Attempt to load Hugging Face tokenizer for bge-m3 (local files only), or None."""
    try:
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception:
            return None
    except Exception:
        return None


# ============================================================================
# SQLite Parents Store
# ============================================================================


def init_parents_db(db_path: Path = DEFAULT_PARENTS_DB) -> sqlite3.Connection:
    """Initialize SQLite database for parent sections and tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS parents (
            parent_id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            title TEXT NOT NULL,
            year INTEGER NOT NULL,
            section TEXT NOT NULL,
            section_type TEXT NOT NULL,
            page_start INTEGER NOT NULL,
            page_end INTEGER NOT NULL,
            text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            is_table INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_parents_paper ON parents(paper_id);"
    )
    conn.commit()
    return conn


def save_parents_to_db(
    parents: List[Dict[str, Any]],
    db_path: Path = DEFAULT_PARENTS_DB,
) -> int:
    """Save parent records into SQLite database. Overwrites duplicates."""
    if not parents:
        return 0

    conn = init_parents_db(db_path)
    records = [
        (
            p["parent_id"],
            p["paper_id"],
            p["title"],
            p["year"],
            p["section"],
            p.get("section_type", "body"),
            p.get("page_start", 1),
            p.get("page_end", 1),
            p["text"],
            p.get("token_count", estimate_tokens(p["text"])),
            1 if p.get("is_table", False) else 0,
        )
        for p in parents
    ]

    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO parents (
                parent_id, paper_id, title, year, section, section_type,
                page_start, page_end, text, token_count, is_table
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            records,
        )
    conn.close()
    return len(records)


def get_parents_by_ids(
    parent_ids: List[str],
    db_path: Path = DEFAULT_PARENTS_DB,
) -> Dict[str, Dict[str, Any]]:
    """Retrieve parent records by their IDs from SQLite database."""
    if not parent_ids:
        return {}

    if not db_path.exists():
        logger.warning(f"Parents DB not found at: {db_path}")
        return {}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    placeholders = ",".join(["?"] * len(parent_ids))
    query = f"""
        SELECT parent_id, paper_id, title, year, section, section_type,
               page_start, page_end, text, token_count, is_table
        FROM parents
        WHERE parent_id IN ({placeholders})
    """
    cursor = conn.execute(query, parent_ids)
    rows = cursor.fetchall()
    conn.close()

    results: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        results[r["parent_id"]] = {
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
    return results


# ============================================================================
# Chunking Logic
# ============================================================================


def split_text_into_tokens_or_words(
    text: str,
    tokenizer: Optional[Any] = None,
) -> Tuple[List[Any], bool]:
    """Tokenize text using tokenizer if available, else regex tokens."""
    if tokenizer is not None:
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return tokens, True
        except Exception:
            pass
    return simple_tokenize(text), False


def decode_tokens_or_words(
    token_slice: List[Any],
    is_hf: bool,
    tokenizer: Optional[Any] = None,
) -> str:
    """Decode a slice of tokens back to string."""
    if is_hf and tokenizer is not None:
        return tokenizer.decode(token_slice, skip_special_tokens=True).strip()
    joined = " ".join(str(t) for t in token_slice).strip()
    return re.sub(r"\s+([.,!?;:])", r"\1", joined)


def split_section_if_too_large(
    section: Dict[str, Any],
    paper_id: str,
    title: str,
    year: int,
    max_parent_tokens: int = DEFAULT_MAX_PARENT_TOKENS,
    tokenizer: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Split section into multiple parent records if it exceeds max_parent_tokens."""
    text = section.get("text", "").strip()
    if not text:
        return []

    tokens, is_hf = split_text_into_tokens_or_words(text, tokenizer=tokenizer)
    total_tokens = len(tokens)

    sec_id = section.get("section_id", f"{paper_id}_sec_0")
    heading = section.get("heading", "Untitled Section")
    sec_type = section.get("section_type", "body")
    p_start = section.get("page_start", 1)
    p_end = section.get("page_end", p_start)

    if total_tokens <= max_parent_tokens:
        return [
            {
                "parent_id": sec_id,
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "section": heading,
                "section_type": sec_type,
                "page_start": p_start,
                "page_end": p_end,
                "text": text,
                "token_count": total_tokens,
                "is_table": False,
            }
        ]

    # Split into contiguous chunks of max_parent_tokens
    parents: List[Dict[str, Any]] = []
    part_idx = 0
    for i in range(0, total_tokens, max_parent_tokens):
        sub_tokens = tokens[i : i + max_parent_tokens]
        sub_text = decode_tokens_or_words(sub_tokens, is_hf=is_hf, tokenizer=tokenizer)
        if not sub_text:
            continue

        part_id = f"{sec_id}_part{part_idx}"
        part_heading = f"{heading} (Part {part_idx + 1})"
        parents.append(
            {
                "parent_id": part_id,
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "section": part_heading,
                "section_type": sec_type,
                "page_start": p_start,
                "page_end": p_end,
                "text": sub_text,
                "token_count": len(sub_tokens),
                "is_table": False,
            }
        )
        part_idx += 1

    return parents


def chunk_parent_into_children(
    parent: Dict[str, Any],
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    child_chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
    tokenizer: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Split a parent record into ~300 token child chunks."""
    text = parent.get("text", "").strip()
    if not text:
        return []

    parent_id = parent["parent_id"]
    paper_id = parent["paper_id"]
    title = parent["title"]
    year = parent["year"]
    section = parent["section"]
    sec_type = parent.get("section_type", "body")
    p_start = parent.get("page_start", 1)
    p_end = parent.get("page_end", p_start)
    is_table = parent.get("is_table", False)

    # If it's a table, keep table as its own single child chunk
    if is_table:
        child_id = f"{parent_id}_c0"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, child_id))
        return [
            {
                "child_id": child_id,
                "point_id": point_id,
                "parent_id": parent_id,
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "section": section,
                "section_type": sec_type,
                "page_start": p_start,
                "page_end": p_end,
                "page": p_start,
                "text": text,
                "is_table": True,
                "chunk_index": 0,
            }
        ]

    tokens, is_hf = split_text_into_tokens_or_words(text, tokenizer=tokenizer)
    total_tokens = len(tokens)

    if total_tokens <= child_chunk_size:
        child_id = f"{parent_id}_c0"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, child_id))
        return [
            {
                "child_id": child_id,
                "point_id": point_id,
                "parent_id": parent_id,
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "section": section,
                "section_type": sec_type,
                "page_start": p_start,
                "page_end": p_end,
                "page": p_start,
                "text": text,
                "is_table": False,
                "chunk_index": 0,
            }
        ]

    step = max(1, child_chunk_size - child_chunk_overlap)
    children: List[Dict[str, Any]] = []
    chunk_idx = 0

    for i in range(0, total_tokens, step):
        sub_tokens = tokens[i : i + child_chunk_size]
        sub_text = decode_tokens_or_words(sub_tokens, is_hf=is_hf, tokenizer=tokenizer)
        if not sub_text:
            continue

        child_id = f"{parent_id}_c{chunk_idx}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, child_id))
        children.append(
            {
                "child_id": child_id,
                "point_id": point_id,
                "parent_id": parent_id,
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "section": section,
                "section_type": sec_type,
                "page_start": p_start,
                "page_end": p_end,
                "page": p_start,
                "text": sub_text,
                "is_table": False,
                "chunk_index": chunk_idx,
            }
        )
        chunk_idx += 1
        if i + child_chunk_size >= total_tokens:
            break

    return children


def process_paper_chunks(
    parsed_paper: Dict[str, Any],
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    child_chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
    max_parent_tokens: int = DEFAULT_MAX_PARENT_TOKENS,
    include_references: bool = False,
    include_appendix: bool = True,
    tokenizer: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate parents and child chunks for a single parsed paper JSON."""
    paper_id = parsed_paper.get("paper_id", "")
    title = parsed_paper.get("title", "Untitled")
    year = parsed_paper.get("year", 0)

    parents: List[Dict[str, Any]] = []
    children: List[Dict[str, Any]] = []

    # 1. Process sections
    for sec in parsed_paper.get("sections", []):
        sec_type = sec.get("section_type", "body")
        # Split section if it exceeds max_parent_tokens
        sec_parents = split_section_if_too_large(
            section=sec,
            paper_id=paper_id,
            title=title,
            year=year,
            max_parent_tokens=max_parent_tokens,
            tokenizer=tokenizer,
        )
        for p in sec_parents:
            parents.append(p)

            # Check if children should be indexed
            if sec_type == "references" and not include_references:
                continue
            if sec_type == "appendix" and not include_appendix:
                continue

            sec_children = chunk_parent_into_children(
                parent=p,
                child_chunk_size=child_chunk_size,
                child_chunk_overlap=child_chunk_overlap,
                tokenizer=tokenizer,
            )
            children.extend(sec_children)

    # 2. Process tables
    for tbl in parsed_paper.get("tables", []):
        md = tbl.get("markdown", "").strip()
        if not md:
            continue

        tbl_id = tbl.get("table_id", f"{paper_id}_tbl_0")
        p_start = tbl.get("page_start", tbl.get("page", 1))
        p_end = tbl.get("page_end", p_start)
        caption = tbl.get("caption") or f"Table (Page {p_start})"

        tbl_parent = {
            "parent_id": tbl_id,
            "paper_id": paper_id,
            "title": title,
            "year": year,
            "section": caption,
            "section_type": "table",
            "page_start": p_start,
            "page_end": p_end,
            "text": md,
            "token_count": estimate_tokens(md),
            "is_table": True,
        }
        parents.append(tbl_parent)

        tbl_children = chunk_parent_into_children(
            parent=tbl_parent,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
            tokenizer=tokenizer,
        )
        children.extend(tbl_children)

    return parents, children


def chunk_all_parsed_papers(
    parsed_dir: Path = DEFAULT_PARSED_DIR,
    parents_db: Path = DEFAULT_PARENTS_DB,
    config: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    save_to_db: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process all parsed papers in parsed_dir into parents and children."""
    if config is None:
        config = load_config()

    chunk_cfg = config.get("chunking", {})
    child_size = chunk_cfg.get("child_chunk_size", DEFAULT_CHILD_CHUNK_SIZE)
    child_overlap = chunk_cfg.get("child_chunk_overlap", DEFAULT_CHILD_CHUNK_OVERLAP)
    max_parent_tokens = chunk_cfg.get("max_parent_tokens", DEFAULT_MAX_PARENT_TOKENS)
    include_refs = chunk_cfg.get("include_references", False)
    include_app = chunk_cfg.get("include_appendix", True)

    model_name = config.get("models", {}).get("embedding", "BAAI/bge-m3")
    tokenizer = get_tokenizer(model_name)

    json_files = sorted(parsed_dir.glob("*.json"))
    if limit is not None and limit > 0:
        json_files = json_files[:limit]

    all_parents: List[Dict[str, Any]] = []
    all_children: List[Dict[str, Any]] = []

    logger.info(
        f"Chunking {len(json_files)} parsed papers (child_size={child_size}, "
        f"max_parent_tokens={max_parent_tokens}, include_refs={include_refs})..."
    )

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                parsed_paper = json.load(f)
            parents, children = process_paper_chunks(
                parsed_paper=parsed_paper,
                child_chunk_size=child_size,
                child_chunk_overlap=child_overlap,
                max_parent_tokens=max_parent_tokens,
                include_references=include_refs,
                include_appendix=include_app,
                tokenizer=tokenizer,
            )
            all_parents.extend(parents)
            all_children.extend(children)
        except Exception as e:
            logger.error(f"Error chunking paper from {jf}: {e}", exc_info=True)

    if save_to_db and all_parents:
        save_parents_to_db(all_parents, db_path=parents_db)
        logger.info(f"Saved {len(all_parents)} parent sections/tables to SQLite: {parents_db}")

    logger.info(
        f"Generated {len(all_parents)} parents and {len(all_children)} children "
        f"across {len(json_files)} papers."
    )
    return all_parents, all_children


def main() -> None:
    """CLI testing helper for structured chunking."""
    parser = argparse.ArgumentParser(description="Run structure-aware parent-child chunking.")
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR)
    parser.add_argument("--parents-db", type=Path, default=DEFAULT_PARENTS_DB)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    parents, children = chunk_all_parsed_papers(
        parsed_dir=args.parsed_dir,
        parents_db=args.parents_db,
        limit=args.limit,
    )
    print(f"\nChunking Summary: {len(parents)} parents stored, {len(children)} children generated.")
    if children:
        print(f"Sample child chunk:\n{json.dumps(children[0], indent=2)}")


if __name__ == "__main__":
    main()
