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
            "collection_v3": "scholar_papers_v3",
            "use_bm25": True,
            "use_reranker": True,
            "rrf_k": 60,
            "dense_candidates": 50,
            "bm25_candidates": 50,
            "rerank_top_n": 20,
            "max_parents_per_paper": None,
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
            "bm25_index": "data/bm25_index.pkl",
            "rerank_cache": ".cache/rerank_cache.json",
        },
        "models": {
            "embedding": "BAAI/bge-m3",
            "reranker": "BAAI/bge-reranker-v2-m3",
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


def extract_table_label(caption: str, fallback_label: str) -> str:
    """Extract 'Table N' short label requiring a number or roman numeral after 'Table'.
    
    Strips trailing punctuation (colons, periods, dashes, whitespace).
    """
    caption_clean = caption.strip()
    m = re.match(r"^Table\s+([0-9]+|[IVXLCDM]+)\b", caption_clean, re.IGNORECASE)
    if m:
        return m.group(0).strip().rstrip(".:-–— \t")
    return fallback_label


def find_enclosing_section(
    tbl: Dict[str, Any],
    sections: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the section in which the table appears, based on page numbering."""
    tbl_page = tbl.get("page_start", tbl.get("page", 1))
    candidates = [s for s in sections if s.get("page_start", 1) <= tbl_page]
    if candidates:
        return candidates[-1]
    if sections:
        return sections[0]
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


def count_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """Return token count of text using tokenizer if available, else regex estimation."""
    if not text or not text.strip():
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return estimate_tokens(text)


ABBREVIATIONS = {
    "e.g.", "i.e.", "et al.", "al.", "fig.", "figs.", "tab.", "tabs.",
    "eq.", "eqs.", "ref.", "refs.", "sec.", "secs.", "vol.", "vols.",
    "no.", "nos.", "pp.", "p.", "vs.", "approx.", "dept.", "univ.",
    "dr.", "prof.", "mr.", "mrs.", "ms.", "cf.", "ed.", "eds.",
}


def split_into_sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Split text into sentence spans (start_idx, end_idx) in text.
    
    Handles paragraph breaks, punctuation with academic abbreviations,
    decimals, and citation brackets. Each span (start, end) corresponds
    to non-whitespace text.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    candidate_splits: List[int] = []

    # 1. Paragraph boundaries (\n\s*\n+)
    for m in re.finditer(r"\n\s*\n+", text):
        candidate_splits.append(m.end())

    # 2. Sentence boundary punctuation [.!?]
    for m in re.finditer(r"([.!?])(['\")\]]*)(?=\s+|$)", text):
        punct = m.group(1)
        punct_pos = m.start(1)
        end_pos = m.end()

        if punct == ".":
            # Check for ellipsis (preceded or followed by another dot)
            if (punct_pos > 0 and text[punct_pos - 1] == ".") or (
                punct_pos + 1 < len(text) and text[punct_pos + 1] == "."
            ):
                continue

            # Check for decimals (e.g. 3.14)
            if punct_pos > 0 and punct_pos + 1 < len(text):
                if text[punct_pos - 1].isdigit() and text[punct_pos + 1].isdigit():
                    continue

            # Check preceding word for abbreviations
            preceding_match = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*)\.$", text[: punct_pos + 1])
            if preceding_match:
                word = preceding_match.group(1).lower() + "."
                two_words_match = re.search(r"\b(et\s+al)\.$", text[: punct_pos + 1], re.IGNORECASE)
                if word in ABBREVIATIONS or (two_words_match is not None):
                    continue
                # Single letter initial (e.g. "J. Smith")
                if len(preceding_match.group(1)) == 1 and preceding_match.group(1).isupper():
                    continue

            # Check if preceded only by digits at line start or whitespace (e.g., "1. Introduction")
            num_match = re.search(r"(?:^|\n|\s)\d+\.$", text[: punct_pos + 1])
            if num_match:
                continue

            # Check following non-whitespace character: if lowercase, likely not a sentence boundary
            after_text = text[end_pos:].lstrip()
            if after_text and after_text[0].islower():
                continue

        candidate_splits.append(end_pos)

    # Sort and deduplicate candidate split positions
    split_points = sorted(set(candidate_splits))
    if not split_points or split_points[-1] < len(text):
        split_points.append(len(text))

    spans: List[Tuple[int, int]] = []
    curr = 0
    for sp in split_points:
        if sp <= curr:
            continue
        seg = text[curr:sp]
        l_trim = len(seg) - len(seg.lstrip())
        r_trim = len(seg) - len(seg.rstrip())
        start = curr + l_trim
        end = sp - r_trim
        if start < end:
            spans.append((start, end))
        curr = sp

    if not spans:
        l_trim = len(text) - len(text.lstrip())
        r_trim = len(text) - len(text.rstrip())
        start = l_trim
        end = len(text) - r_trim
        if start < end:
            spans.append((start, end))

    return spans


def split_overlong_sentence_at_words(
    text: str,
    s_start: int,
    s_end: int,
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    tokenizer: Optional[Any] = None,
) -> List[Tuple[int, int]]:
    """Split an over-long sentence into sub-spans at word boundaries, each <= child_chunk_size."""
    s_text = text[s_start:s_end]
    word_matches = list(re.finditer(r"\S+", s_text))
    if not word_matches:
        return [(s_start, s_end)]

    sub_spans: List[Tuple[int, int]] = []
    w_start = 0
    n_words = len(word_matches)

    while w_start < n_words:
        w_end = w_start
        best_end = w_start
        while w_end < n_words:
            cand_start = s_start + word_matches[w_start].start()
            cand_end = s_start + word_matches[w_end].end()
            cand_text = text[cand_start:cand_end]
            t_count = count_tokens(cand_text, tokenizer=tokenizer)
            if t_count <= child_chunk_size:
                best_end = w_end
                w_end += 1
            else:
                break

        # If even a single word exceeded child_chunk_size, include it
        if best_end < w_start:
            best_end = w_start

        span_start = s_start + word_matches[w_start].start()
        span_end = s_start + word_matches[best_end].end()
        sub_spans.append((span_start, span_end))
        w_start = best_end + 1

    return sub_spans


def chunk_parent_into_children(
    parent: Dict[str, Any],
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    child_chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
    tokenizer: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Split a parent record into ~300 token child chunks.

    1. Tables are kept as standalone children (chunk_index=0), embedding caption and markdown together.
    2. Non-table sections pack whole sentences up to ~300 tokens with 1-sentence overlap.
    3. Over-long sentences exceeding ~300 tokens are split strictly at word boundaries.
    4. Every child chunk is a contiguous span of parent text and never starts or ends mid-word.
    """
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

    # 1. If it's a table, keep table as its own single standalone child chunk
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

    parent_text = parent["text"]
    total_tokens = count_tokens(parent_text, tokenizer=tokenizer)

    # 2. If entire section fits within child_chunk_size, return single child chunk
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
                "text": parent_text.strip(),
                "is_table": False,
                "chunk_index": 0,
            }
        ]

    # 3. Split parent text into sentence spans
    sentence_spans = split_into_sentence_spans(parent_text)
    if not sentence_spans:
        sentence_spans = [(0, len(parent_text))]

    # 4. Pack sentences with 1-sentence overlap
    chunk_spans: List[Tuple[int, int]] = []
    i = 0
    n_sentences = len(sentence_spans)

    while i < n_sentences:
        s_start, s_end = sentence_spans[i]
        s_text = parent_text[s_start:s_end]
        s_tokens = count_tokens(s_text, tokenizer=tokenizer)

        # Over-long sentence: split at word boundaries
        if s_tokens > child_chunk_size:
            overlong_spans = split_overlong_sentence_at_words(
                text=parent_text,
                s_start=s_start,
                s_end=s_end,
                child_chunk_size=child_chunk_size,
                tokenizer=tokenizer,
            )
            chunk_spans.extend(overlong_spans)
            i += 1
            continue

        # Pack whole sentences up to child_chunk_size
        j = i
        while j + 1 < n_sentences:
            cand_end = sentence_spans[j + 1][1]
            cand_text = parent_text[s_start:cand_end]
            if count_tokens(cand_text, tokenizer=tokenizer) <= child_chunk_size:
                j += 1
            else:
                break

        chunk_spans.append((s_start, sentence_spans[j][1]))

        # Advance with 1-sentence overlap
        if j == n_sentences - 1:
            break
        elif j > i:
            i = j
        else:
            i += 1

    # 5. Build child records
    children: List[Dict[str, Any]] = []
    for chunk_idx, (start, end) in enumerate(chunk_spans):
        child_text = parent_text[start:end]
        if not child_text.strip():
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
                "text": child_text,
                "is_table": False,
                "chunk_index": chunk_idx,
            }
        )

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
    sections = parsed_paper.get("sections", [])
    for tbl in parsed_paper.get("tables", []):
        md = tbl.get("markdown", "").strip()
        if not md:
            continue

        tbl_id = tbl.get("table_id", f"{paper_id}_tbl_0")
        p_start = tbl.get("page_start", tbl.get("page", 1))
        p_end = tbl.get("page_end", p_start)
        raw_caption = (tbl.get("caption") or "").strip()

        # Derive fallback if no numeric/roman Table label in caption
        tbl_idx_match = re.search(r"_tbl_(\d+)$", tbl_id)
        fallback = (
            f"Table {int(tbl_idx_match.group(1)) + 1}"
            if tbl_idx_match
            else f"Table (Page {p_start})"
        )
        short_label = extract_table_label(raw_caption, fallback)

        # Prepend caption to markdown table text
        full_text = f"{raw_caption}\n\n{md}".strip() if raw_caption else md

        # Inherit section_type from enclosing section
        enclosing = find_enclosing_section(tbl, sections)
        tbl_sec_type = enclosing.get("section_type", "body") if enclosing else "body"

        tbl_parent = {
            "parent_id": tbl_id,
            "paper_id": paper_id,
            "title": title,
            "year": year,
            "section": short_label,
            "section_type": tbl_sec_type,
            "page_start": p_start,
            "page_end": p_end,
            "text": full_text,
            "token_count": estimate_tokens(full_text),
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
