#!/usr/bin/env python3
"""Docling-based PDF parser for Scholar papers.

Extracts structured sections, headings, body text, tables (as markdown), and
page numbers (page_start, page_end) with OCR disabled and table structure enabled.
Saves parsed output as JSON in data/parsed/ (one file per paper) so parsing only
runs once. Provides a --limit argument and post-parsing stats summary.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scholar.parse_docling")

DEFAULT_DATA_DIR = Path("data")
DEFAULT_PDFS_DIR = DEFAULT_DATA_DIR / "pdfs"
DEFAULT_PARSED_DIR = DEFAULT_DATA_DIR / "parsed"
DEFAULT_METADATA_FILE = DEFAULT_DATA_DIR / "metadata.json"


def sanitize_filename(name: str) -> str:
    """Sanitize paper id to match downloaded PDF filenames."""
    clean = re.sub(r'[\\/*?:"<>|]', "_", name)
    return clean.strip()


def simple_tokenize(text: str) -> List[str]:
    """Tokenize text by words and punctuation for token estimation."""
    return re.findall(r"\w+|[^\w\s]", text)


def estimate_tokens(text: str) -> int:
    """Return estimated token count of text."""
    return len(simple_tokenize(text))


def classify_section_type(heading: str) -> str:
    """Classify section type as 'references', 'appendix', or 'body'."""
    h = heading.strip().lower()
    h_clean = re.sub(r"^[\d\.\s\-]+", "", h).strip()
    if re.search(r"^(references|bibliography|works cited|literature cited)", h_clean):
        return "references"
    if re.search(r"^(appendix|appendices|supplementary|supplemental)", h_clean):
        return "appendix"
    return "body"


def load_metadata_map(metadata_file: Path) -> Dict[str, Dict[str, Any]]:
    """Load metadata.json mapping paper_id to metadata record (title, year, etc.)."""
    if not metadata_file.exists():
        logger.warning(f"Metadata file not found: {metadata_file}")
        return {}

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        return {r["id"]: r for r in records if "id" in r}
    except Exception as e:
        logger.error(f"Error reading metadata file {metadata_file}: {e}")
        return {}


def get_docling_converter() -> Any:
    """Initialize Docling DocumentConverter with OCR disabled and tables enabled."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        return converter
    except ImportError as e:
        raise ImportError(
            f"Docling is not installed: {e}. "
            "Please install docling in your environment: pip install docling"
        ) from e


def extract_page_from_item(item: Any, default_page: int = 1) -> int:
    """Extract 1-indexed page number from Docling item provenance if available."""
    prov = getattr(item, "prov", None)
    if prov and isinstance(prov, list) and len(prov) > 0:
        first_prov = prov[0]
        page = getattr(first_prov, "page_no", getattr(first_prov, "page", None))
        if page is not None and isinstance(page, int):
            return page
    return default_page


def parse_docling_document(
    doc: Any,
    paper_id: str,
    title: str,
    year: int,
) -> Dict[str, Any]:
    """Extract structured sections, tables, and page ranges from a Docling document."""
    sections: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []

    # 1. Extract tables as markdown
    doc_tables = getattr(doc, "tables", []) or []
    for tbl_idx, tbl in enumerate(doc_tables):
        tbl_page = extract_page_from_item(tbl, default_page=1)
        # Convert table to markdown
        md_text = ""
        if hasattr(tbl, "export_to_markdown"):
            md_text = tbl.export_to_markdown()
        elif hasattr(tbl, "to_markdown"):
            md_text = tbl.to_markdown()
        else:
            md_text = str(tbl)

        caption = ""
        if hasattr(tbl, "caption_text"):
            try:
                caption = tbl.caption_text(doc) or ""
            except Exception:
                caption = ""
        elif hasattr(tbl, "caption"):
            caption = str(tbl.caption or "")

        tables.append(
            {
                "table_id": f"{paper_id}_tbl_{tbl_idx}",
                "page": tbl_page,
                "page_start": tbl_page,
                "page_end": tbl_page,
                "caption": caption.strip(),
                "markdown": md_text.strip(),
            }
        )

    # 2. Iterate document items to group into sections
    current_heading: Optional[str] = None
    current_paragraphs: List[str] = []
    current_pages: List[int] = []
    page_paragraphs: Dict[int, List[str]] = {}  # Fallback for page-by-page grouping
    detected_headings_count = 0

    # Retrieve iterable items
    if hasattr(doc, "iterate_items"):
        items = list(doc.iterate_items())
    elif hasattr(doc, "body"):
        items = [(item, 0) for item in getattr(doc.body, "children", [])]
    else:
        items = []

    for entry in items:
        # iterate_items returns (item, level) or just item
        item = entry[0] if isinstance(entry, (tuple, list)) else entry

        # Check item label
        label = getattr(item, "label", None)
        label_str = str(label).lower() if label is not None else ""
        item_text = getattr(item, "text", "")
        if not item_text:
            continue
        item_text = item_text.strip()
        if not item_text:
            continue

        item_page = extract_page_from_item(item, default_page=1)
        page_paragraphs.setdefault(item_page, []).append(item_text)

        is_heading = (
            "section_header" in label_str
            or "heading" in label_str
            or "title" in label_str
        )

        if is_heading:
            detected_headings_count += 1
            # Save previous section if exists
            if current_heading is not None and current_paragraphs:
                p_start = min(current_pages) if current_pages else 1
                p_end = max(current_pages) if current_pages else p_start
                sec_text = "\n\n".join(current_paragraphs).strip()
                sections.append(
                    {
                        "section_id": f"{paper_id}_sec_{len(sections)}",
                        "heading": current_heading,
                        "section_type": classify_section_type(current_heading),
                        "page_start": p_start,
                        "page_end": p_end,
                        "text": sec_text,
                        "token_count": estimate_tokens(sec_text),
                    }
                )

            current_heading = item_text
            current_paragraphs = []
            current_pages = [item_page]
        else:
            # Paragraph or body text
            current_paragraphs.append(item_text)
            current_pages.append(item_page)

    # Save final section if exists
    if current_heading is not None and current_paragraphs:
        p_start = min(current_pages) if current_pages else 1
        p_end = max(current_pages) if current_pages else p_start
        sec_text = "\n\n".join(current_paragraphs).strip()
        sections.append(
            {
                "section_id": f"{paper_id}_sec_{len(sections)}",
                "heading": current_heading,
                "section_type": classify_section_type(current_heading),
                "page_start": p_start,
                "page_end": p_end,
                "text": sec_text,
                "token_count": estimate_tokens(sec_text),
            }
        )

    # 3. Fallback: if no section headings detected across the paper
    has_fallback = False
    if detected_headings_count == 0 or len(sections) == 0:
        logger.warning(
            f"No section headings detected in paper '{paper_id}'. "
            "Falling back to 1 parent per page."
        )
        has_fallback = True
        sections = []
        for page_num in sorted(page_paragraphs.keys()):
            p_text = "\n\n".join(page_paragraphs[page_num]).strip()
            if not p_text:
                continue
            sections.append(
                {
                    "section_id": f"{paper_id}_sec_p{page_num}",
                    "heading": f"Page {page_num}",
                    "section_type": "body",
                    "page_start": page_num,
                    "page_end": page_num,
                    "text": p_text,
                    "token_count": estimate_tokens(p_text),
                }
            )

    return {
        "paper_id": paper_id,
        "title": title,
        "year": year,
        "has_heading_fallback": has_fallback,
        "sections": sections,
        "tables": tables,
    }


def parse_pdf_file(
    pdf_path: Path,
    paper_id: str,
    title: str,
    year: int,
    converter: Any,
) -> Dict[str, Any]:
    """Parse a single PDF file using Docling converter."""
    res = converter.convert(str(pdf_path))
    doc = getattr(res, "document", res)
    return parse_docling_document(doc=doc, paper_id=paper_id, title=title, year=year)


def process_all_pdfs(
    data_dir: Path = DEFAULT_DATA_DIR,
    pdfs_dir: Path = DEFAULT_PDFS_DIR,
    parsed_dir: Path = DEFAULT_PARSED_DIR,
    metadata_file: Path = DEFAULT_METADATA_FILE,
    limit: Optional[int] = None,
    force: bool = False,
    converter: Optional[Any] = None,
) -> int:
    """Parse PDFs and save structured JSON files into parsed_dir."""
    parsed_dir.mkdir(parents=True, exist_ok=True)
    metadata_map = load_metadata_map(metadata_file)

    if not metadata_map:
        logger.error(f"No metadata found in {metadata_file}. Run ingestion first.")
        return 0

    items_to_process = list(metadata_map.items())
    if limit is not None and limit > 0:
        items_to_process = items_to_process[:limit]

    logger.info(
        f"Starting Docling parsing: {len(items_to_process)} papers "
        f"(force={force}, parsed_dir={parsed_dir})"
    )

    if converter is None:
        converter = get_docling_converter()

    processed_count = 0
    skipped_count = 0

    for paper_id, meta in tqdm(items_to_process, desc="Parsing papers"):
        safe_id = sanitize_filename(paper_id)
        out_file = parsed_dir / f"{safe_id}.json"

        if out_file.exists() and not force:
            skipped_count += 1
            continue

        pdf_path = pdfs_dir / f"{safe_id}.pdf"
        if not pdf_path.exists():
            pdf_path = pdfs_dir / f"{paper_id}.pdf"

        if not pdf_path.exists():
            logger.warning(f"PDF not found for paper '{paper_id}': {pdf_path}")
            continue

        title = meta.get("title", "Untitled")
        year = meta.get("year", 0)

        try:
            parsed_data = parse_pdf_file(
                pdf_path=pdf_path,
                paper_id=paper_id,
                title=title,
                year=year,
                converter=converter,
            )
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
            processed_count += 1
        except Exception as e:
            logger.error(f"Failed to parse PDF for {paper_id}: {e}", exc_info=True)

    logger.info(
        f"Parsing finished: {processed_count} newly parsed, "
        f"{skipped_count} skipped (already cached)."
    )
    return processed_count


def print_parsing_stats(parsed_dir: Path = DEFAULT_PARSED_DIR) -> None:
    """Print limit-free sanity-check stats of all parsed JSON files in parsed_dir."""
    if not parsed_dir.exists():
        print(f"\n[Stats] Parsed directory does not exist: {parsed_dir}")
        return

    json_files = sorted(parsed_dir.glob("*.json"))
    if not json_files:
        print(f"\n[Stats] No parsed JSON files found in {parsed_dir}")
        return

    total_papers = len(json_files)
    total_sections = 0
    total_tables = 0
    total_tokens = 0
    total_fallbacks = 0

    print("\n" + "=" * 94)
    print(f"PARSING STATS SUMMARY ({total_papers} papers in {parsed_dir})")
    print("=" * 94)
    header = (
        f"{'Paper ID':<16} | {'Title':<30} | {'Secs':>4} | {'Tbls':>4} | "
        f"{'Tokens':>8} | {'Refs':<4} | {'Headings'}"
    )
    print(header)
    print("-" * 94)

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        paper_id = data.get("paper_id", jf.stem)
        title = data.get("title", "Untitled")
        truncated_title = (title[:27] + "...") if len(title) > 30 else title
        sections = data.get("sections", [])
        tables = data.get("tables", [])
        has_fallback = data.get("has_heading_fallback", False)

        sec_count = len(sections)
        tbl_count = len(tables)
        paper_tokens = sum(s.get("token_count", estimate_tokens(s.get("text", ""))) for s in sections)
        has_refs = any(s.get("section_type") == "references" for s in sections)

        total_sections += sec_count
        total_tables += tbl_count
        total_tokens += paper_tokens
        if has_fallback:
            total_fallbacks += 1

        refs_str = "Yes" if has_refs else "No"
        heading_str = "Fallback (Page)" if has_fallback else "Detected"

        print(
            f"{paper_id:<16} | {truncated_title:<30} | {sec_count:>4} | {tbl_count:>4} | "
            f"{paper_tokens:>8} | {refs_str:<4} | {heading_str}"
        )

    print("-" * 94)
    print(
        f"TOTALS: {total_papers} papers | {total_sections} sections | "
        f"{total_tables} tables | {total_tokens:,} tokens | "
        f"{total_fallbacks} papers used page fallback"
    )
    print("=" * 94 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Parse PDFs from data/pdfs using Docling into structured JSON in data/parsed/."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root data directory (default: data).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers to parse (for testing on a few papers).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-parse PDFs even if JSON output already exists in data/parsed/.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print sanity-check stats for all parsed papers and exit.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for parse_docling."""
    args = parse_args()

    parsed_dir = args.data_dir / "parsed"
    pdfs_dir = args.data_dir / "pdfs"
    metadata_file = args.data_dir / "metadata.json"

    if args.stats:
        print_parsing_stats(parsed_dir=parsed_dir)
        return

    process_all_pdfs(
        data_dir=args.data_dir,
        pdfs_dir=pdfs_dir,
        parsed_dir=parsed_dir,
        metadata_file=metadata_file,
        limit=args.limit,
        force=args.force,
    )

    # Always print limit-free stats summary after parsing
    print_parsing_stats(parsed_dir=parsed_dir)


if __name__ == "__main__":
    main()
