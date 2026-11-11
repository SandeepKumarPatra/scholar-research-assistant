#!/usr/bin/env python3
"""arXiv Paper Downloader for Scholar.

Downloads papers and metadata on topics like 'retrieval augmented generation'
and 'LLM hallucination' from the arXiv API while strictly observing rate limits.
"""

import argparse
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scholar.ingestion")

ARXIV_API_URL = "http://export.arxiv.org/api/query"
DEFAULT_QUERIES = [
    "retrieval augmented generation",
    "LLM hallucination",
]
DEFAULT_DELAY = 3.0  # seconds required between requests to respect arXiv rate limits
USER_AGENT = "scholar-arxiv-ingestion/1.0 (https://github.com/scholar-project; mailto:scholar@example.com)"

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class RateLimiter:
    """Ensures at least `delay` seconds between external network calls."""

    def __init__(self, delay: float = DEFAULT_DELAY):
        self.delay = delay
        self.last_call_timestamp: float = 0.0

    def wait(self) -> None:
        """Sleep if needed to ensure the minimum delay between consecutive calls."""
        now = time.time()
        elapsed = now - self.last_call_timestamp
        if elapsed < self.delay:
            sleep_duration = self.delay - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_duration:.2f}s...")
            time.sleep(sleep_duration)
        self.last_call_timestamp = time.time()


def sanitize_filename(name: str) -> str:
    """Sanitize paper id to make it safe for filesystem storage."""
    # Replace slashes and colons that might exist in old-style arXiv IDs (e.g., cs/0101001)
    clean = re.sub(r'[\\/*?:"<>|]', "_", name)
    return clean.strip()


def clean_text(text: Optional[str]) -> str:
    """Normalize whitespace and strip newlines from parsed XML text."""
    if not text:
        return ""
    return " ".join(text.split())


def extract_paper_id(raw_id: str) -> str:
    """Extract canonical arXiv ID from entry id URL.

    Example: 'http://arxiv.org/abs/2312.12345v1' -> '2312.12345v1'
    """
    if "/abs/" in raw_id:
        return raw_id.split("/abs/")[-1]
    return raw_id.rstrip("/").split("/")[-1]


def parse_atom_entry(entry: ET.Element) -> Dict[str, Any]:
    """Parse a single Atom <entry> XML element into paper metadata."""
    raw_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
    paper_id = extract_paper_id(raw_id)

    raw_title = entry.findtext("atom:title", default="", namespaces=ATOM_NS)
    title = clean_text(raw_title)

    raw_summary = entry.findtext("atom:summary", default="", namespaces=ATOM_NS)
    abstract = clean_text(raw_summary)

    published = entry.findtext("atom:published", default="", namespaces=ATOM_NS).strip()
    year = 0
    if published and len(published) >= 4 and published[:4].isdigit():
        year = int(published[:4])

    authors = [
        clean_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
        for author in entry.findall("atom:author", namespaces=ATOM_NS)
    ]
    authors = [a for a in authors if a]

    canonical_url = f"https://arxiv.org/abs/{paper_id}"

    # Extract PDF URL
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    for link in entry.findall("atom:link", namespaces=ATOM_NS):
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            link_href = link.attrib.get("href", "")
            if link_href:
                pdf_url = link_href
                if not pdf_url.endswith(".pdf"):
                    pdf_url = f"{pdf_url}.pdf"
                break

    return {
        "id": paper_id,
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": abstract,
        "url": canonical_url,
        "_pdf_url": pdf_url,
    }


def query_arxiv(
    search_query: str,
    max_results: int,
    start: int = 0,
    rate_limiter: Optional[RateLimiter] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Execute an arXiv API search query and return parsed entries and total results."""
    if rate_limiter:
        rate_limiter.wait()

    params = {
        "search_query": f'all:"{search_query}"',
        "start": start,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
    logger.info(f"Querying arXiv API: '{search_query}' (start={start}, max_results={max_results})")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            xml_data = response.read()
    except urllib.error.URLError as e:
        logger.error(f"Failed to query arXiv API: {e}")
        return [], 0

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML from arXiv: {e}")
        return [], 0

    # Total results if present
    total_results_elem = root.find(
        "{http://a9.com/-/spec/opensearch/1.1/}totalResults"
    )
    total_results = (
        int(total_results_elem.text)
        if total_results_elem is not None and total_results_elem.text
        else 0
    )

    entries = root.findall("atom:entry", namespaces=ATOM_NS)
    papers = [parse_atom_entry(entry) for entry in entries]
    return papers, total_results


def download_pdf(
    pdf_url: str,
    output_path: Path,
    rate_limiter: Optional[RateLimiter] = None,
) -> bool:
    """Download a PDF from the given URL to output_path respecting rate limits."""
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info(f"PDF already exists, skipping: {output_path.name}")
        return True

    if rate_limiter:
        rate_limiter.wait()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading PDF: {output_path.name} from {pdf_url}")

    req = urllib.request.Request(
        pdf_url,
        headers={"User-Agent": USER_AGENT},
    )

    temp_path = output_path.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "wb") as f:
            while chunk := response.read(64 * 1024):
                f.write(chunk)
        temp_path.replace(output_path)
        logger.info(f"Saved: {output_path.name} ({output_path.stat().st_size / 1024:.1f} KB)")
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.error(f"Error downloading PDF {pdf_url}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def load_metadata(metadata_file: Path) -> List[Dict[str, Any]]:
    """Load existing metadata if present."""
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read existing metadata from {metadata_file}: {e}")
    return []


def save_metadata(metadata: List[Dict[str, Any]], metadata_file: Path) -> None:
    """Save paper metadata to JSON file."""
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    # Clean internal helper keys before saving
    clean_records = []
    for item in metadata:
        clean_record = {
            "id": item.get("id", ""),
            "title": item.get("title", ""),
            "authors": item.get("authors", []),
            "year": item.get("year", 0),
            "abstract": item.get("abstract", ""),
            "url": item.get("url", ""),
        }
        clean_records.append(clean_record)

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(clean_records, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(clean_records)} metadata records to {metadata_file}")


def download_papers(
    queries: List[str],
    max_papers: int = 20,
    output_dir: Path = Path("data/pdfs"),
    metadata_file: Path = Path("data/metadata.json"),
    delay: float = DEFAULT_DELAY,
) -> List[Dict[str, Any]]:
    """Main ingestion coordinator.

    Queries arXiv across topics, downloads PDFs, and updates metadata.
    """
    rate_limiter = RateLimiter(delay=delay)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing metadata
    existing_records = load_metadata(metadata_file)
    known_ids: Set[str] = {r["id"] for r in existing_records if "id" in r}
    metadata_map: Dict[str, Dict[str, Any]] = {r["id"]: r for r in existing_records if "id" in r}

    target_per_query = max(1, (max_papers + len(queries) - 1) // len(queries))
    collected_papers: List[Dict[str, Any]] = []

    logger.info(f"Starting arXiv paper ingestion. Target: {max_papers} papers across {len(queries)} queries.")

    for query in queries:
        if len(collected_papers) >= max_papers:
            break

        remaining = max_papers - len(collected_papers)
        fetch_limit = min(target_per_query, remaining)

        papers, _ = query_arxiv(
            search_query=query,
            max_results=fetch_limit * 2,  # Fetch extra to handle potential duplicates
            rate_limiter=rate_limiter,
        )

        query_added = 0
        for paper in papers:
            paper_id = paper["id"]
            if paper_id in metadata_map:
                continue

            metadata_map[paper_id] = paper
            collected_papers.append(paper)
            query_added += 1

            if len(collected_papers) >= max_papers or query_added >= fetch_limit:
                break

    if not collected_papers:
        logger.info("No new papers to download.")
        return list(metadata_map.values())

    logger.info(f"Found {len(collected_papers)} new paper(s) to process.")

    # Download PDFs
    downloaded_count = 0
    for idx, paper in enumerate(collected_papers, 1):
        paper_id = paper["id"]
        safe_id = sanitize_filename(paper_id)
        pdf_path = output_dir / f"{safe_id}.pdf"
        pdf_url = paper.get("_pdf_url", f"https://arxiv.org/pdf/{paper_id}.pdf")

        logger.info(f"[{idx}/{len(collected_papers)}] Processing: {paper['title'][:60]}... ({paper_id})")
        success = download_pdf(pdf_url, pdf_path, rate_limiter=rate_limiter)
        if success:
            downloaded_count += 1

    # Save merged metadata
    all_metadata = list(metadata_map.values())
    save_metadata(all_metadata, metadata_file)

    logger.info(f"Ingestion complete: {downloaded_count}/{len(collected_papers)} PDFs downloaded. Total tracked papers: {len(all_metadata)}.")
    return all_metadata


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Download arXiv papers on 'retrieval augmented generation' and 'LLM hallucination'."
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=20,
        help="Maximum number of papers to download (default: 20)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory (default: data)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Delay in seconds between arXiv requests to comply with rate limits (default: 3.0)",
    )
    parser.add_argument(
        "--query",
        type=str,
        action="append",
        dest="queries",
        help="Search query to run (can be specified multiple times; defaults to 'retrieval augmented generation' and 'LLM hallucination')",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    queries = args.queries if args.queries else DEFAULT_QUERIES
    output_dir = args.data_dir / "pdfs"
    metadata_file = args.data_dir / "metadata.json"

    download_papers(
        queries=queries,
        max_papers=args.max_papers,
        output_dir=output_dir,
        metadata_file=metadata_file,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
