"""Unit tests for ingestion/download_arxiv.py."""

import json
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

from ingestion.download_arxiv import (
    ATOM_NS,
    DEFAULT_DELAY,
    DEFAULT_QUERIES,
    RateLimiter,
    build_arg_parser,
    clean_text,
    download_papers,
    download_pdf,
    extract_paper_id,
    load_metadata,
    parse_atom_entry,
    sanitize_filename,
    save_metadata,
)

SAMPLE_ENTRY_XML = """
<entry xmlns="http://www.w3.org/2005/Atom">
    <id>http://arxiv.org/abs/2312.00999v1</id>
    <published>2023-12-02T10:00:00Z</published>
    <title>
        Retrieval-Augmented Generation for
        Knowledge-Intensive NLP
    </title>
    <summary>
        This paper explores retrieval-augmented generation (RAG)
        to mitigate hallucinations in large language models.
    </summary>
    <author>
        <name>Alice Smith</name>
    </author>
    <author>
        <name>Bob Jones</name>
    </author>
    <link title="pdf" rel="related" type="application/pdf" href="http://arxiv.org/pdf/2312.00999v1"/>
</entry>
"""


class TestDownloadArxiv(unittest.TestCase):
    """Test suite for arXiv paper ingestion module."""

    def test_sanitize_filename(self):
        self.assertEqual(sanitize_filename("2312.00999v1"), "2312.00999v1")
        self.assertEqual(sanitize_filename("cs/0101001"), "cs_0101001")
        self.assertEqual(sanitize_filename('bad:file*name?"<>|'), "bad_file_name_____")

    def test_clean_text(self):
        raw = "   Line one\n\n   line two \t  with   spaces  "
        self.assertEqual(clean_text(raw), "Line one line two with spaces")
        self.assertEqual(clean_text(None), "")

    def test_extract_paper_id(self):
        self.assertEqual(
            extract_paper_id("http://arxiv.org/abs/2312.00999v1"), "2312.00999v1"
        )
        self.assertEqual(extract_paper_id("http://arxiv.org/abs/math/0101001"), "math/0101001")
        self.assertEqual(extract_paper_id("2312.00999"), "2312.00999")

    def test_parse_atom_entry(self):
        entry_elem = ET.fromstring(SAMPLE_ENTRY_XML)
        paper = parse_atom_entry(entry_elem)

        self.assertEqual(paper["id"], "2312.00999v1")
        self.assertEqual(
            paper["title"],
            "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
        )
        self.assertEqual(paper["authors"], ["Alice Smith", "Bob Jones"])
        self.assertEqual(paper["year"], 2023)
        self.assertIn("mitigate hallucinations", paper["abstract"])
        self.assertEqual(paper["url"], "https://arxiv.org/abs/2312.00999v1")
        self.assertEqual(paper["_pdf_url"], "http://arxiv.org/pdf/2312.00999v1.pdf")

    def test_metadata_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta_path = Path(tmp_dir) / "data" / "metadata.json"
            sample_data = [
                {
                    "id": "2312.00999v1",
                    "title": "Sample Paper",
                    "authors": ["Alice"],
                    "year": 2023,
                    "abstract": "Sample abstract",
                    "url": "https://arxiv.org/abs/2312.00999v1",
                    "_pdf_url": "http://arxiv.org/pdf/2312.00999v1",  # Internal key should be stripped
                }
            ]

            save_metadata(sample_data, meta_path)
            self.assertTrue(meta_path.exists())

            loaded = load_metadata(meta_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["id"], "2312.00999v1")
            self.assertEqual(loaded[0]["year"], 2023)
            # Internal key should not be present in saved file
            self.assertNotIn("_pdf_url", loaded[0])

    def test_rate_limiter(self):
        limiter = RateLimiter(delay=0.05)
        start = time.time()
        limiter.wait()
        limiter.wait()
        elapsed = time.time() - start
        self.assertGreaterEqual(elapsed, 0.04)

    def test_cli_argument_parsing(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        self.assertEqual(args.max_papers, 20)
        self.assertEqual(args.delay, DEFAULT_DELAY)
        self.assertEqual(args.data_dir, Path("data"))

        custom_args = parser.parse_args(
            ["--max-papers", "10", "--delay", "5.0", "--query", "hallucination"]
        )
        self.assertEqual(custom_args.max_papers, 10)
        self.assertEqual(custom_args.delay, 5.0)
        self.assertEqual(custom_args.queries, ["hallucination"])

    @patch("ingestion.download_arxiv.download_pdf")
    @patch("ingestion.download_arxiv.query_arxiv")
    def test_download_papers_coordinator(self, mock_query, mock_download):
        mock_query.return_value = (
            [
                {
                    "id": "2312.00001",
                    "title": "Paper 1",
                    "authors": ["Author 1"],
                    "year": 2023,
                    "abstract": "Abstract 1",
                    "url": "https://arxiv.org/abs/2312.00001",
                    "_pdf_url": "https://arxiv.org/pdf/2312.00001.pdf",
                },
                {
                    "id": "2312.00002",
                    "title": "Paper 2",
                    "authors": ["Author 2"],
                    "year": 2023,
                    "abstract": "Abstract 2",
                    "url": "https://arxiv.org/abs/2312.00002",
                    "_pdf_url": "https://arxiv.org/pdf/2312.00002.pdf",
                },
            ],
            2,
        )
        mock_download.return_value = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_dir = tmp_path / "pdfs"
            meta_file = tmp_path / "metadata.json"

            papers = download_papers(
                queries=["retrieval augmented generation"],
                max_papers=2,
                output_dir=out_dir,
                metadata_file=meta_file,
                delay=0.0,
            )

            self.assertEqual(len(papers), 2)
            self.assertEqual(mock_download.call_count, 2)
            self.assertTrue(meta_file.exists())


if __name__ == "__main__":
    unittest.main()
