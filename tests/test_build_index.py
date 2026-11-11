"""Tests for ingestion/build_index.py."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ingestion.build_index import (
    chunk_text,
    extract_pages_from_pdf,
    prepare_documents,
    sanitize_filename,
    simple_tokenize,
)


class TestBuildIndex(unittest.TestCase):
    def test_sanitize_filename(self):
        self.assertEqual(sanitize_filename("2411.18583v1"), "2411.18583v1")
        self.assertEqual(sanitize_filename("cs/0101001"), "cs_0101001")
        self.assertEqual(sanitize_filename("a/b:c*d?e"), "a_b_c_d_e")

    def test_chunk_text_empty(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\t  "), [])

    def test_chunk_text_short(self):
        short_text = "This is a brief text with only a few words."
        chunks = chunk_text(short_text, chunk_size=800, chunk_overlap=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], short_text)

    def test_chunk_text_sliding_window(self):
        # Create text with 250 words
        words = [f"word{i}" for i in range(250)]
        text = " ".join(words)

        # Chunk size 100, overlap 20 -> step 80
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        # 250 words / 80 step -> ~4 chunks (0-100, 80-180, 160-250)
        self.assertGreater(len(chunks), 1)
        # Check first chunk contains first word
        self.assertIn("word0", chunks[0])
        # Check last chunk contains last word
        self.assertIn("word249", chunks[-1])

    @patch("ingestion.build_index.pypdf.PdfReader")
    def test_extract_pages_from_pdf(self, mock_reader_class):
        mock_reader = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_reader.pages = [mock_page1, mock_page2]
        mock_reader_class.return_value = mock_reader

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)

        try:
            pages = extract_pages_from_pdf(pdf_path)
            self.assertEqual(len(pages), 2)
            self.assertEqual(pages[0], (1, "Page 1 content"))
            self.assertEqual(pages[1], (2, "Page 2 content"))
        finally:
            if pdf_path.exists():
                pdf_path.unlink()

    @patch("ingestion.build_index.extract_pages_from_pdf")
    def test_prepare_documents(self, mock_extract):
        mock_extract.return_value = [
            (1, "Page 1 text about retrieval augmented generation."),
            (2, "Page 2 text discussing hallucination mitigation."),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdfs_dir = Path(tmp_dir)
            # Create a fake PDF file
            (pdfs_dir / "2411.18583v1.pdf").touch()

            metadata_map = {
                "2411.18583v1": {
                    "id": "2411.18583v1",
                    "title": "Automated Literature Review",
                    "year": 2024,
                }
            }

            docs = prepare_documents(
                metadata_map=metadata_map,
                pdfs_dir=pdfs_dir,
                chunk_size=800,
                chunk_overlap=100,
            )

            self.assertEqual(len(docs), 2)
            self.assertEqual(docs[0]["metadata"]["paper_id"], "2411.18583v1")
            self.assertEqual(docs[0]["metadata"]["title"], "Automated Literature Review")
            self.assertEqual(docs[0]["metadata"]["page"], 1)
            self.assertEqual(docs[1]["metadata"]["page"], 2)


if __name__ == "__main__":
    unittest.main()
