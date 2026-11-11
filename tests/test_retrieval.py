"""Tests for retrieval/naive.py."""

import unittest
from unittest.mock import MagicMock

from retrieval.naive import retrieve


class TestNaiveRetrieval(unittest.TestCase):
    def test_retrieve_empty_query(self):
        results = retrieve("")
        self.assertEqual(results, [])

        results_spaces = retrieve("   ")
        self.assertEqual(results_spaces, [])

    def test_retrieve_with_mock_client_and_model(self):
        # Mock embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1] * 1024

        # Mock Qdrant client
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_col.name = "scholar_papers"
        mock_client.get_collections.return_value.collections = [mock_col]

        # Mock hit
        mock_hit1 = MagicMock()
        mock_hit1.score = 0.89
        mock_hit1.payload = {
            "paper_id": "2411.18583v1",
            "title": "Automated Literature Review",
            "year": 2024,
            "page": 2,
            "text": "Retrieval Augmented Generation reduces hallucination rates.",
        }

        mock_hit2 = MagicMock()
        mock_hit2.score = 0.75
        mock_hit2.payload = {
            "paper_id": "2610.22520v1",
            "title": "Understanding LLM Hallucinations",
            "year": 2026,
            "page": 1,
            "text": "Hardware soft errors also trigger hallucinations.",
        }

        mock_client.search.return_value = [mock_hit1, mock_hit2]

        results = retrieve(
            query="What causes hallucinations in LLMs?",
            top_k=5,
            client=mock_client,
            model=mock_model,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["paper_id"], "2411.18583v1")
        self.assertEqual(results[0]["title"], "Automated Literature Review")
        self.assertEqual(results[0]["page"], 2)
        self.assertAlmostEqual(results[0]["score"], 0.89)

        self.assertEqual(results[1]["paper_id"], "2610.22520v1")
        self.assertEqual(results[1]["page"], 1)

        # Check call arguments
        mock_client.search.assert_called_once()
        _, kwargs = mock_client.search.call_args
        self.assertEqual(kwargs["limit"], 5)
        self.assertEqual(kwargs["collection_name"], "scholar_papers")


if __name__ == "__main__":
    unittest.main()
