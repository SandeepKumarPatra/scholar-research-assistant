"""Tests for cli.py."""

import io
import unittest
from unittest.mock import patch

from cli import main


class TestCLI(unittest.TestCase):
    @patch("sys.argv", ["cli.py", "What is RAG?", "--mode", "naive"])
    @patch("cli.retrieve")
    @patch("cli.answer_question")
    def test_cli_naive_run(self, mock_answer, mock_retrieve):
        mock_retrieve.return_value = [
            {
                "paper_id": "2411.18583v1",
                "title": "Automated Review",
                "page": 1,
                "text": "RAG snippet",
                "score": 0.85,
            }
        ]
        mock_answer.return_value = (
            "RAG stands for Retrieval-Augmented Generation [Automated Review, Page 1].",
            ["[Automated Review, Page 1]"],
        )

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main()
            output = mock_stdout.getvalue()

            self.assertTrue(mock_retrieve.called)
            self.assertIn("ANSWER:", output)
            self.assertIn("RAG stands for Retrieval-Augmented Generation", output)
            self.assertIn("SOURCES:", output)
            self.assertIn("[Automated Review, Page 1]", output)

    @patch("sys.argv", ["cli.py", "What causes hallucination?", "--mode", "parent_child"])
    @patch("cli.retrieve_parent_child")
    @patch("cli.answer_question")
    def test_cli_parent_child_run(self, mock_answer, mock_ret_pc):
        mock_ret_pc.return_value = [
            {
                "paper_id": "2411.18583v1",
                "title": "Understanding Hallucinations",
                "section": "1 Introduction",
                "page_start": 1,
                "page_end": 2,
                "page": 1,
                "text": "Parent section context.",
                "score": 0.92,
            }
        ]
        mock_answer.return_value = (
            "Hallucinations occur due to soft errors [Understanding Hallucinations, Pages 1-2].",
            ["[Understanding Hallucinations, Pages 1-2]"],
        )

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main()
            output = mock_stdout.getvalue()

            self.assertTrue(mock_ret_pc.called)
            self.assertIn("ANSWER:", output)
            self.assertIn("Hallucinations occur due to soft errors", output)
            self.assertIn("SOURCES:", output)
            self.assertIn("[Understanding Hallucinations, Pages 1-2]", output)

    @patch("sys.argv", ["cli.py", "Unindexed question", "--mode", "naive"])
    @patch("cli.retrieve")
    def test_cli_no_contexts(self, mock_retrieve):
        mock_retrieve.return_value = []

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout, self.assertRaises(SystemExit) as cm:
            main()

        self.assertEqual(cm.exception.code, 0)
        output = mock_stdout.getvalue()
        self.assertIn("No relevant contexts found", output)


if __name__ == "__main__":
    unittest.main()
