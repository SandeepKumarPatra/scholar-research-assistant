"""Tests for cli.py."""

import io
import unittest
from unittest.mock import patch

from cli import main


class TestCLI(unittest.TestCase):
    @patch("sys.argv", ["cli.py", "What is RAG?"])
    @patch("cli.retrieve")
    @patch("cli.answer_question")
    def test_cli_successful_run(self, mock_answer, mock_retrieve):
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

            self.assertIn("ANSWER:", output)
            self.assertIn("RAG stands for Retrieval-Augmented Generation", output)
            self.assertIn("SOURCES:", output)
            self.assertIn("[Automated Review, Page 1]", output)

    @patch("sys.argv", ["cli.py", "Unindexed question"])
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
