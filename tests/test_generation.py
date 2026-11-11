"""Tests for generation/answer.py (with mocked Gemini calls)."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from generation.answer import (
    DiskResponseCache,
    RateLimiter,
    answer_question,
    build_prompt,
    call_gemini_with_retry,
    is_rate_limit_error,
)


class TestGeneration(unittest.TestCase):
    def test_rate_limiter_interval(self):
        limiter = RateLimiter(max_per_minute=600)  # 0.1s interval for fast test
        t0 = time.time()
        limiter.wait()
        limiter.wait()
        elapsed = time.time() - t0
        self.assertGreaterEqual(elapsed, 0.08)

    def test_disk_response_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "test_cache.json"
            cache = DiskResponseCache(cache_file=cache_file)

            contexts = [
                {
                    "paper_id": "2411.18583v1",
                    "title": "Automated Review",
                    "page": 1,
                    "text": "Some text",
                }
            ]

            # Cache miss
            self.assertIsNone(cache.get("gemini-model", "test question", contexts))

            # Cache set
            cache.set("gemini-model", "test question", contexts, "Cached answer text")

            # Cache hit
            ans = cache.get("gemini-model", "test question", contexts)
            self.assertEqual(ans, "Cached answer text")

            # Different question -> cache miss
            self.assertIsNone(cache.get("gemini-model", "different question", contexts))

    def test_build_prompt_structure(self):
        contexts = [
            {
                "paper_id": "2411.18583v1",
                "title": "Automated Literature Review",
                "page": 3,
                "text": "RAG reduces hallucinations by grounding answers in retrieved documents.",
            }
        ]
        prompt = build_prompt("How does RAG help?", contexts)

        self.assertIn("Automated Literature Review, Page 3", prompt)
        self.assertIn("How does RAG help?", prompt)
        self.assertIn("Answer ONLY from the retrieved context", prompt)
        self.assertIn("[Title, Page X]", prompt)

    def test_is_rate_limit_error(self):
        class Mock429Error(Exception):
            def __init__(self):
                self.code = 429

        self.assertTrue(is_rate_limit_error(Mock429Error()))
        self.assertTrue(is_rate_limit_error(Exception("429 Resource has been exhausted")))
        self.assertTrue(is_rate_limit_error(Exception("RESOURCE_EXHAUSTED")))
        self.assertFalse(is_rate_limit_error(Exception("Invalid argument: bad prompt")))

    def test_call_gemini_with_retry_success(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is the generated answer."
        mock_client.models.generate_content.return_value = mock_response

        limiter = RateLimiter(max_per_minute=6000)
        answer = call_gemini_with_retry(
            client=mock_client,
            model="gemini-2.5-flash",
            prompt="Hello?",
            rate_limiter=limiter,
        )

        self.assertEqual(answer, "This is the generated answer.")
        mock_client.models.generate_content.assert_called_once()

    def test_call_gemini_with_retry_on_429(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Answer after 429 retry."

        # Fail once with 429, then succeed
        mock_client.models.generate_content.side_effect = [
            Exception("429 RESOURCE_EXHAUSTED"),
            mock_response,
        ]

        limiter = RateLimiter(max_per_minute=6000)
        answer = call_gemini_with_retry(
            client=mock_client,
            model="gemini-2.5-flash",
            prompt="Test 429 retry",
            rate_limiter=limiter,
            max_retries=2,
            initial_backoff=0.01,  # Fast backoff for tests
        )

        self.assertEqual(answer, "Answer after 429 retry.")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)

    def test_answer_question_end_to_end_mocked(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            "RAG mitigates hallucinations [Automated Literature Review, Page 2]."
        )
        mock_client.models.generate_content.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = DiskResponseCache(cache_file=Path(tmp_dir) / "cache.json")
            limiter = RateLimiter(max_per_minute=6000)

            contexts = [
                {
                    "paper_id": "2411.18583v1",
                    "title": "Automated Literature Review",
                    "page": 2,
                    "text": "RAG reduces hallucinations by grounding answers.",
                }
            ]

            # 1. Fresh call
            ans, sources = answer_question(
                question="How does RAG reduce hallucination?",
                contexts=contexts,
                model="gemini-2.5-flash",
                client=mock_client,
                cache=cache,
                rate_limiter=limiter,
            )

            self.assertEqual(
                ans, "RAG mitigates hallucinations [Automated Literature Review, Page 2]."
            )
            self.assertEqual(sources, ["[Automated Literature Review, Page 2]"])
            self.assertEqual(mock_client.models.generate_content.call_count, 1)

            # 2. Cached call -> should not invoke mock_client again
            ans2, sources2 = answer_question(
                question="How does RAG reduce hallucination?",
                contexts=contexts,
                model="gemini-2.5-flash",
                client=mock_client,
                cache=cache,
                rate_limiter=limiter,
            )

            self.assertEqual(ans2, ans)
            self.assertEqual(sources2, sources)
            self.assertEqual(mock_client.models.generate_content.call_count, 1)


if __name__ == "__main__":
    unittest.main()
