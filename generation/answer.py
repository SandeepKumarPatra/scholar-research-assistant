#!/usr/bin/env python3
"""Generation module for scholar.

Generates answers using Google GenAI SDK (google-genai), grounded strictly in
retrieved contexts, citing sources as [title, page], with 12 req/min throttling,
exponential backoff on HTTP 429, and disk response caching.
"""

import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger("scholar.generation")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CACHE_FILE_PATH = Path(".cache/generation_cache.json")
MAX_REQUESTS_PER_MINUTE = 12
MIN_REQUEST_INTERVAL = 60.0 / MAX_REQUESTS_PER_MINUTE  # 5.0 seconds


class RateLimiter:
    """Enforces a maximum of 12 requests per minute and minimum spacing."""

    def __init__(self, max_per_minute: int = MAX_REQUESTS_PER_MINUTE):
        self.max_per_minute = max_per_minute
        self.min_interval = 60.0 / max_per_minute
        self.last_call_time = 0.0
        self.call_timestamps: List[float] = []

    def wait(self) -> None:
        """Throttle requests to satisfy the rate limit."""
        now = time.time()

        # Enforce minimum interval between consecutive requests
        elapsed_since_last = now - self.last_call_time
        if elapsed_since_last < self.min_interval:
            sleep_needed = self.min_interval - elapsed_since_last
            logger.info(f"Throttling: sleeping {sleep_needed:.2f}s for minimum interval...")
            time.sleep(sleep_needed)
            now = time.time()

        # Sliding window check for max_per_minute in last 60 seconds
        window_start = now - 60.0
        self.call_timestamps = [t for t in self.call_timestamps if t > window_start]

        if len(self.call_timestamps) >= self.max_per_minute:
            sleep_needed = 60.0 - (now - self.call_timestamps[0]) + 0.1
            if sleep_needed > 0:
                logger.info(f"Throttling: rate limit ({self.max_per_minute}/min) reached. Sleeping {sleep_needed:.2f}s...")
                time.sleep(sleep_needed)
                now = time.time()
                self.call_timestamps = [t for t in self.call_timestamps if t > (now - 60.0)]

        self.last_call_time = now
        self.call_timestamps.append(now)


class DiskResponseCache:
    """Simple thread-safe JSON disk cache for LLM generations."""

    def __init__(self, cache_file: Path = CACHE_FILE_PATH):
        self.cache_file = cache_file
        self._ensure_cache_file()

    def _ensure_cache_file(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_file.exists():
            try:
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump({}, f)
            except OSError as e:
                logger.warning(f"Could not initialize cache file: {e}")

    def _compute_key(self, model: str, question: str, contexts: List[Dict[str, Any]]) -> str:
        # Create deterministic fingerprint of question and context chunks
        context_fingerprint = "|".join(
            f"{c.get('paper_id')}:{c.get('page')}:{hashlib.md5(c.get('text', '').encode('utf-8')).hexdigest()}"
            for c in contexts
        )
        combined = f"{model}:::{question.strip()}:::{context_fingerprint}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def get(self, model: str, question: str, contexts: List[Dict[str, Any]]) -> Optional[str]:
        if not self.cache_file.exists():
            return None
        key = self._compute_key(model, question, contexts)
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                entry = data.get(key)
                if entry and isinstance(entry, dict):
                    return entry.get("answer")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read response cache: {e}")
        return None

    def set(self, model: str, question: str, contexts: List[Dict[str, Any]], answer: str) -> None:
        key = self._compute_key(model, question, contexts)
        self._ensure_cache_file()
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

        data[key] = {
            "answer": answer,
            "timestamp": time.time(),
            "model": model,
            "question": question,
        }

        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.warning(f"Failed to write response to cache: {e}")


# Global instances
_global_rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)
_global_cache = DiskResponseCache()


def format_citation(ctx: Dict[str, Any]) -> str:
    """Format citation string supporting single page or page ranges."""
    title = ctx.get("title", "Unknown Title")
    p_start = ctx.get("page_start", ctx.get("page", 1))
    p_end = ctx.get("page_end", p_start)
    if p_start == p_end:
        return f"[{title}, Page {p_start}]"
    return f"[{title}, Pages {p_start}-{p_end}]"


def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    """Build grounded RAG prompt strictly enforcing source citations."""
    context_sections = []
    for idx, ctx in enumerate(contexts, 1):
        cite_str = format_citation(ctx)
        paper_id = ctx.get("paper_id", "")
        text = ctx.get("text", "").strip()
        context_sections.append(
            f"--- SOURCE {cite_str} (ID: {paper_id}) ---\n{text}"
        )

    context_str = "\n\n".join(context_sections)
    sample_cite = format_citation(contexts[0]) if contexts else "[Title, Page 1]"

    prompt = f"""You are a rigorous academic AI research assistant for the Scholar project.
Your task is to answer the user's question using ONLY the provided retrieved context snippets below.

CRITICAL INSTRUCTIONS:
1. Grounding: Answer ONLY from the retrieved context. Do NOT extrapolate or assume facts not present.
2. If the context does not contain enough information to answer the question, state:
   "I cannot answer this question based on the provided papers."
3. Citations: You MUST cite every key claim or fact using the exact format:
   [Title, Page X] or [Title, Pages X-Y] (e.g. {sample_cite})
   matching the sources provided in the context.

Retrieved Context:
{context_str}

Question: {question}

Answer:"""
    return prompt


def is_rate_limit_error(error: Exception) -> bool:
    """Check if exception represents an HTTP 429 / RESOURCE_EXHAUSTED rate limit error."""
    error_str = str(error)
    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
        return True
    code = getattr(error, "code", None)
    if code in (429, "429", "RESOURCE_EXHAUSTED"):
        return True
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True
    return False


def call_gemini_with_retry(
    client: Any,
    model: str,
    prompt: str,
    rate_limiter: RateLimiter,
    max_retries: int = 4,
    initial_backoff: float = 5.0,
) -> str:
    """Call Google GenAI SDK generate_content with throttling and exponential backoff on 429."""
    backoff = initial_backoff

    for attempt in range(max_retries + 1):
        rate_limiter.wait()
        try:
            logger.info(f"Calling Gemini API (model={model}, attempt={attempt + 1})...")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            return str(response).strip()

        except Exception as e:
            if is_rate_limit_error(e) and attempt < max_retries:
                jitter = random.uniform(0.5, 1.5)
                sleep_time = backoff + jitter
                logger.warning(
                    f"Received 429/Rate Limit error: {e}. "
                    f"Retrying in {sleep_time:.2f}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(sleep_time)
                backoff *= 2.0
            else:
                logger.error(f"Gemini API invocation failed: {e}")
                raise


def answer_question(
    question: str,
    contexts: List[Dict[str, Any]],
    model: Optional[str] = None,
    client: Optional[Any] = None,
    cache: Optional[DiskResponseCache] = None,
    rate_limiter: Optional[RateLimiter] = None,
    use_cache: bool = True,
) -> Tuple[str, List[str]]:
    """Generate answer from retrieved context with caching, citations, and throttling.

    Returns:
        Tuple of (answer_text, formatted_sources_list)
    """
    model_name = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    cache = cache or _global_cache
    rate_limiter = rate_limiter or _global_rate_limiter

    # Prepare formatted sources list for reference
    unique_sources = []
    seen = set()
    for ctx in contexts:
        cite_str = format_citation(ctx)
        if cite_str not in seen:
            seen.add(cite_str)
            unique_sources.append(cite_str)

    if not contexts:
        return "I cannot answer this question because no relevant context was found.", []

    # Check disk cache
    if use_cache:
        cached_answer = cache.get(model_name, question, contexts)
        if cached_answer is not None:
            logger.info("Found cached answer on disk. Returning without API call.")
            return cached_answer, unique_sources

    prompt = build_prompt(question, contexts)

    # Initialize google-genai client if not provided
    if client is None:
        try:
            from google import genai

            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                client = genai.Client(api_key=api_key)
            else:
                client = genai.Client()
        except Exception as e:
            logger.error(f"Failed to initialize google-genai client: {e}")
            raise RuntimeError(
                f"Could not initialize Google GenAI Client: {e}. "
                "Ensure GEMINI_API_KEY is configured in your .env file."
            ) from e

    answer = call_gemini_with_retry(
        client=client,
        model=model_name,
        prompt=prompt,
        rate_limiter=rate_limiter,
    )

    # Cache response to disk
    if use_cache:
        cache.set(model_name, question, contexts, answer)

    return answer, unique_sources
