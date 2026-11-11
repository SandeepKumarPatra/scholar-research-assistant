# Architecture and Design Decisions (DECISIONS.md)

This document records the design and architectural choices for the **scholar** naive RAG baseline.

---

## 1. Document Extraction & Preprocessing

- **Tool**: `pypdf` (`PdfReader`)
- **Granularity**: Text extracted page-by-page.
- **Rationale**: Extracting text per page ensures that every chunk is directly tied to an unambiguous physical page number in the original PDF. This guarantees reliable citations formatted as `[title, page]` without heuristic page-boundary guessing.

---

## 2. Chunking Strategy

- **Chunk Size**: 800 tokens
- **Chunk Overlap**: 100 tokens
- **Stride**: 700 tokens
- **Tokenizer**: `BAAI/bge-m3` tokenizer via Hugging Face `transformers` (with a regex/whitespace fallback when offline).
- **Scope**: Chunking is scoped to each page (or contiguous text per page). If a page has fewer than 800 tokens, it forms a single chunk. If it exceeds 800 tokens, a sliding window of 800 tokens with 100 tokens of overlap is applied.
- **Payload Metadata**: Each chunk stores:
  - `id` (paper ID, e.g. `2411.18583v1`)
  - `title`
  - `year`
  - `page` (1-based physical page number)
  - `chunk_index` (index of chunk within the paper/page)
  - `text` (raw text chunk for LLM prompt context)

---

## 3. Embedding Model & Vector Storage

- **Embedding Model**: `BAAI/bge-m3` via `sentence-transformers`
  - Output dimension: 1024
  - Metric: Cosine similarity
- **Vector Database**: `Qdrant` in embedded file mode (`QdrantClient(path="./qdrant_data")`)
  - Collection name: `scholar_papers`
  - Point IDs: Deterministic UUID5 derived from `{paper_id}_{page}_{chunk_index}` to ensure idempotent re-indexing.
- **Rationale**: Local file mode requires no external server/daemon, runs on Windows PowerShell out-of-the-box, persists across executions, and provides fast local search.

---

## 4. Retrieval

- **Approach**: Naive dense retrieval.
- **Depth**: Top 5 chunks (`top_k=5`) ranked by cosine similarity against the query embedding.
- **Output**: Ranked chunks containing text and metadata (`paper_id`, `title`, `year`, `page`, `score`).

---

## 5. Generation & LLM Integration

- **SDK**: Official `google-genai` SDK (`from google import genai; client = genai.Client()`).
- **Configuration**:
  - Model loaded dynamically from `GEMINI_MODEL` in `.env` (defaults to `gemini-2.5-flash`).
  - API Key loaded from `GEMINI_API_KEY` in `.env` or system environment.
- **Grounding & Citation Policy**:
  - Strict system prompt instructing the model to answer *only* using information provided in the retrieved context.
  - If the context does not contain enough information to answer, state clearly: *"I cannot answer this question based on the provided papers."*
  - Sources must be cited as `[title, page]` corresponding to the chunks provided in context.

---

## 6. Rate Limiting, Throttling & Resilience

- **Throttle**: Hard rate limit of **12 requests per minute** (minimum 5.0 seconds spacing between outgoing calls).
- **Retry on 429**: Catches HTTP 429 (`RESOURCE_EXHAUSTED` / `APIError`) and retries using exponential backoff with jitter (initial delay 5s, doubling up to 4 attempts).
- **Disk Caching**:
  - Cache location: `.cache/generation_cache.json`.
  - Key: SHA256 hash of `(model_name + question + sorted context hashes)`.
  - Cache hits bypass the Gemini API entirely and do not consume rate limit quota.

---

## 7. CLI Interface

- Simple command-line invocation:
  ```bash
  python cli.py "How does retrieval augmented generation reduce hallucination?"
  ```
- Outputs the generated answer along with an explicit list of source references `[title, page]`.
