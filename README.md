# Scholar

**Scholar** is an academic paper ingestion, retrieval, and evaluation toolkit designed for AI research, specifically focused on Retrieval-Augmented Generation (RAG) and LLM hallucination research.

---

## Directory Structure

```text
scholar/
├── ingestion/                  # Data collection & indexing pipelines
│   ├── __init__.py
│   ├── download_arxiv.py       # arXiv paper & metadata downloader
│   └── build_index.py          # PDF text extraction, chunking & Qdrant indexing
├── retrieval/                  # Dense vector search & retrieval
│   ├── __init__.py
│   └── naive.py                # Top-5 dense retrieval using BAAI/bge-m3
├── generation/                 # RAG prompt construction and LLM generation
│   ├── __init__.py
│   └── answer.py               # Google GenAI grounded generation with citations
├── eval/                       # Hallucination and retrieval quality evaluation
│   └── __init__.py
├── app/                        # Application / UI entry points
│   └── __init__.py
├── tests/                      # Unit and integration test suite
│   ├── __init__.py
│   ├── test_download_arxiv.py  # Tests for arXiv ingestion
│   ├── test_build_index.py     # Tests for PDF extraction & chunking
│   ├── test_retrieval.py       # Tests for dense vector retrieval
│   ├── test_generation.py      # Tests for Gemini grounding, rate limits & cache
│   └── test_cli.py             # Tests for CLI entry point
├── data/                       # Ingested datasets
│   ├── pdfs/                   # Downloaded PDF papers (.gitignore tracked)
│   └── metadata.json           # JSON catalog of downloaded papers
├── qdrant_data/                # Local file-mode Qdrant vector database
├── .cache/                     # Local disk cache for generated responses
├── .env.example                # Example environment variables
├── DECISIONS.md                # Architecture and design decisions
├── requirements.txt            # Python dependencies
├── cli.py                      # CLI entry point for RAG question answering
└── README.md                   # Project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.12+ (or 3.8+)
- Windows PowerShell / Linux / macOS

### Setup Virtual Environment

```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Copy the example `.env.example` file to `.env` and insert your Gemini API Key:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

---

## Usage Guide

### 1. Ingest Papers from arXiv

Download papers on **"retrieval augmented generation"** and **"LLM hallucination"**:

```powershell
python ingestion/download_arxiv.py --max-papers 20
```

- Downloads PDFs to `data/pdfs/`.
- Saves structured metadata (`id`, `title`, `authors`, `year`, `abstract`, `url`) to `data/metadata.json`.
- Adheres to arXiv's 3-second rate limit between requests.

### 2. Build the Vector Index

Extract text per page using `pypdf`, split into fixed 800-token chunks (100 token overlap), embed with `BAAI/bge-m3`, and index into a local Qdrant collection:

```powershell
python ingestion/build_index.py
```

Optional flags:
- `--recreate`: Recreate collection from scratch.
- `--chunk-size 800`: Token chunk size (default: 800).
- `--chunk-overlap 100`: Token overlap between chunks (default: 100).
- `--qdrant-path ./qdrant_data`: Storage location for local Qdrant database.

### 3. Retrieve Contexts (Standalone)

Test dense retrieval independently:

```powershell
python retrieval/naive.py "How does RAG mitigate hallucinations?" --top-k 5
```

### 4. Ask Questions via CLI

Run the full RAG pipeline through the command line:

```powershell
python cli.py "What causes hallucinations in large language models according to recent studies?"
```

Output format:
```text
============================================================
ANSWER:
============================================================
Large language models can hallucinate due to software-level factors
as well as hardware-level soft errors [Understanding LLM Hallucinations Caused by Soft Errors, Page 1].
...

============================================================
SOURCES:
============================================================
- [Understanding LLM Hallucinations Caused by Soft Errors, Page 1]
- [Automated Literature Review Using NLP Techniques, Page 2]
============================================================
```

---

## Architectural Highlights

Refer to [`DECISIONS.md`](DECISIONS.md) for full rationale:

- **Provenance & Citations**: Chunks are tied to specific physical pages (`[title, page]`), enabling verifiable inline academic citations.
- **Strict Grounding**: The LLM is instructed to answer *strictly* from retrieved context and explicitly refuse to fabricate answers when context is insufficient.
- **Rate Throttling & Resilience**: Throttled to 12 requests/minute with exponential backoff on HTTP 429 (`RESOURCE_EXHAUSTED`).
- **Disk Caching**: Responses are cached on disk (`.cache/generation_cache.json`) keyed by a deterministic hash of the model, question, and retrieved contexts.
- **Embedded Vector DB**: Uses local file-mode Qdrant (`./qdrant_data`), requiring no external Docker containers or remote services.

---

## Running Tests

Run the complete test suite using `pytest`:

```powershell
pytest -v tests/
```

Or using standard Python `unittest`:

```powershell
python -m unittest discover tests
```
