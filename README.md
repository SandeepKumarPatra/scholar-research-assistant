# Scholar

**Scholar** is an academic paper ingestion, retrieval, and evaluation toolkit designed for AI research, specifically focused on Retrieval-Augmented Generation (RAG) and LLM hallucination research.

---

## Directory Structure

```text
scholar/
├── ingestion/                  # Data collection & indexing pipelines
│   ├── __init__.py
│   ├── download_arxiv.py       # arXiv paper & metadata downloader
│   ├── build_index.py          # Phase 1: Naive PDF extraction & Qdrant indexing
│   ├── parse_docling.py        # Phase 2: Docling structured PDF parser & disk caching
│   ├── chunk_structured.py     # Phase 2: Structure-aware parent-child chunking & SQLite store
│   ├── build_index_v2.py       # Phase 2: Child embedding & Qdrant scholar_papers_v2 indexer
│   ├── refresh_parents.py      # Standalone metadata refresh (post-References appendix & table inheritance)
│   └── reembed_tables.py       # Optional script to re-embed table child chunks into scholar_papers_v2
├── retrieval/                  # Dense vector search & retrieval
│   ├── __init__.py
│   ├── naive.py                # Phase 1: Naive dense retrieval (scholar_papers)
│   ├── parent_child.py         # Phase 2: Parent-child retrieval & deduplication (scholar_papers_v2/v3)
│   ├── bm25.py                 # Phase 3: Lexical BM25 retrieval over child chunks
│   ├── hybrid.py               # Phase 3: Dense + BM25 Reciprocal Rank Fusion (RRF)
│   └── rerank.py               # Phase 3: Cross-encoder reranking (BAAI/bge-reranker-v2-m3)
├── generation/                 # RAG prompt construction and LLM generation
│   ├── __init__.py
│   └── answer.py               # Google GenAI grounded generation with citations & page ranges
├── eval/                       # Hallucination and retrieval quality evaluation
│   └── __init__.py
├── scripts/                    # Evaluation & comparison scripts
│   └── compare_modes.py        # Phase 3: Multi-mode retrieval comparison without LLM calls
├── app/                        # Application / UI entry points
│   └── __init__.py
├── tests/                      # Unit and integration test suite
│   ├── __init__.py
│   ├── test_download_arxiv.py  # Tests for arXiv ingestion
│   ├── test_build_index.py     # Tests for naive PDF extraction & chunking
│   ├── test_parse_docling.py   # Tests for Docling parser & stats
│   ├── test_chunk_structured.py# Tests for structured parent-child chunking & SQLite
│   ├── test_build_index_v2.py  # Tests for V2 child vector indexing
│   ├── test_refresh_parents.py # Tests for metadata refresh and ID invariance
│   ├── test_reembed_tables.py  # Tests for table vector and payload re-embedding
│   ├── test_retrieval.py       # Tests for naive dense retrieval
│   ├── test_parent_child_retrieval.py # Tests for parent-child retrieval, candidate expansion & debug
│   ├── test_bm25.py            # Phase 3: Tests for BM25 tokenization, build, verification & search
│   ├── test_hybrid.py          # Phase 3: Tests for RRF fusion, parent merging & filtering
│   ├── test_rerank.py          # Phase 3: Tests for cross-encoder reranking, token truncation & disk cache
│   ├── test_compare_modes.py   # Phase 3: Tests for multi-mode retrieval benchmark runner
│   ├── test_generation.py      # Tests for Gemini grounding & cache
│   └── test_cli.py             # Tests for CLI entry point across all 4 modes and debug
├── data/                       # Ingested datasets (.gitignore tracked)
│   ├── pdfs/                   # Downloaded PDF papers
│   ├── parsed/                 # Phase 2: Docling parsed JSON files
│   ├── parents.db              # Phase 2: SQLite database of parent sections & tables
│   └── metadata.json           # JSON catalog of downloaded papers
├── qdrant_data/                # Local file-mode Qdrant vector database
├── .cache/                     # Local disk cache for generated responses
├── .env.example                # Example environment variables
├── config.yaml                 # Central project configuration & ablation toggles
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

Copy `.env.example` to `.env` and set your Gemini API Key:

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

---

### 2. Phase 2: Structured Parsing & Parent-Child Pipeline

Phase 2 replaces naive sliding-window chunking with structure-aware parent-child chunking.

#### Step 2a: Parse PDFs with Docling (Cached to Disk)
Parse PDFs into structured JSON under `data/parsed/` (sections, headings, body text, tables in markdown, page numbers):

```powershell
# Parse papers (OCR disabled for speed, tables enabled)
python ingestion/parse_docling.py --limit 5

# Sanity-check stats across all parsed papers
python ingestion/parse_docling.py --stats
```

#### Step 2b: Build Index V3 (Sentence-Boundary Chunks into scholar_papers_v3)
Splits sections larger than `max_parent_tokens` (1500) into bounded parents, packs whole sentences up to ~300 tokens with 1-sentence overlap (and word-boundary splitting for over-long sentences) into `scholar_papers_v3`, and saves parents to SQLite `data/parents.db`:

```powershell
# Build new v3 collection (default: scholar_papers_v3)
python ingestion/build_index_v2.py --collection scholar_papers_v3 --limit 5

# Or build into scholar_papers_v2 for ablation comparison
python ingestion/build_index_v2.py --collection scholar_papers_v2 --limit 5
```

#### Step 2c: Test Parent-Child Retrieval Standalone
```powershell
# Basic retrieval (defaults to scholar_papers_v3 from config.yaml)
python retrieval/parent_child.py "How does RAG mitigate hallucinations?" --top-k 5

# Diagnostic debug mode (prints top 10 child hits before parent merging)
python retrieval/parent_child.py "How does RAG mitigate hallucinations?" --top-k 5 --debug
```

#### Step 2d: Refresh Metadata (Post-References Appendices & Table Labels)
Refreshes `data/parsed/*.json` and `data/parents.db` without modifying Qdrant or re-embedding:
```powershell
python ingestion/refresh_parents.py
```
- Creates a timestamped backup (`data/parents_<timestamp>.db.bak`).
- Tags all sections after References heading as `appendix`.
- Formats table parents as `"Table N"` short labels and preserves full captions in `text`.
- Tables inherit `section_type` from the enclosing section (tables after References become `appendix`).
- Verifies ID invariance (aborts and restores backup on mismatch).
- Prints a before/after comparison table with token percentages and `>40%` warnings.

#### Step 2e: Optional Table Re-embedding
To update vector representations of table chunks with their full captions:
```powershell
python ingestion/reembed_tables.py --limit 5
```

---

### 3. Phase 3: Hybrid Search & Cross-Encoder Reranking

Phase 3 builds lexical BM25 and cross-encoder reranking on top of the `scholar_papers_v3` child chunks.

#### Step 3a: Build BM25 Child Index
Builds a persistent BM25 index over the child chunks in `scholar_papers_v3`, tokenized to lowercase with hyphen and number preservation:
```powershell
python retrieval/bm25.py --build
```
- Verifies child counts and IDs match the Qdrant collection on load.
- Saves index to `data/bm25_index.pkl`.

#### Step 3b: Test Hybrid & Rerank Standalone
```powershell
# BM25 standalone search over child chunks
python retrieval/bm25.py "ChainPoll hallucination" --top-k 10

# Hybrid dense + BM25 RRF fusion
python retrieval/hybrid.py "What is ChainPoll?" --top-k 5 --debug

# Hybrid + Cross-Encoder reranking
python retrieval/rerank.py "What is ChainPoll?" --top-k 5 --debug
```

#### Step 3c: Compare All Modes (Zero LLM Calls)
Run retrieval-only comparison across all 4 modes on the 5 benchmark questions:
```powershell
python scripts/compare_modes.py
```
- Outputs timing metrics and top-5 parent sources per mode to `compare_modes.md`.

---

### 4. Phase 1: Naive Baseline (Legacy)

The original naive baseline remains completely intact:

```powershell
# Build naive index (fixed 800-token chunks into "scholar_papers")
python ingestion/build_index.py

# Retrieve with naive dense search
python retrieval/naive.py "How does RAG mitigate hallucinations?" --top-k 5
```

---

### 5. Ask Questions via CLI

Run the full grounded RAG pipeline through the command line in any mode:

```powershell
# Phase 3: Hybrid Rerank (Recommended)
python cli.py "What is ChainPoll?" --mode hybrid_rerank

# Phase 3: Hybrid Rerank with Stage Diagnostics
python cli.py "What is ChainPoll?" --mode hybrid_rerank --debug

# Phase 3: Hybrid RRF (Dense + BM25)
python cli.py "What is ChainPoll?" --mode hybrid

# Phase 2: Parent-Child Structure-Aware RAG
python cli.py "What causes hallucinations in large language models according to recent studies?" --mode parent_child

# Phase 1: Naive Baseline RAG
python cli.py "What causes hallucinations in large language models according to recent studies?" --mode naive
```

Output format:
```text
============================================================
ANSWER:
============================================================
Large language models hallucinate due to software-level factors
as well as hardware soft errors [Understanding LLM Hallucinations Caused by Soft Errors, Pages 1-3].
...

============================================================
SOURCES:
============================================================
- [Understanding LLM Hallucinations Caused by Soft Errors, Pages 1-3]
- [Automated Literature Review Using NLP Techniques, Page 2]
============================================================
```

---

## Configuration & Ablation Studies

Central settings can be adjusted in [`config.yaml`](config.yaml) to run controlled ablation experiments:

```yaml
retrieval:
  mode: "parent_child"               # Options: "naive" | "parent_child" | "hybrid" | "hybrid_rerank"
  top_k: 5
  collection_naive: "scholar_papers"
  collection_v2: "scholar_papers_v2"
  collection_v3: "scholar_papers_v3"
  use_bm25: true
  use_reranker: true
  rrf_k: 60
  dense_candidates: 50
  bm25_candidates: 50
  rerank_top_n: 20
  max_parents_per_paper: null        # null = off; integer limits parents from same paper

chunking:
  child_chunk_size: 300              # Target child chunk size in tokens
  child_chunk_overlap: 30            # Child token overlap
  max_parent_tokens: 1500            # Split parents larger than this
  include_references: false          # Exclude bibliography from child search index
  include_appendix: false            # Include appendix sections

storage:
  qdrant_path: "./qdrant_data"
  parsed_dir: "data/parsed"
  parents_db: "data/parents.db"
  bm25_index: "data/bm25_index.pkl"
  rerank_cache: ".cache/rerank_cache.json"

models:
  embedding: "BAAI/bge-m3"
  reranker: "BAAI/bge-reranker-v2-m3"
```

---

## Architectural Highlights

Refer to [`DECISIONS.md`](DECISIONS.md) for full rationale:

- **Structure Understanding**: Uses Docling with OCR disabled and table structure enabled to extract authentic academic paper structure.
- **Parent-Child Retrieval**: Searches granular ~300 token child chunks to maximize dense retrieval precision, but returns complete parent sections (up to 1500 tokens) to provide rich, coherent context to the LLM.
- **Lexical BM25 & Hyphen Preservation**: Regex tokenization preserves domain terms (e.g. `bge-reranker-v2-m3`, `gpt-3.5-turbo`) for keyword precision.
- **Reciprocal Rank Fusion (RRF)**: Combines dense semantic candidate ranks and lexical BM25 ranks ($k=60$) to surface relevant evidence before reranking.
- **Cross-Encoder Reranking**: Evaluates query against individual child texts (truncated to 512 tokens) with local disk score caching.
- **Page Range Citations**: Tracks `page_start` and `page_end` for accurate academic citations (`[Title, Pages X-Y]`).
- **Disk Caching**: Parsed document structures (`data/parsed/*.json`), LLM responses (`.cache/generation_cache.json`), and reranking scores (`.cache/rerank_cache.json`) are cached on disk.
- **Pure SQLite Storage**: Parent records are stored in an indexed SQLite database (`data/parents.db`).
- **Zero Regression**: Baseline collection `scholar_papers` and `retrieval/naive.py` remain untouched.

---

## Running Tests

Run the complete test suite (81 unit and integration tests) using `pytest`:

```powershell
pytest -v tests/
```
