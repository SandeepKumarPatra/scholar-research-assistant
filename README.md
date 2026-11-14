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
│   └── parent_child.py         # Phase 2: Parent-child retrieval & deduplication (scholar_papers_v2)
├── generation/                 # RAG prompt construction and LLM generation
│   ├── __init__.py
│   └── answer.py               # Google GenAI grounded generation with citations & page ranges
├── eval/                       # Hallucination and retrieval quality evaluation
│   └── __init__.py
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
│   ├── test_generation.py      # Tests for Gemini grounding & cache
│   └── test_cli.py             # Tests for CLI entry point (naive, parent_child, and debug)
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

#### Step 2b: Build Index V2 (Parents in SQLite, Children in Qdrant)
Splits sections larger than `max_parent_tokens` (1500) into bounded parents, embeds ~300 token children with `BAAI/bge-m3` into `scholar_papers_v2`, and saves parents to SQLite `data/parents.db`:

```powershell
python ingestion/build_index_v2.py --limit 5
```

#### Step 2c: Test Parent-Child Retrieval Standalone
```powershell
# Basic retrieval
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

### 3. Phase 1: Naive Baseline (Legacy)

The original naive baseline remains completely intact:

```powershell
# Build naive index (fixed 800-token chunks into "scholar_papers")
python ingestion/build_index.py

# Retrieve with naive dense search
python retrieval/naive.py "How does RAG mitigate hallucinations?" --top-k 5
```

---

### 4. Ask Questions via CLI

Run the full grounded RAG pipeline through the command line in either mode:

```powershell
# Phase 2: Parent-Child Structure-Aware RAG (Default)
python cli.py "What causes hallucinations in large language models according to recent studies?" --mode parent_child

# Phase 2 with Debug Diagnostics
python cli.py "What causes hallucinations in large language models according to recent studies?" --mode parent_child --debug

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
  mode: "parent_child"               # Options: "naive" | "parent_child"
  top_k: 5
  collection_naive: "scholar_papers"
  collection_v2: "scholar_papers_v2"

chunking:
  child_chunk_size: 300              # Target child chunk size in tokens
  child_chunk_overlap: 30            # Child token overlap
  max_parent_tokens: 1500            # Split parents larger than this
  include_references: false          # Exclude bibliography from child search index
  include_appendix: true             # Include appendix sections

storage:
  qdrant_path: "./qdrant_data"
  parsed_dir: "data/parsed"
  parents_db: "data/parents.db"
```

---

## Architectural Highlights

Refer to [`DECISIONS.md`](DECISIONS.md) for full rationale:

- **Structure Understanding**: Uses Docling with OCR disabled and table structure enabled to extract authentic academic paper structure.
- **Parent-Child Retrieval**: Searches granular ~300 token child chunks to maximize dense retrieval precision, but returns complete parent sections (up to 1500 tokens) to provide rich, coherent context to the LLM.
- **Page Range Citations**: Tracks `page_start` and `page_end` for accurate academic citations (`[Title, Pages X-Y]`).
- **Disk Caching**: Both parsed document structures (`data/parsed/*.json`) and LLM responses (`.cache/generation_cache.json`) are cached on disk.
- **Pure SQLite Storage**: Parent records are stored in an indexed SQLite database (`data/parents.db`).
- **Zero Regression**: Baseline collection `scholar_papers` and `retrieval/naive.py` remain untouched.

---

## Running Tests

Run the complete test suite (59 unit and integration tests) using `pytest`:

```powershell
pytest -v tests/
```
