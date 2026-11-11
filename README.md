# Scholar

**Scholar** is an academic paper ingestion, retrieval, and evaluation toolkit designed for AI research, specifically focused on Retrieval-Augmented Generation (RAG) and LLM hallucination research.

---

## Directory Structure

```text
scholar/
├── ingestion/                  # Data collection & arXiv scraping pipelines
│   ├── __init__.py
│   └── download_arxiv.py       # arXiv paper & metadata downloader
├── retrieval/                  # Indexing, embedding, and vector search modules
│   └── __init__.py
├── generation/                 # RAG prompt construction and LLM generation
│   └── __init__.py
├── eval/                       # Hallucination and retrieval quality evaluation
│   └── __init__.py
├── app/                        # Application / UI entry points
│   └── __init__.py
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   └── test_download_arxiv.py
├── data/                       # Ingested datasets (PDFs, JSON metadata)
│   ├── pdfs/                   # Downloaded PDF papers (.gitignore tracked)
│   └── metadata.json           # JSON catalog of downloaded papers
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Installation

### Prerequisites
- Python 3.8+

### Setup
Clone the repository and install dependencies:

```bash
# Optional: create a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note**: `ingestion/download_arxiv.py` uses only Python standard libraries (`urllib`, `xml.etree.ElementTree`, `json`, `pathlib`, `argparse`) and can run out-of-the-box without external dependencies.

---

## Usage

### Ingesting arXiv Papers

To download research papers on **"retrieval augmented generation"** and **"LLM hallucination"**:

```bash
python ingestion/download_arxiv.py --max-papers 20
```

#### Command-Line Options

| Argument | Default | Description |
|---|---|---|
| `--max-papers` | `20` | Total number of papers to fetch across topics. |
| `--data-dir` | `data` | Directory where `pdfs/` and `metadata.json` are stored. |
| `--delay` | `3.0` | Delay (in seconds) between requests to respect arXiv rate limits. |
| `--query` | *default queries* | Custom query string(s). Can be specified multiple times. |

#### Example: Download 50 papers with a 4-second delay

```bash
python ingestion/download_arxiv.py --max-papers 50 --delay 4.0
```

#### Example: Custom search topic

```bash
python ingestion/download_arxiv.py --max-papers 10 --query "chain of thought reasoning"
```

---

## arXiv Rate Limiting Compliance

Per the [arXiv API Terms of Use](https://arxiv.org/help/api/tou):
- Calls to the arXiv API must not exceed **one request every 3 seconds**.
- PDF downloads must be sequential with polite rate delays.
- A descriptive `User-Agent` header is sent with all requests.

The `download_arxiv.py` script enforces this automatically using a built-in `RateLimiter` class (default: 3-second delay between all API calls and PDF downloads).

---

## Output Data Format

### PDFs
PDF files are saved to `data/pdfs/{id}.pdf` (e.g. `data/pdfs/2411.18583v1.pdf`). If a paper ID contains slashes, they are sanitized to underscores (e.g. `cs_0101001.pdf`).

### Metadata (`data/metadata.json`)
Metadata for all downloaded papers is persisted in `data/metadata.json`. Duplicate downloads are prevented by checking existing IDs.

```json
[
  {
    "id": "2411.18583v1",
    "title": "Automated Literature Review Using NLP Techniques and LLM-Based Retrieval",
    "authors": [
      "Jane Doe",
      "John Smith"
    ],
    "year": 2024,
    "abstract": "In this work, we investigate retrieval-augmented generation...",
    "url": "https://arxiv.org/abs/2411.18583v1"
  }
]
```

---

## Running Tests

Run the test suite using `unittest`:

```bash
python -m unittest discover tests
```

Or using `pytest`:

```bash
pytest tests/
```
