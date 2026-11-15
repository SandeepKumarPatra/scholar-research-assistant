# Architecture and Design Decisions (DECISIONS.md)

This document records the design and architectural choices for the **scholar** project, spanning Phase 1 (Naive Baseline) and Phase 2 (Structured Parsing & Parent-Child Chunking).

---

## 1. Document Extraction & Preprocessing (Phase 1 Baseline)

- **Tool**: `pypdf` (`PdfReader`)
- **Granularity**: Text extracted page-by-page.
- **Rationale**: Extracting text per page ensures that every chunk is directly tied to an unambiguous physical page number in the original PDF. This guarantees reliable citations formatted as `[title, page]` without heuristic page-boundary guessing.

---

## 2. Chunking Strategy (Phase 1 Baseline)

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

## 3. Embedding Model & Vector Storage (Phase 1 Baseline)

- **Embedding Model**: `BAAI/bge-m3` via `sentence-transformers`
  - Output dimension: 1024
  - Metric: Cosine similarity
- **Vector Database**: `Qdrant` in embedded file mode (`QdrantClient(path="./qdrant_data")`)
  - Collection name: `scholar_papers`
  - Point IDs: Deterministic UUID5 derived from `{paper_id}_{page}_{chunk_index}` to ensure idempotent re-indexing.
- **Rationale**: Local file mode requires no external server/daemon, runs on Windows PowerShell out-of-the-box, persists across executions, and provides fast local search.

---

## 4. Retrieval (Phase 1 Baseline)

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
  - Sources cited as `[title, page]` or `[title, pages X-Y]` corresponding to retrieved contexts.

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

- Command-line invocation supporting `--mode naive` and `--mode parent_child`:
  ```bash
  python cli.py "How does retrieval augmented generation reduce hallucination?" --mode parent_child
  ```
- Outputs the generated answer along with an explicit list of source references.

---

## 8. Structured PDF Parsing with Docling (Phase 2)

- **Tool**: `docling` (`DocumentConverter`)
- **OCR Disabled**: Born-digital academic PDFs have embedded text streams. Disabling OCR (`pipeline_options.do_ocr = False`) dramatically accelerates parsing speed, reduces memory consumption, and avoids pulling heavy OCR model weights.
- **Table Structure Extraction**: Enabled (`pipeline_options.do_table_structure = True`). Tables are parsed into structured markdown to retain cell relationships and column alignments.
- **Section & Heading Identification**: Traverses Docling document items, identifying section headers, paragraphs, and page boundaries (`page_start`, `page_end`).
- **Heading Fallback**: If Docling detects no headings for a paper (e.g. unconventional formatting), the parser automatically falls back to 1 parent per page and emits a warning log.
- **Disk Caching**: Parsed outputs are serialized as JSON in `data/parsed/{paper_id}.json`. Subsequent indexing or chunking reads from disk cache, ensuring expensive Docling parsing executes only once per paper.
- **Metadata Authority**: Paper `title` and `year` are pulled directly from `data/metadata.json` rather than inferred from PDF cover heuristics.
- **Section Classification**: Each section is tagged with `section_type`:
  - `"body"`: Core academic content.
  - `"references"`: Bibliography and works cited sections.
  - `"appendix"`: Supplementary materials and appendices.

---

## 9. Structure-Aware Parent-Child Chunking (Phase 2)

- **Hierarchical Representation**:
  - **Parent**: Represents the full semantic context of an entire section or table.
  - **Child**: Granular ~300 token slices of the parent section, embedded for dense vector search.
- **Parent Splitting Cap (`max_parent_tokens: 1500`)**:
  - Sections larger than 1500 tokens are split into bounded parent chunks (`_part0`, `_part1`, etc.).
  - Prevents oversized sections from dominating the LLM context window during retrieval.
  - Child chunks are linked directly to their respective parent slice.
- **Table Handling**:
  - Tables are maintained as standalone parent and child units in markdown format, preventing tabular data from fragmenting across arbitrary token windows.
- **Reference Section Filtering**:
  - Reference sections are preserved in the parsed JSON but excluded from the child index by default (`include_references: false`). This prevents dense retrieval from surfacing raw bibliography entries instead of meaningful explanatory prose.

---

## 10. Vector & Parent Storage Architecture (Phase 2)

- **Isolation from Baseline**:
  - The naive baseline Qdrant collection (`scholar_papers`) and `retrieval/naive.py` remain untouched.
  - New child vectors are indexed into a separate Qdrant collection: `scholar_papers_v2`.
- **Pure SQLite Parent Store**:
  - Parent records are stored exclusively in `data/parents.db` with primary key indexing on `parent_id`.
  - Schema:
    ```sql
    CREATE TABLE parents (
        parent_id TEXT PRIMARY KEY,
        paper_id TEXT NOT NULL,
        title TEXT NOT NULL,
        year INTEGER NOT NULL,
        section TEXT NOT NULL,
        section_type TEXT NOT NULL,
        page_start INTEGER NOT NULL,
        page_end INTEGER NOT NULL,
        text TEXT NOT NULL,
        token_count INTEGER NOT NULL,
        is_table INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX idx_parents_paper ON parents(paper_id);
    ```
  - Enables $O(1)$ indexed retrieval of complete parent sections without memory bloat.

---

## 11. Parent-Child Retrieval & Citation Ranges (Phase 2)

- **Search Workflow**:
  1. Dense query embedding using `BAAI/bge-m3`.
  2. Candidate search in `scholar_papers_v2` over granular child chunks.
  3. Extraction of `parent_id`s and aggregation of the maximum child similarity score per parent.
  4. Lookup of deduplicated parent sections from `data/parents.db`.
  5. Ranking parents by score descending and slicing to `top_k`.
- **Page Range Citations**:
  - Citations dynamically adapt to page coverage:
    - Single page: `[{title}, Page {page_start}]`
    - Multi-page section: `[{title}, Pages {page_start}-{page_end}]`

---

## 12. Configuration & Ablation Framework

- Centralized configuration in `config.yaml`:
  - `retrieval.mode`: `"parent_child"` or `"naive"`
  - `retrieval.top_k`: Context budget
  - `chunking.child_chunk_size`: 300
  - `chunking.max_parent_tokens`: 1500
  - `chunking.include_references`: false
  - `chunking.include_appendix`: false
- Every component supports CLI flag overrides, enabling reproducible ablation benchmarks between naive fixed-size chunking and structure-aware parent-child chunking.

---

## 13. Appendix Detection & Table Section Type Inheritance

- **Strict References Detection**:
  - Detects reference headings using exact matching: `^(references|bibliography|works cited|literature cited)$` after stripping optional section numbering and punctuation.
  - Avoids false-positive prefix matches on headings like "References to Prior Work".
- **Post-References Appendix Heuristic**:
  - All sections appearing after the References heading are classified as `section_type: "appendix"`, accurately capturing lettered appendices (e.g., `A Overview of HalluLens`, `B Implementation Details`).
- **Table Section Type Inheritance**:
  - Table parents dynamically inherit `section_type` from the enclosing section in which they appear based on document reading order and page coverage.
  - Tables located after References become `section_type: "appendix"` (with `is_table: True`), ensuring that appendix filtering toggles correctly exclude both appendix prose and appendix tables.

---

## 14. Query-Time Section Filtering & Iterative Candidate Expansion

- **Query-Time Filtering**:
  - `include_appendix` and `include_references` are evaluated at query time in `retrieval/parent_child.py`, rather than at index time.
  - Enables instant ablation toggling without re-embedding child vectors in Qdrant.
- **Iterative Candidate Expansion (Cap at $top\_k \times 10$)**:
  - Vector search initially fetches $top\_k \times 4$ candidate child chunks.
  - If post-filtering by `section_type` leaves fewer than $top\_k$ unique parents, the candidate pool automatically expands in steps up to a ceiling of $top\_k \times 10$ (or until the collection runs out of hits).
  - Guarantees $top\_k$ relevant parents are returned even when top dense matches belong to excluded sections.

---

## 15. Table Representation, Diagnostics, and Metadata Refresh

- **Table Short Labeling & Caption Preservation**:
  - Table section names are normalized to short labels `"Table N"` requiring a number or roman numeral (`^Table\s+([0-9]+|[IVXLCDM]+)\b`), with trailing punctuation stripped.
  - The entire original caption is preserved in the table parent `text` (`f"{caption}\n\n{markdown}"`).
- **Retrieval vs LLM Context Distinction**:
  - Prepending captions to table text in `parents.db` immediately enhances the LLM prompt context upon retrieval.
  - To update dense vector retrieval representations with captions, the optional script `ingestion/reembed_tables.py` re-embeds only table child chunks into `scholar_papers_v2` with updated vectors and Qdrant payloads.
- **Diagnostic `--debug` Flag**:
  - Available on `cli.py` and `retrieval/parent_child.py`.
  - Prints the top 10 child hits before parent merging with full-precision raw scores, child IDs, parent IDs, parent headings, and section types resolved directly from `parents.db` (bypassing stale vector payloads).
- **Standalone Metadata Refresh (`ingestion/refresh_parents.py`)**:
  - Safely updates `data/parsed/*.json` and `data/parents.db` with zero Qdrant interactions.
  - Creates a timestamped backup (`parents_<timestamp>.db.bak`) prior to execution.
  - Enforces parent ID invariance: if the set of `parent_id`s differs after the refresh, the backup is automatically restored and execution aborts.
  - Generates a before/after audit report showing appendix and reference parent counts per paper, with token percentage calculations and warnings for papers exceeding 40% non-body tokens.

---

## 16. Sentence-Boundary Child Chunking & Collection V3 (`scholar_papers_v3`)

- **Motivation**:
  - The previous token-slicing approach cut text at arbitrary subword or token boundaries, which frequently truncated words mid-token (e.g. `hallu` / `cination`) and modified text formatting during token detokenization/decoding.
- **Whole-Sentence Packing with 1-Sentence Overlap**:
  - Sentences are detected using academic boundary analysis handling abbreviations (`e.g.`, `i.e.`, `et al.`, `Fig.`, `Tab.`, `vs.`, `Dr.`, etc.), citation numbers, and decimals.
  - Whole sentences are packed up to ~300 tokens (`child_chunk_size`).
  - Consecutive chunks share a 1-sentence overlap (the next chunk begins at the last sentence of the preceding chunk).
  - Child text is sliced directly by character offsets `(start, end)` from `parent["text"]`, guaranteeing that every child is an authentic, 100% contiguous substring of its parent text (`child["text"] in parent["text"] == True`).
- **Over-Long Sentence Word-Boundary Splitting**:
  - Any single sentence exceeding ~300 tokens is split strictly at word boundaries into contiguous slices $\le 300$ tokens, ensuring chunks never start or end mid-word.
- **Standalone Table Children with Captions**:
  - Tables are kept as standalone child chunks (`_c0`) with the caption and markdown table embedded together in `text`.
- **Collection Isolation (`scholar_papers_v3`)**:
  - The new sentence-boundary chunks are indexed into a new collection: `scholar_papers_v3` (`collection_v3` in `config.yaml`).
  - `build_index_v2.py` accepts `--collection` (defaulting to `scholar_papers_v3`), allowing `scholar_papers_v2` to remain intact for ablation comparisons.
- **Parent ID Invariance**:
  - `parent_id` generation for sections and tables remains completely unchanged, preserving references across SQLite `data/parents.db` and downstream evaluations.

---

## 17. BM25 Lexical Indexing over Child Chunks & Hyphen-Preserving Regex (Phase 3)

- **Motivation**:
  - Dense embedding retrieval often struggles with exact keyword queries, proper nouns, model names (e.g. `gpt-3.5-turbo`, `bge-reranker-v2-m3`), and benchmark acronyms (e.g. `ChainPoll`, `MiHR`, `ERBench`).
- **Tokenization**:
  - Regex pattern: `re.findall(r"(?u)\b[a-z0-9]+(?:-[a-z0-9]+)*\b", text.lower())`
  - Normalizes text to lowercase while explicitly keeping numbers and hyphenated compounds as single lexical units.
- **Index Persistence & Verification Gate**:
  - Built offline using `python retrieval/bm25.py --build` and saved to `data/bm25_index.pkl`.
  - Loaded once per process (`_CACHED_BM25_INDEXES`).
  - Strict verification at load time: verifies both child count and child IDs against the Qdrant collection (`scholar_papers_v3`). If the vector database and BM25 index drift, retrieval aborts with an unambiguous message requiring `--build`.

---

## 18. Reciprocal Rank Fusion (RRF) & Candidate Pool Sizing (Phase 3)

- **Algorithm**:
  - Dense candidate retrieval returns top 50 child chunks (`dense_candidates: 50`).
  - BM25 candidate retrieval returns top 50 child chunks (`bm25_candidates: 50`).
  - Merged using Reciprocal Rank Fusion:
    $$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{k + \text{rank}_m(d)}$$
  - Constant $k = 60$ (`rrf_k` in `config.yaml`) balances early vs late ranks and avoids arbitrary score normalization across cosine distance and BM25 unbounded scores.
- **Top-30 Child Candidate Selection**:
  - Retains the top 30 unique fused child chunks after sorting by RRF score descending.
  - Merged to parent sections by taking the highest child RRF score per parent (`merge_children_to_parents`).
  - Query-time section filtering (`include_appendix`, `include_references`) is maintained.

---

## 19. Cross-Encoder Reranking at Child Chunk Level (Phase 3)

- **Model Selection & Architecture**:
  - Cross-encoder: `BAAI/bge-reranker-v2-m3` configured via `models.reranker` in `config.yaml` to allow seamless swapping with lighter models (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`).
  - Loaded once per process via module cache.
- **Child-Level Scoring vs Parent-Level**:
  - Reranking scores `(query, child_text)` rather than full parent text. Child chunks are focused semantic units (~300 tokens), preventing long parent sections from causing truncation bias or dilution in cross-attention layers.
  - Text is truncated to $\le 512$ tokens (`max_length=512`).
- **Disk Caching**:
  - Scores cached to disk at `.cache/rerank_cache.json` keyed by `(query, child_id)`.
  - Avoids recomputing cross-encoder inference across iterative queries, identical sub-questions, and ablation runs.
- **Parent Score Merging**:
  - Parent section score equals the maximum child cross-encoder score among its constituent retrieved children.

---

## 20. Multi-Paper Diversity & Retrieval Benchmark Comparison (Phase 3)

- **Diversity Capping (`max_parents_per_paper`)**:
  - Optional toggle in `config.yaml` (`max_parents_per_paper`, default `null` = off).
  - When set to a positive integer $M$, caps the number of parent contexts admitted from any single paper, preventing long papers from crowding out relevant evidence from other publications.
- **Multi-Mode Benchmark (`scripts/compare_modes.py`)**:
  - Runs all 4 modes (`naive`, `parent_child`, `hybrid`, `hybrid_rerank`) over standard benchmark questions.
  - Measures end-to-end retrieval latency in milliseconds without incurring Gemini LLM API costs or rate limit overhead.
  - Writes structured markdown reports to `compare_modes.md` showing top 5 sources and timing comparisons across phases.
