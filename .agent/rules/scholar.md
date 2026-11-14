# Scholar project rules

## Code style
- Python 3.12, type hints, ruff formatting
- Every module gets pytest tests; mock heavy models and all API calls
- Log retrieval results with scores
- Update README.md after each task
- Log design decisions in DECISIONS.md

## Environment
- Always use the project virtual environment at .venv; install packages
  only inside it, never globally
- Keep requirements.txt up to date when adding packages
- Never hardcode API keys; read from .env
- Never commit .env, data/pdfs/, data/parsed/, qdrant_data/, .cache/ or .venv/;
  keep .gitignore current

## Gemini API
- Use the google-genai SDK (NOT google-generativeai)
- Read the model name from GEMINI_MODEL in .env
- Throttle Gemini calls to max 12 requests/minute
- Retry with exponential backoff on 429 errors
- Cache every LLM response on disk (keyed by prompt hash)

## Pipeline rules
- Never modify retrieval/naive.py or the "scholar_papers" collection;
  new work goes in new files and new collections
- Every new component must be toggleable via config.yaml (needed for ablations)
- Slow steps (parsing, embedding) must cache results to disk, show a tqdm
  progress bar, and support a --limit argument for testing on few papers
- Load embedding models once per process, not once per question

## Working style
- Show a plan first and wait for approval before building
- Do not run long jobs (parsing, indexing) automatically; tell me the command
  and let me run it