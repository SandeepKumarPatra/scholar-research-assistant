- Python 3.12, type hints, ruff formatting
- Use a virtual environment (.venv) and requirements.txt
- Every module gets pytest tests
- Never hardcode API keys; read from .env
- Log retrieval results with scores
- Update README.md after each task
- Log design decisions in DECISIONS.md
- Use the google-genai SDK (NOT google-generativeai) for the Gemini API
- Read the model name from GEMINI_MODEL in .env; add retry with exponential
  backoff for rate-limit (429) errors
- Use the google-genai SDK (NOT google-generativeai) for the Gemini API
- Read the model name from GEMINI_MODEL in .env
- Throttle Gemini calls to max 12 requests/minute
- Retry with exponential backoff on 429 errors
- Cache every LLM response on disk (keyed by prompt hash)
- Never modify the baseline code in retrieval/naive.py or the
  "scholar_papers" collection; new work goes in new files/collections
- Every new component must be toggleable via config.yaml (needed for ablations)