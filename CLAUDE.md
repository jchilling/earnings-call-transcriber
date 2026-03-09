# Earnings Call Transcriber — Asia Markets

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

---

## Project Overview

An end-to-end pipeline that automatically discovers, downloads, transcribes, and analyzes earnings call audio for publicly listed companies across Asian markets (Taiwan, Hong Kong, Japan, South Korea, Singapore, India). Built as both a personal research tool and a portfolio project demonstrating applied AI in fundamental equities research.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌────────────┐
│ Source       │───▶│ Audio        │───▶│ Transcription   │───▶│ Analysis     │───▶│ Storage &  │
│ Discovery    │    │ Downloader   │    │ Engine          │    │ Engine       │    │ API        │
└─────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘    └────────────┘
  - IR pages         - yt-dlp           - Whisper (local)      - Claude API       - PostgreSQL
  - Exchange feeds   - requests         - Whisper API           - Structured       - FastAPI
  - RSS/webhooks     - selenium           (fallback)             extraction       - Full-text search
```

## Tech Stack
- **Language**: Python 3.11+
- **Transcription**: OpenAI Whisper (large-v3) — local-first, API fallback
- **LLM Analysis**: Anthropic Claude API (claude-sonnet-4-20250514)
- **Web Framework**: FastAPI
- **Database**: PostgreSQL with pgvector for semantic search
- **Task Queue**: Celery + Redis for async transcription jobs
- **Package Manager**: Poetry
- **Testing**: pytest with async support
- **Linting**: ruff

## Directory Structure
```
earnings-call-transcriber/
├── CLAUDE.md                    # This file
├── pyproject.toml
├── README.md
├── .env.example
├── tasks/                       # Task tracking (Claude-managed)
│   ├── todo.md                  # Current plan with checkable items
│   └── lessons.md               # Accumulated lessons from corrections
├── src/
│   ├── __init__.py
│   ├── config.py                # Settings via pydantic-settings
│   ├── exceptions.py            # Custom exception hierarchy
│   ├── models/                  # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── company.py           # Company, Exchange, Sector
│   │   ├── earnings_call.py     # EarningsCall, Transcript, AudioFile
│   │   └── analysis.py          # CallSummary, KeyMetric, SentimentScore
│   ├── sources/                 # Market-specific scrapers
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base scraper
│   │   ├── taiwan.py            # TWSE/MOPS scraper
│   │   ├── hongkong.py          # HKEX scraper
│   │   ├── japan.py             # TDnet/JPX scraper
│   │   ├── korea.py             # DART/KRX scraper
│   │   └── singapore.py         # SGX scraper
│   ├── audio/                   # Audio download & preprocessing
│   │   ├── __init__.py
│   │   ├── downloader.py        # Multi-source audio fetcher
│   │   └── preprocessor.py      # Noise reduction, format conversion
│   ├── transcription/           # Speech-to-text
│   │   ├── __init__.py
│   │   ├── whisper_local.py     # Local Whisper inference
│   │   ├── whisper_api.py       # OpenAI Whisper API fallback
│   │   └── diarization.py       # Speaker diarization (pyannote)
│   ├── analysis/                # LLM-powered analysis
│   │   ├── __init__.py
│   │   ├── summarizer.py        # Call summary generation
│   │   ├── metrics_extractor.py # Revenue, margins, guidance extraction
│   │   ├── sentiment.py         # Management tone / sentiment analysis
│   │   └── prompts/             # Prompt templates (Jinja2)
│   │       ├── summarize.j2
│   │       ├── extract_metrics.j2
│   │       └── sentiment.j2
│   ├── api/                     # FastAPI endpoints
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── companies.py
│   │   │   ├── transcripts.py
│   │   │   └── analysis.py
│   │   └── schemas.py           # Pydantic request/response models
│   └── tasks/                   # Celery async tasks
│       ├── __init__.py
│       ├── transcribe.py
│       └── analyze.py
├── tests/
│   ├── conftest.py
│   ├── test_sources/
│   ├── test_transcription/
│   └── test_analysis/
├── scripts/
│   ├── seed_companies.py        # Seed DB with company master data
│   └── backfill.py              # Backfill historical calls
├── agents/                      # Custom Claude Code subagent definitions
│   ├── scraper-agent.md
│   ├── transcription-agent.md
│   ├── analysis-agent.md
│   └── test-agent.md
└── docker/
    ├── Dockerfile
    └── docker-compose.yml       # App + PostgreSQL + Redis
```

## Coding Conventions
- Type hints on all function signatures
- Docstrings in Google style
- Async-first: use `async def` for I/O-bound operations
- All config via environment variables (never hardcode API keys)
- Database migrations via Alembic
- Error handling: custom exception hierarchy in `src/exceptions.py`
- Logging: structured JSON logging via `structlog`

## Key Design Decisions
- **Local Whisper first**: Minimize API costs; fall back to Whisper API for languages where local model underperforms
- **Speaker diarization**: Critical for earnings calls — need to distinguish CEO, CFO, analysts
- **Prompt templates as files**: Keep prompts version-controlled and editable without code changes
- **Market-specific scrapers**: Each Asian market has unique IR disclosure patterns; no one-size-fits-all

## Important Context
- Primary target markets: Taiwan (TWSE), Hong Kong (HKEX), Japan (TSE), South Korea (KRX)
- Many calls are in local languages (Mandarin, Cantonese, Japanese, Korean) — Whisper handles these but quality varies
- Taiwan's MOPS system (mops.twse.com.tw) is the primary data source for Taiwanese companies
- Hong Kong companies often use third-party webcast services
- Japanese companies frequently publish text-based summaries (決算短信) alongside audio — consider ingesting both

## Commands
- `poetry install` — install dependencies
- `poetry run pytest` — run tests
- `poetry run pytest --cov=src --cov-report=term-missing` — tests with coverage
- `poetry run uvicorn src.api.main:app --reload` — start dev server
- `docker compose up` — start full stack
- `poetry run celery -A src.tasks worker` — start task worker

## Multi-Agent Workflow
See `agents/` directory for subagent definitions. The project is designed for parallel development:
- **scraper-agent**: Works on `src/sources/` — one market at a time
- **transcription-agent**: Works on `src/transcription/` and `src/audio/`
- **analysis-agent**: Works on `src/analysis/` and prompt templates
- **test-agent**: Works on `tests/` — writes tests for completed modules
Each agent should use worktree isolation to avoid conflicts.
