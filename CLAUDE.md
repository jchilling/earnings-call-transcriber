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

### 7. Teaching 
- Treat the user as a junior software engineer not familiar with the techniques / frameworks / architecture + system design you use. 
- Treat development like pair programming and teach the user your design choices and techniques / frameworks / architecture by writing to `tasks/lessions.md`

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

## Product End Goal

### Core Experience
A web UI where investment professionals can search, filter, and analyze earnings call transcripts across Asian markets. The system should feel like a Bloomberg terminal for earnings calls — fast, dense, and queryable.

### Search & Discovery
- **Filter by**: country/exchange, industry/sector (GICS classification), market cap range, stock name, stock ticker code
- **Full-text search** across all transcripts (e.g. "inventory writedown" returns every call where management discussed it)
- **Semantic search** via pgvector embeddings (e.g. "margin pressure from raw materials" finds relevant segments even without exact keyword match)
- **Date range filters**: quarterly (Q1-Q4), fiscal year, or custom date range
- **Language filter**: show only English-language calls, or include translated summaries of local-language calls

### Transcript View
- Raw transcribed text with speaker labels (CEO, CFO, Analyst Name — Firm)
- Timestamps linked to audio playback position (click a paragraph, hear that segment)
- Highlight key financial figures mentioned (revenue, EPS, guidance) inline
- Side-by-side: original language transcript + English translation for non-English calls
- Download transcript as PDF, DOCX, or plain text

### Analysis Dashboard (per company, per call)
- **Executive Summary**: 3-paragraph overview (results, outlook, strategic moves)
- **Key Metrics Table**: Revenue, operating income, net income, margins, EPS — actual vs consensus vs prior quarter/year
- **Guidance Tracker**: Management's forward guidance extracted and stored quarter-over-quarter, visualized as a time series chart showing guidance revisions (raised, maintained, lowered)
- **Sentiment Score**: Management confidence index (1-10) with trend over past quarters
- **Notable Quotes**: Top 5 most significant management statements per call, tagged by topic
- **Q&A Breakdown**: Each analyst question + management response, summarized with topic tags
- **Red Flags / Signals**: Automated detection of language shifts (hedging, deflection, unusual qualifiers) compared to prior calls

### Cross-Company / Portfolio Views
- **Sector Heatmap**: Aggregate sentiment scores across companies in a sector — quickly spot which management teams are turning cautious
- **Guidance Trend Dashboard**: Compare guidance revisions across competitors (e.g. all Taiwan semiconductor companies' revenue guidance over 4 quarters)
- **Earnings Calendar**: Upcoming calls with auto-scheduled transcription jobs, filterable by watchlist
- **Watchlist**: Save companies and get notifications when new transcripts are available
- **Comparative Analysis**: Select 2-5 companies and compare key metrics, sentiment, guidance side-by-side

### Data Export & Integration
- **API access**: RESTful API for all data (transcripts, analysis, metrics) so users can pull into their own models/spreadsheets
- **CSV/Excel export**: Filtered search results, metrics tables, guidance history
- **Webhook notifications**: Alert when a new transcript for a watchlisted company is ready

### What Makes This Useful for Fundamental Research
- **Speed**: Earnings calls drop during market hours across overlapping Asian time zones. Having auto-transcription within minutes of a call ending means analysts can react faster than reading a broker note
- **Coverage breadth**: Most sell-side research covers only large caps. This tool transcribes mid/small-cap calls that have no English coverage, giving an information edge
- **Longitudinal tracking**: Management credibility is measured by comparing what they said they'd do vs what happened. The guidance tracker makes this trivial
- **Language barrier removal**: A Korean-language call from a $2B battery component maker has no English transcript anywhere. This tool creates one automatically
- **Pattern detection at scale**: "Which semiconductor companies mentioned 'AI' more than 3x in their latest call?" is a 2-second query instead of reading 40 transcripts


## MVP Scope

### Guiding Principle
Ship the smallest thing that demonstrates the full pipeline end-to-end, for one market, before expanding. A working demo with 10 transcripts beats a half-built system that covers 4 markets.

### MVP: Taiwan Market, Single-Company Flow

**What's in:**

1. **One market scraper** — Taiwan (MOPS/TWSE) only
   - Discover earnings call audio URLs for a given stock ticker
   - Support both live HLS streams (.m3u8) and archived audio files
   - Seed with ~20 large-cap Taiwan tickers (TSMC, MediaTek, Hon Hai, Delta, etc.)

2. **Transcription pipeline** — Whisper local, Mandarin + English
   - Download audio → preprocess to 16kHz WAV → transcribe via Whisper large-v3
   - Speaker diarization via pyannote (label speakers, no role mapping yet)
   - Store raw transcript segments with timestamps in PostgreSQL

3. **Analysis for one call** — Claude API summarization
   - Executive summary (3 paragraphs)
   - Key metrics extraction (revenue, margins, EPS, guidance) as structured JSON
   - Sentiment score (1-10)
   - Store analysis results linked to transcript in DB

4. **Minimal web UI** — read-only, single-page app
   - Search bar: filter by stock ticker or company name
   - Company page: list of available earnings calls by date
   - Call detail page: transcript text with speaker labels + analysis panel alongside
   - No auth, no user accounts — just a local tool

5. **API** — FastAPI, 3 endpoints
   - `GET /companies` — list companies with filters (ticker, name)
   - `GET /companies/{ticker}/calls` — list earnings calls for a company
   - `GET /calls/{id}` — full transcript + analysis for a single call

6. **Docker Compose** — one command to run everything
   - PostgreSQL + Redis + FastAPI app + Celery worker
   - `docker compose up` and it works

**What's NOT in MVP:**

- Hong Kong, Japan, Korea, Singapore scrapers (Phase 2)
- Full-text search / semantic search (Phase 2)
- Guidance tracker time series charts (Phase 2)
- Cross-company comparison views (Phase 3)
- Watchlists, notifications, earnings calendar (Phase 3)
- Red flag / language shift detection (Phase 3)
- Audio playback synced to transcript (Phase 3)
- User accounts, auth, multi-tenancy (Phase 3)
- CSV/Excel export (Phase 2)
- Side-by-side translation view (Phase 2)
- PDF/DOCX transcript download (Phase 2)

### MVP Build Order

Build in this exact sequence. Each step produces something testable.

```
Step 1: Foundation          ✅ DONE (project scaffold, models, config)
   │
Step 2: Taiwan scraper      ← START HERE
   │    Discover + download audio for one ticker (2330.TW / TSMC)
   │    Verify: audio file saved locally
   │
Step 3: Transcription
   │    Whisper local → raw transcript with timestamps
   │    Verify: readable transcript text from TSMC call
   │
Step 4: Analysis
   │    Claude API → summary + metrics + sentiment
   │    Verify: structured JSON output from transcript
   │
Step 5: Storage
   │    Save everything to PostgreSQL via SQLAlchemy
   │    Verify: data persists, queryable via psql
   │
Step 6: API
   │    FastAPI endpoints serving stored data
   │    Verify: curl returns JSON for all 3 endpoints
   │
Step 7: UI
   │    React or plain HTML/JS frontend
   │    Verify: can search TSMC, see transcript + analysis in browser
   │
Step 8: Docker
   │    Compose file wiring everything together
   │    Verify: git clone → docker compose up → working app
   │
Step 9: Seed data
        Backfill 2-3 quarters of calls for 10-20 Taiwan tickers
        Verify: browsable corpus of real data
```

### MVP Success Criteria
- Someone can `docker compose up`, open a browser, search for "TSMC", and read a transcribed + analyzed earnings call within 60 seconds
- The transcript has speaker labels and timestamps
- The analysis panel shows a summary, key metrics, and sentiment score
- Total audio-to-analysis latency for a new call: under 30 minutes

### Post-MVP Phases

**Phase 2: Depth** — make the Taiwan experience great
- Full-text + semantic search across all transcripts
- Guidance tracker (quarterly time series)
- Side-by-side Mandarin/English translation
- Add 2nd market (Hong Kong — most relevant for Dymon Asia)
- CSV export

**Phase 3: Breadth** — expand coverage and features
- Japan + Korea scrapers
- Cross-company comparison dashboards
- Sector heatmaps
- Earnings calendar with auto-scheduling
- Watchlists + notifications
- Red flag detection (language shift analysis)

**Phase 4: Polish** — production readiness
- User auth (if needed for team use)
- Audio playback synced to transcript
- Performance optimization for large corpus
- Monitoring, alerting, error recovery
