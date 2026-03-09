# Project Scaffold — Initial Setup

## Plan
Scaffold the Python project foundation: package config, core modules, models, and Docker services.

## Tasks

- [x] 1. Create `pyproject.toml` with all dependencies (FastAPI, SQLAlchemy, Whisper, Celery, etc.)
- [x] 2. Create `src/__init__.py`
- [x] 3. Create `src/config.py` — pydantic-settings for DB, Redis, API keys
- [x] 4. Create `src/exceptions.py` — custom exception hierarchy
- [x] 5. Create `src/sources/__init__.py` and `src/sources/base.py` — abstract base scraper
- [x] 6. Create `src/models/__init__.py`, `base.py`, `company.py`, `earnings_call.py`, `analysis.py`
- [x] 7. Create `docker/docker-compose.yml` — PostgreSQL (pgvector) + Redis
- [x] 8. Create `.env.example` with all required env vars
- [x] 9. Verify: poetry resolves deps, all files parse cleanly

## Review
- Poetry resolves 158 packages successfully (dry-run)
- All 7 Python files pass syntax validation
- pgvector/pgvector:pg16 image used for native vector support
- Models use UUID PKs, timestamp mixins, proper FK relationships
- Circular imports between models handled via bottom-of-file imports
