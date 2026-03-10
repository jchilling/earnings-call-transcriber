# Whisper Local Transcription Module

## Plan
Build the local Whisper transcription module (`src/transcription/whisper_local.py`) with supporting infrastructure.

## Tasks

- [x] 1. Create `src/transcription/__init__.py` with `TranscriptSegment` and `TranscriptionResult` dataclasses
- [x] 2. Create `src/audio/__init__.py`
- [x] 3. Create `src/audio/preprocessor.py` — ffmpeg wrapper to convert to 16kHz mono WAV
- [x] 4. Create `src/transcription/whisper_local.py` — local Whisper inference with chunking
- [x] 5. Create `tests/test_transcription/` with unit tests (20 tests)
- [x] 6. Verify: all files parse, ruff passes, 20/20 tests pass

## Review
- All 7 new files pass syntax validation and ruff lint
- 20/20 tests pass (7 preprocessor + 13 whisper_local)
- Note: tests need `PYTHONPATH=.` since pyproject.toml doesn't configure src as a package root
- Whisper import is lazy (inside `_get_model`) so the module works even if openai-whisper isn't installed
- Audio chunking at 30-min boundaries with 30s overlap for long earnings calls
- Thread pool executor used for CPU-bound Whisper inference (keeps async loop responsive)
