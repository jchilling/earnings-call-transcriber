# Transcription Pipeline with Speaker Diarization & Name Extraction

## Plan
Full pipeline: faster-whisper transcription + pyannote diarization + speaker name extraction.

## Tasks

### Dependency & Config
- [x] 1. Swap `openai-whisper` → `faster-whisper` in pyproject.toml
- [x] 2. Add `hf_token` setting to `src/config.py`
- [x] 3. `poetry install` — verify faster-whisper installs

### Data Models
- [x] 4. Add `speaker_name` field to `TranscriptSegment`
- [x] 5. Add `SpeakerInfo` dataclass to `src/transcription/__init__.py`
- [x] 6. Add `speakers` dict to `TranscriptionResult`

### Transcription
- [x] 7. Rewrite `whisper_local.py` for faster-whisper API (WhisperModel, segment iterator)

### Diarization
- [x] 8. Create `src/transcription/diarization.py` — pyannote speaker diarization

### Speaker Identification
- [x] 9. Create `src/transcription/speaker_identification.py` — regex name extraction

### Pipeline Orchestrator
- [x] 10. Create `src/transcription/pipeline.py` — parallel transcription + diarization + alignment

### Agent Config
- [x] 11. Update `agents/transcription-agent.md` with new files and workflow

### LLM Speaker ID + VAD Fallback + TXT Output
- [x] 14. Add `deepseek_api_key` and `deepseek_model` to `src/config.py`
- [x] 15. Create `src/transcription/vad_diarization.py` — silence-gap speaker turn detection
- [x] 16. Create `src/transcription/speaker_identification_llm.py` — DeepSeek LLM speaker ID
- [x] 17. Update `src/transcription/pipeline.py` — integrate VAD fallback + LLM enhancement
- [x] 18. Update `scripts/test_pipeline.py` — add TXT output alongside JSON

### DeepSeek Reasoner Correction
- [x] 19. Add `deepseek_reasoner_model` to config
- [x] 20. Add reasoner correction pass to `speaker_identification_llm.py`
- [x] 21. Create isolated test script `scripts/test_speaker_id.py`
- [x] 22. Optimize: focused transition windows instead of full transcript
- [x] 23. Integrate correction pass into pipeline

### Testing
- [x] 12. Run pipeline on `tsmc_q3_2025.mp3` — verify output
- [x] 13. Dump output to JSON + TXT for inspection

## Review
- Pipeline tested on TSMC Q3 2025 earnings call (62 min, 567 segments)
- VAD detected 23 speaker IDs → deepseek-chat identified 6/7 → merge-by-name collapsed to 7 unique
- deepseek-reasoner corrected 28 mis-assigned segments around speaker handoffs
- Final: Jeff Su (246), C.C. Wei (217), Wendell Huang (74), Gokul Hariharan (17), Arthur (9), Operator (2), Krish Sankar (2)
- Total LLM cost: ~$0.03 per call
