# Lessons

## Testing
- **Isolate what you're iterating on**: When prompt-engineering LLM calls, don't rerun the whole pipeline (3+ min transcription). Write a standalone test script that loads cached JSON output and only exercises the LLM step. Same applies to any component — test it in isolation first. See `scripts/test_speaker_id.py` for the pattern.
- **Use `--correction-only` flag pattern**: Build test scripts with flags to skip expensive steps and only run the part you're iterating on.

## LLM Integration (DeepSeek)
- **Don't ask LLMs to merge speaker IDs from text**: Without audio, the LLM can't distinguish speakers — it over-merges everything into 2-3 speakers. Instead: ask LLM to identify-only (assign names per ID independently), then merge-by-name programmatically.
- **DeepSeek auto-caches prefixes**: No opt-in needed. Put stable content (transcript) in system message, variable content (instructions) in user message. Cache hit = $0.07/M vs $0.27/M (74% savings). Second call on same transcript gets 99%+ cache hit rate.
- **deepseek-reasoner has no system message**: Only user messages supported. Combine context + instructions into one user message. The prefix still auto-caches.
- **deepseek-reasoner needs focused input**: The reasoning chain is very verbose — it will analyze every segment you give it. For a 567-segment transcript, 16K output tokens wasn't enough. Solution: only send ~60 segments around transition windows (handoff keywords), not the full transcript.
- **Two-pass speaker ID architecture**: (1) deepseek-chat identifies names per VAD speaker ID (cheap, ~$0.005). (2) deepseek-reasoner corrects assignments around handoff points (smarter, ~$0.024). Total ~$0.03 per 1-hour call.

## Audio/Transcription
- **yt-dlp vs ffmpeg**: Never use yt-dlp for large direct video URLs. Use ffmpeg stream extraction instead.
- **pyannote on CPU is impractical**: 30+ minutes for a 1-hour file. Check `torch.cuda.is_available()` and use VAD fallback (silence-gap detection) when no CUDA GPU.
- **VAD diarization over-segments**: Silence gaps create 20+ speaker IDs for ~11 real speakers. The LLM identify → merge-by-name approach handles this well.
- **Tests need PYTHONPATH**: Run pytest with `PYTHONPATH=.` since pyproject.toml uses `packages = [{include = "src"}]`
