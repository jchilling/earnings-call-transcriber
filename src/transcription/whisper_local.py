"""Local Whisper inference for speech-to-text transcription.

Uses OpenAI's Whisper model (large-v3 by default) running locally.
Handles long audio (>30 min) by chunking with overlap to avoid truncation.
"""

import asyncio
import functools
from pathlib import Path
from typing import Any

import structlog

from src.audio.preprocessor import convert_to_wav, get_audio_duration
from src.config import settings
from src.exceptions import WhisperModelError
from src.transcription import TranscriptionResult, TranscriptSegment

logger = structlog.get_logger(__name__)

# Whisper natively handles 30-second windows; for very long audio we chunk
# at this threshold to manage memory and avoid quality degradation.
CHUNK_DURATION_SEC = 1800  # 30 minutes per chunk
CHUNK_OVERLAP_SEC = 30  # 30 seconds overlap to avoid cutting mid-sentence

# Singleton model cache to avoid reloading on every call
_model_cache: dict[str, Any] = {}


def _get_model(model_name: str, device: str) -> Any:
    """Load and cache a Whisper model.

    Args:
        model_name: Whisper model size (e.g. "large-v3", "medium", "base").
        device: Compute device ("cpu", "cuda", "mps").

    Returns:
        Loaded whisper model instance.

    Raises:
        WhisperModelError: If the model cannot be loaded.
    """
    cache_key = f"{model_name}:{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        import whisper
    except ImportError as e:
        raise WhisperModelError(
            "openai-whisper is not installed. Install with: pip install openai-whisper"
        ) from e

    logger.info("loading_whisper_model", model=model_name, device=device)
    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as e:
        raise WhisperModelError(f"Failed to load Whisper model '{model_name}': {e}") from e

    _model_cache[cache_key] = model
    logger.info("whisper_model_loaded", model=model_name, device=device)
    return model


def _transcribe_sync(
    audio_path: str,
    model_name: str,
    device: str,
    language: str | None,
    initial_prompt: str | None,
) -> dict[str, Any]:
    """Run Whisper transcription synchronously (called in executor).

    Args:
        audio_path: Path to 16kHz mono WAV file.
        model_name: Whisper model name.
        device: Compute device.
        language: ISO language code to force, or None for auto-detect.
        initial_prompt: Optional prompt to condition the model (e.g. company/speaker names).

    Returns:
        Raw Whisper result dict with keys: text, segments, language.
    """
    model = _get_model(model_name, device)

    decode_options: dict[str, Any] = {
        "verbose": False,
        "word_timestamps": False,
    }
    if language:
        decode_options["language"] = language
    if initial_prompt:
        decode_options["initial_prompt"] = initial_prompt

    return model.transcribe(audio_path, **decode_options)


def _segments_from_whisper(raw_segments: list[dict[str, Any]]) -> list[TranscriptSegment]:
    """Convert Whisper's raw segment dicts to TranscriptSegment objects.

    Args:
        raw_segments: List of segment dicts from whisper.transcribe().

    Returns:
        List of TranscriptSegment instances.
    """
    segments = []
    for seg in raw_segments:
        segments.append(
            TranscriptSegment(
                text=seg["text"].strip(),
                start_time=seg["start"],
                end_time=seg["end"],
                confidence=seg.get("avg_logprob", 0.0),
            )
        )
    return segments


async def _split_audio(audio_path: Path, chunk_duration: int, overlap: int) -> list[Path]:
    """Split a long audio file into overlapping chunks using ffmpeg.

    Args:
        audio_path: Path to the source WAV file.
        chunk_duration: Duration of each chunk in seconds.
        overlap: Overlap between consecutive chunks in seconds.

    Returns:
        List of paths to chunk files (caller responsible for cleanup).
    """
    total_duration = await get_audio_duration(audio_path)
    if total_duration <= chunk_duration:
        return [audio_path]

    chunks: list[Path] = []
    start = 0.0
    chunk_idx = 0

    while start < total_duration:
        chunk_path = audio_path.parent / f"{audio_path.stem}_chunk{chunk_idx:03d}.wav"
        end = min(start + chunk_duration, total_duration)
        duration = end - start

        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", str(start),
            "-t", str(duration),
            "-c:a", "pcm_s16le",
            str(chunk_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.warning("chunk_split_failed", chunk=chunk_idx, error=stderr.decode()[-200:])
            break

        chunks.append(chunk_path)
        start += chunk_duration - overlap
        chunk_idx += 1

    logger.info("audio_split", chunks=len(chunks), total_duration=total_duration)
    return chunks


def _merge_chunk_segments(
    all_chunk_segments: list[list[TranscriptSegment]],
    chunk_duration: int,
    overlap: int,
) -> list[TranscriptSegment]:
    """Merge segments from multiple chunks, handling overlap deduplication.

    For overlapping regions, we keep segments from the earlier chunk since
    they tend to have better context from preceding audio.

    Args:
        all_chunk_segments: Segments per chunk, in order.
        chunk_duration: Duration used per chunk.
        overlap: Overlap duration between chunks.

    Returns:
        Merged list of TranscriptSegment with corrected timestamps.
    """
    if len(all_chunk_segments) <= 1:
        return all_chunk_segments[0] if all_chunk_segments else []

    merged: list[TranscriptSegment] = []
    step = chunk_duration - overlap

    for chunk_idx, chunk_segments in enumerate(all_chunk_segments):
        time_offset = chunk_idx * step

        for seg in chunk_segments:
            absolute_start = seg.start_time + time_offset
            absolute_end = seg.end_time + time_offset

            # Skip segments that fall within the overlap zone of the previous chunk.
            # The previous chunk already covered this region.
            if chunk_idx > 0 and seg.start_time < overlap:
                continue

            merged.append(
                TranscriptSegment(
                    text=seg.text,
                    start_time=absolute_start,
                    end_time=absolute_end,
                    speaker_id=seg.speaker_id,
                    language=seg.language,
                    confidence=seg.confidence,
                )
            )

    return merged


async def transcribe_audio(
    audio_path: str | Path,
    *,
    language: str | None = None,
    initial_prompt: str | None = None,
    model_name: str | None = None,
    device: str | None = None,
    preprocess: bool = True,
) -> TranscriptionResult:
    """Transcribe an audio file using local Whisper.

    This is the main entry point for local transcription. It handles:
    1. Preprocessing to 16kHz mono WAV (if needed)
    2. Chunking long audio with overlap
    3. Running inference in a thread pool (Whisper is CPU/GPU bound)
    4. Merging chunks and returning structured results

    Args:
        audio_path: Path to the audio file (any format ffmpeg supports).
        language: ISO language code to force (e.g. "zh", "ja", "ko", "en").
            If None, Whisper auto-detects from the first 30 seconds.
        initial_prompt: Optional conditioning prompt. Useful for providing
            company names, speaker names, or domain terminology to improve
            recognition accuracy.
        model_name: Override the configured Whisper model.
        device: Override the configured compute device.
        preprocess: Whether to convert to 16kHz mono WAV first. Set False
            if the file is already preprocessed.

    Returns:
        TranscriptionResult with segments, full text, detected language, etc.

    Raises:
        WhisperModelError: If the model cannot be loaded or inference fails.
    """
    audio_path = Path(audio_path)
    model_name = model_name or settings.whisper_model
    device = device or settings.whisper_device

    logger.info(
        "transcription_started",
        audio=str(audio_path),
        model=model_name,
        language=language,
    )

    # Step 1: Preprocess
    wav_path = audio_path
    temp_wav = False
    if preprocess:
        wav_path = await convert_to_wav(audio_path)
        temp_wav = (wav_path != audio_path)

    try:
        # Step 2: Get duration and decide on chunking
        duration = await get_audio_duration(wav_path)

        chunks = await _split_audio(wav_path, CHUNK_DURATION_SEC, CHUNK_OVERLAP_SEC)
        is_chunked = len(chunks) > 1

        # Step 3: Transcribe each chunk in the thread pool
        loop = asyncio.get_running_loop()
        all_chunk_segments: list[list[TranscriptSegment]] = []
        detected_language: str = ""

        for i, chunk_path in enumerate(chunks):
            logger.info(
                "transcribing_chunk",
                chunk=i + 1,
                total=len(chunks),
                path=str(chunk_path),
            )

            try:
                result = await loop.run_in_executor(
                    None,
                    functools.partial(
                        _transcribe_sync,
                        str(chunk_path),
                        model_name,
                        device,
                        language,
                        initial_prompt,
                    ),
                )
            except WhisperModelError:
                raise
            except Exception as e:
                raise WhisperModelError(
                    f"Whisper inference failed on chunk {i}: {e}"
                ) from e

            # Capture detected language from the first chunk
            if i == 0:
                detected_language = result.get("language", "")

            all_chunk_segments.append(_segments_from_whisper(result.get("segments", [])))

        # Step 4: Merge chunks
        if is_chunked:
            segments = _merge_chunk_segments(
                all_chunk_segments, CHUNK_DURATION_SEC, CHUNK_OVERLAP_SEC
            )
        else:
            segments = all_chunk_segments[0] if all_chunk_segments else []

        # Set language on all segments
        segments = [
            TranscriptSegment(
                text=seg.text,
                start_time=seg.start_time,
                end_time=seg.end_time,
                speaker_id=seg.speaker_id,
                language=detected_language or language,
                confidence=seg.confidence,
            )
            for seg in segments
        ]

        full_text = " ".join(seg.text for seg in segments)

        result_obj = TranscriptionResult(
            segments=segments,
            full_text=full_text,
            language=detected_language or language or "",
            model_used=model_name,
            duration_seconds=duration,
        )

        logger.info(
            "transcription_completed",
            segments=len(segments),
            language=result_obj.language,
            duration=duration,
        )

        return result_obj

    finally:
        # Clean up temp WAV and chunk files
        if temp_wav and wav_path.exists() and wav_path != audio_path:
            wav_path.unlink(missing_ok=True)

        if len(chunks) > 1:
            for chunk_path in chunks:
                if chunk_path != wav_path and chunk_path.exists():
                    chunk_path.unlink(missing_ok=True)
