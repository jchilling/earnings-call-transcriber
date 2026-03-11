"""Transcription pipeline orchestrator.

Runs Whisper transcription and speaker diarization (pyannote or VAD fallback),
aligns results, extracts speaker names via regex + optional LLM enhancement,
and returns a complete TranscriptionResult with named speakers.
"""

import asyncio
from pathlib import Path

import structlog

from src.audio.preprocessor import convert_to_wav
from src.config import settings
from src.transcription import SpeakerInfo, TranscriptionResult, TranscriptSegment
from src.transcription.diarization import DiarizationSegment, diarize_audio
from src.transcription.speaker_identification import extract_speaker_names
from src.transcription.speaker_identification_llm import (
    correct_speaker_assignments,
    enhance_speaker_names,
)
from src.transcription.vad_diarization import assign_speakers_by_vad
from src.transcription.whisper_local import transcribe_audio

logger = structlog.get_logger(__name__)


def _has_cuda() -> bool:
    """Check if CUDA GPU is available for pyannote."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _align_segments(
    transcript_segments: list[TranscriptSegment],
    diarization_segments: list[DiarizationSegment],
) -> list[TranscriptSegment]:
    """Align transcription segments with diarization speaker labels.

    For each transcript segment, finds the diarization segment with the most
    temporal overlap and assigns that speaker_id. This handles cases where
    speaker boundaries don't perfectly match Whisper segment boundaries.

    Args:
        transcript_segments: Segments from Whisper transcription.
        diarization_segments: Speaker turns from pyannote diarization.

    Returns:
        New list of TranscriptSegment with speaker_id populated.
    """
    if not diarization_segments:
        return transcript_segments

    aligned: list[TranscriptSegment] = []

    for seg in transcript_segments:
        seg_mid = (seg.start_time + seg.end_time) / 2
        best_speaker: str | None = None
        best_overlap: float = 0.0

        for diar in diarization_segments:
            # Calculate overlap between transcript segment and diarization turn
            overlap_start = max(seg.start_time, diar.start_time)
            overlap_end = min(seg.end_time, diar.end_time)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar.speaker_id

        # Fallback: if no overlap found, use the diarization segment
        # whose midpoint is closest to the transcript segment midpoint
        if best_speaker is None:
            best_speaker = min(
                diarization_segments,
                key=lambda d: abs((d.start_time + d.end_time) / 2 - seg_mid),
            ).speaker_id

        aligned.append(
            TranscriptSegment(
                text=seg.text,
                start_time=seg.start_time,
                end_time=seg.end_time,
                speaker_id=best_speaker,
                speaker_name=seg.speaker_name,
                language=seg.language,
                confidence=seg.confidence,
            )
        )

    return aligned


def _apply_speaker_names(
    segments: list[TranscriptSegment],
    speakers: dict[str, SpeakerInfo],
) -> list[TranscriptSegment]:
    """Apply resolved speaker names to transcript segments.

    Args:
        segments: Aligned segments with speaker_id set.
        speakers: Map of speaker_id → SpeakerInfo with resolved names.

    Returns:
        New list of TranscriptSegment with speaker_name populated.
    """
    return [
        TranscriptSegment(
            text=seg.text,
            start_time=seg.start_time,
            end_time=seg.end_time,
            speaker_id=seg.speaker_id,
            speaker_name=speakers.get(seg.speaker_id, SpeakerInfo(id="")).name
            if seg.speaker_id
            else None,
            language=seg.language,
            confidence=seg.confidence,
        )
        for seg in segments
    ]


def _merge_speakers_by_name(
    segments: list[TranscriptSegment],
    speakers: dict[str, SpeakerInfo],
) -> tuple[list[TranscriptSegment], dict[str, SpeakerInfo]]:
    """Merge VAD speaker IDs that were identified as the same person.

    After LLM identification, multiple speaker IDs may have the same name
    (e.g. SPEAKER_00 and SPEAKER_07 are both "Jeff Su"). This merges them
    into a single canonical ID (the one with the most segments).

    Args:
        segments: Segments with speaker IDs.
        speakers: Speaker info with LLM-identified names.

    Returns:
        Tuple of (merged_segments, merged_speakers).
    """
    # Group speaker IDs by name
    name_to_sids: dict[str, list[str]] = {}
    for sid, info in speakers.items():
        if info.name:
            name_to_sids.setdefault(info.name, []).append(sid)

    # Build merge map: for each name with multiple IDs, keep the one with most segments
    merge_map: dict[str, str] = {}
    for name, sids in name_to_sids.items():
        if len(sids) <= 1:
            continue
        # Canonical = the ID with the most segments
        canonical = max(sids, key=lambda s: speakers[s].segments_count)
        for sid in sids:
            if sid != canonical:
                merge_map[sid] = canonical

    if not merge_map:
        return segments, speakers

    # Apply merges to segments
    merged_segments = [
        TranscriptSegment(
            text=seg.text,
            start_time=seg.start_time,
            end_time=seg.end_time,
            speaker_id=merge_map.get(seg.speaker_id, seg.speaker_id),
            speaker_name=seg.speaker_name,
            language=seg.language,
            confidence=seg.confidence,
        )
        for seg in segments
    ]

    # Rebuild speaker dict with merged counts
    from collections import Counter
    counts = Counter(seg.speaker_id for seg in merged_segments if seg.speaker_id)

    merged_speakers: dict[str, SpeakerInfo] = {}
    for sid, count in counts.items():
        info = speakers.get(sid, SpeakerInfo(id=sid))
        merged_speakers[sid] = SpeakerInfo(
            id=sid,
            name=info.name,
            title=info.title,
            segments_count=count,
        )

    logger.info(
        "speakers_merged_by_name",
        merges=len(merge_map),
        speakers_before=len(speakers),
        speakers_after=len(merged_speakers),
    )

    return merged_segments, merged_speakers


async def transcribe_with_diarization(
    audio_path: str | Path,
    *,
    language: str | None = None,
    initial_prompt: str | None = None,
    model_name: str | None = None,
    device: str | None = None,
    hf_token: str | None = None,
) -> TranscriptionResult:
    """Full transcription pipeline: transcribe + diarize + identify speakers.

    Runs Whisper transcription and pyannote diarization in parallel on the
    same preprocessed audio, then:
    1. Aligns transcript segments with speaker turns
    2. Extracts speaker names from self-introductions
    3. Maps speaker IDs to real names

    Args:
        audio_path: Path to the audio file (any format ffmpeg supports).
        language: ISO language code to force, or None for auto-detect.
        initial_prompt: Conditioning prompt for Whisper (company/speaker names).
        model_name: Override the configured Whisper model.
        device: Override the configured compute device.
        hf_token: HuggingFace token for pyannote. Falls back to settings.

    Returns:
        TranscriptionResult with speaker-attributed, named segments.

    Raises:
        WhisperModelError: If transcription fails.
        DiarizationError: If diarization fails.
    """
    audio_path = Path(audio_path)

    logger.info("pipeline_started", audio=str(audio_path))

    # Step 0: Preprocess audio once (both Whisper and pyannote use the same file)
    wav_path = await convert_to_wav(audio_path)
    temp_wav = wav_path != audio_path

    try:
        # Step 1: Run transcription (and diarization in parallel if HF token available)
        transcription_task = transcribe_audio(
            wav_path,
            language=language,
            initial_prompt=initial_prompt,
            model_name=model_name,
            device=device,
            preprocess=False,  # already preprocessed
        )

        # Resolve HF token — use pyannote only if token available AND GPU present
        resolved_hf_token = hf_token or settings.hf_token
        diarization_segments: list[DiarizationSegment] = []
        use_pyannote = bool(resolved_hf_token) and _has_cuda()

        if use_pyannote:
            diarization_task = diarize_audio(wav_path, hf_token=resolved_hf_token)
            transcription_result, diarization_segments = await asyncio.gather(
                transcription_task,
                diarization_task,
            )
        else:
            reason = "no CUDA GPU" if resolved_hf_token else "no HF_TOKEN"
            logger.info(
                "pyannote_skipped",
                reason=f"{reason} — using VAD-based diarization fallback",
            )
            transcription_result = await transcription_task

        logger.info(
            "parallel_tasks_completed",
            transcript_segments=len(transcription_result.segments),
            diarization_segments=len(diarization_segments),
        )

        # Step 2: Align transcript segments with speaker turns
        if diarization_segments:
            # Pyannote path: align Whisper segments with pyannote speaker turns
            aligned_segments = _align_segments(
                transcription_result.segments,
                diarization_segments,
            )
        else:
            # VAD fallback: assign speakers based on silence gaps
            aligned_segments = assign_speakers_by_vad(transcription_result.segments)

        # Step 3: Extract speaker names from introductions (regex)
        speakers = extract_speaker_names(aligned_segments)

        # Step 4: LLM speaker identification (DeepSeek)
        aligned_segments, speakers = await enhance_speaker_names(
            aligned_segments, speakers,
        )

        # Step 5: Merge VAD speaker IDs that the LLM identified as the same person
        aligned_segments, speakers = _merge_speakers_by_name(
            aligned_segments, speakers,
        )

        # Step 6: Apply speaker names to segments
        named_segments = _apply_speaker_names(aligned_segments, speakers)

        # Step 7: Reasoner correction pass (fixes mid-block speaker handoffs)
        named_segments, speakers = await correct_speaker_assignments(
            named_segments, speakers,
        )

        # Build final result
        full_text = " ".join(seg.text for seg in named_segments)
        result = TranscriptionResult(
            segments=named_segments,
            full_text=full_text,
            language=transcription_result.language,
            model_used=transcription_result.model_used,
            duration_seconds=transcription_result.duration_seconds,
            speakers=speakers,
        )

        logger.info(
            "pipeline_completed",
            segments=len(named_segments),
            speakers_total=len(speakers),
            speakers_identified=sum(1 for s in speakers.values() if s.name),
            duration=result.duration_seconds,
        )

        return result

    finally:
        if temp_wav and wav_path.exists() and wav_path != audio_path:
            wav_path.unlink(missing_ok=True)
