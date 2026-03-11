"""VAD-based speaker diarization fallback.

When pyannote is unavailable (no GPU, no HF token), approximates speaker
turns by detecting silence gaps between Whisper segments. A gap longer
than the threshold suggests a speaker change.
"""

import structlog

from src.transcription import TranscriptSegment

logger = structlog.get_logger(__name__)


def assign_speakers_by_vad(
    segments: list[TranscriptSegment],
    silence_threshold: float = 2.0,
) -> list[TranscriptSegment]:
    """Assign speaker IDs based on silence gaps between segments.

    Walks through segments chronologically. When the gap between one
    segment's end and the next segment's start exceeds the threshold,
    increments the speaker counter. This is a rough heuristic — it won't
    catch mid-sentence speaker changes, but it's free (zero extra compute)
    and gives the LLM speaker identifier something to work with.

    Args:
        segments: Whisper transcript segments (no speaker_id yet).
        silence_threshold: Minimum gap in seconds to trigger a new speaker.

    Returns:
        New list of TranscriptSegment with speaker_id populated.
    """
    if not segments:
        return segments

    current_speaker = 0
    result: list[TranscriptSegment] = []

    for i, seg in enumerate(segments):
        if i > 0:
            gap = seg.start_time - segments[i - 1].end_time
            if gap >= silence_threshold:
                current_speaker += 1

        result.append(
            TranscriptSegment(
                text=seg.text,
                start_time=seg.start_time,
                end_time=seg.end_time,
                speaker_id=f"SPEAKER_{current_speaker:02d}",
                speaker_name=seg.speaker_name,
                language=seg.language,
                confidence=seg.confidence,
            )
        )

    unique_speakers = len(set(s.speaker_id for s in result))
    logger.info(
        "vad_diarization_completed",
        total_segments=len(result),
        speakers_detected=unique_speakers,
        silence_threshold=silence_threshold,
    )
    return result
