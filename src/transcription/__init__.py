"""Speech-to-text transcription pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TranscriptSegment:
    """A single segment of transcribed speech.

    Attributes:
        text: The transcribed text for this segment.
        start_time: Start time in seconds from the beginning of the audio.
        end_time: End time in seconds from the beginning of the audio.
        speaker_id: Identifier for the speaker (e.g. "SPEAKER_00" from diarization).
        speaker_name: Resolved human name (e.g. "Jeff Su") after name extraction.
        language: Detected language code (e.g. "zh", "ja", "en").
        confidence: Average confidence score from Whisper (0.0–1.0).
        vocal_metrics: Per-segment vocal/prosodic features (from GPU pipeline).
    """

    text: str
    start_time: float
    end_time: float
    speaker_id: str | None = None
    speaker_name: str | None = None
    language: str | None = None
    confidence: float = 0.0
    vocal_metrics: dict[str, Any] | None = None


@dataclass(frozen=True)
class SpeakerInfo:
    """Information about an identified speaker.

    Attributes:
        id: Diarization speaker ID (e.g. "SPEAKER_00").
        name: Resolved human name, or None if unknown.
        title: Role/title (e.g. "CFO", "Director of IR"), or None.
        segments_count: Number of transcript segments attributed to this speaker.
    """

    id: str
    name: str | None = None
    title: str | None = None
    segments_count: int = 0


@dataclass
class TranscriptionResult:
    """Complete result from a transcription run.

    Attributes:
        segments: Ordered list of transcript segments.
        full_text: Concatenated text from all segments.
        language: Primary detected language.
        model_used: Whisper model identifier used for transcription.
        duration_seconds: Total audio duration in seconds.
        speakers: Map of speaker_id → SpeakerInfo with resolved names.
        vocal_profiles: Per-speaker aggregated vocal metrics (from GPU pipeline).
        speaker_embeddings: Speaker embedding vectors for cross-call re-identification.
        audio_quality: Audio quality metrics for the full recording.
    """

    segments: list[TranscriptSegment] = field(default_factory=list)
    full_text: str = ""
    language: str = ""
    model_used: str = ""
    duration_seconds: float = 0.0
    speakers: dict[str, SpeakerInfo] = field(default_factory=dict)
    vocal_profiles: list[dict[str, Any]] = field(default_factory=list)
    speaker_embeddings: dict[str, list[float]] = field(default_factory=dict)
    audio_quality: dict[str, Any] | None = None
