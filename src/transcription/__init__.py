"""Speech-to-text transcription pipeline."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TranscriptSegment:
    """A single segment of transcribed speech.

    Attributes:
        text: The transcribed text for this segment.
        start_time: Start time in seconds from the beginning of the audio.
        end_time: End time in seconds from the beginning of the audio.
        speaker_id: Identifier for the speaker (set after diarization).
        language: Detected language code (e.g. "zh", "ja", "en").
        confidence: Average confidence score from Whisper (0.0–1.0).
    """

    text: str
    start_time: float
    end_time: float
    speaker_id: str | None = None
    language: str | None = None
    confidence: float = 0.0


@dataclass
class TranscriptionResult:
    """Complete result from a transcription run.

    Attributes:
        segments: Ordered list of transcript segments.
        full_text: Concatenated text from all segments.
        language: Primary detected language.
        model_used: Whisper model identifier used for transcription.
        duration_seconds: Total audio duration in seconds.
    """

    segments: list[TranscriptSegment] = field(default_factory=list)
    full_text: str = ""
    language: str = ""
    model_used: str = ""
    duration_seconds: float = 0.0
