"""Custom exception hierarchy for the earnings call transcriber."""


class EarningsTranscriberError(Exception):
    """Base exception for all application errors."""


# --- Source / Scraper errors ---


class SourceError(EarningsTranscriberError):
    """Base error for source discovery and scraping."""


class ScraperConnectionError(SourceError):
    """Failed to connect to a data source."""


class ScraperParseError(SourceError):
    """Failed to parse response from a data source."""


class RateLimitError(SourceError):
    """Hit rate limit on a data source."""


# --- Audio errors ---


class AudioError(EarningsTranscriberError):
    """Base error for audio download and processing."""


class AudioDownloadError(AudioError):
    """Failed to download audio file."""


class AudioProcessingError(AudioError):
    """Failed to preprocess audio (format conversion, noise reduction)."""


# --- Transcription errors ---


class TranscriptionError(EarningsTranscriberError):
    """Base error for transcription."""


class WhisperModelError(TranscriptionError):
    """Failed to load or run local Whisper model."""


class WhisperAPIError(TranscriptionError):
    """Failed to call OpenAI Whisper API."""


class DiarizationError(TranscriptionError):
    """Failed to perform speaker diarization."""


# --- Analysis errors ---


class AnalysisError(EarningsTranscriberError):
    """Base error for LLM-powered analysis."""


class LLMAPIError(AnalysisError):
    """Failed to call the LLM API."""


class PromptRenderError(AnalysisError):
    """Failed to render a prompt template."""


class ExtractionError(AnalysisError):
    """Failed to extract structured data from LLM response."""


# --- Database errors ---


class DatabaseError(EarningsTranscriberError):
    """Base error for database operations."""


class RecordNotFoundError(DatabaseError):
    """Requested record does not exist."""


class DuplicateRecordError(DatabaseError):
    """Attempted to create a record that already exists."""
