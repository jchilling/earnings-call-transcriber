"""Speaker diarization using pyannote.audio.

Identifies *who* spoke *when* in an audio file, returning labeled time segments.
Uses pyannote/speaker-diarization-3.1 which requires a HuggingFace token
with accepted license agreements.
"""

import asyncio
import functools
from dataclasses import dataclass
from pathlib import Path

import structlog

from src.config import settings
from src.exceptions import DiarizationError

logger = structlog.get_logger(__name__)

# Cache the pipeline — it takes several seconds to load
_pipeline_cache: dict[str, object] = {}


@dataclass(frozen=True)
class DiarizationSegment:
    """A single speaker turn from diarization.

    Attributes:
        speaker_id: Speaker label assigned by pyannote (e.g. "SPEAKER_00").
        start_time: Start of this turn in seconds.
        end_time: End of this turn in seconds.
    """

    speaker_id: str
    start_time: float
    end_time: float


def _load_pipeline(hf_token: str) -> object:
    """Load and cache the pyannote speaker diarization pipeline.

    Args:
        hf_token: HuggingFace API token with pyannote license access.

    Returns:
        Loaded pyannote Pipeline instance.

    Raises:
        DiarizationError: If the pipeline cannot be loaded.
    """
    if "default" in _pipeline_cache:
        return _pipeline_cache["default"]

    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise DiarizationError(
            "pyannote-audio is not installed. Install with: pip install pyannote-audio"
        ) from e

    if not hf_token:
        raise DiarizationError(
            "HuggingFace token is required for pyannote speaker diarization. "
            "Set HF_TOKEN in your .env file."
        )

    logger.info("loading_diarization_pipeline")
    try:
        # torch 2.6+ defaults torch.load to weights_only=True, but pyannote 3.x
        # checkpoints contain custom classes (TorchVersion, Specifications, etc.)
        # that aren't in the safe globals list. Temporarily set weights_only=False
        # for loading trusted pyannote models from HuggingFace.
        import torch
        _original_load = torch.load

        def _patched_load(*args: object, **kwargs: object) -> object:
            kwargs["weights_only"] = False
            return _original_load(*args, **kwargs)

        torch.load = _patched_load

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )

        # Restore original torch.load
        torch.load = _original_load
    except Exception as e:
        torch.load = _original_load  # restore even on failure
        raise DiarizationError(f"Failed to load diarization pipeline: {e}") from e

    _pipeline_cache["default"] = pipeline
    logger.info("diarization_pipeline_loaded")
    return pipeline


def _diarize_sync(audio_path: str, hf_token: str) -> list[DiarizationSegment]:
    """Run speaker diarization synchronously (called in executor).

    Args:
        audio_path: Path to audio file (WAV recommended).
        hf_token: HuggingFace API token.

    Returns:
        List of DiarizationSegment ordered by start time.
    """
    pipeline = _load_pipeline(hf_token)

    logger.info("diarization_started", audio=audio_path)
    diarization = pipeline(audio_path)

    segments: list[DiarizationSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            DiarizationSegment(
                speaker_id=speaker,
                start_time=turn.start,
                end_time=turn.end,
            )
        )

    logger.info("diarization_completed", speakers=len(set(s.speaker_id for s in segments)),
                segments=len(segments))
    return segments


async def diarize_audio(
    audio_path: str | Path,
    hf_token: str | None = None,
) -> list[DiarizationSegment]:
    """Perform speaker diarization on an audio file.

    Runs pyannote's speaker-diarization-3.1 pipeline in a thread pool
    executor since it's CPU/GPU bound.

    Args:
        audio_path: Path to the audio file.
        hf_token: HuggingFace token. Falls back to settings.hf_token.

    Returns:
        List of DiarizationSegment sorted by start time.

    Raises:
        DiarizationError: If diarization fails.
    """
    audio_path = Path(audio_path)
    hf_token = hf_token or settings.hf_token

    loop = asyncio.get_running_loop()
    try:
        segments = await loop.run_in_executor(
            None,
            functools.partial(_diarize_sync, str(audio_path), hf_token),
        )
    except DiarizationError:
        raise
    except Exception as e:
        raise DiarizationError(f"Diarization failed: {e}") from e

    return segments
