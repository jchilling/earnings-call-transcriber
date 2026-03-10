"""Audio preprocessing: format conversion and normalization via ffmpeg."""

import asyncio
import shutil
import tempfile
from pathlib import Path

import structlog

from src.exceptions import AudioProcessingError

logger = structlog.get_logger(__name__)

# Whisper expects 16kHz mono audio
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


async def convert_to_wav(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Convert an audio file to 16kHz mono WAV suitable for Whisper.

    Args:
        input_path: Path to the source audio file (any format ffmpeg supports).
        output_path: Where to write the output WAV. If None, writes to a temp file.

    Returns:
        Path to the converted WAV file.

    Raises:
        AudioProcessingError: If ffmpeg is not installed or conversion fails.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise AudioProcessingError(f"Input file does not exist: {input_path}")

    if output_path is None:
        suffix = ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        output_path = Path(tmp.name)
        tmp.close()
    else:
        output_path = Path(output_path)

    if not shutil.which("ffmpeg"):
        raise AudioProcessingError("ffmpeg is not installed or not on PATH")

    cmd = [
        "ffmpeg",
        "-y",  # overwrite output
        "-i", str(input_path),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", str(TARGET_CHANNELS),
        "-c:a", "pcm_s16le",  # 16-bit signed little-endian PCM
        str(output_path),
    ]

    logger.info("converting_audio", input=str(input_path), output=str(output_path))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise AudioProcessingError(
            f"ffmpeg conversion failed (exit {proc.returncode}): {stderr.decode()[-500:]}"
        )

    logger.info("audio_converted", output=str(output_path))
    return output_path


async def get_audio_duration(file_path: str | Path) -> float:
    """Get audio duration in seconds using ffprobe.

    Args:
        file_path: Path to the audio file.

    Returns:
        Duration in seconds.

    Raises:
        AudioProcessingError: If ffprobe fails.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise AudioProcessingError(f"File does not exist: {file_path}")

    if not shutil.which("ffprobe"):
        raise AudioProcessingError("ffprobe is not installed or not on PATH")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise AudioProcessingError(f"ffprobe failed: {stderr.decode()[-500:]}")

    try:
        return float(stdout.decode().strip())
    except ValueError as e:
        raise AudioProcessingError(f"Could not parse duration from ffprobe output: {e}") from e
