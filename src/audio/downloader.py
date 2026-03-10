"""Audio downloader with two backends: ffmpeg (preferred) and yt-dlp (fallback).

ffmpeg streams audio directly from HTTP URLs without downloading the full
video file — critical for large earnings call recordings (often 1–3 GB).
yt-dlp is used as a fallback for sources that need extraction logic
(YouTube, HLS playlists, etc.).
"""

import asyncio
import logging
import re
import shutil
from pathlib import Path

from src.exceptions import AudioDownloadError

logger = logging.getLogger(__name__)

# URL patterns where yt-dlp is needed (it handles auth, extraction, etc.)
_YTDLP_PATTERNS = re.compile(
    r"youtube\.com|youtu\.be|\.m3u8", re.IGNORECASE
)


def _needs_ytdlp(url: str) -> bool:
    """Check if this URL requires yt-dlp instead of plain ffmpeg."""
    return bool(_YTDLP_PATTERNS.search(url))


async def download_audio(
    url: str,
    output_path: str | Path,
    format: str = "mp3",
    timeout_secs: int = 1800,
) -> Path:
    """Download audio from a URL.

    Routing logic:
    - Direct media URLs (.mp4, .mp3, etc.) → ffmpeg stream extraction.
      ffmpeg reads the remote file and writes only the audio track,
      avoiding a full video download.
    - YouTube / HLS (.m3u8) URLs → yt-dlp, which handles authentication,
      playlist resolution, and format selection.

    Args:
        url: Source URL (direct file, M3U8, YouTube, etc.).
        output_path: Where to save the audio. Extension is replaced
            with the requested format.
        format: Output audio format (default: "mp3").
        timeout_secs: Maximum time in seconds (default: 1800 / 30 min).

    Returns:
        Path to the downloaded audio file.

    Raises:
        AudioDownloadError: If download fails or tools are missing.
    """
    if _needs_ytdlp(url):
        return await _download_via_ytdlp(url, output_path, format, timeout_secs)
    return await _download_via_ffmpeg(url, output_path, format, timeout_secs)


async def _download_via_ffmpeg(
    url: str,
    output_path: str | Path,
    format: str,
    timeout_secs: int,
) -> Path:
    """Extract audio from a remote URL using ffmpeg.

    ffmpeg reads the HTTP stream and transcodes only the audio track.
    For a 2 GB video, this downloads ~100–300 MB of audio data instead
    of the full file.
    """
    if not shutil.which("ffmpeg"):
        raise AudioDownloadError("ffmpeg is not installed or not on PATH")

    output_path = Path(output_path).with_suffix(f".{format}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Codec selection based on output format
    codec_map = {
        "mp3": ["libmp3lame", "-q:a", "2"],
        "wav": ["pcm_s16le"],
        "m4a": ["aac", "-b:a", "192k"],
        "flac": ["flac"],
    }
    codec_args = codec_map.get(format, ["libmp3lame", "-q:a", "2"])

    cmd = [
        "ffmpeg",
        "-y",
        "-i", url,
        "-vn",                    # no video
        "-acodec", codec_args[0],
        *codec_args[1:],
        str(output_path),
    ]

    logger.info("ffmpeg_download", url=url, output=str(output_path))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_secs
        )
    except FileNotFoundError:
        raise AudioDownloadError("ffmpeg is not installed or not on PATH")
    except TimeoutError:
        proc.kill()
        raise AudioDownloadError(
            f"ffmpeg download timed out after {timeout_secs}s: {url}"
        )

    if proc.returncode != 0:
        stderr_text = stderr.decode()[-500:] if stderr else "no stderr"
        raise AudioDownloadError(
            f"ffmpeg failed (exit {proc.returncode}): {stderr_text}"
        )

    if not output_path.exists():
        raise AudioDownloadError(f"ffmpeg completed but output not found: {output_path}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("audio_downloaded", path=str(output_path), size_mb=f"{size_mb:.1f}")
    return output_path


async def _download_via_ytdlp(
    url: str,
    output_path: str | Path,
    format: str,
    timeout_secs: int,
) -> Path:
    """Download and extract audio using yt-dlp.

    Used for YouTube URLs and HLS playlists where ffmpeg alone can't
    handle the extraction (authentication, format selection, etc.).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_template = str(output_path.with_suffix(""))

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format", format,
        "--audio-quality", "0",
        "-o", f"{output_template}.%(ext)s",
        url,
    ]

    logger.info("ytdlp_download", url=url, output=str(output_path))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_secs
        )
    except FileNotFoundError:
        raise AudioDownloadError(
            "yt-dlp is not installed or not on PATH. "
            "Install with: pip install yt-dlp"
        )
    except TimeoutError:
        proc.kill()
        raise AudioDownloadError(
            f"yt-dlp download timed out after {timeout_secs}s: {url}"
        )

    if proc.returncode != 0:
        stderr_text = stderr.decode()[-500:] if stderr else "no stderr"
        raise AudioDownloadError(
            f"yt-dlp failed (exit {proc.returncode}): {stderr_text}"
        )

    expected = output_path.with_suffix(f".{format}")
    if expected.exists():
        size_mb = expected.stat().st_size / (1024 * 1024)
        logger.info("audio_downloaded", path=str(expected), size_mb=f"{size_mb:.1f}")
        return expected

    # yt-dlp may produce slightly different filenames
    stem = output_path.stem
    for candidate in output_path.parent.iterdir():
        if candidate.stem == stem and candidate.suffix:
            logger.info("audio_downloaded", path=str(candidate))
            return candidate

    raise AudioDownloadError(
        f"yt-dlp completed but output file not found at {expected}"
    )
