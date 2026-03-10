"""Tests for audio preprocessor module."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.audio.preprocessor import convert_to_wav, get_audio_duration
from src.exceptions import AudioProcessingError


class TestConvertToWav:
    @pytest.mark.asyncio
    async def test_missing_input_file_raises(self, tmp_path: Path):
        with pytest.raises(AudioProcessingError, match="does not exist"):
            await convert_to_wav(tmp_path / "nonexistent.mp3")

    @pytest.mark.asyncio
    async def test_missing_ffmpeg_raises(self, tmp_path: Path):
        input_file = tmp_path / "test.mp3"
        input_file.write_bytes(b"\x00" * 100)

        with patch("src.audio.preprocessor.shutil.which", return_value=None):
            with pytest.raises(AudioProcessingError, match="ffmpeg is not installed"):
                await convert_to_wav(input_file)

    @pytest.mark.asyncio
    async def test_successful_conversion(self, tmp_path: Path):
        input_file = tmp_path / "test.mp3"
        input_file.write_bytes(b"\x00" * 100)
        output_file = tmp_path / "output.wav"

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with (
            patch("src.audio.preprocessor.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch(
                "src.audio.preprocessor.asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ) as mock_exec,
        ):
            result = await convert_to_wav(input_file, output_file)

        assert result == output_file
        # Verify ffmpeg was called with correct args
        call_args = mock_exec.call_args[0]
        assert call_args[0] == "ffmpeg"
        assert "-ar" in call_args
        assert "16000" in call_args
        assert "-ac" in call_args
        assert "1" in call_args

    @pytest.mark.asyncio
    async def test_ffmpeg_failure_raises(self, tmp_path: Path):
        input_file = tmp_path / "test.mp3"
        input_file.write_bytes(b"\x00" * 100)

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"Error: codec not found")
        mock_proc.returncode = 1

        with (
            patch("src.audio.preprocessor.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch(
                "src.audio.preprocessor.asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ),
        ):
            with pytest.raises(AudioProcessingError, match="ffmpeg conversion failed"):
                await convert_to_wav(input_file)


class TestGetAudioDuration:
    @pytest.mark.asyncio
    async def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(AudioProcessingError, match="does not exist"):
            await get_audio_duration(tmp_path / "nonexistent.wav")

    @pytest.mark.asyncio
    async def test_successful_duration_query(self, tmp_path: Path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"3723.45\n", b"")
        mock_proc.returncode = 0

        with (
            patch("src.audio.preprocessor.shutil.which", return_value="/usr/bin/ffprobe"),
            patch(
                "src.audio.preprocessor.asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ),
        ):
            duration = await get_audio_duration(audio_file)

        assert duration == pytest.approx(3723.45)

    @pytest.mark.asyncio
    async def test_ffprobe_failure_raises(self, tmp_path: Path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"Error")
        mock_proc.returncode = 1

        with (
            patch("src.audio.preprocessor.shutil.which", return_value="/usr/bin/ffprobe"),
            patch(
                "src.audio.preprocessor.asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ),
        ):
            with pytest.raises(AudioProcessingError, match="ffprobe failed"):
                await get_audio_duration(audio_file)
