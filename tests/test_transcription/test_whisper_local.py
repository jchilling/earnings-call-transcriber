"""Tests for the local Whisper transcription module."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.exceptions import WhisperModelError
from src.transcription import TranscriptionResult, TranscriptSegment
from src.transcription.whisper_local import (
    CHUNK_DURATION_SEC,
    CHUNK_OVERLAP_SEC,
    _get_model,
    _merge_chunk_segments,
    _model_cache,
    _segments_from_whisper,
    transcribe_audio,
)

# --- Unit tests for pure functions ---


class TestSegmentsFromWhisper:
    def test_converts_raw_segments(self):
        raw = [
            {"text": " Hello world", "start": 0.0, "end": 2.5, "avg_logprob": -0.3},
            {"text": " Revenue grew 15%", "start": 2.5, "end": 5.0, "avg_logprob": -0.2},
        ]
        segments = _segments_from_whisper(raw)
        assert len(segments) == 2
        assert segments[0].text == "Hello world"
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 2.5
        assert segments[0].confidence == -0.3
        assert segments[1].text == "Revenue grew 15%"

    def test_empty_segments(self):
        assert _segments_from_whisper([]) == []

    def test_missing_avg_logprob_defaults_to_zero(self):
        raw = [{"text": " Test", "start": 0.0, "end": 1.0}]
        segments = _segments_from_whisper(raw)
        assert segments[0].confidence == 0.0


class TestMergeChunkSegments:
    def _seg(self, text: str, start: float, end: float) -> TranscriptSegment:
        return TranscriptSegment(text=text, start_time=start, end_time=end)

    def test_single_chunk_returns_as_is(self):
        segs = [self._seg("hello", 0, 1), self._seg("world", 1, 2)]
        merged = _merge_chunk_segments([segs], CHUNK_DURATION_SEC, CHUNK_OVERLAP_SEC)
        assert len(merged) == 2
        assert merged[0].text == "hello"

    def test_empty_input(self):
        assert _merge_chunk_segments([], CHUNK_DURATION_SEC, CHUNK_OVERLAP_SEC) == []

    def test_two_chunks_overlap_dedup(self):
        chunk_dur = 100
        overlap = 10
        step = chunk_dur - overlap  # 90

        # Chunk 0: segments at 0-50 and 50-95
        chunk0 = [self._seg("A", 0, 50), self._seg("B", 50, 95)]
        # Chunk 1: segment at 5 (in overlap zone, should be skipped) and 15 (kept)
        chunk1 = [self._seg("B-dup", 5, 10), self._seg("C", 15, 50)]

        merged = _merge_chunk_segments([chunk0, chunk1], chunk_dur, overlap)

        texts = [s.text for s in merged]
        assert "A" in texts
        assert "B" in texts
        assert "B-dup" not in texts  # overlap zone, skipped
        assert "C" in texts

        # Check time offset applied to chunk 1 segments
        c_seg = [s for s in merged if s.text == "C"][0]
        assert c_seg.start_time == pytest.approx(15 + step)
        assert c_seg.end_time == pytest.approx(50 + step)


# --- Tests for model loading ---


class TestGetModel:
    def setup_method(self):
        _model_cache.clear()

    def teardown_method(self):
        _model_cache.clear()

    @patch("src.transcription.whisper_local.whisper", create=True)
    def test_loads_and_caches_model(self, mock_whisper_module):
        mock_model = MagicMock()
        # We need to patch the import inside the function
        with patch.dict("sys.modules", {"whisper": mock_whisper_module}):
            mock_whisper_module.load_model.return_value = mock_model
            model = _get_model("base", "cpu")
            assert model is mock_model
            # Second call should use cache
            model2 = _get_model("base", "cpu")
            assert model2 is mock_model
            mock_whisper_module.load_model.assert_called_once()

    def test_raises_on_missing_whisper_package(self):
        with patch.dict("sys.modules", {"whisper": None}):
            with pytest.raises(WhisperModelError, match="not installed"):
                _get_model("base", "cpu")


# --- Integration-style tests (mocked Whisper) ---


class TestTranscribeAudio:
    @pytest.fixture
    def fake_wav(self, tmp_path: Path) -> Path:
        """Create a fake WAV file for testing."""
        wav = tmp_path / "test.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)  # minimal fake
        return wav

    @pytest.fixture
    def mock_whisper_result(self):
        return {
            "text": "Good morning. Revenue was strong this quarter.",
            "language": "en",
            "segments": [
                {"text": " Good morning.", "start": 0.0, "end": 1.5, "avg_logprob": -0.25},
                {
                    "text": " Revenue was strong this quarter.",
                    "start": 1.5,
                    "end": 4.0,
                    "avg_logprob": -0.15,
                },
            ],
        }

    @pytest.mark.asyncio
    async def test_basic_transcription(self, fake_wav, mock_whisper_result):
        with (
            patch(
                "src.transcription.whisper_local.convert_to_wav",
                new_callable=AsyncMock,
                return_value=fake_wav,
            ),
            patch(
                "src.transcription.whisper_local.get_audio_duration",
                new_callable=AsyncMock,
                return_value=120.0,
            ),
            patch(
                "src.transcription.whisper_local._transcribe_sync",
                return_value=mock_whisper_result,
            ),
        ):
            result = await transcribe_audio(fake_wav, language="en")

        assert isinstance(result, TranscriptionResult)
        assert result.language == "en"
        assert result.model_used == "large-v3"
        assert len(result.segments) == 2
        assert "Good morning" in result.full_text
        assert "Revenue was strong" in result.full_text
        assert result.duration_seconds == 120.0

    @pytest.mark.asyncio
    async def test_auto_language_detection(self, fake_wav, mock_whisper_result):
        mock_whisper_result["language"] = "zh"

        with (
            patch(
                "src.transcription.whisper_local.convert_to_wav",
                new_callable=AsyncMock,
                return_value=fake_wav,
            ),
            patch(
                "src.transcription.whisper_local.get_audio_duration",
                new_callable=AsyncMock,
                return_value=60.0,
            ),
            patch(
                "src.transcription.whisper_local._transcribe_sync",
                return_value=mock_whisper_result,
            ),
        ):
            result = await transcribe_audio(fake_wav)

        assert result.language == "zh"
        for seg in result.segments:
            assert seg.language == "zh"

    @pytest.mark.asyncio
    async def test_initial_prompt_passed_through(self, fake_wav, mock_whisper_result):
        with (
            patch(
                "src.transcription.whisper_local.convert_to_wav",
                new_callable=AsyncMock,
                return_value=fake_wav,
            ),
            patch(
                "src.transcription.whisper_local.get_audio_duration",
                new_callable=AsyncMock,
                return_value=60.0,
            ),
            patch(
                "src.transcription.whisper_local._transcribe_sync",
                return_value=mock_whisper_result,
            ) as mock_sync,
        ):
            await transcribe_audio(
                fake_wav,
                initial_prompt="TSMC quarterly earnings call. CEO C.C. Wei speaking.",
            )

        # Verify initial_prompt was passed
        call_args = mock_sync.call_args
        assert call_args[0][4] == "TSMC quarterly earnings call. CEO C.C. Wei speaking."

    @pytest.mark.asyncio
    async def test_whisper_failure_raises_model_error(self, fake_wav):
        with (
            patch(
                "src.transcription.whisper_local.convert_to_wav",
                new_callable=AsyncMock,
                return_value=fake_wav,
            ),
            patch(
                "src.transcription.whisper_local.get_audio_duration",
                new_callable=AsyncMock,
                return_value=60.0,
            ),
            patch(
                "src.transcription.whisper_local._transcribe_sync",
                side_effect=RuntimeError("CUDA OOM"),
            ),
        ):
            with pytest.raises(WhisperModelError, match="inference failed"):
                await transcribe_audio(fake_wav)

    @pytest.mark.asyncio
    async def test_skip_preprocessing(self, fake_wav, mock_whisper_result):
        with (
            patch(
                "src.transcription.whisper_local.convert_to_wav",
                new_callable=AsyncMock,
            ) as mock_convert,
            patch(
                "src.transcription.whisper_local.get_audio_duration",
                new_callable=AsyncMock,
                return_value=60.0,
            ),
            patch(
                "src.transcription.whisper_local._transcribe_sync",
                return_value=mock_whisper_result,
            ),
        ):
            await transcribe_audio(fake_wav, preprocess=False)

        mock_convert.assert_not_called()
