"""Unit tests for the HiNet OTT Live client with mocked HTTP."""

import json
from datetime import datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from src.sources.hinet_ott import (
    HINET_BACKEND_BASE,
    HiNetOTTClient,
    HiNetVideo,
)

# --- Sample responses ---

SAMPLE_JSON_RESPONSE = json.dumps([
    {
        "id": "vid001",
        "title": "2025Q4 法說會",
        "timestamp": "20260116140000",
        "thumbnail": "https://example.com/thumb.jpg",
    },
    {
        "id": "vid002",
        "title": "2025Q3 法說會",
        "timestamp": "20251017140000",
        "thumbnail": "",
    },
])

SAMPLE_HTML_RESPONSE = """
<html><body>
<div class="video-list">
  <a href="/watch/tsmc/20260116140000">2025Q4 法說會</a>
  <a href="/watch/tsmc/20251017140000">2025Q3 法說會</a>
</div>
</body></html>
"""

EMPTY_RESPONSE = """<html><body><p>No videos found</p></body></html>"""


def _mock_response(text: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=text,
        request=httpx.Request("GET", f"{HINET_BACKEND_BASE}/tsmc/"),
    )


# --- Tests ---


class TestListVideos:
    @pytest.mark.asyncio
    async def test_json_response(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(SAMPLE_JSON_RESPONSE)

        client = HiNetOTTClient(http_client=mock_client)
        videos = await client.list_videos("tsmc")

        assert len(videos) == 2
        assert videos[0].timestamp == "20260116140000"
        assert videos[0].title == "2025Q4 法說會"
        assert videos[1].timestamp == "20251017140000"

    @pytest.mark.asyncio
    async def test_html_response(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(SAMPLE_HTML_RESPONSE)

        client = HiNetOTTClient(http_client=mock_client)
        videos = await client.list_videos("tsmc")

        assert len(videos) == 2
        # Sorted newest first
        assert videos[0].timestamp == "20260116140000"

    @pytest.mark.asyncio
    async def test_empty_response(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_response(EMPTY_RESPONSE)

        client = HiNetOTTClient(http_client=mock_client)
        videos = await client.list_videos("tsmc")

        assert videos == []

    @pytest.mark.asyncio
    async def test_http_error_returns_empty(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        client = HiNetOTTClient(http_client=mock_client)
        videos = await client.list_videos("tsmc")

        assert videos == []


class TestGetM3U8Url:
    def test_known_company(self):
        client = HiNetOTTClient()
        url = client.get_m3u8_url("tsmc", "20260116140000")

        assert "tsmcvod-ott2b.cdn.hinet.net" in url
        assert "tsmc" in url
        assert "20260116140000" in url
        assert url.endswith("playlist.m3u8")

    def test_custom_cdn_host(self):
        client = HiNetOTTClient()
        url = client.get_m3u8_url("tsmc", "20260116140000", cdn_host="custom.cdn.net")

        assert "custom.cdn.net" in url
        assert "tsmc" in url

    def test_unknown_company_uses_default_pattern(self):
        client = HiNetOTTClient()
        url = client.get_m3u8_url("newco", "20260116140000")

        # Should use a default CDN hostname pattern
        assert "newco" in url
        assert url.endswith("playlist.m3u8")


class TestMatchVideoByDate:
    def test_exact_match(self):
        client = HiNetOTTClient()
        videos = [
            HiNetVideo("v1", "Q4", datetime(2026, 1, 16), "20260116140000"),
            HiNetVideo("v2", "Q3", datetime(2025, 10, 17), "20251017140000"),
        ]

        match = client.match_video_by_date(videos, datetime(2026, 1, 16))
        assert match is not None
        assert match.video_id == "v1"

    def test_close_match_within_tolerance(self):
        client = HiNetOTTClient()
        videos = [
            HiNetVideo("v1", "Q4", datetime(2026, 1, 16), "20260116140000"),
        ]

        # 2 days off — within default tolerance of 3
        match = client.match_video_by_date(videos, datetime(2026, 1, 18))
        assert match is not None
        assert match.video_id == "v1"

    def test_no_match_outside_tolerance(self):
        client = HiNetOTTClient()
        videos = [
            HiNetVideo("v1", "Q4", datetime(2026, 1, 16), "20260116140000"),
        ]

        match = client.match_video_by_date(videos, datetime(2026, 2, 20))
        assert match is None

    def test_picks_closest_date(self):
        client = HiNetOTTClient()
        videos = [
            HiNetVideo("v1", "Q4", datetime(2026, 1, 16), "20260116140000"),
            HiNetVideo("v2", "Q4b", datetime(2026, 1, 17), "20260117140000"),
        ]

        match = client.match_video_by_date(videos, datetime(2026, 1, 17))
        assert match is not None
        assert match.video_id == "v2"

    def test_skips_videos_without_date(self):
        client = HiNetOTTClient()
        videos = [
            HiNetVideo("v1", "Q4", None, "20260116140000"),
            HiNetVideo("v2", "Q3", datetime(2025, 10, 17), "20251017140000"),
        ]

        match = client.match_video_by_date(videos, datetime(2025, 10, 17))
        assert match is not None
        assert match.video_id == "v2"


class TestJsonVideoParsing:
    def test_parse_json_with_date_code(self):
        client = HiNetOTTClient()
        item = {"id": "abc", "title": "Test", "date_code": "20260116"}
        video = client._parse_json_video(item)
        assert video is not None
        assert video.timestamp == "20260116000000"
        assert video.date == datetime(2026, 1, 16)

    def test_parse_json_short_timestamp_rejected(self):
        client = HiNetOTTClient()
        item = {"id": "abc", "title": "Test", "timestamp": "2026"}
        video = client._parse_json_video(item)
        assert video is None


class TestContextManager:
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with HiNetOTTClient() as client:
            assert client is not None
        # After exit, client should be closed
        assert client._client is None
