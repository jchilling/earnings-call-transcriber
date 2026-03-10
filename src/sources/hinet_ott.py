"""HiNet OTT Live client for Taiwan earnings call video streams.

Many large Taiwanese companies (TSMC, TCC Group, etc.) host their
investor conference webcasts on HiNet's OTT Live platform
(ottlive.hinet.net). This module queries the HiNet backend API to
discover videos and construct HLS (M3U8) playlist URLs for download.

CDN URL pattern:
    https://{cdn_host}/vod_{slug}/_definst_/smil:{slug}/{slug}w_{timestamp}/hd-hls-pc.smil/playlist.m3u8

The backend API provides a video listing at:
    GET https://ottlive.hinet.net/backend/company/{slug}/
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)

# Known company slug → CDN host mappings.
# Discovered by inspecting HiNet OTT Live player network requests.
HINET_COMPANIES: dict[str, dict[str, str]] = {
    "tsmc": {
        "name": "台積電 (TSMC)",
        "ticker": "2330",
        "cdn_host": "tsmcvod-ott2b.cdn.hinet.net",
    },
}

HINET_BACKEND_BASE = "https://ottlive.hinet.net/backend/company"
HINET_DEFAULT_CDN_PATTERN = (
    "https://{cdn_host}/vod_{slug}/_definst_/"
    "smil:{slug}/{slug}w_{timestamp}/hd-hls-pc.smil/playlist.m3u8"
)


@dataclass
class HiNetVideo:
    """A single video entry from the HiNet OTT backend."""

    video_id: str
    title: str
    date: datetime | None
    timestamp: str  # e.g. "20260116140000" — used to build CDN URL
    thumbnail_url: str = ""


class HiNetOTTClient:
    """Client for the HiNet OTT Live backend API.

    Fetches video listings and constructs HLS M3U8 URLs for a given
    company slug.
    """

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Referer": "https://ottlive.hinet.net/",
                },
            )
        return self._client

    async def list_videos(self, slug: str) -> list[HiNetVideo]:
        """Fetch the video listing for a company from HiNet backend.

        Args:
            slug: Company slug on HiNet (e.g. "tsmc").

        Returns:
            List of HiNetVideo entries, newest first.
        """
        client = await self._get_client()
        url = f"{HINET_BACKEND_BASE}/{slug}/"

        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("HiNet API error for slug=%s: %s", slug, exc)
            return []

        return self._parse_video_list(resp.text, slug)

    def _parse_video_list(self, html: str, slug: str) -> list[HiNetVideo]:
        """Parse HiNet backend response into video entries.

        The backend returns HTML with video entries. Each entry has a
        data attribute or link containing the video timestamp identifier.
        We also try to parse JSON responses if the API returns JSON.
        """
        import json

        videos: list[HiNetVideo] = []

        # Try JSON response first (some endpoints return JSON)
        try:
            data = json.loads(html)
            if isinstance(data, list):
                for item in data:
                    video = self._parse_json_video(item)
                    if video:
                        videos.append(video)
                return videos
            if isinstance(data, dict) and "videos" in data:
                for item in data["videos"]:
                    video = self._parse_json_video(item)
                    if video:
                        videos.append(video)
                return videos
        except (json.JSONDecodeError, TypeError):
            pass

        # Fall back to HTML parsing
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Look for video links/entries with timestamp patterns
        # HiNet uses timestamps like "20260116140000" in URLs
        timestamp_pattern = re.compile(r"(\d{14})")

        for link in soup.find_all("a", href=True):
            href = link["href"]
            match = timestamp_pattern.search(href)
            if match:
                ts = match.group(1)
                title = link.get_text(strip=True) or f"{slug} video {ts}"
                videos.append(
                    HiNetVideo(
                        video_id=ts,
                        title=title,
                        date=self._parse_timestamp(ts),
                        timestamp=ts,
                    )
                )

        # Also check data attributes and div elements
        for el in soup.find_all(attrs={"data-video": True}):
            ts = el["data-video"]
            if timestamp_pattern.match(ts):
                title = el.get_text(strip=True) or f"{slug} video {ts}"
                videos.append(
                    HiNetVideo(
                        video_id=ts,
                        title=title,
                        date=self._parse_timestamp(ts),
                        timestamp=ts,
                    )
                )

        # Sort newest first
        videos.sort(key=lambda v: v.timestamp, reverse=True)
        return videos

    @staticmethod
    def _parse_json_video(item: dict) -> HiNetVideo | None:
        """Parse a single video entry from a JSON response."""
        vid = item.get("id") or item.get("video_id") or ""
        title = item.get("title") or item.get("name") or ""
        ts = item.get("timestamp") or item.get("date_code") or str(vid)

        # Normalize timestamp to 14 digits
        ts_clean = re.sub(r"[^0-9]", "", str(ts))
        if len(ts_clean) < 8:
            return None

        # Pad to 14 digits if needed (date only → add 000000)
        if len(ts_clean) == 8:
            ts_clean += "000000"
        elif len(ts_clean) < 14:
            ts_clean = ts_clean.ljust(14, "0")

        date = None
        try:
            date = datetime.strptime(ts_clean[:8], "%Y%m%d")
        except ValueError:
            pass

        return HiNetVideo(
            video_id=str(vid) or ts_clean,
            title=title or f"video {ts_clean}",
            date=date,
            timestamp=ts_clean[:14],
            thumbnail_url=item.get("thumbnail", ""),
        )

    @staticmethod
    def _parse_timestamp(ts: str) -> datetime | None:
        """Parse a 14-digit timestamp string to datetime."""
        try:
            return datetime.strptime(ts[:8], "%Y%m%d")
        except ValueError:
            return None

    def get_m3u8_url(self, slug: str, timestamp: str, cdn_host: str | None = None) -> str:
        """Construct the HLS M3U8 playlist URL for a video.

        Args:
            slug: Company slug (e.g. "tsmc").
            timestamp: 14-digit timestamp (e.g. "20260116140000").
            cdn_host: CDN hostname. If None, looks up from HINET_COMPANIES.

        Returns:
            Full M3U8 playlist URL.
        """
        if cdn_host is None:
            company = HINET_COMPANIES.get(slug, {})
            cdn_host = company.get("cdn_host", f"{slug}vod-ott2b.cdn.hinet.net")

        return HINET_DEFAULT_CDN_PATTERN.format(
            cdn_host=cdn_host,
            slug=slug,
            timestamp=timestamp,
        )

    def match_video_by_date(
        self, videos: list[HiNetVideo], target_date: datetime, max_days_diff: int = 3
    ) -> HiNetVideo | None:
        """Find the video closest to a target date.

        Args:
            videos: List of HiNetVideo entries.
            target_date: The earnings call date to match.
            max_days_diff: Maximum days difference to consider a match.

        Returns:
            Best matching video, or None if no match within tolerance.
        """
        best: HiNetVideo | None = None
        best_diff = float("inf")

        for video in videos:
            if video.date is None:
                continue
            diff = abs((video.date - target_date).days)
            if diff <= max_days_diff and diff < best_diff:
                best = video
                best_diff = diff

        return best

    async def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "HiNetOTTClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
