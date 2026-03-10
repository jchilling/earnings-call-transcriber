"""Multi-strategy audio URL resolver for Taiwan earnings calls.

Tries multiple strategies in priority order to find a working audio
URL for a given earnings call. Strategies are pluggable — new sources
can be added without modifying existing code.

Strategy priority:
1. HiNet OTT Live — HLS streams for companies with known HiNet presence
2. MOPS links — Direct webcast/video URLs from MOPS table columns
3. Company IR page — Scrape individual IR websites for media links
4. YouTube — Search company YouTube channel for the call
"""

import logging
import re
from abc import ABC, abstractmethod

import httpx

from src.sources.base import EarningsCallInfo
from src.sources.hinet_ott import HiNetOTTClient

logger = logging.getLogger(__name__)


# --- Company metadata for audio resolution ---

COMPANY_METADATA: dict[str, dict] = {
    "2330": {  # TSMC
        "name": "台積電",
        "hinet_slug": "tsmc",
        "hinet_cdn": "tsmcvod-ott2b.cdn.hinet.net",
        "ir_url": "https://investor.tsmc.com",
        "youtube_channel": None,
    },
    "1101": {  # TCC Group
        "name": "台泥",
        "hinet_slug": None,
        "ir_url": "https://www.tccgroup.com.tw/investor",
        "youtube_channel": None,
    },
    "2317": {  # Foxconn
        "name": "鴻海",
        "hinet_slug": None,
        "ir_url": "https://www.foxconn.com/en-us/investor-relations",
        "youtube_channel": None,
    },
    "2308": {  # Delta Electronics
        "name": "台達電",
        "hinet_slug": None,
        "ir_url": "https://www.deltaww.com/en-US/ir",
        "youtube_channel": None,
    },
    "2454": {  # MediaTek
        "name": "聯發科",
        "hinet_slug": None,
        "ir_url": "https://corp.mediatek.com/investor-relations",
        "youtube_channel": None,
    },
}


class AudioStrategy(ABC):
    """Interface for audio URL resolution strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    async def can_handle(self, call_info: EarningsCallInfo) -> bool:
        """Check if this strategy might work for the given call.

        Args:
            call_info: Earnings call metadata.

        Returns:
            True if this strategy should be attempted.
        """

    @abstractmethod
    async def resolve(self, call_info: EarningsCallInfo) -> str | None:
        """Try to resolve an audio/video URL for the call.

        Args:
            call_info: Earnings call metadata.

        Returns:
            Audio/video URL if found, None otherwise.
        """


class HiNetOTTStrategy(AudioStrategy):
    """Resolve audio via HiNet OTT Live platform.

    Works for companies with a known HiNet slug (e.g. TSMC → "tsmc").
    Queries the HiNet backend for video listings, matches by date,
    and constructs an HLS M3U8 URL.
    """

    def __init__(self, hinet_client: HiNetOTTClient | None = None) -> None:
        self._client = hinet_client

    @property
    def name(self) -> str:
        return "HiNet OTT Live"

    async def _get_client(self) -> HiNetOTTClient:
        if self._client is None:
            self._client = HiNetOTTClient()
        return self._client

    async def can_handle(self, call_info: EarningsCallInfo) -> bool:
        meta = COMPANY_METADATA.get(call_info.ticker, {})
        return meta.get("hinet_slug") is not None

    async def resolve(self, call_info: EarningsCallInfo) -> str | None:
        meta = COMPANY_METADATA.get(call_info.ticker, {})
        slug = meta.get("hinet_slug")
        if not slug:
            return None

        cdn_host = meta.get("hinet_cdn")
        client = await self._get_client()

        try:
            videos = await client.list_videos(slug)
        except Exception:
            logger.warning("HiNet list_videos failed for slug=%s", slug, exc_info=True)
            return None

        if not videos:
            logger.info("No HiNet videos found for slug=%s", slug)
            return None

        match = client.match_video_by_date(videos, call_info.call_date)
        if match is None:
            logger.info(
                "No HiNet video matching date %s for slug=%s",
                call_info.call_date.date(),
                slug,
            )
            return None

        url = client.get_m3u8_url(slug, match.timestamp, cdn_host=cdn_host)
        logger.info("HiNet resolved: %s → %s", call_info.ticker, url)
        return url


class MOPSLinkStrategy(AudioStrategy):
    """Resolve audio from MOPS webcast/video columns.

    MOPS columns 8 (公司網站) and 9 (影音連結) sometimes contain
    direct URLs to webcast platforms or video files.
    """

    @property
    def name(self) -> str:
        return "MOPS Direct Link"

    async def can_handle(self, call_info: EarningsCallInfo) -> bool:
        # Check if we have a webcast URL or video_info from MOPS
        if call_info.webcast_url:
            return True
        video_info = call_info.metadata.get("video_info", "")
        return bool(video_info and "http" in video_info)

    async def resolve(self, call_info: EarningsCallInfo) -> str | None:
        # Try video_info first (column 9 — more likely to be direct media)
        video_info = call_info.metadata.get("video_info", "")
        if video_info and "http" in video_info:
            # Extract URL from text (may contain surrounding text)
            url_match = re.search(r"https?://\S+", video_info)
            if url_match:
                url = url_match.group(0).rstrip(")")
                logger.info("MOPS video_info resolved: %s → %s", call_info.ticker, url)
                return url

        # Try webcast_url (column 8)
        if call_info.webcast_url and _looks_like_media_url(call_info.webcast_url):
            logger.info(
                "MOPS webcast_url resolved: %s → %s",
                call_info.ticker,
                call_info.webcast_url,
            )
            return call_info.webcast_url

        return None


class IRPageStrategy(AudioStrategy):
    """Resolve audio by scraping the company's IR page.

    Visits the company's investor relations page and looks for links
    to audio/video files or embedded media players related to earnings calls.
    """

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._client = http_client

    @property
    def name(self) -> str:
        return "IR Page Scraping"

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
                },
            )
        return self._client

    async def can_handle(self, call_info: EarningsCallInfo) -> bool:
        meta = COMPANY_METADATA.get(call_info.ticker, {})
        return bool(meta.get("ir_url"))

    async def resolve(self, call_info: EarningsCallInfo) -> str | None:
        meta = COMPANY_METADATA.get(call_info.ticker, {})
        ir_url = meta.get("ir_url")
        if not ir_url:
            return None

        client = await self._get_client()

        try:
            resp = await client.get(ir_url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("IR page fetch failed for %s: %s", ir_url, exc)
            return None

        return self._find_media_url(resp.text, call_info)

    @staticmethod
    def _find_media_url(html: str, call_info: EarningsCallInfo) -> str | None:
        """Search HTML for audio/video links matching the call date."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        media_extensions = re.compile(r"\.(mp3|mp4|m3u8|wav|m4a|webm)", re.IGNORECASE)
        call_year = str(call_info.call_date.year)
        call_date_str = call_info.call_date.strftime("%Y%m%d")

        # Look for direct media links
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if media_extensions.search(href):
                # Prefer links near the call date
                if call_date_str in href or call_year in href:
                    logger.info("IR page found media link: %s", href)
                    return href

        # Look for iframe/embed sources (webcast players)
        for tag in soup.find_all(["iframe", "embed", "source", "video"]):
            src = tag.get("src") or tag.get("data-src") or ""
            if src and ("http" in src or src.startswith("//")):
                if media_extensions.search(src) or "webcast" in src.lower():
                    full_url = src if src.startswith("http") else f"https:{src}"
                    logger.info("IR page found embedded media: %s", full_url)
                    return full_url

        return None


class YouTubeStrategy(AudioStrategy):
    """Resolve audio via YouTube search using yt-dlp.

    Searches for the company's earnings call on YouTube and returns
    a yt-dlp extractable URL.
    """

    @property
    def name(self) -> str:
        return "YouTube Search"

    async def can_handle(self, call_info: EarningsCallInfo) -> bool:
        # YouTube is always worth trying as a last resort
        return True

    async def resolve(self, call_info: EarningsCallInfo) -> str | None:
        import asyncio

        meta = COMPANY_METADATA.get(call_info.ticker, {})
        company_name = meta.get("name") or call_info.company_name
        date_str = call_info.call_date.strftime("%Y")
        quarter = call_info.fiscal_quarter or ""
        q_str = f"Q{quarter}" if quarter else ""

        search_query = f"{company_name} 法說會 {q_str} {date_str}".strip()

        # Use yt-dlp to search YouTube (ytsearch1: returns first result)
        cmd = [
            "yt-dlp",
            "--no-download",
            "--print", "webpage_url",
            f"ytsearch1:{search_query}",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except (TimeoutError, FileNotFoundError) as exc:
            logger.warning("yt-dlp search failed: %s", exc)
            return None

        if proc.returncode != 0:
            logger.debug("yt-dlp search returned no results for: %s", search_query)
            return None

        url = stdout.decode().strip()
        if url and url.startswith("http"):
            logger.info("YouTube resolved: %s → %s", search_query, url)
            return url

        return None


def _looks_like_media_url(url: str) -> bool:
    """Check if a URL looks like it points to audio/video content."""
    media_patterns = [
        r"\.(mp3|mp4|m3u8|wav|m4a|webm|flv|avi)",
        r"youtube\.com",
        r"youtu\.be",
        r"webcast",
        r"livestream",
        r"ottlive",
        r"player",
    ]
    url_lower = url.lower()
    return any(re.search(p, url_lower) for p in media_patterns)


class AudioResolver:
    """Dispatches audio URL resolution across multiple strategies.

    Tries each strategy in priority order until one returns a URL.
    Caches which strategy worked per ticker so subsequent calls for the
    same company try the winning strategy first (avoids wasted network
    calls to strategies that won't work for that company).

    The cache is in-memory only — it lives for the lifetime of this
    resolver instance. No persistence across process restarts.
    """

    def __init__(
        self,
        strategies: list[AudioStrategy] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if strategies is not None:
            self._strategies = strategies
        else:
            self._strategies = [
                HiNetOTTStrategy(),
                MOPSLinkStrategy(),
                IRPageStrategy(http_client=http_client),
                YouTubeStrategy(),
            ]
        # ticker → strategy name that last succeeded
        self._strategy_cache: dict[str, str] = {}

    @property
    def strategies(self) -> list[AudioStrategy]:
        return self._strategies

    @property
    def strategy_cache(self) -> dict[str, str]:
        """Read-only view of the ticker → strategy name cache."""
        return dict(self._strategy_cache)

    def _get_strategy_order(self, ticker: str) -> list[AudioStrategy]:
        """Return strategies ordered with the cached winner first.

        If ticker "1101" previously resolved via "MOPS Direct Link",
        returns [MOPSLinkStrategy, HiNetOTTStrategy, IRPageStrategy, YouTubeStrategy]
        so we try the known-good strategy first without skipping fallbacks.
        """
        cached_name = self._strategy_cache.get(ticker)
        if cached_name is None:
            return list(self._strategies)

        # Move the cached strategy to the front
        cached = None
        rest = []
        for s in self._strategies:
            if s.name == cached_name:
                cached = s
            else:
                rest.append(s)

        if cached is not None:
            return [cached, *rest]
        return list(self._strategies)

    async def resolve(self, call_info: EarningsCallInfo) -> str | None:
        """Try each strategy in order to find an audio URL.

        If a strategy succeeds, it's cached for this ticker so it's
        tried first on subsequent calls.

        Args:
            call_info: Earnings call metadata from MOPS discovery.

        Returns:
            Audio/video URL if any strategy succeeds, None otherwise.
        """
        ordered = self._get_strategy_order(call_info.ticker)

        for strategy in ordered:
            try:
                if not await strategy.can_handle(call_info):
                    logger.debug(
                        "Strategy %s cannot handle %s", strategy.name, call_info.ticker
                    )
                    continue

                logger.info(
                    "Trying strategy %s for %s (%s)",
                    strategy.name,
                    call_info.ticker,
                    call_info.call_date.date(),
                )
                url = await strategy.resolve(call_info)
                if url:
                    # Cache the winning strategy for this ticker
                    self._strategy_cache[call_info.ticker] = strategy.name
                    logger.info(
                        "Resolved via %s: %s → %s",
                        strategy.name,
                        call_info.ticker,
                        url,
                    )
                    return url

                logger.debug("Strategy %s returned no result", strategy.name)

            except Exception:
                logger.warning(
                    "Strategy %s failed for %s",
                    strategy.name,
                    call_info.ticker,
                    exc_info=True,
                )
                continue

        logger.info("No audio URL found for %s (%s)", call_info.ticker, call_info.call_date.date())
        return None
