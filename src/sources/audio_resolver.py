"""Multi-strategy audio URL resolver for Taiwan earnings calls.

Tries multiple strategies in priority order to find a working audio
URL for a given earnings call. Strategies are pluggable — new sources
can be added without modifying existing code.

Strategy priority (per-company, defined in data/companies.yaml):
1. HiNet OTT Live — HLS streams for companies with known HiNet presence
2. MOPS Direct Link — URLs from MOPS webcast/video columns
3. IR Page — Scrape company IR page with per-company instructions

For unregistered tickers, all strategies are tried in default order.
"""

import logging
import re
from abc import ABC, abstractmethod

import httpx

from src.sources.base import EarningsCallInfo
from src.sources.hinet_ott import HiNetOTTClient

logger = logging.getLogger(__name__)


class AudioStrategy(ABC):
    """Interface for audio URL resolution strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Machine identifier matching YAML config (e.g. 'hinet_ott')."""

    @abstractmethod
    async def can_handle(
        self, call_info: EarningsCallInfo, params: dict | None = None
    ) -> bool:
        """Check if this strategy might work for the given call."""

    @abstractmethod
    async def resolve(
        self, call_info: EarningsCallInfo, params: dict | None = None
    ) -> str | None:
        """Try to resolve an audio/video URL for the call."""


class HiNetOTTStrategy(AudioStrategy):
    """Resolve audio via HiNet OTT Live platform.

    Params (from registry):
        slug: Company slug on HiNet (e.g. "tsmc")
        cdn_host: CDN hostname for M3U8 URLs
    """

    def __init__(self, hinet_client: HiNetOTTClient | None = None) -> None:
        self._client = hinet_client

    @property
    def name(self) -> str:
        return "HiNet OTT Live"

    @property
    def strategy_id(self) -> str:
        return "hinet_ott"

    async def _get_client(self) -> HiNetOTTClient:
        if self._client is None:
            self._client = HiNetOTTClient()
        return self._client

    async def can_handle(
        self, call_info: EarningsCallInfo, params: dict | None = None
    ) -> bool:
        return bool(params and params.get("slug"))

    async def resolve(
        self, call_info: EarningsCallInfo, params: dict | None = None
    ) -> str | None:
        if not params:
            return None
        slug = params.get("slug")
        if not slug:
            return None

        cdn_host = params.get("cdn_host")
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

    MOPS columns 8 and 9 sometimes contain direct URLs to webcast
    platforms or video files. Rarely useful in practice — most contain
    links to IR portal webpages rather than actual media.
    """

    @property
    def name(self) -> str:
        return "MOPS Direct Link"

    @property
    def strategy_id(self) -> str:
        return "mops_link"

    async def can_handle(
        self, call_info: EarningsCallInfo, params: dict | None = None
    ) -> bool:
        if call_info.webcast_url:
            return True
        video_info = call_info.metadata.get("video_info", "")
        return bool(video_info and "http" in video_info)

    async def resolve(
        self, call_info: EarningsCallInfo, params: dict | None = None
    ) -> str | None:
        # Try video_info first (column 9 — more likely to be direct media)
        video_info = call_info.metadata.get("video_info", "")
        if video_info and "http" in video_info:
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

    Params (from registry):
        url_template: URL with {year} and {quarter} placeholders
        link_text: Anchor text to search for (e.g. "音訊播放")
        follow_links: If true, follow matching anchors to find media on second page
        media_pattern: Regex for matching media URLs (overrides default)
    """

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._client = http_client

    @property
    def name(self) -> str:
        return "IR Page Scraping"

    @property
    def strategy_id(self) -> str:
        return "ir_page"

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

    async def can_handle(
        self, call_info: EarningsCallInfo, params: dict | None = None
    ) -> bool:
        return bool(params and params.get("url_template"))

    async def resolve(
        self, call_info: EarningsCallInfo, params: dict | None = None
    ) -> str | None:
        if not params:
            return None

        url_template = params.get("url_template")
        if not url_template:
            return None

        # Fill in placeholders
        fiscal_year = call_info.fiscal_year or call_info.call_date.year
        fiscal_quarter = call_info.fiscal_quarter or 1
        ir_url = url_template.format(year=fiscal_year, quarter=fiscal_quarter)

        client = await self._get_client()
        media_pattern = params.get("media_pattern")
        link_text = params.get("link_text")
        follow_links = params.get("follow_links", False)

        try:
            resp = await client.get(ir_url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("IR page fetch failed for %s: %s", ir_url, exc)
            return None

        # If follow_links + link_text: two-hop resolution
        if follow_links and link_text:
            result = await self._two_hop_resolve(
                resp.text, ir_url, link_text, media_pattern, call_info, client
            )
            if result:
                return result

        # Single-page media search
        return self._find_media_url(
            resp.text, call_info, media_pattern=media_pattern
        )

    async def _two_hop_resolve(
        self,
        html: str,
        base_url: str,
        link_text: str,
        media_pattern: str | None,
        call_info: EarningsCallInfo,
        client: httpx.AsyncClient,
    ) -> str | None:
        """Find an anchor by text, follow the link, search for media."""
        from urllib.parse import urljoin

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Find anchor matching link_text (partial match)
        target_link = None
        for a_tag in soup.find_all("a", href=True):
            if link_text in a_tag.get_text():
                target_link = a_tag["href"]
                break

        if not target_link:
            logger.debug("No link matching '%s' found on %s", link_text, base_url)
            # Fall through to single-page search
            return None

        # Resolve relative URL
        follow_url = urljoin(base_url, target_link)
        logger.info("Following link '%s' → %s", link_text, follow_url)

        try:
            resp2 = await client.get(follow_url)
            resp2.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Two-hop follow failed for %s: %s", follow_url, exc)
            return None

        return self._find_media_url(
            resp2.text, call_info, media_pattern=media_pattern
        )

    @staticmethod
    def _find_media_url(
        html: str,
        call_info: EarningsCallInfo,
        media_pattern: str | None = None,
    ) -> str | None:
        """Search HTML for audio/video links matching the call date."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        if media_pattern:
            extensions_re = re.compile(media_pattern, re.IGNORECASE)
        else:
            extensions_re = re.compile(
                r"\.(mp3|mp4|m3u8|wav|m4a|webm)", re.IGNORECASE
            )

        call_year = str(call_info.call_date.year)
        call_date_str = call_info.call_date.strftime("%Y%m%d")

        # Look for direct media links
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if extensions_re.search(href):
                if call_date_str in href or call_year in href:
                    logger.info("IR page found media link: %s", href)
                    return href

        # Look for iframe/embed sources (webcast players)
        for tag in soup.find_all(["iframe", "embed", "source", "video"]):
            src = tag.get("src") or tag.get("data-src") or ""
            if src and ("http" in src or src.startswith("//")):
                if extensions_re.search(src) or "webcast" in src.lower():
                    full_url = src if src.startswith("http") else f"https:{src}"
                    logger.info("IR page found embedded media: %s", full_url)
                    return full_url

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


# Strategy ID → class mapping for registry-driven instantiation
_STRATEGY_CLASSES: dict[str, type[AudioStrategy]] = {
    "hinet_ott": HiNetOTTStrategy,
    "ir_page": IRPageStrategy,
    "mops_link": MOPSLinkStrategy,
}


class AudioResolver:
    """Dispatches audio URL resolution across multiple strategies.

    When a CompanyRegistry is provided, uses per-company strategy configs
    from YAML. For unregistered tickers, falls back to trying all
    strategies in default order.

    Caches which strategy worked per ticker so subsequent calls for the
    same company try the winning strategy first.
    """

    def __init__(
        self,
        strategies: list[AudioStrategy] | None = None,
        http_client: httpx.AsyncClient | None = None,
        registry: "CompanyRegistry | None" = None,
    ) -> None:
        from src.sources.registry import CompanyRegistry

        if strategies is not None:
            self._strategies = strategies
        else:
            self._strategies = [
                HiNetOTTStrategy(),
                MOPSLinkStrategy(),
                IRPageStrategy(http_client=http_client),
            ]
        self._registry = registry
        # Build strategy_id → strategy instance lookup
        self._strategy_map: dict[str, AudioStrategy] = {
            s.strategy_id: s for s in self._strategies
        }
        # ticker → strategy name that last succeeded
        self._strategy_cache: dict[str, str] = {}

    @property
    def strategies(self) -> list[AudioStrategy]:
        return self._strategies

    @property
    def strategy_cache(self) -> dict[str, str]:
        """Read-only view of the ticker → strategy name cache."""
        return dict(self._strategy_cache)

    def _get_strategy_order(
        self, ticker: str
    ) -> list[tuple[AudioStrategy, dict | None]]:
        """Return (strategy, params) pairs in priority order.

        If the ticker is in the registry, uses the registry's strategy
        order with per-company params. Otherwise falls back to all
        strategies in default order with no params.

        The cache promotes the last-successful strategy to the front.
        """
        # Registry-driven order
        if self._registry is not None:
            configs = self._registry.get_audio_strategies(ticker)
            if configs:
                pairs: list[tuple[AudioStrategy, dict | None]] = []
                for cfg in configs:
                    strategy = self._strategy_map.get(cfg.name)
                    if strategy:
                        pairs.append((strategy, dict(cfg.params) if cfg.params else None))
                return self._apply_cache(ticker, pairs)

        # Fallback: all strategies, no params
        pairs = [(s, None) for s in self._strategies]
        return self._apply_cache(ticker, pairs)

    def _apply_cache(
        self,
        ticker: str,
        pairs: list[tuple[AudioStrategy, dict | None]],
    ) -> list[tuple[AudioStrategy, dict | None]]:
        """Move the cached winning strategy to the front."""
        cached_name = self._strategy_cache.get(ticker)
        if cached_name is None:
            return pairs

        cached_pair = None
        rest = []
        for pair in pairs:
            if pair[0].name == cached_name:
                cached_pair = pair
            else:
                rest.append(pair)

        if cached_pair is not None:
            return [cached_pair, *rest]
        return pairs

    async def resolve(self, call_info: EarningsCallInfo) -> str | None:
        """Try each strategy in order to find an audio URL.

        Args:
            call_info: Earnings call metadata from MOPS discovery.

        Returns:
            Audio/video URL if any strategy succeeds, None otherwise.
        """
        ordered = self._get_strategy_order(call_info.ticker)

        for strategy, params in ordered:
            try:
                if not await strategy.can_handle(call_info, params=params):
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
                url = await strategy.resolve(call_info, params=params)
                if url:
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
