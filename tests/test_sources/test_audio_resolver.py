"""Unit tests for the multi-strategy AudioResolver with mocked strategies."""

from datetime import datetime
from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, patch

import pytest

from src.sources.audio_resolver import (
    AudioResolver,
    AudioStrategy,
    HiNetOTTStrategy,
    IRPageStrategy,
    MOPSLinkStrategy,
    _looks_like_media_url,
)
from src.sources.base import EarningsCallInfo
from src.sources.registry import CompanyRegistry

# --- Helpers ---


def _make_call_info(
    ticker: str = "2330",
    company_name: str = "台積電",
    webcast_url: str | None = None,
    video_info: str = "",
    call_date: datetime | None = None,
) -> EarningsCallInfo:
    return EarningsCallInfo(
        company_name=company_name,
        ticker=ticker,
        exchange="TWSE",
        call_date=call_date or datetime(2026, 1, 16),
        webcast_url=webcast_url,
        fiscal_year=2025,
        fiscal_quarter=4,
        metadata={
            "video_info": video_info,
            "source": "MOPS",
        },
    )


class MockStrategy(AudioStrategy):
    """Test helper: configurable mock strategy."""

    def __init__(
        self,
        name: str = "Mock",
        sid: str = "mock",
        can_handle_result: bool = True,
        resolve_result: str | None = None,
        raise_on_resolve: Exception | None = None,
    ):
        self._name = name
        self._sid = sid
        self._can_handle_result = can_handle_result
        self._resolve_result = resolve_result
        self._raise_on_resolve = raise_on_resolve

    @property
    def name(self) -> str:
        return self._name

    @property
    def strategy_id(self) -> str:
        return self._sid

    async def can_handle(
        self, call_info: EarningsCallInfo, params: dict | None = None
    ) -> bool:
        return self._can_handle_result

    async def resolve(
        self, call_info: EarningsCallInfo, params: dict | None = None
    ) -> str | None:
        if self._raise_on_resolve:
            raise self._raise_on_resolve
        return self._resolve_result


# --- AudioResolver tests ---


class TestAudioResolver:
    @pytest.mark.asyncio
    async def test_returns_first_successful_result(self):
        strategies = [
            MockStrategy("S1", "s1", resolve_result=None),
            MockStrategy("S2", "s2", resolve_result="https://example.com/audio.mp3"),
            MockStrategy("S3", "s3", resolve_result="https://example.com/other.mp3"),
        ]
        resolver = AudioResolver(strategies=strategies)
        result = await resolver.resolve(_make_call_info())
        assert result == "https://example.com/audio.mp3"

    @pytest.mark.asyncio
    async def test_returns_none_when_all_fail(self):
        strategies = [
            MockStrategy("S1", "s1", resolve_result=None),
            MockStrategy("S2", "s2", resolve_result=None),
        ]
        resolver = AudioResolver(strategies=strategies)
        result = await resolver.resolve(_make_call_info())
        assert result is None

    @pytest.mark.asyncio
    async def test_skips_strategies_that_cannot_handle(self):
        strategies = [
            MockStrategy("S1", "s1", can_handle_result=False, resolve_result="should-not-be-returned"),
            MockStrategy("S2", "s2", resolve_result="https://correct.com/audio.mp3"),
        ]
        resolver = AudioResolver(strategies=strategies)
        result = await resolver.resolve(_make_call_info())
        assert result == "https://correct.com/audio.mp3"

    @pytest.mark.asyncio
    async def test_continues_after_strategy_exception(self):
        strategies = [
            MockStrategy("S1", "s1", raise_on_resolve=RuntimeError("boom")),
            MockStrategy("S2", "s2", resolve_result="https://fallback.com/audio.mp3"),
        ]
        resolver = AudioResolver(strategies=strategies)
        result = await resolver.resolve(_make_call_info())
        assert result == "https://fallback.com/audio.mp3"

    @pytest.mark.asyncio
    async def test_empty_strategies_returns_none(self):
        resolver = AudioResolver(strategies=[])
        result = await resolver.resolve(_make_call_info())
        assert result is None

    def test_default_strategies_created(self):
        resolver = AudioResolver()
        names = [s.name for s in resolver.strategies]
        assert "HiNet OTT Live" in names
        assert "MOPS Direct Link" in names
        assert "IR Page Scraping" in names

    @pytest.mark.asyncio
    async def test_caches_winning_strategy(self):
        strategies = [
            MockStrategy("S1", "s1", resolve_result=None),
            MockStrategy("S2", "s2", resolve_result="https://example.com/audio.mp3"),
        ]
        resolver = AudioResolver(strategies=strategies)

        await resolver.resolve(_make_call_info(ticker="2330"))
        assert resolver.strategy_cache == {"2330": "S2"}

    @pytest.mark.asyncio
    async def test_cached_strategy_tried_first(self):
        """After S2 wins for ticker 2330, it should be tried first next time."""
        call_log: list[str] = []

        class TrackingStrategy(MockStrategy):
            async def resolve(self, call_info, params=None):
                call_log.append(self._name)
                return await super().resolve(call_info, params=params)

        strategies = [
            TrackingStrategy("S1", "s1", resolve_result=None),
            TrackingStrategy("S2", "s2", resolve_result="https://example.com/audio.mp3"),
        ]
        resolver = AudioResolver(strategies=strategies)

        # First call: tries S1 then S2
        await resolver.resolve(_make_call_info(ticker="2330"))
        assert call_log == ["S1", "S2"]

        # Second call: S2 is tried first (cached), succeeds immediately
        call_log.clear()
        await resolver.resolve(_make_call_info(ticker="2330"))
        assert call_log == ["S2"]

    @pytest.mark.asyncio
    async def test_cache_falls_back_if_cached_strategy_fails(self):
        """If the cached strategy stops working, fall back to others."""
        resolve_count = 0

        class FlippingStrategy(MockStrategy):
            async def resolve(self, call_info, params=None):
                nonlocal resolve_count
                resolve_count += 1
                if resolve_count <= 1:
                    return "https://example.com/audio.mp3"
                return None

        strategies = [
            MockStrategy("S1", "s1", resolve_result=None),
            FlippingStrategy("S2", "s2"),
        ]
        resolver = AudioResolver(strategies=strategies)

        # First call: S1 returns None, S2 succeeds
        result1 = await resolver.resolve(_make_call_info(ticker="2330"))
        assert result1 == "https://example.com/audio.mp3"
        assert resolver.strategy_cache["2330"] == "S2"

        # Second call: cached S2 fails, falls back to S1 which also returns None
        result2 = await resolver.resolve(_make_call_info(ticker="2330"))
        assert result2 is None

    @pytest.mark.asyncio
    async def test_different_tickers_cached_independently(self):
        strategies = [
            MockStrategy("S1", "s1", resolve_result="https://s1.com/audio.mp3"),
        ]
        resolver = AudioResolver(strategies=strategies)

        await resolver.resolve(_make_call_info(ticker="2330"))
        await resolver.resolve(_make_call_info(ticker="1101"))

        assert resolver.strategy_cache == {"2330": "S1", "1101": "S1"}

    @pytest.mark.asyncio
    async def test_no_cache_when_all_fail(self):
        strategies = [MockStrategy("S1", "s1", resolve_result=None)]
        resolver = AudioResolver(strategies=strategies)

        await resolver.resolve(_make_call_info(ticker="2330"))
        assert resolver.strategy_cache == {}


# --- Registry-driven resolution tests ---


class TestRegistryDrivenResolution:
    @pytest.fixture
    def registry_yaml(self, tmp_path: Path) -> Path:
        content = dedent("""\
            companies:
              - ticker: "2330"
                name: "TSMC"
                name_local: "台積電"
                exchange: "TWSE"
                market_type: "sii"
                sector: "Semiconductors"
                language: "zh"
                ir_url: "https://investor.tsmc.com"
                audio:
                  strategies:
                    - name: "hinet_ott"
                      params:
                        slug: "tsmc"
                        cdn_host: "tsmcvod-ott2b.cdn.hinet.net"
                    - name: "mops_link"
        """)
        p = tmp_path / "companies.yaml"
        p.write_text(content)
        return p

    @pytest.mark.asyncio
    async def test_registry_strategies_used_when_available(self, registry_yaml: Path):
        """When a ticker is in the registry, only its configured strategies are tried."""
        call_log: list[tuple[str, dict | None]] = []

        class LoggingHiNet(HiNetOTTStrategy):
            async def can_handle(self, call_info, params=None):
                return bool(params and params.get("slug"))

            async def resolve(self, call_info, params=None):
                call_log.append(("hinet_ott", params))
                return "https://cdn.example.com/playlist.m3u8"

        class LoggingMOPS(MOPSLinkStrategy):
            async def can_handle(self, call_info, params=None):
                return True

            async def resolve(self, call_info, params=None):
                call_log.append(("mops_link", params))
                return None

        registry = CompanyRegistry(path=registry_yaml)
        strategies = [
            LoggingHiNet(),
            LoggingMOPS(),
            IRPageStrategy(),    # NOT in registry for 2330
        ]
        resolver = AudioResolver(strategies=strategies, registry=registry)
        result = await resolver.resolve(_make_call_info(ticker="2330"))

        assert result == "https://cdn.example.com/playlist.m3u8"
        # Only hinet_ott was tried (it succeeded), MOPS/IR were not in registry
        assert len(call_log) == 1
        assert call_log[0][0] == "hinet_ott"
        assert call_log[0][1]["slug"] == "tsmc"

    @pytest.mark.asyncio
    async def test_unregistered_ticker_uses_all_strategies(self, registry_yaml: Path):
        """For tickers not in the registry, all strategies are tried."""
        registry = CompanyRegistry(path=registry_yaml)
        strategies = [
            MockStrategy("S1", "hinet_ott", resolve_result=None),
            MockStrategy("S2", "mops_link", resolve_result="https://found.com/audio.mp3"),
        ]
        resolver = AudioResolver(strategies=strategies, registry=registry)
        result = await resolver.resolve(_make_call_info(ticker="9999"))

        assert result == "https://found.com/audio.mp3"

    @pytest.mark.asyncio
    async def test_params_passed_to_strategy(self, registry_yaml: Path):
        """Registry params are forwarded to strategy.resolve()."""
        received_params = {}

        class ParamCapture(HiNetOTTStrategy):
            async def can_handle(self, call_info, params=None):
                return True

            async def resolve(self, call_info, params=None):
                received_params.update(params or {})
                return "https://example.com/audio.m3u8"

        registry = CompanyRegistry(path=registry_yaml)
        resolver = AudioResolver(
            strategies=[ParamCapture(), MOPSLinkStrategy()],
            registry=registry,
        )
        await resolver.resolve(_make_call_info(ticker="2330"))

        assert received_params["slug"] == "tsmc"
        assert received_params["cdn_host"] == "tsmcvod-ott2b.cdn.hinet.net"


# --- HiNetOTTStrategy tests ---


class TestHiNetOTTStrategy:
    @pytest.mark.asyncio
    async def test_can_handle_with_params(self):
        strategy = HiNetOTTStrategy()
        assert await strategy.can_handle(
            _make_call_info(), params={"slug": "tsmc"}
        ) is True

    @pytest.mark.asyncio
    async def test_cannot_handle_without_params(self):
        strategy = HiNetOTTStrategy()
        assert await strategy.can_handle(_make_call_info(), params=None) is False
        assert await strategy.can_handle(_make_call_info(), params={}) is False

    @pytest.mark.asyncio
    async def test_resolve_with_matching_video(self):
        from src.sources.hinet_ott import HiNetOTTClient, HiNetVideo

        mock_client = AsyncMock(spec=HiNetOTTClient)
        mock_client.list_videos.return_value = [
            HiNetVideo("v1", "Q4", datetime(2026, 1, 16), "20260116140000"),
        ]
        mock_client.match_video_by_date.return_value = HiNetVideo(
            "v1", "Q4", datetime(2026, 1, 16), "20260116140000"
        )
        mock_client.get_m3u8_url.return_value = "https://cdn.example.com/playlist.m3u8"

        strategy = HiNetOTTStrategy(hinet_client=mock_client)
        result = await strategy.resolve(
            _make_call_info(),
            params={"slug": "tsmc", "cdn_host": "tsmcvod-ott2b.cdn.hinet.net"},
        )
        assert result == "https://cdn.example.com/playlist.m3u8"

    @pytest.mark.asyncio
    async def test_resolve_no_matching_video(self):
        from src.sources.hinet_ott import HiNetOTTClient

        mock_client = AsyncMock(spec=HiNetOTTClient)
        mock_client.list_videos.return_value = []
        mock_client.match_video_by_date.return_value = None

        strategy = HiNetOTTStrategy(hinet_client=mock_client)
        result = await strategy.resolve(
            _make_call_info(), params={"slug": "tsmc"}
        )
        assert result is None


# --- MOPSLinkStrategy tests ---


class TestMOPSLinkStrategy:
    @pytest.mark.asyncio
    async def test_can_handle_with_webcast_url(self):
        strategy = MOPSLinkStrategy()
        info = _make_call_info(webcast_url="https://example.com/webcast")
        assert await strategy.can_handle(info) is True

    @pytest.mark.asyncio
    async def test_can_handle_with_video_info(self):
        strategy = MOPSLinkStrategy()
        info = _make_call_info(video_info="https://example.com/video.mp4")
        assert await strategy.can_handle(info) is True

    @pytest.mark.asyncio
    async def test_cannot_handle_no_links(self):
        strategy = MOPSLinkStrategy()
        info = _make_call_info(webcast_url=None, video_info="")
        assert await strategy.can_handle(info) is False

    @pytest.mark.asyncio
    async def test_resolve_video_info_url(self):
        strategy = MOPSLinkStrategy()
        info = _make_call_info(video_info="Watch at https://stream.example.com/live.m3u8")
        result = await strategy.resolve(info)
        assert result == "https://stream.example.com/live.m3u8"

    @pytest.mark.asyncio
    async def test_resolve_webcast_media_url(self):
        strategy = MOPSLinkStrategy()
        info = _make_call_info(
            webcast_url="https://webcast.example.com/player",
            video_info="",
        )
        result = await strategy.resolve(info)
        assert result == "https://webcast.example.com/player"

    @pytest.mark.asyncio
    async def test_resolve_non_media_webcast_url_returns_none(self):
        strategy = MOPSLinkStrategy()
        info = _make_call_info(
            webcast_url="https://investor.company.com/about",
            video_info="",
        )
        result = await strategy.resolve(info)
        assert result is None


# --- IRPageStrategy tests ---


class TestIRPageStrategy:
    @pytest.mark.asyncio
    async def test_can_handle_with_params(self):
        strategy = IRPageStrategy()
        assert await strategy.can_handle(
            _make_call_info(),
            params={"url_template": "https://ir.company.com/q/{year}"},
        ) is True

    @pytest.mark.asyncio
    async def test_cannot_handle_without_params(self):
        strategy = IRPageStrategy()
        assert await strategy.can_handle(_make_call_info(), params=None) is False
        assert await strategy.can_handle(_make_call_info(), params={}) is False

    def test_find_media_url_with_mp3(self):
        html = '<a href="https://ir.company.com/20260116_call.mp3">Audio</a>'
        info = _make_call_info(call_date=datetime(2026, 1, 16))
        result = IRPageStrategy._find_media_url(html, info)
        assert result == "https://ir.company.com/20260116_call.mp3"

    def test_find_media_url_with_iframe(self):
        html = '<iframe src="https://webcast.example.com/player?id=123"></iframe>'
        info = _make_call_info()
        result = IRPageStrategy._find_media_url(html, info)
        assert result == "https://webcast.example.com/player?id=123"

    def test_find_media_url_no_match(self):
        html = "<html><body><p>No media here</p></body></html>"
        info = _make_call_info()
        result = IRPageStrategy._find_media_url(html, info)
        assert result is None

    def test_find_media_url_with_custom_pattern(self):
        """Custom media_pattern from registry matches different URL patterns."""
        html = '<a href="https://drive.google.com/file/d/2026_q4/view">Download</a>'
        info = _make_call_info(call_date=datetime(2026, 1, 16))
        result = IRPageStrategy._find_media_url(
            html, info, media_pattern=r"drive\.google\.com|\.(mp4|mp3)"
        )
        assert result == "https://drive.google.com/file/d/2026_q4/view"


# --- Utility tests ---


class TestLooksLikeMediaUrl:
    def test_mp3_url(self):
        assert _looks_like_media_url("https://example.com/file.mp3") is True

    def test_m3u8_url(self):
        assert _looks_like_media_url("https://cdn.example.com/playlist.m3u8") is True

    def test_youtube_url(self):
        assert _looks_like_media_url("https://www.youtube.com/watch?v=abc") is True

    def test_webcast_url(self):
        assert _looks_like_media_url("https://webcast.example.com/live") is True

    def test_plain_html_url(self):
        assert _looks_like_media_url("https://www.company.com/about") is False

    def test_ottlive_url(self):
        assert _looks_like_media_url("https://ottlive.hinet.net/something") is True
