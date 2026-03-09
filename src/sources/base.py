"""Abstract base class for market-specific earnings call scrapers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import httpx


@dataclass
class EarningsCallInfo:
    """Discovered earnings call metadata from a source."""

    company_name: str
    ticker: str
    exchange: str
    call_date: datetime
    audio_url: str | None = None
    webcast_url: str | None = None
    language: str = "en"
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    metadata: dict = field(default_factory=dict)


class BaseScraper(ABC):
    """Abstract base for market-specific scrapers.

    Each subclass implements discovery logic for a specific exchange or
    regulatory filing system. The scraper is responsible for finding
    earnings call announcements and returning structured metadata.
    """

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={"User-Agent": "EarningsCallTranscriber/0.1"},
            )
        return self._client

    @property
    @abstractmethod
    def exchange_code(self) -> str:
        """Short code for the exchange (e.g. 'TWSE', 'HKEX')."""

    @property
    @abstractmethod
    def supported_languages(self) -> list[str]:
        """ISO 639-1 language codes this scraper may encounter."""

    @abstractmethod
    async def discover_calls(
        self,
        start_date: datetime,
        end_date: datetime,
        tickers: list[str] | None = None,
    ) -> list[EarningsCallInfo]:
        """Discover earnings calls published in the given date range.

        Args:
            start_date: Earliest filing/publication date to include.
            end_date: Latest filing/publication date to include.
            tickers: Optional filter to specific tickers.

        Returns:
            List of discovered earnings call metadata.
        """

    @abstractmethod
    async def get_audio_url(self, call_info: EarningsCallInfo) -> str | None:
        """Resolve the direct audio download URL for a discovered call.

        Args:
            call_info: Metadata from discover_calls.

        Returns:
            Direct URL to the audio file, or None if unavailable.
        """

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "BaseScraper":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
