"""Company registry — loads company metadata and audio strategies from YAML.

Centralizes all per-company configuration (ticker, name, exchange, audio
fetch instructions) into a single YAML file. Adding a new company means
adding a YAML entry, zero code changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Valid strategy names that can appear in companies.yaml
KNOWN_STRATEGIES = {"hinet_ott", "ir_page", "mops_link"}

# Project root — two levels up from src/sources/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_REGISTRY_PATH = _PROJECT_ROOT / "data" / "companies.yaml"


@dataclass
class AudioStrategyConfig:
    """A single audio resolution strategy with its parameters."""

    name: str
    params: dict[str, str | bool] = field(default_factory=dict)


@dataclass
class CompanyConfig:
    """Company metadata and audio resolution configuration."""

    ticker: str
    name: str
    name_local: str
    exchange: str
    market_type: str  # "sii" (listed) or "otc"
    sector: str
    language: str
    ir_url: str
    audio_strategies: list[AudioStrategyConfig] = field(default_factory=list)
    gics_sub_industry: str = ""
    market_cap_usd_b: float | None = None


class CompanyRegistry:
    """Loads and caches company configs from a YAML registry file.

    Usage:
        registry = CompanyRegistry()  # loads data/companies.yaml
        config = registry.get("2330")
        strategies = registry.get_audio_strategies("2330")
        tickers = registry.list_tickers(market_type="otc")
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_REGISTRY_PATH
        self._companies: dict[str, CompanyConfig] = {}
        self._load()

    def _load(self) -> None:
        """Load and validate the YAML registry file."""
        if not self._path.exists():
            logger.warning("Company registry not found: %s", self._path)
            return

        with open(self._path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "companies" not in data:
            logger.warning("Empty or malformed registry: %s", self._path)
            return

        for entry in data["companies"]:
            try:
                config = self._parse_entry(entry)
                self._companies[config.ticker] = config
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "Skipping malformed company entry %s: %s",
                    entry.get("ticker", "?"),
                    exc,
                )

        logger.info(
            "Loaded %d companies from %s", len(self._companies), self._path
        )

    def _parse_entry(self, entry: dict) -> CompanyConfig:
        """Parse a single company dict from YAML into a CompanyConfig."""
        strategies: list[AudioStrategyConfig] = []
        audio_block = entry.get("audio", {})
        for s in audio_block.get("strategies", []):
            name = s["name"]
            if name not in KNOWN_STRATEGIES:
                logger.warning(
                    "Unknown strategy '%s' for ticker %s — skipping",
                    name,
                    entry["ticker"],
                )
                continue
            strategies.append(
                AudioStrategyConfig(name=name, params=s.get("params", {}))
            )

        return CompanyConfig(
            ticker=entry["ticker"],
            name=entry["name"],
            name_local=entry.get("name_local", ""),
            exchange=entry["exchange"],
            market_type=entry.get("market_type", "sii"),
            sector=entry.get("sector", ""),
            language=entry.get("language", "zh"),
            ir_url=entry.get("ir_url", ""),
            audio_strategies=strategies,
            gics_sub_industry=entry.get("gics_sub_industry", ""),
            market_cap_usd_b=entry.get("market_cap_usd_b"),
        )

    def get(self, ticker: str) -> CompanyConfig | None:
        """Look up a company by ticker. Returns None if not registered."""
        return self._companies.get(ticker)

    def list_tickers(
        self,
        exchange: str | None = None,
        market_type: str | None = None,
    ) -> list[str]:
        """List all registered tickers, optionally filtered.

        Args:
            exchange: Filter by exchange code (e.g. "TWSE").
            market_type: Filter by market type ("sii" or "otc").

        Returns:
            Sorted list of matching ticker strings.
        """
        tickers = []
        for config in self._companies.values():
            if exchange and config.exchange != exchange:
                continue
            if market_type and config.market_type != market_type:
                continue
            tickers.append(config.ticker)
        return sorted(tickers)

    def get_audio_strategies(self, ticker: str) -> list[AudioStrategyConfig]:
        """Get the ordered list of audio strategies for a ticker.

        Returns an empty list if the ticker is not registered.
        """
        config = self._companies.get(ticker)
        if config is None:
            return []
        return config.audio_strategies

    def __len__(self) -> int:
        return len(self._companies)

    def __contains__(self, ticker: str) -> bool:
        return ticker in self._companies
