"""Per-company scraper registry.

Maps tickers to scraper classes and provides industry-based lookups.
Add new scrapers here as they're built.
"""

from importlib import import_module

from scripts.scrapers.base_scraper import BaseAudioScraper

# ticker → (module_path, class_name)
SCRAPER_REGISTRY: dict[str, tuple[str, str]] = {
    "3105": ("scripts.scrapers.get_3105_audio", "WinSemiScraper"),
}

# industry → list of tickers
INDUSTRY_MAP: dict[str, list[str]] = {
    "semiconductor": ["3105"],
}


def get_scraper(ticker: str) -> BaseAudioScraper:
    """Instantiate the scraper for a given ticker."""
    if ticker not in SCRAPER_REGISTRY:
        raise ValueError(f"No scraper registered for ticker {ticker}. Available: {list(SCRAPER_REGISTRY.keys())}")

    module_path, class_name = SCRAPER_REGISTRY[ticker]
    module = import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def get_tickers_for_industry(industry: str) -> list[str]:
    """Get all registered tickers for an industry."""
    industry_lower = industry.lower()
    if industry_lower not in INDUSTRY_MAP:
        raise ValueError(f"Unknown industry '{industry}'. Available: {list(INDUSTRY_MAP.keys())}")
    return INDUSTRY_MAP[industry_lower]


def list_scrapers() -> list[dict]:
    """List all registered scrapers with metadata."""
    result = []
    for ticker, (module_path, class_name) in SCRAPER_REGISTRY.items():
        module = import_module(module_path)
        cls = getattr(module, class_name)
        result.append({
            "ticker": ticker,
            "company": cls.COMPANY_NAME,
            "industry": cls.INDUSTRY,
        })
    return result
