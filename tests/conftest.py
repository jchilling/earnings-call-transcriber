"""Shared test fixtures."""

import pytest

from src.sources.taiwan import TaiwanScraper


@pytest.fixture
def scraper():
    """Create a TaiwanScraper with a short rate limit for testing."""
    return TaiwanScraper(rate_limit_secs=0.0)
