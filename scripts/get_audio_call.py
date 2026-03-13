"""Parent orchestrator for downloading earnings call audio.

Usage:
    # Single ticker, single year
    python scripts/get_audio_call.py --ticker 3105 --year 2024

    # Single ticker, year range (full backfill)
    python scripts/get_audio_call.py --ticker 3105 --start-year 2013 --end-year 2025

    # All scrapers for an industry
    python scripts/get_audio_call.py --industry semiconductor --year 2024

    # List available scrapers
    python scripts/get_audio_call.py --list
"""

import argparse
import asyncio
import logging
import sys

from scripts.scrapers import get_scraper, get_tickers_for_industry, list_scrapers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_ticker(ticker: str, start_year: int, end_year: int) -> dict:
    """Run scraper for a single ticker across a year range."""
    scraper = get_scraper(ticker)
    logger.info("=== %s (%s) — %d to %d ===", scraper.COMPANY_NAME, ticker, start_year, end_year)

    results = await scraper.download_range(start_year, end_year)

    # Summary
    ok = sum(1 for r in results if r.audio_status == "ok")
    unavail = sum(1 for r in results if "unavail" in r.audio_status)
    errors = sum(1 for r in results if r.audio_status == "error")
    pdfs_ok = sum(1 for r in results for p in r.pdfs if p.get("status") == "ok")

    logger.info(
        "Summary for %s: %d audio OK, %d unavailable, %d errors, %d PDFs downloaded",
        ticker, ok, unavail, errors, pdfs_ok,
    )

    return {
        "ticker": ticker,
        "company": scraper.COMPANY_NAME,
        "audio_ok": ok,
        "audio_unavailable": unavail,
        "audio_errors": errors,
        "pdfs_downloaded": pdfs_ok,
        "total_quarters": len(results),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Download earnings call audio and PDFs")
    parser.add_argument("--ticker", type=str, help="Company ticker (e.g. 3105)")
    parser.add_argument("--industry", type=str, help="Industry group (e.g. semiconductor)")
    parser.add_argument("--year", type=int, help="Single year to download")
    parser.add_argument("--start-year", type=int, help="Start of year range")
    parser.add_argument("--end-year", type=int, help="End of year range")
    parser.add_argument("--list", action="store_true", help="List available scrapers")

    args = parser.parse_args()

    if args.list:
        scrapers = list_scrapers()
        print("\nAvailable scrapers:")
        for s in scrapers:
            print(f"  {s['ticker']}  {s['company']:30s}  [{s['industry']}]")
        return

    if not args.ticker and not args.industry:
        parser.error("Specify --ticker or --industry (or --list to see available)")

    # Determine year range
    if args.year:
        start_year = end_year = args.year
    elif args.start_year and args.end_year:
        start_year, end_year = args.start_year, args.end_year
    else:
        parser.error("Specify --year or both --start-year and --end-year")
        return  # unreachable, but helps type checker

    # Determine tickers
    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = get_tickers_for_industry(args.industry)

    # Run
    summaries = []
    for ticker in tickers:
        summary = await run_ticker(ticker, start_year, end_year)
        summaries.append(summary)

    # Final summary
    if len(summaries) > 1:
        print("\n=== Overall Summary ===")
        for s in summaries:
            print(
                f"  {s['ticker']} ({s['company']}): "
                f"{s['audio_ok']}/{s['total_quarters']} audio, "
                f"{s['pdfs_downloaded']} PDFs"
            )


if __name__ == "__main__":
    asyncio.run(main())
