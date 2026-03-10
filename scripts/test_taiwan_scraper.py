"""Live integration test for the Taiwan MOPS scraper.

Queries MOPS for investor conference (法說會) announcements for the
top TWSE companies. Intended for manual validation, not CI.

Usage:
    poetry run python scripts/test_taiwan_scraper.py
    poetry run python scripts/test_taiwan_scraper.py --months 3 --tickers 2330 2317
    poetry run python scripts/test_taiwan_scraper.py --rate-limit 3.0
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta

from src.sources.taiwan import TOP_TAIWAN_TICKERS, TaiwanScraper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live test: query MOPS for 法說會 announcements"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Number of months to look back (default: 6)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Specific tickers to query (default: top 10)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=2.0,
        help="Seconds between MOPS requests (default: 2.0)",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.months * 30)
    tickers = args.tickers or list(TOP_TAIWAN_TICKERS.keys())

    print(f"Querying MOPS for {len(tickers)} tickers")
    print(f"Date range: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")
    print(f"Rate limit: {args.rate_limit}s between requests")
    print("-" * 80)

    async with TaiwanScraper(rate_limit_secs=args.rate_limit) as scraper:
        results = await scraper.discover_calls(start_date, end_date, tickers)

    if not results:
        print("\nNo conferences found.")
        return 1

    print(f"\nFound {len(results)} conference(s):\n")
    print(f"{'Ticker':<8} {'Company':<25} {'Date':<12} {'Time':<8} {'Venue'}")
    print("-" * 80)

    for info in results:
        time_str = info.metadata.get("time", "")
        venue = info.metadata.get("venue", "")
        print(
            f"{info.ticker:<8} {info.company_name:<25} "
            f"{info.call_date:%Y-%m-%d}   {time_str:<8} {venue}"
        )

    print(f"\nTotal: {len(results)} conferences from {len(tickers)} tickers")

    # Dump full results to JSON for inspection
    output_path = "scripts/test_taiwan_output.json"
    serializable = []
    for info in results:
        d = asdict(info)
        d["call_date"] = info.call_date.isoformat()
        serializable.append(d)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\nFull results written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
