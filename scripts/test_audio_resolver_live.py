"""Live integration test for the audio resolver.

Discovers earnings calls from MOPS, then tries to resolve audio URLs
using the registry-driven AudioResolver. Results are dumped to JSON.

Usage:
    PYTHONPATH=. python scripts/test_audio_resolver_live.py
    PYTHONPATH=. python scripts/test_audio_resolver_live.py --tickers 2330 3081
    PYTHONPATH=. python scripts/test_audio_resolver_live.py --months 3
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta

from src.sources.audio_resolver import AudioResolver
from src.sources.registry import CompanyRegistry
from src.sources.taiwan import TaiwanScraper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live test: MOPS discovery → audio resolution"
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
        help="Specific tickers to query (default: all registered)",
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

    registry = CompanyRegistry()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.months * 30)
    tickers = args.tickers or registry.list_tickers(exchange="TWSE")

    print("=== Audio Resolver Live Test ===")
    print(f"Registry: {len(registry)} companies loaded")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Date range: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")
    print(f"Rate limit: {args.rate_limit}s")
    print("-" * 80)

    # Step 1: Discover calls from MOPS
    print("\n[1/2] Discovering calls from MOPS...")
    async with TaiwanScraper(
        rate_limit_secs=args.rate_limit, registry=registry
    ) as scraper:
        calls = await scraper.discover_calls(start_date, end_date, tickers)

    if not calls:
        print("No conferences found on MOPS.")
        return 1

    print(f"Found {len(calls)} conference(s)")

    # Step 2: Resolve audio for each call
    print("\n[2/2] Resolving audio URLs...")
    resolver = AudioResolver(registry=registry)

    results = []
    for i, call in enumerate(calls, 1):
        config = registry.get(call.ticker)
        strategy_names = (
            [s.name for s in config.audio_strategies] if config else ["(unregistered)"]
        )

        print(
            f"\n  [{i}/{len(calls)}] {call.ticker} {call.company_name} "
            f"({call.call_date:%Y-%m-%d})"
            f"  strategies={strategy_names}"
        )

        audio_url = await resolver.resolve(call)

        result = {
            "ticker": call.ticker,
            "company_name": call.company_name,
            "call_date": call.call_date.isoformat(),
            "fiscal_year": call.fiscal_year,
            "fiscal_quarter": call.fiscal_quarter,
            "market_type": config.market_type if config else "unknown",
            "webcast_url": call.webcast_url,
            "video_info": call.metadata.get("video_info", ""),
            "resolved_audio_url": audio_url,
            "resolution_strategy": resolver.strategy_cache.get(call.ticker),
        }

        if audio_url:
            print(f"    ✓ Audio URL: {audio_url}")
            print(f"    Strategy: {result['resolution_strategy']}")
        else:
            print("    ✗ No audio URL found")

        results.append(result)

    # Summary
    resolved_count = sum(1 for r in results if r["resolved_audio_url"])
    print(f"\n{'=' * 80}")
    print(f"Results: {resolved_count}/{len(results)} calls resolved with audio URLs")

    # Strategy breakdown
    strategy_counts: dict[str, int] = {}
    for r in results:
        s = r.get("resolution_strategy") or "None"
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    print("\nBy strategy:")
    for strategy, count in sorted(strategy_counts.items()):
        print(f"  {strategy}: {count}")

    # Dump to JSON
    output_path = "data/test_registry_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
