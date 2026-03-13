"""End-to-end test: registry → MOPS discovery → audio resolution → download.

Tests the full pipeline for a small set of companies to verify that
the registry-driven approach actually produces downloadable audio files.

Usage:
    PYTHONPATH=. python scripts/test_registry_e2e.py
    PYTHONPATH=. python scripts/test_registry_e2e.py --tickers 2330
    PYTHONPATH=. python scripts/test_registry_e2e.py --tickers 2330 3081 3105
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from src.audio.downloader import download_audio
from src.sources.audio_resolver import AudioResolver
from src.sources.registry import CompanyRegistry
from src.sources.taiwan import TaiwanScraper

OUTPUT_DIR = Path("data/test_e2e")
RESULTS_FILE = OUTPUT_DIR / "test_registry_e2e_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E2E test: registry → discover → resolve → download"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["2330"],
        help="Tickers to test (default: 2330)",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=3,
        help="Months to look back (default: 3)",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=1,
        help="Max calls to download per ticker (default: 1)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only resolve URLs, don't download",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=300,
        help="Download timeout in seconds (default: 300 / 5 min)",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    registry = CompanyRegistry()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.months * 30)

    print("=" * 80)
    print("END-TO-END REGISTRY TEST")
    print("=" * 80)
    print(f"Registry: {len(registry)} companies loaded")
    print(f"Testing tickers: {args.tickers}")
    print(f"Date range: {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d}")
    print(f"Max calls per ticker: {args.max_calls}")
    print(f"Download: {'SKIP' if args.skip_download else 'YES'}")
    print()

    # Validate tickers are in registry
    for t in args.tickers:
        config = registry.get(t)
        if config:
            strategies = [s.name for s in config.audio_strategies]
            print(f"  {t} {config.name_local} ({config.name}) — "
                  f"market_type={config.market_type}, strategies={strategies}")
        else:
            print(f"  {t} — NOT IN REGISTRY (will use fallback strategies)")
    print()

    # Step 1: MOPS Discovery
    print("[1/3] Discovering calls from MOPS...")
    async with TaiwanScraper(rate_limit_secs=2.0, registry=registry) as scraper:
        calls = await scraper.discover_calls(start_date, end_date, args.tickers)

    if not calls:
        print("  No conferences found. Try a longer --months window.")
        return 1

    print(f"  Found {len(calls)} conference(s):")
    for c in calls:
        print(f"    {c.ticker} {c.company_name} — {c.call_date:%Y-%m-%d} "
              f"Q{c.fiscal_quarter} FY{c.fiscal_year}")
    print()

    # Step 2: Audio URL Resolution
    print("[2/3] Resolving audio URLs...")
    resolver = AudioResolver(registry=registry)

    results = []
    calls_to_download = []

    for call in calls:
        audio_url = await resolver.resolve(call)
        strategy_used = resolver.strategy_cache.get(call.ticker)

        result = {
            "ticker": call.ticker,
            "company_name": call.company_name,
            "call_date": call.call_date.isoformat(),
            "fiscal_quarter": call.fiscal_quarter,
            "fiscal_year": call.fiscal_year,
            "market_type": (registry.get(call.ticker) or type('', (), {'market_type': 'unknown'})).market_type,
            "audio_url": audio_url,
            "strategy": strategy_used,
            "downloaded": False,
            "file_path": None,
            "file_size_mb": None,
            "error": None,
        }

        if audio_url:
            print(f"  ✓ {call.ticker} {call.call_date:%Y-%m-%d} → {audio_url[:80]}...")
            print(f"    Strategy: {strategy_used}")
            # Limit downloads per ticker
            ticker_queued = sum(
                1 for _, _, _ in calls_to_download
                if True  # counted below
            )
            ticker_queued = sum(
                1 for c, _, _ in calls_to_download
                if c.ticker == call.ticker
            )
            if ticker_queued < args.max_calls:
                calls_to_download.append((call, audio_url, len(results)))
        else:
            print(f"  ✗ {call.ticker} {call.call_date:%Y-%m-%d} — no audio URL found")

        results.append(result)

    print()

    # Step 3: Download
    if args.skip_download:
        print("[3/3] Download SKIPPED (--skip-download)")
    elif not calls_to_download:
        print("[3/3] No audio URLs to download")
    else:
        print(f"[3/3] Downloading {len(calls_to_download)} audio file(s)...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for call, url, idx in calls_to_download:
            filename = f"{call.ticker}_{call.company_name}_Q{call.fiscal_quarter}_{call.fiscal_year}_{call.call_date:%Y%m%d}"
            output_path = OUTPUT_DIR / filename

            print(f"\n  Downloading {call.ticker} {call.call_date:%Y-%m-%d}...")
            print(f"    URL: {url[:100]}")
            print(f"    Output: {output_path}")

            try:
                result_path = await download_audio(
                    url, output_path, format="mp3",
                    timeout_secs=args.download_timeout,
                )
                size_mb = result_path.stat().st_size / (1024 * 1024)
                results[idx]["downloaded"] = True
                results[idx]["file_path"] = str(result_path)
                results[idx]["file_size_mb"] = round(size_mb, 1)
                print(f"    ✓ Downloaded: {result_path.name} ({size_mb:.1f} MB)")
            except Exception as exc:
                results[idx]["error"] = str(exc)
                print(f"    ✗ Download failed: {exc}")

    # Summary
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    total = len(results)
    resolved = sum(1 for r in results if r["audio_url"])
    downloaded = sum(1 for r in results if r["downloaded"])
    print(f"  Calls discovered: {total}")
    print(f"  URLs resolved:    {resolved}/{total}")
    print(f"  Files downloaded: {downloaded}/{resolved if resolved else total}")

    if any(r["error"] for r in results):
        print("\n  Errors:")
        for r in results:
            if r["error"]:
                print(f"    {r['ticker']} {r['call_date']}: {r['error']}")

    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results written to {RESULTS_FILE}")

    return 0 if downloaded > 0 or args.skip_download else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
