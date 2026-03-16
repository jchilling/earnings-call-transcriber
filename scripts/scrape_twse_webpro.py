"""TWSE WebPro 法人說明會 (investor conference) audio scraper.

Scrapes earnings call audio from TWSE's WebPro VOD platform. All 2,674 videos
are TWSE-hosted with HLS streams on Wowza — no authentication required.

Usage:
    PYTHONPATH=. python scripts/scrape_twse_webpro.py
    PYTHONPATH=. python scripts/scrape_twse_webpro.py --tickers 2330 2317
    PYTHONPATH=. python scripts/scrape_twse_webpro.py --limit 50
    PYTHONPATH=. python scripts/scrape_twse_webpro.py --upload-hf
    PYTHONPATH=. python scripts/scrape_twse_webpro.py --dry-run
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────

LISTING_URL = "https://webpro.twse.com.tw/WebPortal/service/vodChannel/categoryMaterialList"
OUTPUT_DIR = Path("data/audio/twse_webpro")
RESULTS_FILE = OUTPUT_DIR / "download_results.json"
PAGE_SIZE = 100

# ─── Helpers ────────────────────────────────────────────────────────────────

def infer_quarter(event_date: str) -> tuple[int, int]:
    """Infer fiscal quarter from event date (YYYY-MM-DD ...).

    Returns (quarter, fiscal_year).
    """
    date_part = event_date[:10]
    year, month, _ = date_part.split("-")
    year = int(year)
    month = int(month)
    if month <= 3:
        return 4, year - 1
    elif month <= 6:
        return 1, year
    elif month <= 9:
        return 2, year
    else:
        return 3, year


def make_output_path(ticker: str, event_date: str, guid: str) -> Path:
    """Build output path: data/audio/twse_webpro/{ticker}/{ticker}_Q{q}_{date}_{guid_short}.mp3"""
    date_part = event_date[:10].replace("-", "")
    quarter, _ = infer_quarter(event_date)
    guid_short = guid[:8]
    filename = f"{ticker}_Q{quarter}_{date_part}_{guid_short}.mp3"
    return OUTPUT_DIR / ticker / filename


def extract_ticker(material: dict) -> str:
    """Extract stock ticker from material entry."""
    # agentUserName is the ticker (e.g. "1504", "2330")
    return material.get("agentUserName", "unknown").strip()


# ─── Core Scraper ───────────────────────────────────────────────────────────

class TWSEWebProScraper:
    """Scrapes earnings call audio from TWSE WebPro VOD."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                verify=False,  # TWSE has SSL cert issues (Missing Subject Key Identifier)
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Listing ─────────────────────────────────────────────────────────

    async def list_all_materials(self) -> list[dict]:
        """Fetch all VIDEO materials from 法人說明會 category, paginated."""
        client = await self._get_client()
        all_materials: list[dict] = []
        page = 1

        while True:
            resp = await client.post(
                LISTING_URL,
                data={
                    "categoryId": 101,
                    "vodChannelId": 101,
                    "returnType": "json",
                    "platform": "web",
                    "pageNumber": page,
                    "pagingSize": PAGE_SIZE,
                    "order": "eventDate",
                    "sortOrder": "desc",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            materials_wrapper = data.get("result", {}).get("materials", {})
            materials = materials_wrapper.get("material", [])

            if not materials:
                break

            all_materials.extend(materials)
            logger.info("Fetched page %d: %d items (total: %d)", page, len(materials), len(all_materials))

            if len(materials) < PAGE_SIZE:
                break

            page += 1
            await asyncio.sleep(0.5)

        logger.info("Total materials: %d", len(all_materials))
        return all_materials

    # ── Download ────────────────────────────────────────────────────────

    async def check_hls_exists(self, hls_url: str) -> bool:
        """Quick HEAD check to see if the HLS playlist exists (skip 404s fast)."""
        try:
            client = await self._get_client()
            resp = await client.head(hls_url, timeout=10)
            return resp.status_code == 200
        except Exception:
            return False

    async def download_audio(self, hls_url: str, output_path: Path) -> str:
        """Download HLS stream to MP3 via ffmpeg.

        Returns: "ok", "not_found" (404), or "ffmpeg_failed".
        """
        # Quick check if the HLS stream exists before spawning ffmpeg
        if not await self.check_hls_exists(hls_url):
            logger.debug("HLS 404: %s", output_path.name)
            return "not_found"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", hls_url,
            "-vn",
            "-acodec", "libmp3lame",
            "-q:a", "2",
            str(output_path),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)

        if proc.returncode != 0:
            err = stderr.decode()[-500:]
            logger.error("ffmpeg failed for %s: %s", output_path.name, err)
            return "ffmpeg_failed"

        if output_path.exists() and output_path.stat().st_size > 0:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info("Downloaded %s (%.1f MB)", output_path.name, size_mb)
            return "ok"

        logger.error("ffmpeg produced empty file for %s", output_path.name)
        return "ffmpeg_failed"

    # ── Results tracking ────────────────────────────────────────────────

    @staticmethod
    def load_results() -> dict:
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                return json.load(f)
        return {"downloads": {}, "errors": {}, "stats": {}}

    @staticmethod
    def save_results(results: dict) -> None:
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Main run loop ───────────────────────────────────────────────────

    async def run(
        self,
        tickers: list[str] | None = None,
        limit: int | None = None,
        dry_run: bool = False,
        upload_hf: bool = False,
        concurrency: int = 5,
    ) -> None:
        """Main scraper entry point."""
        # 1. List all materials
        materials = await self.list_all_materials()

        # 2. Filter to VIDEO type only (should already be, but be safe)
        materials = [m for m in materials if m.get("type") == "VIDEO"]

        # 3. Filter by tickers
        if tickers:
            ticker_set = set(tickers)
            materials = [m for m in materials if extract_ticker(m) in ticker_set]
            logger.info("Filtered to %d materials for tickers: %s", len(materials), tickers)

        # 4. Apply limit
        if limit:
            materials = materials[:limit]
            logger.info("Limited to %d materials", limit)

        # 5. Dry run
        if dry_run:
            print(f"\n{'─' * 90}")
            print(f"DRY RUN — {len(materials)} videos to download:")
            print(f"{'─' * 90}")
            for m in materials:
                ticker = extract_ticker(m)
                event_date = m["eventDate"][:10]
                quarter, fy = infer_quarter(m["eventDate"])
                duration = m.get("contentLength", 0)
                duration_min = duration // 60 if duration else "?"
                company = m.get("agentSimpleName", "")
                output = make_output_path(ticker, m["eventDate"], m["guid"])
                exists = "EXISTS" if output.exists() else "NEW"
                print(
                    f"  [{exists}] {ticker:>6s} {company:12s} "
                    f"{event_date}  Q{quarter} FY{fy}  ~{duration_min}min  "
                    f"views={m.get('clickCount', 0):>5d}  {m['guid'][:12]}..."
                )
            unique_tickers = len(set(extract_ticker(m) for m in materials))
            print(f"\n{len(materials)} videos across {unique_tickers} tickers")
            return

        # 6. Build download queue — skip already-done items first
        results = self.load_results()
        queue: list[dict] = []
        skipped_count = 0

        for m in materials:
            guid = m["guid"]
            ticker = extract_ticker(m)
            event_date = m["eventDate"][:10]
            output_path = make_output_path(ticker, m["eventDate"], guid)

            # Skip if in results and file exists
            if guid in results["downloads"]:
                existing_path = Path(results["downloads"][guid].get("path", ""))
                if existing_path.exists() and existing_path.stat().st_size > 0:
                    skipped_count += 1
                    continue

            # Skip if file exists on disk but not in results
            if output_path.exists() and output_path.stat().st_size > 0:
                results["downloads"][guid] = {
                    "path": str(output_path),
                    "ticker": ticker,
                    "company": m.get("agentSimpleName", ""),
                    "event_date": event_date,
                    "guid": guid,
                    "duration_sec": m.get("contentLength", 0),
                }
                skipped_count += 1
                continue

            # Get HLS URL
            hls_url = m.get("iosPathHD") or m.get("hlsPathHD") or m.get("iosPath") or m.get("hlsPath")
            if not hls_url:
                logger.warning("No HLS URL for %s %s %s", ticker, event_date, guid)
                results["errors"][guid] = "no_hls_url"
                continue

            queue.append({"material": m, "hls_url": hls_url, "output_path": output_path})

        self.save_results(results)
        logger.info("Skipped %d already-downloaded. Queue: %d to download (concurrency=%d)",
                     skipped_count, len(queue), concurrency)

        # 7. Concurrent download loop
        sem = asyncio.Semaphore(concurrency)
        downloaded_count = 0
        error_count = 0
        lock = asyncio.Lock()

        not_found_count = 0

        async def download_one(item: dict) -> None:
            nonlocal downloaded_count, error_count, not_found_count
            m = item["material"]
            guid = m["guid"]
            ticker = extract_ticker(m)
            event_date = m["eventDate"][:10]
            output_path = item["output_path"]

            async with sem:
                status = await self.download_audio(item["hls_url"], output_path)

            async with lock:
                if status == "ok":
                    results["downloads"][guid] = {
                        "path": str(output_path),
                        "ticker": ticker,
                        "company": m.get("agentSimpleName", ""),
                        "event_date": event_date,
                        "guid": guid,
                        "name": m.get("name", ""),
                        "duration_sec": m.get("contentLength", 0),
                        "industry": m.get("industryName1", ""),
                        "hls_url": item["hls_url"],
                    }
                    downloaded_count += 1
                elif status == "not_found":
                    results["errors"][guid] = "not_found"
                    not_found_count += 1
                else:
                    results["errors"][guid] = "ffmpeg_failed"
                    error_count += 1

                # Save and log progress every download
                self.save_results(results)
                total = len(queue)
                done = downloaded_count + error_count + not_found_count
                if total > 0:
                    pct = done / total * 100
                    logger.info(
                        "Progress: %d/%d (%.0f%%) — %d ok, %d not_found, %d errors, %d skipped",
                        done, total, pct, downloaded_count, not_found_count, error_count, skipped_count,
                    )

        # Run all downloads concurrently with semaphore limiting
        tasks = [asyncio.create_task(download_one(item)) for item in queue]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Final stats
        results["stats"] = {
            "total_listed": len(materials),
            "downloaded": downloaded_count,
            "skipped": skipped_count,
            "not_found": not_found_count,
            "errors": error_count,
        }
        self.save_results(results)

        print(f"\n{'─' * 80}")
        print(f"Done! Downloaded: {downloaded_count}, Not found: {not_found_count}, "
              f"Skipped: {skipped_count}, Errors: {error_count}")
        print(f"Results saved to: {RESULTS_FILE}")
        print(f"{'─' * 80}")

        # 8. Upload to HuggingFace
        if upload_hf:
            await self._upload_huggingface()

        await self.close()

    # ── HuggingFace upload ──────────────────────────────────────────────

    async def _upload_huggingface(self) -> None:
        """Upload downloaded audio files to HuggingFace."""
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            token_path = Path.home() / ".cache" / "huggingface" / "token"
            if token_path.exists():
                hf_token = token_path.read_text().strip()
                logger.info("Using cached HF token from %s", token_path)

        if not hf_token:
            logger.error("No HF token found. Set HF_TOKEN or run `huggingface-cli login`")
            return

        try:
            from huggingface_hub import HfApi
        except ImportError:
            logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
            return

        repo_id = os.environ.get("HF_REPO", "jchilling/taiwan-earnings-calls")
        api = HfApi(token=hf_token)

        try:
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        except Exception as e:
            logger.error("Failed to create/access HF repo %s: %s", repo_id, e)
            return

        # Upload each ticker directory under twse_webpro/ prefix on HF
        ticker_dirs = sorted([d for d in OUTPUT_DIR.iterdir() if d.is_dir()])

        for ticker_dir in ticker_dirs:
            mp3_files = list(ticker_dir.glob("*.mp3"))
            if not mp3_files:
                continue

            logger.info("Uploading %d files from %s to HF...", len(mp3_files), ticker_dir.name)
            try:
                api.upload_folder(
                    folder_path=str(ticker_dir),
                    repo_id=repo_id,
                    repo_type="dataset",
                    path_in_repo=f"twse_webpro/{ticker_dir.name}",
                    allow_patterns="*.mp3",
                )
                logger.info("Uploaded %s", ticker_dir.name)
            except Exception as e:
                logger.error("Failed to upload %s: %s", ticker_dir.name, e)

        # Upload results file
        try:
            api.upload_file(
                path_or_fileobj=str(RESULTS_FILE),
                path_in_repo="twse_webpro/download_results.json",
                repo_id=repo_id,
                repo_type="dataset",
            )
        except Exception as e:
            logger.error("Failed to upload results: %s", e)

        logger.info("HuggingFace upload complete!")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape earnings call audio from TWSE WebPro 法人說明會"
    )
    parser.add_argument(
        "--tickers", nargs="+",
        help="Filter to specific stock tickers (e.g. 2330 2317)"
    )
    parser.add_argument(
        "--limit", type=int,
        help="Max number of videos to process"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List videos without downloading"
    )
    parser.add_argument(
        "--upload-hf", action="store_true",
        help="Upload downloaded files to HuggingFace after scraping"
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Number of parallel ffmpeg downloads (default: 5)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    scraper = TWSEWebProScraper()
    asyncio.run(
        scraper.run(
            tickers=args.tickers,
            limit=args.limit,
            dry_run=args.dry_run,
            upload_hf=args.upload_hf,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
