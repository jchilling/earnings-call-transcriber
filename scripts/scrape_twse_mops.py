"""TWSE WebPro MOPS法說會 audio scraper.

Scrapes earnings call audio from TWSE's MOPS法說會 section (categoryId=148).
~4,825 entries from 2018-present. Direct MP4/MP3 URLs on irconference.twse.com.tw.
Audio extracted via ffmpeg (stream extraction, no re-encoding for MP3 sources).

Usage:
    PYTHONPATH=. python scripts/scrape_twse_mops.py
    PYTHONPATH=. python scripts/scrape_twse_mops.py --tickers 2330 2317
    PYTHONPATH=. python scripts/scrape_twse_mops.py --limit 50
    PYTHONPATH=. python scripts/scrape_twse_mops.py --dry-run
    PYTHONPATH=. python scripts/scrape_twse_mops.py --concurrency 10
"""

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────

LISTING_URL = "https://webpro.twse.com.tw/WebPortal/service/vodChannel/categoryMaterialList"
CATEGORY_ID = 148  # MOPS法說會
OUTPUT_DIR = Path("data/audio/twse_mops")
RESULTS_FILE = OUTPUT_DIR / "download_results.json"
PAGE_SIZE = 100

# ─── Helpers ────────────────────────────────────────────────────────────────

def infer_quarter(event_date: str) -> tuple[int, int]:
    """Infer fiscal quarter from event date (YYYY-MM-DD ...).

    Returns (quarter, fiscal_year).
    """
    date_part = event_date[:10]
    year, month, _ = date_part.split("-")
    year, month = int(year), int(month)
    if month <= 3:
        return 4, year - 1
    elif month <= 6:
        return 1, year
    elif month <= 9:
        return 2, year
    else:
        return 3, year


def make_output_path(ticker: str, event_date: str, guid: str) -> Path:
    """Build output path: data/audio/twse_mops/{ticker}/{ticker}_Q{q}_{date}_{guid_short}.mp3"""
    date_part = event_date[:10].replace("-", "")
    quarter, _ = infer_quarter(event_date)
    guid_short = guid[:8]
    filename = f"{ticker}_Q{quarter}_{date_part}_{guid_short}.mp3"
    return OUTPUT_DIR / ticker / filename


def extract_ticker(material: dict) -> str:
    return material.get("agentUserName", "unknown").strip()


def get_media_url(material: dict) -> str | None:
    """Extract direct MP4/MP3 URL from material entry."""
    return material.get("webLinkPath") or material.get("linkPath")


# ─── Core Scraper ───────────────────────────────────────────────────────────

class TWSEMopsScraper:
    """Scrapes earnings call audio from TWSE WebPro MOPS法說會."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                verify=False,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Listing ─────────────────────────────────────────────────────────

    async def list_all_materials(self) -> list[dict]:
        """Fetch all materials from MOPS法說會 category, paginated."""
        client = await self._get_client()
        all_materials: list[dict] = []
        page = 1

        while True:
            resp = await client.post(
                LISTING_URL,
                data={
                    "categoryId": CATEGORY_ID,
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

    async def check_url_exists(self, url: str) -> bool:
        """Quick HEAD check to see if the media URL exists."""
        try:
            client = await self._get_client()
            resp = await client.head(url, timeout=15)
            return resp.status_code == 200
        except Exception:
            return False

    async def download_audio(self, media_url: str, output_path: Path) -> str:
        """Download MP4/MP3 and extract audio to MP3 via ffmpeg.

        Returns: "ok", "not_found" (404), or "ffmpeg_failed".
        """
        if not await self.check_url_exists(media_url):
            logger.debug("404: %s", output_path.name)
            return "not_found"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use https for all URLs (some old entries use http)
        media_url = media_url.replace("http://", "https://")

        # For MP3 sources, copy directly. For MP4, extract audio.
        is_mp3_source = media_url.lower().endswith(".mp3")
        if is_mp3_source:
            cmd = [
                "ffmpeg", "-y",
                "-i", media_url,
                "-c", "copy",
                str(output_path),
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", media_url,
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
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=1800)

        if proc.returncode != 0:
            err = stderr.decode()[-500:]
            logger.error("ffmpeg failed for %s: %s", output_path.name, err)
            if output_path.exists():
                output_path.unlink()
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
        concurrency: int = 5,
    ) -> None:
        """Main scraper entry point."""
        materials = await self.list_all_materials()

        # Filter to LINK type (MOPS法說會 uses LINK, not VIDEO)
        materials = [m for m in materials if m.get("type") == "LINK"]

        if tickers:
            ticker_set = set(tickers)
            materials = [m for m in materials if extract_ticker(m) in ticker_set]
            logger.info("Filtered to %d materials for tickers: %s", len(materials), tickers)

        if limit:
            materials = materials[:limit]
            logger.info("Limited to %d materials", limit)

        if dry_run:
            print(f"\n{'─' * 90}")
            print(f"DRY RUN — {len(materials)} MOPS法說會 entries:")
            print(f"{'─' * 90}")
            for m in materials:
                ticker = extract_ticker(m)
                event_date = m["eventDate"][:10]
                quarter, fy = infer_quarter(m["eventDate"])
                company = m.get("agentSimpleName", "")
                url = get_media_url(m) or "NO_URL"
                ext = url.rsplit(".", 1)[-1].upper() if url != "NO_URL" else "?"
                output = make_output_path(ticker, m["eventDate"], m["guid"])
                exists = "EXISTS" if output.exists() else "NEW"
                print(
                    f"  [{exists}] {ticker:>6s} {company:12s} "
                    f"{event_date}  Q{quarter} FY{fy}  {ext:4s} "
                    f"views={m.get('clickCount', 0):>5d}  {m['guid'][:12]}..."
                )
            unique_tickers = len(set(extract_ticker(m) for m in materials))
            print(f"\n{len(materials)} entries across {unique_tickers} tickers")
            return

        # Build download queue
        results = self.load_results()
        queue: list[dict] = []
        skipped_count = 0

        for m in materials:
            guid = m["guid"]
            ticker = extract_ticker(m)
            event_date = m["eventDate"][:10]
            output_path = make_output_path(ticker, m["eventDate"], guid)

            if guid in results["downloads"]:
                existing_path = Path(results["downloads"][guid].get("path", ""))
                if existing_path.exists() and existing_path.stat().st_size > 0:
                    skipped_count += 1
                    continue

            if output_path.exists() and output_path.stat().st_size > 0:
                results["downloads"][guid] = {
                    "path": str(output_path),
                    "ticker": ticker,
                    "company": m.get("agentSimpleName", ""),
                    "event_date": event_date,
                    "guid": guid,
                }
                skipped_count += 1
                continue

            # Skip if previously errored as not_found (permanent)
            if guid in results["errors"] and results["errors"][guid] == "not_found":
                skipped_count += 1
                continue

            media_url = get_media_url(m)
            if not media_url:
                logger.warning("No media URL for %s %s %s", ticker, event_date, guid)
                results["errors"][guid] = "no_url"
                continue

            queue.append({"material": m, "media_url": media_url, "output_path": output_path})

        self.save_results(results)
        logger.info("Skipped %d already-done. Queue: %d to download (concurrency=%d)",
                     skipped_count, len(queue), concurrency)

        # Concurrent download
        sem = asyncio.Semaphore(concurrency)
        downloaded_count = 0
        error_count = 0
        not_found_count = 0
        lock = asyncio.Lock()

        async def download_one(item: dict) -> None:
            nonlocal downloaded_count, error_count, not_found_count
            m = item["material"]
            guid = m["guid"]
            ticker = extract_ticker(m)
            event_date = m["eventDate"][:10]
            output_path = item["output_path"]

            async with sem:
                status = await self.download_audio(item["media_url"], output_path)

            async with lock:
                if status == "ok":
                    results["downloads"][guid] = {
                        "path": str(output_path),
                        "ticker": ticker,
                        "company": m.get("agentSimpleName", ""),
                        "event_date": event_date,
                        "guid": guid,
                        "name": m.get("name", ""),
                        "industry": m.get("industryName1", ""),
                        "media_url": item["media_url"],
                    }
                    downloaded_count += 1
                elif status == "not_found":
                    results["errors"][guid] = "not_found"
                    not_found_count += 1
                else:
                    results["errors"][guid] = "ffmpeg_failed"
                    error_count += 1

                self.save_results(results)
                total = len(queue)
                done = downloaded_count + error_count + not_found_count
                if total > 0:
                    pct = done / total * 100
                    logger.info(
                        "Progress: %d/%d (%.0f%%) — %d ok, %d not_found, %d errors",
                        done, total, pct, downloaded_count, not_found_count, error_count,
                    )

        tasks = [asyncio.create_task(download_one(item)) for item in queue]
        await asyncio.gather(*tasks, return_exceptions=True)

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

        await self.close()


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape earnings call audio from TWSE WebPro MOPS法說會"
    )
    parser.add_argument("--tickers", nargs="+", help="Filter to specific stock tickers")
    parser.add_argument("--limit", type=int, help="Max number of entries to process")
    parser.add_argument("--dry-run", action="store_true", help="List entries without downloading")
    parser.add_argument("--concurrency", type=int, default=5, help="Parallel downloads (default: 5)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    scraper = TWSEMopsScraper()
    asyncio.run(scraper.run(tickers=args.tickers, limit=args.limit, dry_run=args.dry_run, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
