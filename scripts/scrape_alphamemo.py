"""AlphaMemo.ai earnings call scraper — transcripts + audio.

Scrapes transcripts and audio from AlphaMemo.ai's free transcript library
(2,871+ Taiwan earnings calls). Always saves transcript JSON; downloads audio
when HLS stream is available (~7% of transcripts). Optionally uploads to HuggingFace.

Usage:
    PYTHONPATH=. python scripts/scrape_alphamemo.py
    PYTHONPATH=. python scripts/scrape_alphamemo.py --tickers 2330 3715
    PYTHONPATH=. python scripts/scrape_alphamemo.py --limit 50
    PYTHONPATH=. python scripts/scrape_alphamemo.py --upload-hf
    PYTHONPATH=. python scripts/scrape_alphamemo.py --dry-run
"""

import argparse
import asyncio
import json
import logging
import os
import random
import string
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────

SUPABASE_URL = "https://api.alphamemo.ai"
SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVmbGR6dGNjY3Robm5qYmViYmFoIiwi"
    "cm9sZSI6ImFub24iLCJpYXQiOjE3NDE5NTIwNTMsImV4cCI6MjA1NzUyODA1M30."
    "XJDInKWn10xUag0bl0Cu3ZwQ2nQ61ZAL_ClajR22t_I"
)
CDN_BASE = "https://cdn.alphamemo.ai"
OUTPUT_DIR = Path("data/audio/alphamemo")
RESULTS_FILE = OUTPUT_DIR / "download_results.json"

# Columns we're allowed to read via RLS
LISTING_COLUMNS = "id,stock_name,stock_number,audio_date,audio_length_ceil_sec,deck_zh,deck_en"


# ─── Helpers ────────────────────────────────────────────────────────────────

def infer_quarter(audio_date: str) -> tuple[int, int]:
    """Infer fiscal quarter from audio_date (YYYY-MM-DD).

    Earnings calls happen after the quarter ends:
      Jan-Mar call → reporting Q4 of previous year
      Apr-Jun call → reporting Q1 of current year
      Jul-Sep call → reporting Q2 of current year
      Oct-Dec call → reporting Q3 of current year

    Returns (quarter, fiscal_year).
    """
    year, month, _ = audio_date.split("-")
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


def make_base_path(stock_number: str, audio_date: str) -> Path:
    """Build base output path (without extension): data/audio/alphamemo/{ticker}/{ticker}_Q{q}_{date}"""
    quarter, _ = infer_quarter(audio_date)
    date_compact = audio_date.replace("-", "")
    basename = f"{stock_number}_Q{quarter}_{date_compact}"
    return OUTPUT_DIR / stock_number / basename


def format_cookie_header(cookies: dict[str, str]) -> str:
    """Format CloudFront cookies into a Cookie header string."""
    return "; ".join(f"{k}={v}" for k, v in cookies.items())


# ─── Core Scraper ───────────────────────────────────────────────────────────

class AlphaMemoScraper:
    """Scrapes earnings call transcripts and audio from AlphaMemo.ai."""

    def __init__(self) -> None:
        self.access_token: str | None = None
        self.refresh_token: str | None = None
        self.token_expires_at: float = 0
        self.cf_cookies: dict[str, str] = {}
        self.cf_cookies_expires_at: float = 0
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Auth ────────────────────────────────────────────────────────────

    async def authenticate(self) -> None:
        """Authenticate via refresh token, env credentials, or signup."""
        # Priority 1: Saved refresh token (from Google OAuth browser login)
        auth_file = Path("/tmp/alphamemo_auth.json")
        if auth_file.exists():
            auth = json.loads(auth_file.read_text())
            if auth.get("refresh_token"):
                logger.info("Using saved refresh token from %s", auth_file)
                try:
                    await self._refresh_auth_with_token(auth["refresh_token"])
                    return
                except httpx.HTTPStatusError:
                    logger.warning("Saved refresh token failed, trying other methods...")

        # Priority 2: Environment variable credentials
        email = os.environ.get("ALPHAMEMO_EMAIL")
        password = os.environ.get("ALPHAMEMO_PASSWORD")

        if email and password:
            try:
                await self._login(email, password)
                return
            except httpx.HTTPStatusError:
                logger.info("Login failed, trying signup...")

        # Priority 3: Random signup (may fail if signup is disabled)
        rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        email = f"scraper_{rand}@proton.me"
        password = f"Scr4p3r_{rand}!"
        logger.info("Trying signup as %s", email)
        await self._signup(email, password)

    async def _signup(self, email: str, password: str) -> None:
        client = await self._get_client()
        resp = await client.post(
            f"{SUPABASE_URL}/auth/v1/signup",
            headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
            json={"email": email, "password": password},
        )
        resp.raise_for_status()
        data = resp.json()
        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        self.token_expires_at = time.time() + data.get("expires_in", 3600) - 60
        logger.info("Authenticated (signup) — token expires in ~%ds", data.get("expires_in", 3600))

    async def _login(self, email: str, password: str) -> None:
        client = await self._get_client()
        resp = await client.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
            headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
            json={"email": email, "password": password},
        )
        resp.raise_for_status()
        data = resp.json()
        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        self.token_expires_at = time.time() + data.get("expires_in", 3600) - 60
        logger.info("Authenticated (login) — token expires in ~%ds", data.get("expires_in", 3600))

    async def _refresh_auth_with_token(self, refresh_token: str) -> None:
        """Refresh using a specific refresh token (e.g. from saved file)."""
        self.refresh_token = refresh_token
        await self._refresh_auth()

    async def _refresh_auth(self) -> None:
        """Refresh the access token using the refresh token."""
        client = await self._get_client()
        resp = await client.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=refresh_token",
            headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
            json={"refresh_token": self.refresh_token},
        )
        resp.raise_for_status()
        data = resp.json()
        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        self.token_expires_at = time.time() + data.get("expires_in", 3600) - 60
        # Persist updated tokens so next run can reuse them
        auth_file = Path("/tmp/alphamemo_auth.json")
        auth_file.write_text(json.dumps({
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
        }))
        logger.info("Token refreshed (saved to %s)", auth_file)

    async def _ensure_auth(self) -> None:
        """Refresh token if expired. Re-authenticates from scratch on failure."""
        if time.time() > self.token_expires_at:
            try:
                await self._refresh_auth()
            except httpx.HTTPStatusError:
                logger.warning("Token refresh failed, re-authenticating...")
                await self.authenticate()

    # ── Listing ─────────────────────────────────────────────────────────

    async def list_transcripts(self) -> list[dict]:
        """Fetch all transcripts from the free_transcripts table (paginated)."""
        client = await self._get_client()
        all_records: list[dict] = []
        page_size = 1000
        offset = 0

        while True:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/free_transcripts",
                params={
                    "select": LISTING_COLUMNS,
                    "is_accessed": "eq.true",
                    "order": "stock_number.asc,audio_date.desc",
                },
                headers={
                    "apikey": SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                    "Range": f"{offset}-{offset + page_size - 1}",
                    "Prefer": "count=exact",
                },
            )
            resp.raise_for_status()
            records = resp.json()
            all_records.extend(records)

            content_range = resp.headers.get("content-range", "")
            if "/" in content_range:
                total = int(content_range.split("/")[1])
                logger.info("Fetched %d/%d transcripts", len(all_records), total)
                if len(all_records) >= total:
                    break
            else:
                break

            offset += page_size
            await asyncio.sleep(0.5)

        logger.info("Total transcripts: %d", len(all_records))
        return all_records

    # ── Transcript detail + cookies ─────────────────────────────────────

    async def get_transcript_detail(self, transcript_id: str) -> dict:
        """Call get_transcript edge function.

        Returns the full response dict with 'metadata', 'content', and 'fileFormat'.
        Also updates self.cf_cookies with CloudFront signed cookies from response headers.
        """
        await self._ensure_auth()
        client = await self._get_client()

        resp = await client.post(
            f"{SUPABASE_URL}/functions/v1/get_transcript",
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            },
            json={"transcriptId": transcript_id},
        )
        resp.raise_for_status()

        # Extract CloudFront cookies from Set-Cookie headers
        for cookie_header in resp.headers.get_list("set-cookie"):
            if cookie_header.startswith("CloudFront-"):
                name, _, rest = cookie_header.partition("=")
                value = rest.split(";")[0]
                self.cf_cookies[name] = value

        if self.cf_cookies:
            self.cf_cookies_expires_at = time.time() + 3500

        return resp.json()

    # ── Download ────────────────────────────────────────────────────────

    def save_transcript(self, data: dict, output_path: Path) -> None:
        """Save transcript JSON to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        size_kb = output_path.stat().st_size / 1024
        logger.info("Saved transcript %s (%.1f KB)", output_path.name, size_kb)

    async def download_audio(self, streaming_key: str, output_path: Path) -> bool:
        """Download HLS stream to MP3 via ffmpeg. Returns True on success."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        url = f"{CDN_BASE}/{streaming_key}"
        cookie_str = format_cookie_header(self.cf_cookies)

        cmd = [
            "ffmpeg", "-y",
            "-headers", f"Cookie: {cookie_str}\r\n",
            "-i", url,
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
            logger.error("ffmpeg failed for %s: %s", streaming_key, err)
            return False

        if output_path.exists() and output_path.stat().st_size > 0:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info("Downloaded audio %s (%.1f MB)", output_path.name, size_mb)
            return True

        logger.error("ffmpeg produced empty file for %s", streaming_key)
        return False

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
        lang_pref: str = "zh",
    ) -> None:
        """Main scraper entry point."""
        # 1. Authenticate
        await self.authenticate()

        # 2. List all transcripts
        transcripts = await self.list_transcripts()

        # 3. Filter by tickers
        if tickers:
            ticker_set = set(tickers)
            transcripts = [t for t in transcripts if t["stock_number"] in ticker_set]
            logger.info("Filtered to %d transcripts for tickers: %s", len(transcripts), tickers)

        # 4. Apply limit
        if limit:
            transcripts = transcripts[:limit]
            logger.info("Limited to %d transcripts", limit)

        # 5. Dry run
        if dry_run:
            print(f"\n{'─' * 80}")
            print(f"DRY RUN — {len(transcripts)} transcripts to process:")
            print(f"{'─' * 80}")
            for t in transcripts:
                quarter, fy = infer_quarter(t["audio_date"])
                duration = t.get("audio_length_ceil_sec", 0)
                duration_min = duration // 60 if duration else "?"
                base = make_base_path(t["stock_number"], t["audio_date"])
                has_transcript = base.with_suffix(".json").exists()
                has_audio = base.with_suffix(".mp3").exists()
                status = "DONE" if has_transcript else "NEW"
                audio_tag = "+audio" if has_audio else ""
                print(
                    f"  [{status}{audio_tag:>6s}] {t['stock_number']} {t['stock_name']:20s} "
                    f"{t['audio_date']}  Q{quarter} FY{fy}  ~{duration_min}min  "
                    f"id={t['id'][:8]}..."
                )
            unique_tickers = len(set(t["stock_number"] for t in transcripts))
            print(f"\n{len(transcripts)} calls across {unique_tickers} tickers")
            return

        # 6. Download loop
        results = self.load_results()
        transcript_count = 0
        audio_count = 0
        skipped_count = 0
        error_count = 0

        for i, t in enumerate(transcripts):
            transcript_id = t["id"]
            stock_number = t["stock_number"]
            audio_date = t["audio_date"]
            result_key = f"{stock_number}_{audio_date}"

            base_path = make_base_path(stock_number, audio_date)
            transcript_path = base_path.with_suffix(".json")
            audio_path = base_path.with_suffix(".mp3")

            # Skip if transcript already saved
            if result_key in results["downloads"] and transcript_path.exists():
                skipped_count += 1
                continue

            # Also skip if transcript file exists on disk but not in results
            if transcript_path.exists() and transcript_path.stat().st_size > 0:
                results["downloads"][result_key] = {
                    "transcript_path": str(transcript_path),
                    "audio_path": str(audio_path) if audio_path.exists() else None,
                    "stock_number": stock_number,
                    "stock_name": t["stock_name"],
                    "audio_date": audio_date,
                    "transcript_id": transcript_id,
                    "has_audio": audio_path.exists(),
                }
                self.save_results(results)
                skipped_count += 1
                continue

            # Fetch transcript detail from API (retry once on 401)
            try:
                full_response = await self.get_transcript_detail(transcript_id)
                metadata = full_response.get("metadata", {})
                streaming_key = metadata.get("streaming_audio_key")

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    # Token expired mid-run — re-auth and retry once
                    logger.warning("Got 401, re-authenticating and retrying...")
                    await self.authenticate()
                    try:
                        full_response = await self.get_transcript_detail(transcript_id)
                        metadata = full_response.get("metadata", {})
                        streaming_key = metadata.get("streaming_audio_key")
                    except Exception as retry_err:
                        logger.error("Retry failed for %s %s: %s", stock_number, audio_date, retry_err)
                        results["errors"][result_key] = f"retry_failed"
                        self.save_results(results)
                        error_count += 1
                        continue
                elif e.response.status_code == 429:
                    wait = 30 + random.uniform(0, 10)
                    logger.warning("Rate limited, waiting %.0fs...", wait)
                    await asyncio.sleep(wait)
                    results["errors"][result_key] = "rate_limited"
                    self.save_results(results)
                    error_count += 1
                    continue
                else:
                    logger.error(
                        "API error for %s %s: %s %s",
                        stock_number, audio_date, e.response.status_code, e.response.text[:200],
                    )
                    results["errors"][result_key] = f"api_{e.response.status_code}"
                    self.save_results(results)
                    error_count += 1
                    continue

            except Exception as e:
                logger.error("Error getting transcript %s %s: %s", stock_number, audio_date, e)
                results["errors"][result_key] = str(e)
                self.save_results(results)
                error_count += 1
                continue

            # Always save transcript JSON
            self.save_transcript(full_response, transcript_path)
            transcript_count += 1

            # Download audio if streaming key exists
            has_audio = False
            if streaming_key:
                has_audio = await self.download_audio(streaming_key, audio_path)
                if has_audio:
                    audio_count += 1

            # Record result
            results["downloads"][result_key] = {
                "transcript_path": str(transcript_path),
                "audio_path": str(audio_path) if has_audio else None,
                "stock_number": stock_number,
                "stock_name": t["stock_name"],
                "audio_date": audio_date,
                "transcript_id": transcript_id,
                "has_audio": has_audio,
                "streaming_key": streaming_key,
            }
            self.save_results(results)

            # Progress
            processed = transcript_count + error_count
            total = len(transcripts) - skipped_count
            if total > 0:
                pct = processed / total * 100
                logger.info(
                    "Progress: %d/%d (%.0f%%) — %d transcripts, %d with audio, %d errors, %d skipped",
                    processed, total, pct, transcript_count, audio_count, error_count, skipped_count,
                )

            # Rate limit: 2-3s between API calls
            delay = 2.0 + random.uniform(0, 1.0)
            await asyncio.sleep(delay)

        # Final stats
        results["stats"] = {
            "total_listed": len(transcripts),
            "transcripts_saved": transcript_count,
            "audio_downloaded": audio_count,
            "skipped": skipped_count,
            "errors": error_count,
        }
        self.save_results(results)

        print(f"\n{'─' * 80}")
        print(f"Done! Transcripts: {transcript_count}, Audio: {audio_count}, "
              f"Skipped: {skipped_count}, Errors: {error_count}")
        print(f"Results saved to: {RESULTS_FILE}")
        print(f"{'─' * 80}")

        # 7. Upload to HuggingFace
        if upload_hf:
            await self._upload_huggingface()

        await self.close()

    # ── HuggingFace upload ──────────────────────────────────────────────

    async def _upload_huggingface(self) -> None:
        """Upload downloaded transcripts and audio to HuggingFace."""
        hf_token = os.environ.get("HF_TOKEN")

        # Fall back to cached token from `huggingface-cli login`
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

        # Ensure repo exists
        try:
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        except Exception as e:
            logger.error("Failed to create/access HF repo %s: %s", repo_id, e)
            return

        # Upload entire alphamemo directory (all ticker subdirs with .json + .mp3)
        logger.info("Uploading to HuggingFace repo: %s", repo_id)
        ticker_dirs = sorted([d for d in OUTPUT_DIR.iterdir() if d.is_dir()])

        for ticker_dir in ticker_dirs:
            files = list(ticker_dir.glob("*.json")) + list(ticker_dir.glob("*.mp3"))
            if not files:
                continue

            n_json = len(list(ticker_dir.glob("*.json")))
            n_mp3 = len(list(ticker_dir.glob("*.mp3")))
            logger.info("Uploading %s: %d transcripts, %d audio files", ticker_dir.name, n_json, n_mp3)
            try:
                api.upload_folder(
                    folder_path=str(ticker_dir),
                    repo_id=repo_id,
                    repo_type="dataset",
                    path_in_repo=ticker_dir.name,
                    allow_patterns=["*.json", "*.mp3"],
                )
                logger.info("Uploaded %s", ticker_dir.name)
            except Exception as e:
                logger.error("Failed to upload %s: %s", ticker_dir.name, e)

        # Also upload the results file
        try:
            api.upload_file(
                path_or_fileobj=str(RESULTS_FILE),
                path_in_repo="download_results.json",
                repo_id=repo_id,
                repo_type="dataset",
            )
            logger.info("Uploaded download_results.json")
        except Exception as e:
            logger.error("Failed to upload results file: %s", e)

        logger.info("HuggingFace upload complete!")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape earnings call transcripts + audio from AlphaMemo.ai"
    )
    parser.add_argument(
        "--tickers", nargs="+",
        help="Filter to specific stock numbers (e.g. 2330 3715)"
    )
    parser.add_argument(
        "--limit", type=int,
        help="Max number of transcripts to process"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List transcripts without downloading"
    )
    parser.add_argument(
        "--upload-hf", action="store_true",
        help="Upload downloaded files to HuggingFace after scraping"
    )
    parser.add_argument(
        "--lang", default="zh", choices=["zh", "en"],
        help="Preferred language for audio (default: zh)"
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

    scraper = AlphaMemoScraper()
    asyncio.run(
        scraper.run(
            tickers=args.tickers,
            limit=args.limit,
            dry_run=args.dry_run,
            upload_hf=args.upload_hf,
            lang_pref=args.lang,
        )
    )


if __name__ == "__main__":
    main()
