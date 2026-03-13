"""Base class for per-company audio scrapers.

Each company gets its own scraper subclass that knows how to:
1. Discover audio + PDF URLs from that company's IR page
2. Download audio (via yt-dlp for GDrive, ffmpeg for direct) and PDFs (httpx)
3. Track download results in a JSON file for idempotent reruns
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


class QuarterResult:
    """Result of downloading one quarter's earnings call materials."""

    def __init__(self, year: int, quarter: int) -> None:
        self.year = year
        self.quarter = quarter
        self.audio_url: str | None = None
        self.audio_source: str | None = None  # "gdrive", "zucast", "direct", etc.
        self.audio_path: str | None = None
        self.audio_status: str = "pending"  # "ok", "skipped", "error", "unavailable"
        self.pdfs: list[dict] = []  # [{type, url, path, status}]
        self.error: str | None = None

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "quarter": self.quarter,
            "audio_url": self.audio_url,
            "audio_source": self.audio_source,
            "audio_path": self.audio_path,
            "audio_status": self.audio_status,
            "pdfs": self.pdfs,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QuarterResult":
        r = cls(d["year"], d["quarter"])
        r.audio_url = d.get("audio_url")
        r.audio_source = d.get("audio_source")
        r.audio_path = d.get("audio_path")
        r.audio_status = d.get("audio_status", "pending")
        r.pdfs = d.get("pdfs", [])
        r.error = d.get("error")
        return r


class BaseAudioScraper(ABC):
    """Abstract base class for per-company audio scrapers."""

    TICKER: str = ""
    COMPANY_NAME: str = ""
    INDUSTRY: str = ""

    def __init__(self, output_dir: Path | None = None) -> None:
        if output_dir is None:
            output_dir = Path("data/audio") / f"{self.TICKER}_{self._dir_slug()}"
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results_path = self.output_dir / "download_results.json"
        self._existing_results: dict[str, QuarterResult] = self._load_results()

    def _dir_slug(self) -> str:
        return self.COMPANY_NAME.lower().replace(" ", "_")

    def _quarter_key(self, year: int, quarter: int) -> str:
        return f"{year}_Q{quarter}"

    def _load_results(self) -> dict[str, QuarterResult]:
        if not self._results_path.exists():
            return {}
        with open(self._results_path) as f:
            data = json.load(f)
        return {
            self._quarter_key(r["year"], r["quarter"]): QuarterResult.from_dict(r)
            for r in data.get("results", [])
        }

    def _save_results(self) -> None:
        data = {
            "ticker": self.TICKER,
            "company": self.COMPANY_NAME,
            "results": [r.to_dict() for r in self._existing_results.values()],
        }
        with open(self._results_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @abstractmethod
    async def get_quarters(self, year: int) -> list[QuarterResult]:
        """Discover available quarters for a given year.

        Returns list of QuarterResult with audio_url, audio_source, and pdfs
        populated but not yet downloaded (status="pending").
        """

    async def download_quarter(self, result: QuarterResult) -> QuarterResult:
        """Download audio + PDFs for one quarter. Skips if already downloaded."""
        key = self._quarter_key(result.year, result.quarter)

        # Check if already fully downloaded (audio ok + all PDFs ok)
        existing = self._existing_results.get(key)
        if existing and existing.audio_status == "ok":
            all_pdfs_ok = all(p.get("status") == "ok" for p in existing.pdfs)
            if all_pdfs_ok:
                logger.info("Skipping %s Q%d %d — already downloaded", self.TICKER, result.quarter, result.year)
                return existing
            # Audio ok but PDFs need retry — merge PDF list from fresh discovery
            result.audio_path = existing.audio_path
            result.audio_status = "ok"

        # Download audio
        if result.audio_url:
            audio_filename = f"{self.TICKER}_Q{result.quarter}_{result.year}.mp3"
            audio_path = self.output_dir / audio_filename

            if audio_path.exists() and audio_path.stat().st_size > 0:
                logger.info("Audio file exists: %s", audio_path)
                result.audio_path = str(audio_path)
                result.audio_status = "ok"
            else:
                try:
                    if result.audio_source == "gdrive":
                        await self._download_gdrive_audio(result.audio_url, audio_path)
                    elif result.audio_source == "zucast":
                        result.audio_status = "zucast_unavailable"
                        result.error = "Zucast streams require dynamic auth — skipped"
                        logger.warning("Skipping Zucast URL for %s Q%d %d", self.TICKER, result.quarter, result.year)
                    elif result.audio_source == "direct":
                        await self._download_direct_audio(result.audio_url, audio_path)
                    else:
                        # Try yt-dlp as generic fallback
                        await self._download_gdrive_audio(result.audio_url, audio_path)

                    if audio_path.exists() and audio_path.stat().st_size > 0:
                        result.audio_path = str(audio_path)
                        result.audio_status = "ok"
                    elif result.audio_status != "zucast_unavailable":
                        result.audio_status = "error"
                        result.error = result.error or "Download produced empty file"
                except Exception as e:
                    result.audio_status = "error"
                    result.error = str(e)
                    logger.error("Audio download failed for %s Q%d %d: %s", self.TICKER, result.quarter, result.year, e)
        else:
            result.audio_status = "unavailable"
            result.error = "No audio URL found"

        # Download PDFs
        for pdf in result.pdfs:
            if pdf.get("status") == "ok" and pdf.get("path") and Path(pdf["path"]).exists():
                continue
            if pdf.get("url"):
                pdf_filename = f"{self.TICKER}_Q{result.quarter}_{result.year}_{pdf['type']}.pdf"
                pdf_path = self.output_dir / pdf_filename
                try:
                    await self._download_pdf(pdf["url"], pdf_path)
                    pdf["path"] = str(pdf_path)
                    pdf["status"] = "ok"
                except Exception as e:
                    pdf["status"] = "error"
                    pdf["error"] = str(e)
                    logger.error("PDF download failed: %s", e)

        # Save progress
        self._existing_results[key] = result
        self._save_results()
        return result

    async def download_year(self, year: int) -> list[QuarterResult]:
        """Discover and download all quarters for a year."""
        logger.info("Fetching quarters for %s %d...", self.COMPANY_NAME, year)
        quarters = await self.get_quarters(year)
        if not quarters:
            logger.info("No quarters found for %s %d", self.COMPANY_NAME, year)
            return []

        results = []
        for q in quarters:
            r = await self.download_quarter(q)
            results.append(r)
            status_icon = "+" if r.audio_status == "ok" else "!" if "unavail" in r.audio_status else "x"
            logger.info(
                "  [%s] Q%d %d — audio: %s, pdfs: %d",
                status_icon, r.quarter, r.year, r.audio_status, len([p for p in r.pdfs if p.get("status") == "ok"]),
            )

        return results

    async def download_range(self, start_year: int, end_year: int) -> list[QuarterResult]:
        """Download all quarters across a year range."""
        all_results = []
        for year in range(start_year, end_year + 1):
            results = await self.download_year(year)
            all_results.extend(results)
        return all_results

    @staticmethod
    async def _download_gdrive_audio(url: str, output_path: Path) -> None:
        """Download audio from Google Drive using yt-dlp."""
        # Normalize GDrive URL to /view format for yt-dlp
        normalized = url.replace("/preview", "/view")

        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--output", str(output_path.with_suffix(".%(ext)s")),
            "--no-playlist",
            "--quiet",
            normalized,
        ]

        logger.info("yt-dlp downloading: %s", normalized)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

        if proc.returncode != 0:
            err_msg = stderr.decode().strip()
            raise RuntimeError(f"yt-dlp failed (rc={proc.returncode}): {err_msg}")

        # yt-dlp may create file with slightly different name — find it
        if not output_path.exists():
            # Look for any file matching the pattern
            pattern = output_path.stem + ".*"
            candidates = list(output_path.parent.glob(pattern))
            mp3_candidates = [c for c in candidates if c.suffix == ".mp3"]
            if mp3_candidates:
                mp3_candidates[0].rename(output_path)
            elif candidates:
                # If no .mp3 but there's another audio format, rename it
                candidates[0].rename(output_path)

    @staticmethod
    async def _download_direct_audio(url: str, output_path: Path) -> None:
        """Download audio via ffmpeg stream extraction."""
        cmd = [
            "ffmpeg", "-y",
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
            raise RuntimeError(f"ffmpeg failed: {stderr.decode()[-500:]}")

    @staticmethod
    async def _download_pdf(url: str, output_path: Path) -> None:
        """Download a PDF via HTTP."""
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info("PDF exists: %s", output_path)
            return

        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            verify=False,  # Some IR sites have SSL issues
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            output_path.write_bytes(resp.content)
            logger.info("Downloaded PDF: %s (%d bytes)", output_path.name, len(resp.content))
