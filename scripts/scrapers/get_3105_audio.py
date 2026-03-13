"""WIN Semiconductors (3105) earnings call audio scraper.

IR page: https://www.winfoundry.com/en-us/Invest/invest_quarter_reports
Audio sources:
  - 2013-2024: Google Drive /preview URLs (yt-dlp handles natively)
  - 2025+: Mix of Google Drive /view and Zucast (dynamic, skipped)
  - 1Q 2013: Legacy URL (likely broken, skipped)
PDFs: Direct download links via /en-us/Base/DownLoadFile/
"""

import logging
import re
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from scripts.scrapers.base_scraper import BaseAudioScraper, QuarterResult

logger = logging.getLogger(__name__)

# IR page URL template — paginated, but page=1 usually has all quarters for a year
IR_URL_TEMPLATE = (
    "https://www.winfoundry.com/en-us/Invest/invest_quarter_reports"
    "?page=1&year={year}&isOpen=True"
)
BASE_URL = "https://www.winfoundry.com"

# Quarter extraction from link text: "4Q 2025", "1Q2013", etc.
QUARTER_RE = re.compile(r"(\d)\s*Q\s*(\d{4})", re.IGNORECASE)
# Also handle "Q1 2025" format
QUARTER_RE_ALT = re.compile(r"Q\s*(\d)\s*(\d{4})", re.IGNORECASE)


def _classify_audio_url(url: str) -> str:
    """Classify audio URL source type."""
    if "drive.google.com" in url:
        return "gdrive"
    if "zucast.com" in url:
        return "zucast"
    if url.endswith((".mp3", ".mp4", ".m4a", ".wav")):
        return "direct"
    return "other"


def _classify_pdf_type(text: str) -> str:
    """Classify PDF as presentation or press_release based on link text."""
    if "簡報" in text or "presentation" in text.lower() or "slides" in text.lower():
        return "presentation"
    if "新聞稿" in text or "press" in text.lower():
        return "press_release"
    return "document"


def _extract_quarter_year(text: str) -> tuple[int, int] | None:
    """Extract (quarter, year) from text like '4Q 2025' or 'Q1 2025'."""
    # Try "4Q 2025" format first
    m = QUARTER_RE.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Try "Q1 2025" format
    m = QUARTER_RE_ALT.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


class WinSemiScraper(BaseAudioScraper):
    """Scraper for WIN Semiconductors (3105) IR page."""

    TICKER = "3105"
    COMPANY_NAME = "WIN Semiconductors"
    INDUSTRY = "semiconductor"

    async def get_quarters(self, year: int) -> list[QuarterResult]:
        """Fetch IR page for a year, parse audio and PDF links."""
        url = IR_URL_TEMPLATE.format(year=year)
        logger.info("Fetching WIN Semi IR page: %s", url)

        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            verify=False,  # winfoundry.com has SSL issues (missing Subject Key Identifier)
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        return self._parse_quarters(soup, year)

    def _parse_quarters(self, soup: BeautifulSoup, target_year: int) -> list[QuarterResult]:
        """Parse all quarter entries from the IR page HTML.

        Page structure:
            <div class="one_season">
                <h5>4Q 2024</h5>
                <div class="download-btn">
                    <img src="/images/invest/doc_pdf.png" />
                    <a href="...">4Q 2024 法說會 – 新聞稿</a>
                </div>
                <div class="download-btn">
                    <img src="/images/invest/doc_video.png" />
                    <a href="https://drive.google.com/...">Webcast Replay</a>
                </div>
            </div>

        The <img> is a sibling of <a> inside <div class="download-btn">.
        """
        quarter_data: dict[int, QuarterResult] = {}

        # Find all season sections
        for section in soup.find_all("div", class_="one_season"):
            # Extract quarter from the h5 header
            h5 = section.find("h5")
            if not h5:
                continue
            qy = _extract_quarter_year(h5.get_text(strip=True))
            if not qy:
                continue
            quarter, link_year = qy
            if link_year != target_year:
                continue

            if quarter not in quarter_data:
                quarter_data[quarter] = QuarterResult(target_year, quarter)
            result = quarter_data[quarter]

            # Find all download buttons in this section
            for btn in section.find_all("div", class_="download-btn"):
                img = btn.find("img")
                a_tag = btn.find("a", href=True)
                if not a_tag:
                    continue

                href = a_tag["href"]
                full_text = a_tag.get_text(strip=True)
                img_src = img["src"] if img and img.get("src") else ""

                if "doc_video" in img_src:
                    # Audio/video link
                    audio_url = href if href.startswith("http") else urljoin(BASE_URL, href)
                    result.audio_url = audio_url
                    result.audio_source = _classify_audio_url(audio_url)
                    logger.info(
                        "  Q%d %d audio: %s (%s)",
                        quarter, target_year, result.audio_source, audio_url[:80],
                    )

                elif "doc_pdf" in img_src:
                    # PDF link
                    pdf_url = href if href.startswith("http") else urljoin(BASE_URL, href)
                    pdf_type = _classify_pdf_type(full_text)
                    # Avoid duplicate PDF types
                    existing_types = {p["type"] for p in result.pdfs}
                    if pdf_type in existing_types:
                        pdf_type = f"{pdf_type}_{len(result.pdfs) + 1}"
                    result.pdfs.append({
                        "type": pdf_type,
                        "url": pdf_url,
                        "path": None,
                        "status": "pending",
                    })
                    logger.info("  Q%d %d PDF: %s → %s", quarter, target_year, pdf_type, pdf_url[:80])

        # Sort by quarter
        return [quarter_data[q] for q in sorted(quarter_data.keys())]
