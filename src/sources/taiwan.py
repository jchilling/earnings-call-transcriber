"""Taiwan TWSE/MOPS scraper for earnings call (法說會) announcements.

Discovers investor conference metadata from Taiwan's Market Observation
Post System (公開資訊觀測站, mops.twse.com.tw). The new MOPS SPA uses a
two-step API: first POST JSON to get a signed URL, then GET the HTML.
"""

import asyncio
import logging
import ssl
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

from src.exceptions import RateLimitError, ScraperConnectionError, ScraperParseError
from src.sources.base import BaseScraper, EarningsCallInfo

logger = logging.getLogger(__name__)

MOPS_API_URL = "https://mops.twse.com.tw/mops/api/redirectToOld"

# Top 10 TWSE companies by market cap (as of early 2026)
TOP_TAIWAN_TICKERS: dict[str, str] = {
    "2330": "台積電 (TSMC)",
    "2317": "鴻海 (Foxconn)",
    "2308": "台達電 (Delta Electronics)",
    "2454": "聯發科 (MediaTek)",
    "2881": "富邦金 (Fubon FHC)",
    "3711": "日月光投控 (ASE Technology)",
    "2882": "國泰金 (Cathay FHC)",
    "2382": "廣達 (Quanta Computer)",
    "2412": "中華電 (Chunghwa Telecom)",
    "2891": "中信金 (CTBC FHC)",
}

_DEFAULT_RATE_LIMIT_SECS = 2.0
_MAX_RETRIES = 3
_BACKOFF_BASE_SECS = 2.0


def gregorian_to_roc(year: int) -> int:
    """Convert a Gregorian year to ROC (民國) year.

    Args:
        year: Gregorian year (e.g. 2026).

    Returns:
        ROC year (e.g. 115).
    """
    return year - 1911


def roc_to_gregorian(roc_year: int) -> int:
    """Convert an ROC (民國) year to Gregorian year.

    Args:
        roc_year: ROC year (e.g. 115).

    Returns:
        Gregorian year (e.g. 2026).
    """
    return roc_year + 1911


class TaiwanScraper(BaseScraper):
    """Scraper for Taiwan's MOPS system (公開資訊觀測站).

    Uses the new MOPS SPA API (2-step flow):
    1. POST JSON to /mops/api/redirectToOld to get a signed URL
    2. GET the signed URL on mopsov.twse.com.tw for the HTML table
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        rate_limit_secs: float = _DEFAULT_RATE_LIMIT_SECS,
    ) -> None:
        super().__init__(http_client)
        self._rate_limit_secs = rate_limit_secs
        self._last_request_time: float = 0.0

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize HTTP client with relaxed SSL for MOPS.

        MOPS certificates may be missing the Subject Key Identifier
        extension, so we relax strict X509 verification while still
        verifying the certificate chain.
        """
        if self._client is None:
            ctx = ssl.create_default_context()
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                verify=ctx,
            )
        return self._client

    @property
    def exchange_code(self) -> str:
        return "TWSE"

    @property
    def supported_languages(self) -> list[str]:
        return ["zh", "en"]

    async def _rate_limit(self) -> None:
        """Enforce minimum gap between MOPS requests."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit_secs:
            await asyncio.sleep(self._rate_limit_secs - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    async def _fetch_mops_page(
        self, ticker: str, roc_year: int, month: int
    ) -> str:
        """Fetch MOPS investor conference data via the 2-step API.

        Step 1: POST JSON to /mops/api/redirectToOld → get signed URL.
        Step 2: GET the signed URL → get HTML table.

        Args:
            ticker: TWSE stock code (e.g. "2330").
            roc_year: ROC calendar year.
            month: Month number (1-12).

        Returns:
            Raw HTML string containing the conference table.

        Raises:
            RateLimitError: If MOPS returns a security/rate-limit block.
            ScraperConnectionError: If all retries are exhausted.
        """
        client = await self._get_client()
        payload = {
            "apiName": "ajax_t100sb02_1",
            "parameters": {
                "encodeURIComponent": 1,
                "step": 1,
                "firstin": 1,
                "off": 1,
                "TYPEK": "sii",
                "co_id": ticker,
                "year": str(roc_year),
                "month": str(month).zfill(2),
            },
        }
        api_headers = {
            "Content-Type": "application/json",
            "Origin": "https://mops.twse.com.tw",
            "Referer": "https://mops.twse.com.tw/mops/",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            await self._rate_limit()
            try:
                # Step 1: Get signed URL
                r1 = await client.post(
                    MOPS_API_URL,
                    json=payload,
                    headers=api_headers,
                )
                r1.raise_for_status()
                data = r1.json()

                if data.get("code") != 200:
                    msg = data.get("message", "Unknown error")
                    if "過於頻繁" in msg:
                        raise RateLimitError(
                            f"MOPS rate limit for {ticker} ({roc_year}/{month:02d})"
                        )
                    raise ScraperConnectionError(
                        f"MOPS API error for {ticker}: {msg}"
                    )

                signed_url = data.get("result", {}).get("url")
                if not signed_url:
                    raise ScraperParseError(
                        f"No signed URL in MOPS response for {ticker}"
                    )

                # Step 2: Fetch the HTML from the signed URL
                await self._rate_limit()
                r2 = await client.get(signed_url, headers=api_headers)
                r2.raise_for_status()
                html = r2.text

                if "過於頻繁" in html or "PAGE CANNOT BE ACCESSED" in html:
                    raise RateLimitError(
                        f"MOPS rate limit hit for {ticker} ({roc_year}/{month:02d})"
                    )

                return html

            except (RateLimitError, ScraperParseError):
                raise
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                logger.warning(
                    "MOPS HTTP %s for %s (%d/%02d), attempt %d/%d",
                    exc.response.status_code,
                    ticker,
                    roc_year,
                    month,
                    attempt,
                    _MAX_RETRIES,
                )
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning(
                    "MOPS request error for %s (%d/%02d), attempt %d/%d: %s",
                    ticker,
                    roc_year,
                    month,
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )

            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_BACKOFF_BASE_SECS * attempt)

        raise ScraperConnectionError(
            f"Failed to fetch MOPS data for {ticker} after {_MAX_RETRIES} retries"
        ) from last_exc

    def _parse_conference_rows(
        self, html: str, ticker: str
    ) -> list[dict[str, str]]:
        """Parse the MOPS HTML table into a list of row dicts.

        MOPS table columns (as of 2026):
        0: 公司代號, 1: 公司名稱, 2: 召開日期 (ROC), 3: 時間,
        4: 地點, 5: 擇要訊息, 6: 中文簡報PDF, 7: 英文簡報PDF,
        8: 公司網站法說會連結, 9: 影音連結, 10: 其他, 11: 歷年資料

        Args:
            html: Raw HTML response from MOPS.
            ticker: The ticker that was queried (for error context).

        Returns:
            List of dicts with conference metadata. Empty if no data.

        Raises:
            ScraperParseError: If the HTML structure is unexpected.
        """
        if "查無資料" in html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", class_="hasBorder")
        if table is None:
            table = soup.find("table")
        if table is None:
            if not html.strip():
                return []
            raise ScraperParseError(
                f"No table found in MOPS response for {ticker}"
            )

        rows: list[dict[str, str]] = []
        # The real MOPS table has 2 header rows (rowspan=2 + colspan)
        # Find all <tr> that contain <td> (not <th>)
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 6:
                continue

            text = [c.get_text(strip=True) for c in cells]

            # Extract webcast URL from column 8 if present
            webcast_url = ""
            if len(cells) > 8:
                link = cells[8].find("a")
                if link and link.get("href"):
                    webcast_url = link["href"]
                elif text[8].startswith("http"):
                    webcast_url = text[8]

            # Extract video URL from column 9 if present
            video_url = ""
            if len(cells) > 9:
                link = cells[9].find("a")
                if link and link.get("href"):
                    video_url = link["href"]
                elif len(text) > 9 and "http" in text[9]:
                    # May contain multiple URLs in text
                    video_url = text[9]

            rows.append(
                {
                    "company_code": text[0],
                    "company_name": text[1],
                    "date": text[2],
                    "time": text[3],
                    "venue": text[4],
                    "summary": text[5] if len(text) > 5 else "",
                    "webcast_url": webcast_url,
                    "video_info": video_url,
                }
            )

        return rows

    @staticmethod
    def _parse_roc_date(date_str: str) -> datetime | None:
        """Parse an ROC date string like '115/01/15' to a datetime.

        Also handles date ranges like '115/01/15 至 115/01/17' by
        extracting the start date.

        Args:
            date_str: ROC-format date string (single or range).

        Returns:
            datetime object, or None if unparseable.
        """
        try:
            cleaned = date_str.strip()
            # Handle date ranges: "114/09/08 至 114/09/12"
            if "至" in cleaned:
                cleaned = cleaned.split("至")[0].strip()
            parts = cleaned.split("/")
            if len(parts) != 3:
                return None
            roc_year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            return datetime(roc_to_gregorian(roc_year), month, day)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _infer_fiscal_quarter(call_date: datetime) -> tuple[int, int]:
        """Infer the fiscal year and quarter being reported.

        Taiwan companies typically hold 法說會 within 1-2 months after
        quarter-end. A conference in March reports Q4 of the prior year,
        in May reports Q1, in August reports Q2, in November reports Q3.

        Args:
            call_date: Date of the investor conference.

        Returns:
            Tuple of (fiscal_year, fiscal_quarter).
        """
        month = call_date.month
        year = call_date.year

        if month <= 3:
            return (year - 1, 4)
        elif month <= 6:
            return (year, 1)
        elif month <= 9:
            return (year, 2)
        else:
            return (year, 3)

    def _row_to_call_info(
        self, row: dict[str, str], ticker: str
    ) -> EarningsCallInfo | None:
        """Convert a parsed MOPS row dict to an EarningsCallInfo.

        Args:
            row: Dict from _parse_conference_rows.
            ticker: TWSE stock code.

        Returns:
            EarningsCallInfo, or None if the date is unparseable.
        """
        call_date = self._parse_roc_date(row["date"])
        if call_date is None:
            logger.warning(
                "Unparseable date '%s' for ticker %s", row["date"], ticker
            )
            return None

        fiscal_year, fiscal_quarter = self._infer_fiscal_quarter(call_date)
        company_name = row.get("company_name", TOP_TAIWAN_TICKERS.get(ticker, ticker))

        # Use webcast URL if available
        webcast_url = row.get("webcast_url") or None

        return EarningsCallInfo(
            company_name=company_name,
            ticker=ticker,
            exchange="TWSE",
            call_date=call_date,
            audio_url=None,
            webcast_url=webcast_url,
            language="zh",
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            metadata={
                "venue": row.get("venue", ""),
                "summary": row.get("summary", ""),
                "time": row.get("time", ""),
                "video_info": row.get("video_info", ""),
                "source": "MOPS",
            },
        )

    async def discover_calls(
        self,
        start_date: datetime,
        end_date: datetime,
        tickers: list[str] | None = None,
    ) -> list[EarningsCallInfo]:
        """Discover 法說會 announcements from MOPS.

        Iterates over each ticker and each month in the date range,
        querying MOPS and parsing the results. Errors for individual
        tickers are logged but do not halt the overall discovery.

        Args:
            start_date: Earliest conference date to include.
            end_date: Latest conference date to include.
            tickers: Stock codes to query. Defaults to TOP_TAIWAN_TICKERS.

        Returns:
            List of discovered EarningsCallInfo, sorted by date descending.
        """
        if tickers is None:
            tickers = list(TOP_TAIWAN_TICKERS.keys())

        # Build list of (roc_year, month) pairs covering the range
        year_months: list[tuple[int, int]] = []
        current = start_date.replace(day=1)
        while current <= end_date:
            roc_year = gregorian_to_roc(current.year)
            year_months.append((roc_year, current.month))
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        results: list[EarningsCallInfo] = []

        for ticker in tickers:
            for roc_year, month in year_months:
                try:
                    html = await self._fetch_mops_page(ticker, roc_year, month)
                    rows = self._parse_conference_rows(html, ticker)

                    for row in rows:
                        info = self._row_to_call_info(row, ticker)
                        if info is None:
                            continue
                        if start_date <= info.call_date <= end_date:
                            results.append(info)

                except RateLimitError:
                    logger.error(
                        "Rate limited by MOPS while querying %s. Stopping.",
                        ticker,
                    )
                    raise
                except (ScraperConnectionError, ScraperParseError) as exc:
                    logger.warning(
                        "Error scraping %s for %d/%02d: %s",
                        ticker,
                        roc_to_gregorian(roc_year),
                        month,
                        exc,
                    )
                    continue

        results.sort(key=lambda c: c.call_date, reverse=True)
        return results

    async def get_audio_url(self, call_info: EarningsCallInfo) -> str | None:
        """Resolve audio URL for a discovered call.

        MOPS does not host audio files. Audio lives on individual company
        IR pages. This will be implemented in a future phase.

        Returns:
            None (always, in Phase 1).
        """
        return None
