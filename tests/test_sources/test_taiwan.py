"""Unit tests for the Taiwan MOPS scraper with mocked HTTP."""

from datetime import datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from src.exceptions import RateLimitError, ScraperConnectionError, ScraperParseError
from src.sources.base import EarningsCallInfo
from src.sources.taiwan import (
    MOPS_API_URL,
    TaiwanScraper,
    gregorian_to_roc,
    roc_to_gregorian,
)

# --- Sample HTML fragments ---

SAMPLE_TABLE_HTML = """
<html><body>
<table class="hasBorder">
<thead><tr class="tblHead">
<th rowspan="2">公司代號</th><th rowspan="2">公司名稱</th>
<th rowspan="2">召開法人說明會日期</th><th rowspan="2">召開法人說明會時間</th>
<th rowspan="2">召開法人說明會地點</th><th rowspan="2">法人說明會擇要訊息</th>
<th colspan="2">法人說明會簡報內容</th>
<th rowspan="2">公司網站</th><th rowspan="2">影音連結</th>
<th rowspan="2">其他</th><th rowspan="2">歷年</th>
</tr><tr class="tblHead"><th>中文</th><th>英文</th></tr></thead>
<tr><td>2330</td><td>台積電</td><td>115/01/16</td><td>14:00</td>
<td>台北市</td><td>法說會</td><td>a.pdf</td><td>b.pdf</td>
<td>https://investor.tsmc.com</td><td>影音</td><td>無</td><td></td></tr>
<tr><td>2330</td><td>台積電</td><td>114/10/17</td><td>14:00</td>
<td>新竹市</td><td>法說會</td><td></td><td></td>
<td></td><td></td><td></td><td></td></tr>
</table>
</body></html>
"""

EMPTY_RESULT_HTML = """<p>查無資料</p>"""

SINGLE_ROW_HTML = """
<table class="hasBorder">
<thead><tr class="tblHead">
<th>公司代號</th><th>公司名稱</th><th>日期</th><th>時間</th>
<th>地點</th><th>說明</th><th>中文</th><th>英文</th>
<th>網站</th><th>影音</th><th>其他</th><th>歷年</th>
</tr></thead>
<tr><td>2317</td><td>鴻海</td><td>115/03/05</td><td>10:00</td>
<td>土城區</td><td>第四季法說會</td><td></td><td></td>
<td></td><td></td><td></td><td></td></tr>
</table>
"""

NO_TABLE_HTML = """<div>Unexpected response</div>"""

RATE_LIMIT_HTML = """<html><body>您的查詢過於頻繁，請稍後再試。</body></html>"""

# --- Mock helpers ---

SIGNED_URL = "https://mopsov.twse.com.tw/mops/web/ajax_t100sb02_1?parameters=abc123"


def _mock_api_response(url: str = SIGNED_URL) -> httpx.Response:
    """Mock JSON response from the MOPS API (step 1)."""
    import json

    body = json.dumps({
        "code": 200,
        "message": "查詢成功",
        "result": {"url": url},
        "datetime": "115/01/01 00:00:00",
    })
    return httpx.Response(
        status_code=200,
        text=body,
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", MOPS_API_URL),
    )


def _mock_html_response(html: str) -> httpx.Response:
    """Mock HTML response from the signed URL (step 2)."""
    return httpx.Response(
        status_code=200,
        text=html,
        request=httpx.Request("GET", SIGNED_URL),
    )


def _setup_two_step_mock(
    mock_client: AsyncMock,
    html: str = SAMPLE_TABLE_HTML,
) -> None:
    """Configure mock client for the 2-step MOPS API flow."""

    async def side_effect(*args, **kwargs):
        # POST = step 1 (API), GET = step 2 (HTML)
        if kwargs.get("json") is not None:
            return _mock_api_response()
        return _mock_html_response(html)

    mock_client.post.return_value = _mock_api_response()
    mock_client.get.return_value = _mock_html_response(html)


# --- ROC date conversion tests ---


class TestROCDateConversion:
    def test_gregorian_to_roc(self):
        assert gregorian_to_roc(2026) == 115
        assert gregorian_to_roc(2000) == 89
        assert gregorian_to_roc(1911) == 0

    def test_roc_to_gregorian(self):
        assert roc_to_gregorian(115) == 2026
        assert roc_to_gregorian(89) == 2000
        assert roc_to_gregorian(0) == 1911


# --- Parsing tests ---


class TestParseConferenceRows:
    def test_parse_normal_table(self, scraper):
        rows = scraper._parse_conference_rows(SAMPLE_TABLE_HTML, "2330")
        assert len(rows) == 2
        assert rows[0]["company_code"] == "2330"
        assert rows[0]["date"] == "115/01/16"
        assert rows[0]["time"] == "14:00"
        assert rows[0]["venue"] == "台北市"
        assert rows[0]["webcast_url"] == "https://investor.tsmc.com"

    def test_parse_empty_result(self, scraper):
        rows = scraper._parse_conference_rows(EMPTY_RESULT_HTML, "2330")
        assert rows == []

    def test_parse_no_table_raises(self, scraper):
        with pytest.raises(ScraperParseError):
            scraper._parse_conference_rows(NO_TABLE_HTML, "2330")

    def test_parse_blank_html_returns_empty(self, scraper):
        rows = scraper._parse_conference_rows("", "2330")
        assert rows == []

    def test_parse_single_row(self, scraper):
        rows = scraper._parse_conference_rows(SINGLE_ROW_HTML, "2317")
        assert len(rows) == 1
        assert rows[0]["company_name"] == "鴻海"
        assert rows[0]["summary"] == "第四季法說會"


class TestParseROCDate:
    def test_valid_date(self):
        dt = TaiwanScraper._parse_roc_date("115/01/16")
        assert dt == datetime(2026, 1, 16)

    def test_invalid_format(self):
        assert TaiwanScraper._parse_roc_date("invalid") is None

    def test_bad_values(self):
        assert TaiwanScraper._parse_roc_date("115/13/01") is None

    def test_date_range(self):
        """MOPS uses date ranges like '114/09/08 至 114/09/12' for multi-day events."""
        dt = TaiwanScraper._parse_roc_date("114/09/08 至 114/09/12")
        assert dt == datetime(2025, 9, 8)


# --- Fiscal quarter inference ---


class TestInferFiscalQuarter:
    def test_q4_reporting(self):
        assert TaiwanScraper._infer_fiscal_quarter(datetime(2026, 1, 16)) == (2025, 4)
        assert TaiwanScraper._infer_fiscal_quarter(datetime(2026, 3, 15)) == (2025, 4)

    def test_q1_reporting(self):
        assert TaiwanScraper._infer_fiscal_quarter(datetime(2026, 4, 10)) == (2026, 1)
        assert TaiwanScraper._infer_fiscal_quarter(datetime(2026, 6, 30)) == (2026, 1)

    def test_q2_reporting(self):
        assert TaiwanScraper._infer_fiscal_quarter(datetime(2026, 7, 15)) == (2026, 2)
        assert TaiwanScraper._infer_fiscal_quarter(datetime(2026, 9, 30)) == (2026, 2)

    def test_q3_reporting(self):
        assert TaiwanScraper._infer_fiscal_quarter(datetime(2026, 10, 1)) == (2026, 3)
        assert TaiwanScraper._infer_fiscal_quarter(datetime(2026, 12, 31)) == (2026, 3)


# --- Row-to-CallInfo conversion ---


class TestRowToCallInfo:
    def test_normal_conversion(self, scraper):
        row = {
            "company_code": "2330",
            "company_name": "台積電",
            "date": "115/01/16",
            "time": "14:00",
            "venue": "台北市",
            "summary": "法說會",
            "webcast_url": "https://investor.tsmc.com",
            "video_info": "",
        }
        info = scraper._row_to_call_info(row, "2330")
        assert info is not None
        assert info.ticker == "2330"
        assert info.company_name == "台積電"
        assert info.exchange == "TWSE"
        assert info.call_date == datetime(2026, 1, 16)
        assert info.fiscal_year == 2025
        assert info.fiscal_quarter == 4
        assert info.language == "zh"
        assert info.audio_url is None
        assert info.webcast_url == "https://investor.tsmc.com"
        assert info.metadata["venue"] == "台北市"
        assert info.metadata["source"] == "MOPS"

    def test_bad_date_returns_none(self, scraper):
        row = {
            "company_code": "2330",
            "company_name": "台積電",
            "date": "bad/date",
            "time": "14:00",
            "venue": "台北市",
            "summary": "",
            "webcast_url": "",
            "video_info": "",
        }
        assert scraper._row_to_call_info(row, "2330") is None

    def test_empty_webcast_url_becomes_none(self, scraper):
        row = {
            "company_code": "2330",
            "company_name": "台積電",
            "date": "115/01/16",
            "time": "14:00",
            "venue": "台北市",
            "summary": "",
            "webcast_url": "",
            "video_info": "",
        }
        info = scraper._row_to_call_info(row, "2330")
        assert info.webcast_url is None


# --- HTTP fetch tests (mocked 2-step flow) ---


class TestFetchMopsPage:
    @pytest.mark.asyncio
    async def test_successful_fetch(self, scraper):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        _setup_two_step_mock(mock_client, SAMPLE_TABLE_HTML)
        scraper._client = mock_client

        html = await scraper._fetch_mops_page("2330", 115, 1)
        assert "台積電" in html
        mock_client.post.assert_called_once()
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limit_on_html_page(self, scraper):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = _mock_api_response()
        mock_client.get.return_value = _mock_html_response(RATE_LIMIT_HTML)
        scraper._client = mock_client

        with pytest.raises(RateLimitError):
            await scraper._fetch_mops_page("2330", 115, 1)

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self, scraper):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        scraper._client = mock_client

        with pytest.raises(ScraperConnectionError):
            await scraper._fetch_mops_page("2330", 115, 1)

        assert mock_client.post.call_count == 3  # _MAX_RETRIES

    @pytest.mark.asyncio
    async def test_api_error_code(self, scraper):
        import json

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        error_resp = httpx.Response(
            status_code=200,
            text=json.dumps({"code": 500, "message": "Internal error"}),
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", MOPS_API_URL),
        )
        mock_client.post.return_value = error_resp
        scraper._client = mock_client

        with pytest.raises(ScraperConnectionError, match="Internal error"):
            await scraper._fetch_mops_page("2330", 115, 1)


# --- discover_calls integration (mocked HTTP) ---


class TestDiscoverCalls:
    @pytest.mark.asyncio
    async def test_discover_filters_by_date_range(self, scraper):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        _setup_two_step_mock(mock_client, SAMPLE_TABLE_HTML)
        scraper._client = mock_client

        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 31)
        results = await scraper.discover_calls(start, end, tickers=["2330"])

        # Only the 115/01/16 row should pass the date filter
        assert len(results) == 1
        assert results[0].call_date == datetime(2026, 1, 16)

    @pytest.mark.asyncio
    async def test_discover_handles_empty_results(self, scraper):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        _setup_two_step_mock(mock_client, EMPTY_RESULT_HTML)
        scraper._client = mock_client

        start = datetime(2026, 1, 1)
        end = datetime(2026, 3, 31)
        results = await scraper.discover_calls(start, end, tickers=["9999"])
        assert results == []

    @pytest.mark.asyncio
    async def test_discover_isolates_per_ticker_errors(self, scraper):
        """Errors for one ticker should not prevent others from being scraped."""

        async def mock_post(*args, **kwargs):
            data = kwargs.get("json", {})
            ticker = data.get("parameters", {}).get("co_id", "")
            if ticker == "FAIL":
                raise httpx.ConnectError("Connection refused")
            return _mock_api_response()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = mock_post
        mock_client.get.return_value = _mock_html_response(SINGLE_ROW_HTML)
        scraper._client = mock_client

        start = datetime(2026, 3, 1)
        end = datetime(2026, 3, 31)
        results = await scraper.discover_calls(
            start, end, tickers=["FAIL", "2317"]
        )

        assert len(results) >= 1
        assert all(r.ticker == "2317" for r in results)

    @pytest.mark.asyncio
    async def test_discover_results_sorted_descending(self, scraper):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        _setup_two_step_mock(mock_client, SAMPLE_TABLE_HTML)
        scraper._client = mock_client

        start = datetime(2025, 10, 1)
        end = datetime(2026, 2, 28)
        results = await scraper.discover_calls(start, end, tickers=["2330"])

        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].call_date >= results[i + 1].call_date


# --- Context manager ---


class TestContextManager:
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with TaiwanScraper(rate_limit_secs=0.0) as scraper:
            assert scraper.exchange_code == "TWSE"
        assert scraper._client is None


# --- get_audio_url ---


class TestGetAudioUrl:
    @pytest.mark.asyncio
    async def test_always_returns_none(self, scraper):
        info = EarningsCallInfo(
            company_name="台積電",
            ticker="2330",
            exchange="TWSE",
            call_date=datetime(2026, 1, 16),
        )
        assert await scraper.get_audio_url(info) is None


# --- Properties ---


class TestProperties:
    def test_exchange_code(self, scraper):
        assert scraper.exchange_code == "TWSE"

    def test_supported_languages(self, scraper):
        assert scraper.supported_languages == ["zh", "en"]
