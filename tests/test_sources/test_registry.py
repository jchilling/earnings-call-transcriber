"""Unit tests for the CompanyRegistry YAML loader."""

from pathlib import Path
from textwrap import dedent

import pytest

from src.sources.registry import AudioStrategyConfig, CompanyConfig, CompanyRegistry


@pytest.fixture
def sample_yaml(tmp_path: Path) -> Path:
    """Write a minimal companies.yaml and return its path."""
    content = dedent("""\
        companies:
          - ticker: "2330"
            name: "TSMC"
            name_local: "台積電"
            exchange: "TWSE"
            market_type: "sii"
            sector: "Semiconductors"
            gics_sub_industry: "Semiconductors"
            market_cap_usd_b: 900
            language: "zh"
            ir_url: "https://investor.tsmc.com"
            audio:
              strategies:
                - name: "hinet_ott"
                  params:
                    slug: "tsmc"
                    cdn_host: "tsmcvod-ott2b.cdn.hinet.net"
                - name: "ir_page"
                  params:
                    url_template: "https://investor.tsmc.com/q/{year}/q{quarter}"
                    link_text: "重播"
                    follow_links: true

          - ticker: "3081"
            name: "LandMark Optoelectronics"
            name_local: "聯亞光電"
            exchange: "TWSE"
            market_type: "otc"
            sector: "Semiconductors"
            language: "zh"
            ir_url: "https://www.lmoc.com.tw"
            audio:
              strategies:
                - name: "ir_page"
                  params:
                    url_template: "https://www.lmoc.com.tw/page"
                    link_text: "音訊播放"
                    follow_links: true

          - ticker: "9999"
            name: "Bad Company"
            name_local: "壞公司"
            exchange: "TWSE"
            market_type: "sii"
            sector: "Other"
            language: "zh"
            ir_url: ""
            audio:
              strategies:
                - name: "nonexistent_strategy"
    """)
    p = tmp_path / "companies.yaml"
    p.write_text(content)
    return p


class TestCompanyRegistry:
    def test_loads_companies(self, sample_yaml: Path):
        reg = CompanyRegistry(path=sample_yaml)
        assert len(reg) == 3

    def test_get_existing_ticker(self, sample_yaml: Path):
        reg = CompanyRegistry(path=sample_yaml)
        config = reg.get("2330")
        assert config is not None
        assert config.name == "TSMC"
        assert config.name_local == "台積電"
        assert config.exchange == "TWSE"
        assert config.market_type == "sii"
        assert config.market_cap_usd_b == 900
        assert config.gics_sub_industry == "Semiconductors"

    def test_get_missing_ticker(self, sample_yaml: Path):
        reg = CompanyRegistry(path=sample_yaml)
        assert reg.get("0000") is None

    def test_contains(self, sample_yaml: Path):
        reg = CompanyRegistry(path=sample_yaml)
        assert "2330" in reg
        assert "0000" not in reg

    def test_list_tickers_all(self, sample_yaml: Path):
        reg = CompanyRegistry(path=sample_yaml)
        tickers = reg.list_tickers()
        assert sorted(tickers) == ["2330", "3081", "9999"]

    def test_list_tickers_by_market_type(self, sample_yaml: Path):
        reg = CompanyRegistry(path=sample_yaml)
        otc = reg.list_tickers(market_type="otc")
        assert otc == ["3081"]
        sii = reg.list_tickers(market_type="sii")
        assert "2330" in sii
        assert "3081" not in sii

    def test_list_tickers_by_exchange(self, sample_yaml: Path):
        reg = CompanyRegistry(path=sample_yaml)
        twse = reg.list_tickers(exchange="TWSE")
        assert len(twse) == 3
        hkex = reg.list_tickers(exchange="HKEX")
        assert hkex == []

    def test_audio_strategies_loaded(self, sample_yaml: Path):
        reg = CompanyRegistry(path=sample_yaml)
        strategies = reg.get_audio_strategies("2330")
        assert len(strategies) == 2
        assert strategies[0].name == "hinet_ott"
        assert strategies[0].params["slug"] == "tsmc"
        assert strategies[1].name == "ir_page"
        assert strategies[1].params["follow_links"] is True

    def test_audio_strategies_missing_ticker(self, sample_yaml: Path):
        reg = CompanyRegistry(path=sample_yaml)
        assert reg.get_audio_strategies("0000") == []

    def test_unknown_strategy_skipped(self, sample_yaml: Path):
        """Strategies with unknown names are skipped during load."""
        reg = CompanyRegistry(path=sample_yaml)
        strategies = reg.get_audio_strategies("9999")
        assert strategies == []

    def test_missing_file_loads_empty(self, tmp_path: Path):
        reg = CompanyRegistry(path=tmp_path / "nope.yaml")
        assert len(reg) == 0

    def test_empty_file_loads_empty(self, tmp_path: Path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        reg = CompanyRegistry(path=p)
        assert len(reg) == 0

    def test_otc_company_config(self, sample_yaml: Path):
        reg = CompanyRegistry(path=sample_yaml)
        config = reg.get("3081")
        assert config is not None
        assert config.market_type == "otc"
        assert config.name_local == "聯亞光電"


class TestDataclasses:
    def test_audio_strategy_config_defaults(self):
        cfg = AudioStrategyConfig(name="ir_page")
        assert cfg.params == {}

    def test_company_config_defaults(self):
        cfg = CompanyConfig(
            ticker="1234",
            name="Test",
            name_local="測試",
            exchange="TWSE",
            market_type="sii",
            sector="Other",
            language="zh",
            ir_url="",
        )
        assert cfg.audio_strategies == []
        assert cfg.gics_sub_industry == ""
        assert cfg.market_cap_usd_b is None


class TestProductionRegistry:
    """Smoke test: verify the actual data/companies.yaml loads correctly."""

    def test_production_yaml_loads(self):
        reg = CompanyRegistry()  # loads default path
        assert len(reg) >= 10, f"Expected at least 10 companies, got {len(reg)}"
        # TSMC must exist
        tsmc = reg.get("2330")
        assert tsmc is not None
        assert tsmc.name == "TSMC"
        # OTC company must exist
        assert reg.get("3081") is not None
        assert reg.get("3081").market_type == "otc"
