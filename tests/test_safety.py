"""Regression tests for the interlocks that keep real money out of reach.

Three defects are pinned here, each of which failed towards danger:

1. The paper trading gate was `mode == 'paper'` over an unvalidated string, so
   any other value ('Paper', a typo, 'backtest') answered False and the bot
   engine went on to place real orders.
2. Boolean environment flags were compared with `== 'true'`, so
   PAPER_TRADING_MODE=True resolved to LIVE and BINANCE_TESTNET=True resolved
   to PRODUCTION.
3. Nothing checked that the client had actually resolved to a testnet host, so
   a testnet setting that did not reach every endpoint went unnoticed.

Every test here runs with no network: the autouse guards in conftest block
sockets and strip credentials, and the exchange is the recording double.
"""

from __future__ import annotations

import logging

import pytest
import yaml

from timi.utils import config as config_module
from timi.utils.config import (
    Config,
    ConfigurationError,
    DEFAULT_MODE,
    LiveTradingNotUnlockedError,
    Mode,
    UnsafeModeError,
    VALID_MODES,
    assert_live_trading_unlocked,
    assert_mode_matches_endpoint,
    is_testnet_endpoint,
    parse_bool,
    parse_bool_env,
    resolved_endpoint,
    validate_mode,
)


PRODUCTION_HOST = "https://api.binance.com/api/v3"


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Keep a developer's own .env out of the assertions."""

    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _no_live_unlock(monkeypatch):
    """The live unlock starts absent, whatever the shell says."""

    monkeypatch.delenv("TIMI_ALLOW_LIVE", raising=False)


def _write_config(tmp_path, data: dict) -> str:
    """Write a YAML config file and return its path."""

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def _load(tmp_path, data: dict, **kwargs) -> Config:
    """Build a Config from `data`, bypassing the process wide singleton."""

    cfg = object.__new__(Config)
    cfg._config_data = {}
    cfg.load(_write_config(tmp_path, data), **kwargs)
    return cfg


class TestModeAllowList:
    """Defect A: the mode was whatever the file said, unvalidated."""

    @pytest.mark.parametrize(
        "bad",
        [
            "Paper",
            "PAPER",
            "paper_trading",
            "",
            "live ",
            " live",
            "papertrading",
            "Live",
            "BACKTEST",
            "simulation",
        ],
    )
    def test_non_allowed_mode_raises(self, bad):
        with pytest.raises(ConfigurationError):
            validate_mode(bad)

    def test_error_names_the_bad_value_and_the_allowed_set(self):
        with pytest.raises(ConfigurationError) as excinfo:
            validate_mode("paper_trading")

        message = str(excinfo.value)
        assert "paper_trading" in message
        for allowed in VALID_MODES:
            assert allowed in message

    @pytest.mark.parametrize("good", ["paper", "live", "backtest"])
    def test_allowed_modes_are_accepted(self, good):
        assert validate_mode(good) == good

    def test_missing_mode_defaults_to_paper(self):
        assert validate_mode(None) == "paper"
        assert DEFAULT_MODE == Mode.PAPER.value

    def test_non_string_mode_raises(self):
        with pytest.raises(ConfigurationError):
            validate_mode(True)

    @pytest.mark.parametrize("bad", ["Paper", "PAPER", "paper_trading", "", "live "])
    def test_loading_a_bad_mode_refuses_to_start(self, tmp_path, bad):
        with pytest.raises(ConfigurationError):
            _load(tmp_path, {"mode": bad})

    def test_config_property_raises_rather_than_answering_false(self, tmp_path):
        """The heart of defect A: a bad mode must not read as 'not paper'."""

        cfg = _load(tmp_path, {"mode": "paper"})
        cfg._config_data["mode"] = "Paper"  # as a hand edited file would leave it

        with pytest.raises(ConfigurationError):
            cfg.is_paper_trading()

    def test_missing_mode_key_loads_as_paper(self, tmp_path):
        cfg = _load(tmp_path, {"exchange": {"primary": "binance"}})

        assert cfg.mode == "paper"
        assert cfg.is_paper_trading() is True
        assert cfg.is_live_trading() is False

    def test_backtest_still_simulates_orders(self, tmp_path):
        """Backtest is not live, so execution must stay simulated."""

        cfg = _load(tmp_path, {"mode": "backtest"})

        assert cfg.is_backtesting() is True
        assert cfg.is_paper_trading() is True
        assert cfg.is_live_trading() is False


class TestBooleanParsing:
    """Defect B: `== 'true'` meant every other spelling meant danger."""

    @pytest.mark.parametrize(
        "raw", ["true", "True", "TRUE", "1", "yes", "YES", "on", " True "]
    )
    def test_truthy_spellings(self, raw):
        assert parse_bool(raw, "FLAG") is True

    @pytest.mark.parametrize(
        "raw", ["false", "False", "FALSE", "0", "no", "NO", "off", " false "]
    )
    def test_falsey_spellings(self, raw):
        assert parse_bool(raw, "FLAG") is False

    @pytest.mark.parametrize("raw", ["maybe", "t", "y", "2", "none", "-"])
    def test_unrecognised_value_raises(self, raw):
        with pytest.raises(ConfigurationError) as excinfo:
            parse_bool(raw, "FLAG")

        assert "FLAG" in str(excinfo.value)

    def test_unset_env_uses_the_default(self, monkeypatch):
        monkeypatch.delenv("SOME_FLAG", raising=False)

        assert parse_bool_env("SOME_FLAG", True) is True
        assert parse_bool_env("SOME_FLAG", False) is False

    def test_unparseable_env_raises_when_there_is_no_safe_value(self, monkeypatch):
        monkeypatch.setenv("SOME_FLAG", "perhaps")

        with pytest.raises(ConfigurationError):
            parse_bool_env("SOME_FLAG", False)


class TestPaperTradingModeEnv:
    """PAPER_TRADING_MODE must never turn live because of its casing."""

    @pytest.mark.parametrize("raw", ["true", "True", "TRUE", "1", "yes", "on"])
    def test_truthy_values_yield_paper(self, tmp_path, monkeypatch, raw):
        monkeypatch.setenv("PAPER_TRADING_MODE", raw)
        cfg = _load(tmp_path, {"mode": "live"})

        assert cfg.mode == "paper"
        assert cfg.is_paper_trading() is True

    @pytest.mark.parametrize("raw", ["false", "False", "FALSE", "0", "no", "off"])
    def test_falsey_values_yield_live(self, tmp_path, monkeypatch, raw):
        monkeypatch.setenv("PAPER_TRADING_MODE", raw)
        cfg = _load(tmp_path, {"mode": "paper"})

        assert cfg.mode == "live"
        assert cfg.is_paper_trading() is False

    def test_unparseable_value_fails_towards_paper(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("PAPER_TRADING_MODE", "yes please")

        with caplog.at_level(logging.WARNING, logger="timi.utils.config"):
            cfg = _load(tmp_path, {"mode": "live"})

        assert cfg.mode == "paper"
        assert any(
            "PAPER_TRADING_MODE" in record.getMessage() for record in caplog.records
        )


class TestBinanceTestnetEnv:
    """The exact bug: BINANCE_TESTNET=True used to select PRODUCTION."""

    @pytest.mark.parametrize("raw", ["True", "TRUE", "true", "1", "yes", "on"])
    def test_truthy_values_select_testnet(self, tmp_path, monkeypatch, raw):
        monkeypatch.setenv("BINANCE_TESTNET", raw)
        cfg = _load(tmp_path, {"exchange": {"primary": "binance", "testnet": False}})

        assert cfg.exchange.testnet is True

    @pytest.mark.parametrize("raw", ["false", "False", "0", "no"])
    def test_falsey_values_select_production(self, tmp_path, monkeypatch, raw):
        monkeypatch.setenv("BINANCE_TESTNET", raw)
        cfg = _load(tmp_path, {"exchange": {"primary": "binance", "testnet": True}})

        assert cfg.exchange.testnet is False

    def test_unparseable_value_fails_towards_testnet(
        self, tmp_path, monkeypatch, caplog
    ):
        monkeypatch.setenv("BINANCE_TESTNET", "nope!")

        with caplog.at_level(logging.WARNING, logger="timi.utils.config"):
            cfg = _load(tmp_path, {"exchange": {"primary": "binance", "testnet": False}})

        assert cfg.exchange.testnet is True
        assert any("BINANCE_TESTNET" in r.getMessage() for r in caplog.records)

    def test_missing_exchange_section_does_not_crash(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BINANCE_TESTNET", "True")
        cfg = _load(tmp_path, {"mode": "paper"})

        assert cfg.get("exchange.testnet") is True


class TestLiveUnlock:
    """Defect: an interactive prompt was the only barrier to live trading."""

    def test_refused_with_neither_unlock(self):
        with pytest.raises(LiveTradingNotUnlockedError) as excinfo:
            assert_live_trading_unlocked(False)

        message = str(excinfo.value)
        assert "TIMI_ALLOW_LIVE=1" in message
        assert "--i-understand-the-risk" in message

    def test_refused_with_only_the_env_variable(self, monkeypatch):
        monkeypatch.setenv("TIMI_ALLOW_LIVE", "1")

        with pytest.raises(LiveTradingNotUnlockedError) as excinfo:
            assert_live_trading_unlocked(False)

        assert "--i-understand-the-risk" in str(excinfo.value)

    def test_refused_with_only_the_cli_flag(self):
        with pytest.raises(LiveTradingNotUnlockedError) as excinfo:
            assert_live_trading_unlocked(True)

        assert "TIMI_ALLOW_LIVE=1" in str(excinfo.value)

    @pytest.mark.parametrize("raw", ["0", "", "true", "yes", "TIMI_ALLOW_LIVE"])
    def test_env_variable_must_be_exactly_one(self, monkeypatch, raw):
        monkeypatch.setenv("TIMI_ALLOW_LIVE", raw)

        with pytest.raises(LiveTradingNotUnlockedError):
            assert_live_trading_unlocked(True)

    def test_allowed_with_both(self, monkeypatch):
        monkeypatch.setenv("TIMI_ALLOW_LIVE", "1")

        assert assert_live_trading_unlocked(True) is None


class TestEndpointMatchesMode:
    """Defect C: nothing verified where the client would actually send."""

    def test_paper_mode_with_a_production_endpoint_raises(self, fake_ccxt):
        fake_ccxt.set_sandbox_mode(False)

        with pytest.raises(UnsafeModeError) as excinfo:
            assert_mode_matches_endpoint(fake_ccxt, "paper")

        message = str(excinfo.value)
        assert "paper" in message
        assert "api.binance.com" in message

    def test_backtest_mode_with_a_production_endpoint_raises(self, fake_ccxt):
        fake_ccxt.set_sandbox_mode(False)

        with pytest.raises(UnsafeModeError):
            assert_mode_matches_endpoint(fake_ccxt, "backtest")

    def test_paper_mode_with_a_testnet_endpoint_passes(self, fake_ccxt):
        fake_ccxt.set_sandbox_mode(True)

        endpoint = assert_mode_matches_endpoint(fake_ccxt, "paper")

        assert "testnet" in endpoint

    def test_live_mode_may_use_production(self, fake_ccxt):
        fake_ccxt.set_sandbox_mode(False)

        endpoint = assert_mode_matches_endpoint(fake_ccxt, "live")

        assert endpoint == PRODUCTION_HOST

    def test_a_bad_mode_raises_before_the_endpoint_is_considered(self, fake_ccxt):
        fake_ccxt.set_sandbox_mode(True)

        with pytest.raises(ConfigurationError):
            assert_mode_matches_endpoint(fake_ccxt, "Paper")

    def test_wrapped_connector_is_unwrapped(self, fake_ccxt):
        """The system passes a connector that holds the client on `.exchange`."""

        class Connector:
            def __init__(self, client):
                self.exchange = client

        fake_ccxt.set_sandbox_mode(False)

        with pytest.raises(UnsafeModeError):
            assert_mode_matches_endpoint(Connector(fake_ccxt), "paper")

    def test_an_unreadable_endpoint_raises(self):
        class Opaque:
            pass

        with pytest.raises(UnsafeModeError):
            assert_mode_matches_endpoint(Opaque(), "paper")

    def test_partially_overridden_urls_are_caught(self, fake_ccxt):
        """The futures endpoint case: public patched, private left in production."""

        fake_ccxt.urls["api"]["public"] = "https://testnet.binancefuture.com/fapi/v1"
        fake_ccxt.urls["api"]["private"] = "https://fapi.binance.com/fapi/v1"

        with pytest.raises(UnsafeModeError):
            assert_mode_matches_endpoint(fake_ccxt, "paper")

    def test_resolved_endpoint_reports_the_private_url(self, fake_ccxt):
        fake_ccxt.set_sandbox_mode(True)

        assert resolved_endpoint(fake_ccxt) == fake_ccxt.urls["api"]["private"]

    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://testnet.binance.vision/api/v3", True),
            ("https://TESTNET.binancefuture.com/fapi/v1", True),
            ("https://api.binance.com/api/v3", False),
            ("https://fapi.binance.com/fapi/v1", False),
        ],
    )
    def test_testnet_host_detection(self, url, expected):
        assert is_testnet_endpoint(url) is expected

    def test_no_order_was_sent_by_any_of_this(self, fake_ccxt):
        """The checks must be inspection only."""

        fake_ccxt.set_sandbox_mode(True)
        assert_mode_matches_endpoint(fake_ccxt, "paper")

        assert fake_ccxt.created_orders == []
