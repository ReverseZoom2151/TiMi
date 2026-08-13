"""Tests for configuration loading: precedence, gaps and path handling.

The safety interlocks in test_safety.py are only as good as the loader that
feeds them, so the awkward inputs are pinned here: a file with sections
missing, a file with nothing in it at all (which used to leave the config as
None and raise TypeError on the next lookup), and relative paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from timi.utils import config as config_module
from timi.utils.config import Config, ConfigurationError


PROJECT_ROOT = Path(config_module.__file__).parent.parent.parent


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Keep a developer's own .env out of the assertions."""

    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: None)


def _fresh() -> Config:
    """A Config instance that does not share the process wide singleton."""

    cfg = object.__new__(Config)
    cfg._config_data = {}
    return cfg


def _write(path: Path, data) -> str:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


class TestPrecedence:
    """CLI beats environment, environment beats the file."""

    def test_file_alone(self, tmp_path):
        cfg = _fresh()
        cfg.load(_write(tmp_path / "c.yaml", {"mode": "live"}))

        assert cfg.mode == "live"

    def test_env_beats_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PAPER_TRADING_MODE", "true")
        cfg = _fresh()
        cfg.load(_write(tmp_path / "c.yaml", {"mode": "live"}))

        assert cfg.mode == "paper"

    def test_cli_beats_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PAPER_TRADING_MODE", "true")
        cfg = _fresh()
        cfg.load(
            _write(tmp_path / "c.yaml", {"mode": "live"}),
            cli_overrides={"mode": "backtest"},
        )

        assert cfg.mode == "backtest"

    def test_cli_beats_file(self, tmp_path):
        cfg = _fresh()
        cfg.load(
            _write(tmp_path / "c.yaml", {"mode": "paper"}),
            cli_overrides={"mode": "live"},
        )

        assert cfg.mode == "live"

    def test_a_none_cli_override_does_not_erase_the_file(self, tmp_path):
        cfg = _fresh()
        cfg.load(
            _write(tmp_path / "c.yaml", {"mode": "backtest"}),
            cli_overrides={"mode": None},
        )

        assert cfg.mode == "backtest"

    def test_a_bad_cli_override_is_refused(self, tmp_path):
        cfg = _fresh()

        with pytest.raises(ConfigurationError):
            cfg.load(
                _write(tmp_path / "c.yaml", {"mode": "paper"}),
                cli_overrides={"mode": "LIVE"},
            )

    def test_env_testnet_beats_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BINANCE_TESTNET", "False")
        cfg = _fresh()
        cfg.load(
            _write(
                tmp_path / "c.yaml",
                {"mode": "live", "exchange": {"primary": "binance", "testnet": True}},
            )
        )

        assert cfg.exchange.testnet is False


class TestMissingSections:
    """A config file that only says half of what it should."""

    def test_absent_sections_fall_back_to_model_defaults(self, tmp_path):
        cfg = _fresh()
        cfg.load(_write(tmp_path / "c.yaml", {"mode": "paper"}))

        assert cfg.strategy.execution_interval == 1
        assert cfg.risk.max_drawdown == 20.0
        assert cfg.trading_pairs_active == []
        assert cfg.trading_pairs_mainstream == []

    def test_get_returns_the_supplied_default(self, tmp_path):
        cfg = _fresh()
        cfg.load(_write(tmp_path / "c.yaml", {"mode": "paper"}))

        assert cfg.get("risk.initial_capital", 10000) == 10000
        assert cfg.get("llm.semantic.model") is None
        assert cfg.get("nothing.here.at.all", "fallback") == "fallback"

    def test_a_required_field_is_reported_rather_than_guessed(self, tmp_path):
        """An exchange section with no primary must not silently pick one."""

        cfg = _fresh()
        cfg.load(_write(tmp_path / "c.yaml", {"mode": "paper"}))

        with pytest.raises(ValidationError):
            cfg.exchange

    def test_partial_section_keeps_its_defaults(self, tmp_path):
        cfg = _fresh()
        cfg.load(
            _write(
                tmp_path / "c.yaml",
                {"mode": "paper", "exchange": {"primary": "binance"}},
            )
        )

        assert cfg.exchange.primary == "binance"
        assert cfg.exchange.testnet is True


class TestEmptyOrMalformedFile:
    def test_empty_file_loads_as_paper(self, tmp_path):
        """yaml.safe_load returns None here, which used to poison every lookup."""

        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")

        cfg = _fresh()
        cfg.load(str(path))

        assert cfg._config_data == {}
        assert cfg.mode == "paper"
        assert cfg.is_paper_trading() is True
        assert cfg.get("anything") is None

    def test_comments_only_file_loads_as_paper(self, tmp_path):
        path = tmp_path / "comments.yaml"
        path.write_text("# nothing but a comment\n", encoding="utf-8")

        cfg = _fresh()
        cfg.load(str(path))

        assert cfg.mode == "paper"

    def test_a_non_mapping_document_is_refused(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- one\n- two\n", encoding="utf-8")

        cfg = _fresh()
        with pytest.raises(ConfigurationError):
            cfg.load(str(path))


class TestPathHandling:
    def test_relative_path_resolves_against_the_working_directory(
        self, tmp_path, monkeypatch
    ):
        _write(tmp_path / "mine.yaml", {"mode": "backtest"})
        monkeypatch.chdir(tmp_path)

        cfg = _fresh()
        cfg.load("mine.yaml")

        assert cfg.mode == "backtest"

    def test_relative_path_falls_back_to_the_project_root(self, tmp_path, monkeypatch):
        """Running from a subdirectory still finds the shipped config.yaml."""

        monkeypatch.chdir(tmp_path)

        cfg = _fresh()
        cfg.load("config.yaml")

        assert cfg.mode in {"paper", "live", "backtest"}
        assert (PROJECT_ROOT / "config.yaml").exists()

    def test_default_path_is_the_project_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        cfg = _fresh()
        cfg.load()

        assert cfg.get("mode") is not None

    def test_a_missing_file_is_reported_clearly(self, tmp_path):
        cfg = _fresh()

        with pytest.raises(FileNotFoundError):
            cfg.load(str(tmp_path / "does_not_exist.yaml"))
