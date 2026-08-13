"""Tests for the trading logging layer.

The central defect under test: logging a position must never raise, even
when the inputs describe states that have no defined cost basis (a spot
holding with no recorded entry price, a flat position, or momentarily
non-finite market data). See `timi.utils.logging._safe_pnl_pct` and
`TradingLogger.log_position`.

No test here touches the network (enforced globally by `tests/conftest.py`)
and none of them configure real handlers, so nothing is written outside the
structlog capture buffer used below.
"""

from __future__ import annotations

import math

import structlog

from timi.utils.logging import TradingLogger, _safe_pnl_pct


def _make_logger() -> TradingLogger:
    return TradingLogger("test.logging")


# --------------------------------------------------------------------------
# _safe_pnl_pct
# --------------------------------------------------------------------------


def test_safe_pnl_pct_zero_entry_price_is_none():
    """A spot holding with no cost basis has an undefined percentage."""

    assert _safe_pnl_pct(pnl=50.0, entry_price=0.0, size=2.0) is None


def test_safe_pnl_pct_zero_size_is_none():
    assert _safe_pnl_pct(pnl=50.0, entry_price=100.0, size=0.0) is None


def test_safe_pnl_pct_normal_position_reports_correct_number():
    # entry_price * size = 1000, pnl = 50 -> 5%
    result = _safe_pnl_pct(pnl=50.0, entry_price=100.0, size=10.0)
    assert result is not None
    assert math.isclose(result, 5.0)


def test_safe_pnl_pct_non_finite_inputs_are_none():
    assert _safe_pnl_pct(pnl=float("inf"), entry_price=100.0, size=1.0) is None
    assert _safe_pnl_pct(pnl=float("nan"), entry_price=100.0, size=1.0) is None
    assert _safe_pnl_pct(pnl=50.0, entry_price=float("inf"), size=1.0) is None
    assert _safe_pnl_pct(pnl=50.0, entry_price=100.0, size=float("nan")) is None


def test_safe_pnl_pct_dust_denominator_is_none():
    # Not exactly zero, but too small to be a meaningful cost basis.
    assert _safe_pnl_pct(pnl=1.0, entry_price=1e-14, size=1.0) is None


# --------------------------------------------------------------------------
# TradingLogger.log_position
# --------------------------------------------------------------------------


def test_log_position_zero_entry_price_does_not_raise_and_omits_flat_claim():
    """Spot holding with no cost basis: no crash, no fabricated 0%."""

    logger = _make_logger()
    with structlog.testing.capture_logs() as records:
        logger.log_position(
            pair="BTC/USDT",
            size=1.5,
            entry_price=0.0,
            current_price=50_000.0,
            pnl=0.0,
        )

    assert len(records) == 1
    record = records[0]
    assert record["event"] == "position_update"
    # A real 0% reading is a claim of "flat". An undefined cost basis must
    # not be reported the same way.
    assert record["pnl_pct"] is None


def test_log_position_zero_size_does_not_raise():
    logger = _make_logger()
    with structlog.testing.capture_logs() as records:
        logger.log_position(
            pair="ETH/USDT",
            size=0.0,
            entry_price=2_000.0,
            current_price=2_100.0,
            pnl=0.0,
        )

    assert len(records) == 1
    assert records[0]["pnl_pct"] is None


def test_log_position_normal_position_reports_correct_percentage():
    logger = _make_logger()
    with structlog.testing.capture_logs() as records:
        logger.log_position(
            pair="BTC/USDT",
            size=2.0,
            entry_price=100.0,
            current_price=110.0,
            pnl=20.0,
        )

    assert len(records) == 1
    record = records[0]
    # cost basis = 200, pnl = 20 -> 10%
    assert record["pnl_pct"] is not None
    assert math.isclose(record["pnl_pct"], 10.0)


def test_log_position_non_finite_inputs_do_not_raise():
    logger = _make_logger()
    with structlog.testing.capture_logs() as records:
        logger.log_position(
            pair="BTC/USDT",
            size=float("inf"),
            entry_price=100.0,
            current_price=100.0,
            pnl=float("nan"),
        )

    assert len(records) == 1
    assert records[0]["pnl_pct"] is None


# --------------------------------------------------------------------------
# General exception-safety of logging calls
# --------------------------------------------------------------------------


def test_log_trade_does_not_raise_on_normal_input():
    logger = _make_logger()
    with structlog.testing.capture_logs() as records:
        logger.log_trade(action="buy", pair="BTC/USDT", price=50_000.0, quantity=0.1)

    assert len(records) == 1
    assert records[0]["event"] == "trade_executed"


def test_log_error_does_not_raise():
    logger = _make_logger()
    with structlog.testing.capture_logs() as records:
        logger.log_error(ValueError("boom"), context={"pair": "BTC/USDT"})

    assert len(records) == 1
    assert records[0]["error_type"] == "ValueError"


def test_malformed_kwargs_degrade_instead_of_raising():
    """A record that fails to render must degrade to a simpler log line,
    never propagate out of the logging call."""

    class _Unrenderable:
        """Raises whenever anything tries to stringify or repr it."""

        def __str__(self):
            raise RuntimeError("cannot render")

        def __repr__(self):
            raise RuntimeError("cannot render")

    logger = _make_logger()
    # Must not raise.
    logger.log_trade(
        action="buy",
        pair="BTC/USDT",
        price=1.0,
        quantity=1.0,
        weird=_Unrenderable(),
    )


# --------------------------------------------------------------------------
# No secrets in formatted log records
# --------------------------------------------------------------------------


def test_no_credential_leaks_in_formatted_log_record():
    logger = _make_logger()
    secret = "sk-super-secret-value-should-not-appear"

    with structlog.testing.capture_logs() as records:
        logger.log_trade(
            action="buy",
            pair="BTC/USDT",
            price=50_000.0,
            quantity=0.1,
            order_id="abc123",
        )
        logger.log_position(
            pair="BTC/USDT",
            size=1.0,
            entry_price=100.0,
            current_price=110.0,
            pnl=10.0,
        )
        logger.log_error(ValueError("plain error, no secrets"))

    formatted = str(records)
    assert secret not in formatted
