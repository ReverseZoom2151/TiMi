"""Shared fixtures, and the guard rails that make this suite safe to run.

This project places real orders with real money. A test suite for it has one
absolute requirement before any other: running the tests must never be able to
touch an exchange account. Two mechanisms enforce that here, and neither relies
on a test author remembering to do the right thing.

1. `_no_network` is autouse. It patches the socket layer so any attempt to open
   a connection raises immediately, naming the test that tried. A test that
   reaches for the network fails loudly rather than quietly hitting Binance.
   Tests that genuinely need the network must be marked `@pytest.mark.network`,
   which is deselected by default and never run in CI.

2. `_no_credentials` is autouse. It removes every exchange and LLM key from the
   environment for the duration of a test, so even if a request escaped the
   socket guard it would be unauthenticated and could not act on an account.
"""

from __future__ import annotations

import os
import socket
from typing import Any, Dict, List

import pytest


# --------------------------------------------------------------------------
# Guard rails
# --------------------------------------------------------------------------

_CREDENTIAL_VARS = (
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET",
    "CME_API_KEY",
    "CME_API_SECRET",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


class NetworkAccessAttempted(RuntimeError):
    """Raised when a test tries to open a socket."""


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    """Block all outbound sockets unless the test is marked `network`."""

    if request.node.get_closest_marker("network"):
        return

    def _blocked(*args: Any, **kwargs: Any):
        raise NetworkAccessAttempted(
            f"{request.node.nodeid} attempted a network connection. "
            "Mock the exchange, or mark the test with @pytest.mark.network."
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch):
    """Ensure no real API key is visible to code under test."""

    for name in _CREDENTIAL_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _clean_safety_env(monkeypatch):
    """Start every test from a known state for the safety-relevant flags.

    These are read at import time in places, and a developer's own shell
    settings must not change what the suite asserts.
    """

    for name in ("PAPER_TRADING_MODE", "BINANCE_TESTNET", "EMERGENCY_STOP"):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------
# Market data fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def ohlcv_rows() -> List[List[float]]:
    """A well-formed 100 candle OHLCV frame, ascending, with real wicks.

    Long enough for every indicator window in the project (the longest is the
    26 period EMA inside MACD).
    """

    rows = []
    price = 100.0
    for i in range(100):
        base = price + i * 0.10
        rows.append([
            1_700_000_000_000 + i * 60_000,  # ms timestamp, one minute apart
            base,                            # open
            base + 0.25,                     # high
            base - 0.25,                     # low
            base + 0.05,                     # close
            10.0 + i,                        # volume
        ])
    return rows


@pytest.fixture
def short_ohlcv_rows(ohlcv_rows) -> List[List[float]]:
    """Ten candles: fewer than the 14 an ATR needs. A newly listed pair."""

    return ohlcv_rows[:10]


# --------------------------------------------------------------------------
# Exchange double
# --------------------------------------------------------------------------


class FakeCCXTExchange:
    """A stand-in for a ccxt client that records rather than transmits.

    Deliberately not a MagicMock: assertions about what the system would have
    sent to an exchange are the whole point of the exchange tests, so the calls
    are recorded in a form that is easy to assert on, and anything unexpected
    raises rather than silently returning a Mock.
    """

    def __init__(self, markets: Dict[str, Any] | None = None):
        self.urls = {"api": {"public": "https://testnet.binance.vision/api/v3",
                             "private": "https://testnet.binance.vision/api/v3"}}
        self.options: Dict[str, Any] = {}
        self.sandbox = False
        self.created_orders: List[Dict[str, Any]] = []
        self.cancelled_orders: List[Dict[str, Any]] = []
        self.markets = markets or {
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "type": "spot",
                "spot": True,
                "swap": False,
                "precision": {"amount": 5, "price": 2},
                "limits": {
                    "amount": {"min": 0.00001, "max": 9000.0},
                    "price": {"min": 0.01, "max": 1_000_000.0},
                    "cost": {"min": 5.0, "max": None},
                },
            }
        }

    # -- lifecycle ---------------------------------------------------------

    def set_sandbox_mode(self, enabled: bool) -> None:
        self.sandbox = enabled
        host = "testnet.binance.vision" if enabled else "api.binance.com"
        self.urls["api"]["public"] = f"https://{host}/api/v3"
        self.urls["api"]["private"] = f"https://{host}/api/v3"

    def load_markets(self, reload: bool = False) -> Dict[str, Any]:
        return self.markets

    def market(self, symbol: str) -> Dict[str, Any]:
        return self.markets[symbol]

    async def close(self) -> None:
        return None

    # -- precision helpers, mirroring ccxt's contract ----------------------

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        digits = self.markets[symbol]["precision"]["amount"]
        return f"{amount:.{digits}f}"

    def price_to_precision(self, symbol: str, price: float) -> str:
        digits = self.markets[symbol]["precision"]["price"]
        return f"{price:.{digits}f}"

    # -- trading -----------------------------------------------------------

    def create_order(self, symbol, type, side, amount, price=None, params=None):
        record = {
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "price": price,
            "params": params or {},
        }
        self.created_orders.append(record)
        return {
            "id": f"fake-{len(self.created_orders)}",
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "price": price,
            "status": "open",
            "filled": 0.0,
            "timestamp": 1_700_000_000_000,
        }

    def cancel_order(self, order_id, symbol=None, params=None):
        self.cancelled_orders.append({"id": order_id, "symbol": symbol})
        return {"id": order_id, "status": "canceled"}

    def fetch_balance(self, params=None):
        return {
            "USDT": {"free": 1000.0, "used": 0.0, "total": 1000.0},
            "BTC": {"free": 0.5, "used": 0.0, "total": 0.5},
            "free": {"USDT": 1000.0, "BTC": 0.5},
            "total": {"USDT": 1000.0, "BTC": 0.5},
        }

    def fetch_open_orders(self, symbol=None, params=None):
        return []


@pytest.fixture
def fake_ccxt() -> FakeCCXTExchange:
    return FakeCCXTExchange()
