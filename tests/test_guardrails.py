"""Tests that the test suite's own safety net works.

These are meta-tests. They assert that the guard rails in conftest.py are in
force, because every other test in this project relies on them: this codebase
places real orders with real money, and a suite that could reach an exchange
account is not one anybody should run.

If these fail, do not run the rest of the suite until they pass again.
"""

from __future__ import annotations

import os
import socket

import pytest

from conftest import NetworkAccessAttempted


class TestNetworkIsBlocked:
    def test_create_connection_raises(self):
        with pytest.raises(NetworkAccessAttempted):
            socket.create_connection(("api.binance.com", 443), timeout=1)

    def test_socket_connect_raises(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(NetworkAccessAttempted):
                sock.connect(("api.binance.com", 443))
        finally:
            sock.close()

    def test_the_error_names_the_offending_test(self):
        """A blocked call must say which test did it, or debugging is guesswork."""
        with pytest.raises(NetworkAccessAttempted) as excinfo:
            socket.create_connection(("example.com", 80), timeout=1)

        assert "test_the_error_names_the_offending_test" in str(excinfo.value)


class TestCredentialsAreStripped:
    @pytest.mark.parametrize(
        "name",
        [
            "BINANCE_API_KEY",
            "BINANCE_API_SECRET",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ],
    )
    def test_credential_is_absent(self, name):
        assert os.getenv(name) is None

    @pytest.mark.parametrize(
        "name", ["PAPER_TRADING_MODE", "BINANCE_TESTNET", "EMERGENCY_STOP"]
    )
    def test_safety_flag_starts_unset(self, name):
        """A developer's own shell must not change what the suite asserts."""
        assert os.getenv(name) is None


class TestExchangeDouble:
    def test_sandbox_mode_moves_the_endpoint(self, fake_ccxt):
        fake_ccxt.set_sandbox_mode(True)
        assert "testnet" in fake_ccxt.urls["api"]["private"]

        fake_ccxt.set_sandbox_mode(False)
        assert "testnet" not in fake_ccxt.urls["api"]["private"]

    def test_orders_are_recorded_not_sent(self, fake_ccxt):
        fake_ccxt.create_order("BTC/USDT", "limit", "buy", 0.001, 50_000.0)

        assert len(fake_ccxt.created_orders) == 1
        assert fake_ccxt.created_orders[0]["side"] == "buy"

    def test_configured_pair_is_spot(self, fake_ccxt):
        """Mirrors the real exchange: BTC/USDT is spot, not a perpetual."""
        market = fake_ccxt.market("BTC/USDT")

        assert market["spot"] is True
        assert market["swap"] is False
