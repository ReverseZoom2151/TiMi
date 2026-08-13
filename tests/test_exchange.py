"""Tests for the Binance connector: routing, rounding, retries and secrecy.

Every test here runs against the recording double from `conftest.py`. Nothing
in this module may reach the network; the autouse socket guard will fail the
test if it tries.

The double is synchronous while the connector drives ccxt's async client, so
`AsyncCCXTAdapter` presents the recorded calls as coroutines. It holds a
reference to the double itself, so assertions still read
`fake_ccxt.created_orders`.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict

import pytest

import ccxt.async_support as ccxt

from timi.exchange import binance as binance_module
from timi.exchange.base import (
    MinimumNotionalError,
    OrderError,
    OrderSide,
    OrderStatusUnknown,
    OrderType,
    APIError,
)
from timi.exchange.binance import (
    PERMANENT_ERRORS,
    TRANSIENT_ERRORS,
    BinanceExchange,
)


API_KEY = "key-AAAAAAAAAAAAAAAAAAAAAAAA"
API_SECRET = "secret-BBBBBBBBBBBBBBBBBBBBBBBB"

PAIR = "BTC/USDT"

#: Calls the connector awaits. Everything else stays synchronous, matching
#: ccxt, where the precision helpers are plain functions.
_AWAITED = frozenset({
    "load_markets",
    "fetch_ticker",
    "fetch_ohlcv",
    "fetch_balance",
    "fetch_order",
    "fetch_open_orders",
    "fetch_trading_fees",
    "create_order",
    "cancel_order",
})


class AsyncCCXTAdapter:
    """Present the synchronous recording double as an async ccxt client."""

    def __init__(self, inner: Any):
        self.inner = inner

    def __getattr__(self, name: str):
        attr = getattr(self.inner, name)

        if name in _AWAITED and callable(attr) and not inspect.iscoroutinefunction(attr):
            async def _awaited(*args, **kwargs):
                return attr(*args, **kwargs)

            return _awaited

        return attr


@pytest.fixture
def client(fake_ccxt, monkeypatch):
    """A connector wired to the double, in testnet mode, with no retry sleep."""

    return _build(fake_ccxt, monkeypatch, testnet=True)


def _build(fake_ccxt, monkeypatch, testnet: bool) -> BinanceExchange:
    """Construct a connector whose ccxt client is the recording double."""

    adapter = AsyncCCXTAdapter(fake_ccxt)
    captured: Dict[str, Any] = {}

    def _factory(config: Dict[str, Any]):
        captured.update(config)
        return adapter

    monkeypatch.setattr(binance_module.ccxt, "binance", _factory)

    exchange = BinanceExchange(
        api_key=API_KEY,
        api_secret=API_SECRET,
        testnet=testnet,
        retry_attempts=3,
        retry_wait_min=0,
        retry_wait_max=0,
    )
    exchange.constructor_config = captured
    exchange.adapter = adapter
    return exchange


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def test_testnet_resolves_to_a_test_endpoint(client):
    """Defect A: testnet mode must move the endpoint the client will call."""

    assert "testnet" in client.api_endpoint
    assert client.is_sandbox is True
    assert client.adapter.inner.sandbox is True


def test_live_does_not_resolve_to_a_test_endpoint(fake_ccxt, monkeypatch):
    """Live mode must not leave a sandbox URL behind."""

    exchange = _build(fake_ccxt, monkeypatch, testnet=False)

    assert "testnet" not in exchange.api_endpoint
    assert exchange.is_sandbox is False
    assert fake_ccxt.sandbox is False


def test_private_endpoint_alias_matches(client):
    """Start-up assertions may read either name."""

    assert client.private_endpoint == client.api_endpoint


def test_no_futures_market_type_is_configured(client, fake_ccxt):
    """Defect B: BTC/USDT is a spot market, so nothing may ask for futures."""

    config = client.constructor_config
    assert "defaultType" not in config.get("options", {})
    assert "future" not in repr(config)
    assert "swap" not in repr(config)
    assert fake_ccxt.options.get("defaultType") not in ("future", "swap", "delivery")


# --------------------------------------------------------------------------
# Rounding and minimums
# --------------------------------------------------------------------------


async def test_price_and_amount_are_rounded_before_sending(client, fake_ccxt):
    """Defect C: raw floats are rejected by the exchange's lot and tick size."""

    await client.create_order(
        pair=PAIR,
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=0.123456789,
        price=30000.987654,
    )

    assert len(fake_ccxt.created_orders) == 1
    sent = fake_ccxt.created_orders[0]
    assert sent["amount"] == 0.12346  # amount precision is 5 decimals
    assert sent["price"] == 30000.99  # price precision is 2 decimals


async def test_order_below_minimum_notional_never_reaches_the_exchange(client, fake_ccxt):
    """0.0001 BTC at 30000 is 3 USDT, under the 5 USDT minimum."""

    with pytest.raises(MinimumNotionalError) as excinfo:
        await client.create_order(
            pair=PAIR,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=0.0001,
            price=30000.12345,
        )

    message = str(excinfo.value)
    assert "5.0" in message  # names the limit
    assert PAIR in message
    assert fake_ccxt.created_orders == []


async def test_order_below_minimum_amount_never_reaches_the_exchange(client, fake_ccxt):
    """A quantity under the lot minimum is refused, with the limit named."""

    with pytest.raises(MinimumNotionalError) as excinfo:
        await client.create_order(
            pair=PAIR,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=0.000001,
            price=1_000_000.0,
        )

    assert "1e-05" in str(excinfo.value) or "0.00001" in str(excinfo.value)
    assert fake_ccxt.created_orders == []


async def test_market_order_is_sent_as_a_market_order(client, fake_ccxt):
    """A market order carries no price, and is valued using the reference."""

    await client.create_order(
        pair=PAIR,
        order_type=OrderType.MARKET,
        side=OrderSide.SELL,
        quantity=0.01,
        price=30000.0,
    )

    sent = fake_ccxt.created_orders[0]
    assert sent["type"] == "market"
    assert sent["price"] is None


# --------------------------------------------------------------------------
# Order types
# --------------------------------------------------------------------------


async def test_stop_loss_is_sent_as_a_stop_order(client, fake_ccxt):
    """Defect D: a stop loss must not be downgraded to a plain limit."""

    await client.create_order(
        pair=PAIR,
        order_type=OrderType.STOP_LOSS,
        side=OrderSide.SELL,
        quantity=0.5,
        price=29000.123,
        stop_price=29500.456,
    )

    sent = fake_ccxt.created_orders[0]
    assert sent["type"] == "stop_loss_limit"
    assert sent["type"] != "limit"
    assert sent["params"]["triggerPrice"] == 29500.46


async def test_stop_loss_without_a_trigger_is_refused(client, fake_ccxt):
    """Better to refuse than to place an unprotected limit order."""

    with pytest.raises(OrderError):
        await client.create_order(
            pair=PAIR,
            order_type=OrderType.STOP_LOSS,
            side=OrderSide.SELL,
            quantity=0.5,
            price=29000.0,
        )

    assert fake_ccxt.created_orders == []


async def test_every_order_carries_a_client_order_id(client, fake_ccxt):
    """Defect E: without an id there is nothing to reconcile against."""

    await client.create_order(
        pair=PAIR,
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=0.01,
        price=30000.0,
    )

    client_order_id = fake_ccxt.created_orders[0]["params"]["clientOrderId"]
    assert client_order_id.startswith("timi-")
    assert len(client_order_id) <= 36


# --------------------------------------------------------------------------
# Failure semantics
# --------------------------------------------------------------------------


async def test_order_timeout_raises_unknown_status_not_failure(client, fake_ccxt):
    """Defect E: a timeout on POST /order is not a rejection."""

    async def _timeout(*args, **kwargs):
        raise ccxt.RequestTimeout("read timeout")

    client.adapter.create_order = _timeout

    with pytest.raises(OrderStatusUnknown) as excinfo:
        await client.create_order(
            pair=PAIR,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=0.01,
            price=30000.0,
        )

    error = excinfo.value
    assert not isinstance(error, OrderError)  # must not read as "rejected"
    assert "may or" in str(error)
    assert error.client_order_id.startswith("timi-")
    assert error.pair == PAIR


async def test_order_timeout_is_not_retried(client):
    """Repeating a POST /order can open the position twice."""

    calls = []

    async def _timeout(*args, **kwargs):
        calls.append(1)
        raise ccxt.RequestTimeout("read timeout")

    client.adapter.create_order = _timeout

    with pytest.raises(OrderStatusUnknown):
        await client.create_order(
            pair=PAIR,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=0.01,
            price=30000.0,
        )

    assert len(calls) == 1


def test_permanent_errors_are_never_classed_as_transient():
    """The two sets must not overlap, or a permanent error would be retried."""

    for permanent in PERMANENT_ERRORS:
        assert not issubclass(permanent, TRANSIENT_ERRORS)


async def test_permanent_error_is_not_retried(client):
    """Defect F: a bad symbol will still be bad on the third attempt."""

    calls = []

    async def _bad_symbol(*args, **kwargs):
        calls.append(1)
        raise ccxt.BadSymbol("no market symbol BTC/USDT:USDT")

    client.adapter.fetch_ticker = _bad_symbol

    with pytest.raises(APIError):
        await client.get_ticker(PAIR)

    assert len(calls) == 1


async def test_transient_error_is_retried(client):
    """Defect F: a dropped connection deserves another attempt."""

    calls = []

    async def _flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise ccxt.ExchangeNotAvailable("503")
        return {
            "bid": 29999.0,
            "ask": 30001.0,
            "last": 30000.0,
            "quoteVolume": 1234.0,
            "timestamp": 1_700_000_000_000,
        }

    client.adapter.fetch_ticker = _flaky

    ticker = await client.get_ticker(PAIR)

    assert len(calls) == 3
    assert ticker.last == 30000.0


async def test_cancel_order_retries_transient_failures(client):
    """Defect F: cancellation is safe to repeat, so it is bounded-retried."""

    calls = []

    async def _flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise ccxt.RequestTimeout("read timeout")
        return {"id": "abc", "status": "canceled"}

    client.adapter.cancel_order = _flaky

    assert await client.cancel_order("abc", PAIR) is True
    assert len(calls) == 3


async def test_cancel_order_does_not_retry_permanent_failures(client):
    """An unknown order id will not become known by asking again."""

    calls = []

    async def _bad_request(*args, **kwargs):
        calls.append(1)
        raise ccxt.BadRequest("unknown parameter")

    client.adapter.cancel_order = _bad_request

    with pytest.raises(OrderError):
        await client.cancel_order("abc", PAIR)

    assert len(calls) == 1


# --------------------------------------------------------------------------
# Spot inventory
# --------------------------------------------------------------------------


async def test_positions_come_from_the_spot_balance(client):
    """Defect G: spot has no positions, only held base-asset inventory."""

    positions = await client.get_positions(PAIR)

    assert len(positions) == 1
    position = positions[0]
    assert position.pair == PAIR
    assert position.size == 0.5  # the BTC balance
    assert position.is_long is True
    assert position.is_short is False
    assert position.metadata["spot"] is True
    assert position.metadata["entry_price_known"] is False


async def test_positions_without_a_pair_cover_every_holding(client):
    """The quote currency alone is not inventory; the base asset is."""

    positions = await client.get_positions()

    assert [p.pair for p in positions] == [PAIR]


# --------------------------------------------------------------------------
# Secrecy
# --------------------------------------------------------------------------


def test_repr_never_discloses_credentials(client):
    """Defect: a connector printed into a traceback must stay safe."""

    text = repr(client) + str(client)

    assert API_KEY not in text
    assert API_SECRET not in text
    assert "REDACTED" in text


def test_formatted_log_record_never_discloses_credentials(client):
    """A log line built from the connector, or from an echoed error."""

    record = logging.LogRecord(
        name="timi.exchange",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="connector=%s detail=%s",
        args=(client, client.redact(f"rejected key {API_KEY} secret {API_SECRET}")),
        exc_info=None,
    )
    formatted = logging.Formatter("%(message)s").format(record)

    assert API_KEY not in formatted
    assert API_SECRET not in formatted


async def test_echoed_credentials_are_stripped_from_raised_errors(client):
    """Exchanges sometimes quote the submitted key back in an error body."""

    async def _leaky(*args, **kwargs):
        raise ccxt.BadRequest(f"signature invalid for {API_KEY} / {API_SECRET}")

    client.adapter.fetch_ticker = _leaky

    with pytest.raises(APIError) as excinfo:
        await client.get_ticker(PAIR)

    assert API_KEY not in str(excinfo.value)
    assert API_SECRET not in str(excinfo.value)
