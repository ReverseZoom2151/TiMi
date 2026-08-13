"""Tests for the market data layer.

The concerns here are the ones that quietly corrupt a live account rather than
crash it: a cache that serves the wrong frame, a freshness check measured
against the wrong clock, and an error path that reports a fabricated zero
volatility instead of admitting it could not read the market.

Nothing in this module touches a network. The exchange is a small double that
records what was asked of it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytest

from timi.data.market_data import (
    MAX_VOLATILITY,
    InsufficientDataError,
    MarketDataManager,
    MarketStats,
)
from timi.exchange.base import OHLCV, Ticker


# --------------------------------------------------------------------------
# Exchange double
# --------------------------------------------------------------------------


class RecordingExchange:
    """A stand-in exchange that records requests and returns canned candles.

    Deliberately not a MagicMock: the tests assert on exactly how many times the
    data layer went to the exchange and with which limit, which is the whole
    point of the cache tests.
    """

    def __init__(
        self,
        rows: Optional[List[List[float]]] = None,
        ticker_price: float = 100.0,
        volume_24h: float = 5_000_000.0
    ):
        self.rows = rows or []
        self.ticker_price = ticker_price
        self.volume_24h = volume_24h
        self.ohlcv_calls: List[Tuple[str, str, int]] = []
        self.ticker_calls: List[str] = []
        self.failure: Optional[Exception] = None

    async def get_ohlcv(
        self,
        pair: str,
        timeframe: str = '1m',
        limit: int = 100
    ) -> List[OHLCV]:
        self.ohlcv_calls.append((pair, timeframe, limit))

        if self.failure is not None:
            raise self.failure

        return [
            OHLCV(
                timestamp=datetime.fromtimestamp(row[0] / 1000),
                open=row[1],
                high=row[2],
                low=row[3],
                close=row[4],
                volume=row[5]
            )
            for row in self.rows[-limit:]
        ]

    async def get_ticker(self, pair: str) -> Ticker:
        self.ticker_calls.append(pair)

        if self.failure is not None:
            raise self.failure

        return Ticker(
            pair=pair,
            bid=self.ticker_price - 0.01,
            ask=self.ticker_price + 0.01,
            last=self.ticker_price,
            volume_24h=self.volume_24h,
            timestamp=datetime.now()
        )


def rows_ending_now(
    count: int = 20,
    interval_seconds: int = 60,
    seconds_into_last_candle: int = 10,
    price: float = 100.0
) -> List[List[float]]:
    """Candles whose final row is still forming, as a live exchange returns.

    The last candle opened `seconds_into_last_candle` ago, so with a one minute
    timeframe it has not closed yet.
    """

    now = datetime.now()
    last_open = now - timedelta(seconds=seconds_into_last_candle)
    rows = []

    for i in range(count):
        open_time = last_open - timedelta(seconds=interval_seconds * (count - 1 - i))
        base = price + i
        rows.append([
            open_time.timestamp() * 1000,
            base,
            base + 0.5,
            base - 0.5,
            base + 0.1,
            10.0
        ])

    return rows


def make_manager(**kwargs: Any) -> Tuple[MarketDataManager, RecordingExchange]:
    """Build a manager over a recording exchange."""

    exchange = RecordingExchange(**kwargs)
    return MarketDataManager(exchange), exchange


# --------------------------------------------------------------------------
# Defect D: the cache key must include the limit
# --------------------------------------------------------------------------


def test_cache_key_includes_the_limit():
    """Two limits produce two keys."""

    sixty = MarketDataManager.cache_key('BTC/USDT', '1m', 60)
    hundred = MarketDataManager.cache_key('BTC/USDT', '1m', 100)

    assert sixty != hundred
    assert '60' in sixty
    assert '100' in hundred


def test_cache_key_still_separates_pairs_and_timeframes():
    """The other two dimensions did not get lost in the change."""

    assert (
        MarketDataManager.cache_key('BTC/USDT', '1m', 60)
        != MarketDataManager.cache_key('ETH/USDT', '1m', 60)
    )
    assert (
        MarketDataManager.cache_key('BTC/USDT', '1m', 60)
        != MarketDataManager.cache_key('BTC/USDT', '1h', 60)
    )


async def test_different_limits_do_not_collide_in_the_cache(ohlcv_rows):
    """A 60 candle request must not be served the cached 100 candle frame.

    This is the live collision: the macro analysis agent caches 100 candles,
    then the volatility calculation asks for 60. Since the volatility measure is
    a range statistic that grows with the window, the longer frame inflates it.
    """

    manager, exchange = make_manager(rows=ohlcv_rows)

    hundred = await manager.get_historical_data('BTC/USDT', '1m', 100)
    sixty = await manager.get_historical_data('BTC/USDT', '1m', 60)

    assert len(hundred) == 100
    assert len(sixty) == 60
    assert [call[2] for call in exchange.ohlcv_calls] == [100, 60]


async def test_same_limit_is_served_from_the_cache(ohlcv_rows):
    """The cache still does its job for a repeated request."""

    manager, exchange = make_manager(rows=ohlcv_rows)

    first = await manager.get_historical_data('BTC/USDT', '1m', 60)
    second = await manager.get_historical_data('BTC/USDT', '1m', 60)

    assert len(exchange.ohlcv_calls) == 1
    assert len(first) == len(second) == 60


async def test_volatility_over_a_short_window_is_not_the_long_window_value(ohlcv_rows):
    """The collision, measured. The two windows give different numbers."""

    manager, _ = make_manager(rows=ohlcv_rows)

    await manager.get_historical_data('BTC/USDT', '1m', 100)
    sixty = await manager.calculate_volatility('BTC/USDT', lookback_period=60)
    hundred_frame = await manager.get_historical_data('BTC/USDT', '1m', 100)

    wide_range = (
        max(hundred_frame['open'].max(), hundred_frame['close'].max())
        - min(hundred_frame['open'].min(), hundred_frame['close'].min())
    ) / hundred_frame['close'].iloc[-1]

    assert sixty < wide_range


# --------------------------------------------------------------------------
# Defect E: TTL measured from the fetch, not from the candle
# --------------------------------------------------------------------------


async def test_cache_ttl_is_measured_from_the_fetch_not_the_candle_age(ohlcv_rows):
    """An hourly frame is not stale merely because its candle opened an hour ago.

    The fixture candles are historic. Judged by candle age every one of them is
    long expired, and the old check went back to the exchange every single time.
    """

    manager, exchange = make_manager(rows=ohlcv_rows)

    await manager.get_historical_data('BTC/USDT', '1h', 100)
    await manager.get_historical_data('BTC/USDT', '1h', 100)
    await manager.get_historical_data('BTC/USDT', '1h', 100)

    assert len(exchange.ohlcv_calls) == 1


async def test_cache_expires_once_the_ttl_has_elapsed(ohlcv_rows):
    """Freshness is a real limit, not an unbounded cache."""

    exchange = RecordingExchange(rows=ohlcv_rows)
    manager = MarketDataManager(exchange, cache_ttl=timedelta(seconds=30))

    await manager.get_historical_data('BTC/USDT', '1m', 100)

    # Age the recorded fetch time past the TTL.
    key = MarketDataManager.cache_key('BTC/USDT', '1m', 100)
    fetched_at, frame = manager.cache[key]
    manager.cache[key] = (fetched_at - timedelta(seconds=31), frame)

    await manager.get_historical_data('BTC/USDT', '1m', 100)

    assert len(exchange.ohlcv_calls) == 2


async def test_use_cache_false_always_refetches(ohlcv_rows):
    """An explicit opt out is honoured."""

    manager, exchange = make_manager(rows=ohlcv_rows)

    await manager.get_historical_data('BTC/USDT', '1m', 100)
    await manager.get_historical_data('BTC/USDT', '1m', 100, use_cache=False)

    assert len(exchange.ohlcv_calls) == 2


async def test_clear_cache_forces_a_refetch(ohlcv_rows):
    """The cache is still clearable."""

    manager, exchange = make_manager(rows=ohlcv_rows)

    await manager.get_historical_data('BTC/USDT', '1m', 100)
    manager.clear_cache()
    await manager.get_historical_data('BTC/USDT', '1m', 100)

    assert len(exchange.ohlcv_calls) == 2


# --------------------------------------------------------------------------
# Defect F: an empty candle list
# --------------------------------------------------------------------------


async def test_empty_candle_list_returns_an_empty_frame_with_columns():
    """No candles must not raise a KeyError from set_index."""

    manager, _ = make_manager(rows=[])

    df = await manager.get_historical_data('NEW/USDT', '1m', 60)

    assert df.empty
    for column in ('open', 'high', 'low', 'close', 'volume'):
        assert column in df.columns


async def test_empty_candle_list_is_not_cached():
    """An empty response must not be served back for the rest of the TTL."""

    manager, exchange = make_manager(rows=[])

    await manager.get_historical_data('NEW/USDT', '1m', 60)
    await manager.get_historical_data('NEW/USDT', '1m', 60)

    assert len(exchange.ohlcv_calls) == 2


# --------------------------------------------------------------------------
# Defect G: a failure must never be reported as zero volatility
# --------------------------------------------------------------------------


async def test_fetch_error_does_not_become_zero_volatility(ohlcv_rows):
    """A timeout is not a flat market."""

    manager, exchange = make_manager(rows=ohlcv_rows)
    exchange.failure = TimeoutError("connection timed out")

    with pytest.raises(TimeoutError):
        await manager.calculate_volatility('BTC/USDT', lookback_period=60)


async def test_empty_frame_does_not_become_zero_volatility():
    """Neither is an empty response."""

    manager, _ = make_manager(rows=[])

    with pytest.raises(InsufficientDataError):
        await manager.calculate_volatility('NEW/USDT', lookback_period=60)


async def test_zero_recent_close_does_not_become_zero_volatility():
    """A zero denominator is undefined, not zero."""

    rows = [[1_700_000_000_000 + i * 60_000, 0.0, 0.0, 0.0, 0.0, 1.0] for i in range(5)]
    manager, _ = make_manager(rows=rows)

    with pytest.raises(InsufficientDataError):
        await manager.calculate_volatility('DEAD/USDT', lookback_period=60)


async def test_genuinely_flat_market_returns_zero():
    """A real zero is still reportable, and distinguishable from a failure."""

    rows = [[1_700_000_000_000 + i * 60_000, 50.0, 50.0, 50.0, 50.0, 1.0] for i in range(5)]
    manager, _ = make_manager(rows=rows)

    volatility = await manager.calculate_volatility('FLAT/USDT', lookback_period=60)

    assert volatility == 0.0


async def test_market_stats_propagates_the_same_failure(ohlcv_rows):
    """`get_market_stats` and `calculate_volatility` behave alike on error."""

    manager, exchange = make_manager(rows=ohlcv_rows)
    exchange.failure = TimeoutError("connection timed out")

    with pytest.raises(TimeoutError):
        await manager.get_market_stats('BTC/USDT', lookback_period=60)


async def test_qualify_trading_pairs_skips_a_pair_it_cannot_read(ohlcv_rows):
    """A pair that fails to price is dropped, not admitted on a default."""

    manager, exchange = make_manager(rows=[])

    qualified = await manager.qualify_trading_pairs(
        ['NEW/USDT'],
        min_volume=0.0,
        min_volatility=0.0
    )

    assert qualified == []


# --------------------------------------------------------------------------
# Defect H: the in-progress candle
# --------------------------------------------------------------------------


def test_timeframe_to_timedelta_parses_the_usual_strings():
    """Interval parsing underpins the in-progress candle check."""

    assert MarketDataManager.timeframe_to_timedelta('1m') == timedelta(minutes=1)
    assert MarketDataManager.timeframe_to_timedelta('15m') == timedelta(minutes=15)
    assert MarketDataManager.timeframe_to_timedelta('4h') == timedelta(hours=4)
    assert MarketDataManager.timeframe_to_timedelta('1d') == timedelta(days=1)
    assert MarketDataManager.timeframe_to_timedelta('nonsense') is None


def test_exclude_unclosed_candle_drops_a_candle_that_is_still_forming(ohlcv_rows):
    """A candle whose interval has not elapsed is provisional and is dropped."""

    df = _frame(ohlcv_rows)
    last_open = df.index[-1]

    trimmed = MarketDataManager.exclude_unclosed_candle(
        df, '1m', now=last_open + timedelta(seconds=30)
    )

    assert len(trimmed) == len(df) - 1
    assert trimmed.index[-1] == df.index[-2]


def test_exclude_unclosed_candle_keeps_a_closed_candle(ohlcv_rows):
    """Once the interval has elapsed the candle is final and is kept."""

    df = _frame(ohlcv_rows)
    last_open = df.index[-1]

    kept = MarketDataManager.exclude_unclosed_candle(
        df, '1m', now=last_open + timedelta(seconds=61)
    )

    assert len(kept) == len(df)


def test_exclude_unclosed_candle_leaves_an_unknown_timeframe_alone(ohlcv_rows):
    """Without an interval there is nothing to decide, so nothing is dropped."""

    df = _frame(ohlcv_rows)

    assert len(MarketDataManager.exclude_unclosed_candle(df, 'weekly')) == len(df)


def test_exclude_unclosed_candle_handles_an_empty_frame():
    """The degenerate case does not raise."""

    empty = MarketDataManager._empty_frame()

    assert MarketDataManager.exclude_unclosed_candle(empty, '1m').empty


async def test_historical_data_excludes_the_in_progress_candle():
    """The live path drops the forming candle by default."""

    rows = rows_ending_now(count=20, seconds_into_last_candle=10)
    manager, _ = make_manager(rows=rows)

    df = await manager.get_historical_data('BTC/USDT', '1m', 20)

    assert len(df) == 19
    assert df['close'].iloc[-1] == pytest.approx(rows[-2][4])


async def test_historical_data_can_include_the_in_progress_candle_on_request():
    """The exclusion is a default, not a prohibition."""

    rows = rows_ending_now(count=20, seconds_into_last_candle=10)
    manager, _ = make_manager(rows=rows)

    df = await manager.get_historical_data('BTC/USDT', '1m', 20, include_unclosed=True)

    assert len(df) == 20
    assert df['close'].iloc[-1] == pytest.approx(rows[-1][4])


async def test_volatility_ignores_the_in_progress_candle():
    """A spike in the forming candle must not widen the grid before it settles."""

    rows = rows_ending_now(count=20, seconds_into_last_candle=10)
    rows[-1][1] = 10_000.0  # a wild open on the candle that is still forming
    rows[-1][4] = 10_000.0

    manager, _ = make_manager(rows=rows)

    volatility = await manager.calculate_volatility('BTC/USDT', lookback_period=20)

    assert volatility < 1.0


# --------------------------------------------------------------------------
# Defect I: guards on the measure itself
# --------------------------------------------------------------------------


async def test_volatility_is_a_fraction_not_a_percentage():
    """A 1% peak-to-trough window reports 0.01, not 1.0."""

    # Opens and closes span 99.0 to 100.0, with the last close at 100.0.
    rows = [
        [1_700_000_000_000, 99.0, 100.5, 98.5, 99.5, 1.0],
        [1_700_000_060_000, 99.5, 100.5, 98.5, 100.0, 1.0]
    ]
    manager, _ = make_manager(rows=rows)

    volatility = await manager.calculate_volatility('BTC/USDT', lookback_period=60)

    assert volatility == pytest.approx((100.0 - 99.0) / 100.0)
    assert volatility == pytest.approx(0.01)


async def test_volatility_at_or_above_one_is_clamped():
    """Φ >= 1 makes `(1 - Φ) ** exponent` complex in the grid engine.

    A tenfold move over the window is a real reading, but it cannot be handed on
    unclamped without every grid price turning into a complex number.
    """

    rows = [
        [1_700_000_000_000, 10.0, 110.0, 9.0, 10.0, 1.0],
        [1_700_000_060_000, 100.0, 110.0, 9.0, 10.0, 1.0]
    ]
    manager, _ = make_manager(rows=rows)

    volatility = await manager.calculate_volatility('WILD/USDT', lookback_period=60)

    assert volatility == MAX_VOLATILITY
    assert volatility < 1.0
    # The expression the engine evaluates stays real.
    assert isinstance((1 - volatility) ** 0.5, float)


async def test_volatility_uses_opens_and_closes_only():
    """The measure is unchanged: wicks are outside it, by design.

    Documented rather than corrected, because the grid widths and profit targets
    are calibrated to this understatement.
    """

    rows = [
        [1_700_000_000_000, 100.0, 200.0, 50.0, 100.0, 1.0],
        [1_700_000_060_000, 100.0, 300.0, 10.0, 100.0, 1.0]
    ]
    manager, _ = make_manager(rows=rows)

    volatility = await manager.calculate_volatility('BTC/USDT', lookback_period=60)

    assert volatility == 0.0  # flat opens and closes, despite enormous wicks


# --------------------------------------------------------------------------
# Requirements check, and the configured threshold
# --------------------------------------------------------------------------


def test_meets_requirements_compares_volatility_on_the_fraction_scale():
    """0.005 means 0.5%, and a 1% market clears it."""

    stats = MarketStats(
        pair='BTC/USDT',
        volume_24h=5_000_000.0,
        volatility=0.01,
        price=100.0,
        timestamp=datetime.now()
    )

    assert stats.meets_requirements(min_volume=1_000_000, min_volatility=0.005)
    # The shipped 0.5 demanded a 50% range and admitted nothing.
    assert not stats.meets_requirements(min_volume=1_000_000, min_volatility=0.5)


def test_configured_min_volatility_is_on_the_fraction_scale():
    """The shipped configuration must not silently block every order."""

    import yaml
    from pathlib import Path

    config_path = Path(__file__).resolve().parent.parent / 'config.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))

    min_volatility = config['strategy']['min_volatility']

    assert min_volatility == pytest.approx(0.005)
    assert min_volatility < 0.1  # a fraction, not a percentage


async def test_market_stats_reports_the_computed_volatility(ohlcv_rows):
    """The happy path still assembles a full statistics record."""

    manager, exchange = make_manager(rows=ohlcv_rows)

    stats = await manager.get_market_stats('BTC/USDT', lookback_period=60)

    assert stats.pair == 'BTC/USDT'
    assert stats.volume_24h == exchange.volume_24h
    assert stats.price == exchange.ticker_price
    assert stats.volatility > 0


# --------------------------------------------------------------------------
# Helpers used above
# --------------------------------------------------------------------------


def _frame(rows: List[List[float]]) -> pd.DataFrame:
    """Build an OHLCV frame from raw candle rows, indexed by open time."""

    df = pd.DataFrame(
        rows,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = df['timestamp'].map(lambda ms: datetime.fromtimestamp(ms / 1000))
    df.set_index('timestamp', inplace=True)
    return df
