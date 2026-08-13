"""Tests for the technical indicator layer.

Every indicator is checked against a value worked out from its definition, not
merely against "it returned something". An indicator that runs and returns a
wrong number is the failure mode that matters here, because the grid engine
sizes real orders from these readings.

The second theme is short frames. A newly listed pair returns fewer candles
than the longest window, and the layer must say "unknown" rather than "zero".
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import pytest

from timi.data.indicators import TechnicalIndicators


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def frame_from_rows(rows: List[List[float]]) -> pd.DataFrame:
    """Turn raw candle rows into the OHLCV frame the indicators expect."""

    df = pd.DataFrame(
        rows,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = df['timestamp'].map(lambda ms: datetime.fromtimestamp(ms / 1000))
    df.set_index('timestamp', inplace=True)
    return df


def close_only_frame(closes: List[float]) -> pd.DataFrame:
    """A frame carrying just a close series, for the price based indicators."""

    return pd.DataFrame({
        'open': closes,
        'high': [c + 0.5 for c in closes],
        'low': [c - 0.5 for c in closes],
        'close': closes,
        'volume': [1.0] * len(closes)
    })


@pytest.fixture
def full_frame(ohlcv_rows) -> pd.DataFrame:
    """The 100 candle frame as a DataFrame."""

    return frame_from_rows(ohlcv_rows)


@pytest.fixture
def short_frame(short_ohlcv_rows) -> pd.DataFrame:
    """Ten candles: fewer than the 14 an ATR needs."""

    return frame_from_rows(short_ohlcv_rows)


# --------------------------------------------------------------------------
# Simple moving average
# --------------------------------------------------------------------------


def test_sma_matches_hand_computed_mean():
    """SMA over a known series equals the arithmetic mean of the window."""

    df = close_only_frame([float(i) for i in range(1, 11)])  # 1..10

    sma = TechnicalIndicators.calculate_sma(df, period=3)

    # Mean of 8, 9, 10.
    assert sma.iloc[-1] == pytest.approx(9.0)
    # Mean of 1, 2, 3.
    assert sma.iloc[2] == pytest.approx(2.0)


def test_sma_warm_up_is_nan_not_a_partial_average():
    """Rows before the window is full are NaN, not the mean of what exists."""

    df = close_only_frame([float(i) for i in range(1, 11)])

    sma = TechnicalIndicators.calculate_sma(df, period=3)

    assert pd.isna(sma.iloc[0])
    assert pd.isna(sma.iloc[1])


def test_sma_of_fixture_is_exact(full_frame):
    """The fixture closes are an arithmetic progression, so the SMA is exact."""

    # close = 100.05 + 0.10 * i for i in 0..99.
    # SMA(20) at the last row averages i = 80..99, whose mean index is 89.5.
    sma_20 = TechnicalIndicators.calculate_sma(full_frame, 20).iloc[-1]
    # SMA(50) averages i = 50..99, mean index 74.5.
    sma_50 = TechnicalIndicators.calculate_sma(full_frame, 50).iloc[-1]

    assert sma_20 == pytest.approx(100.05 + 0.10 * 89.5)
    assert sma_50 == pytest.approx(100.05 + 0.10 * 74.5)


# --------------------------------------------------------------------------
# Exponential moving average
# --------------------------------------------------------------------------


def test_ema_uses_smoothing_factor_two_over_n_plus_one():
    """EMA is the recursion e_t = alpha * x_t + (1 - alpha) * e_{t-1}."""

    df = close_only_frame([1.0, 2.0, 3.0, 4.0])
    period = 3
    alpha = 2 / (period + 1)  # 0.5

    ema = TechnicalIndicators.calculate_ema(df, period=period)

    expected = 1.0
    for price in (2.0, 3.0, 4.0):
        expected = alpha * price + (1 - alpha) * expected

    assert alpha == pytest.approx(0.5)
    assert ema.iloc[0] == pytest.approx(1.0)  # seeded with the first observation
    assert ema.iloc[1] == pytest.approx(1.5)
    assert ema.iloc[2] == pytest.approx(2.25)
    assert ema.iloc[-1] == pytest.approx(expected)
    assert ema.iloc[-1] == pytest.approx(3.125)


def test_ema_weights_recent_prices_more_heavily_than_sma():
    """On a rising series the EMA leads the SMA of the same period."""

    df = close_only_frame([float(i) for i in range(1, 21)])

    ema = TechnicalIndicators.calculate_ema(df, period=10).iloc[-1]
    sma = TechnicalIndicators.calculate_sma(df, period=10).iloc[-1]

    assert ema > sma


# --------------------------------------------------------------------------
# RSI
# --------------------------------------------------------------------------


def test_rsi_is_one_hundred_on_a_monotonic_rise(full_frame):
    """No down moves at all means no losses, so RSI sits at its ceiling."""

    rsi = TechnicalIndicators.calculate_rsi(full_frame)

    assert rsi.iloc[-1] == pytest.approx(100.0)


def test_rsi_is_bounded_zero_to_one_hundred():
    """RSI never leaves 0..100, whatever the series does."""

    rng = np.random.default_rng(11)
    closes = list(100 + np.cumsum(rng.normal(0, 1.5, 200)))

    rsi = TechnicalIndicators.calculate_rsi(close_only_frame(closes))
    defined = rsi.dropna()

    assert len(defined) > 0
    assert defined.min() >= 0.0
    assert defined.max() <= 100.0


def test_rsi_is_low_on_a_monotonic_fall():
    """The mirror case: only losses drives RSI to its floor."""

    closes = [float(200 - i) for i in range(60)]

    rsi = TechnicalIndicators.calculate_rsi(close_only_frame(closes))

    assert rsi.iloc[-1] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# MACD
# --------------------------------------------------------------------------


def test_macd_line_is_fast_ema_minus_slow_ema(full_frame):
    """MACD is the difference of two EMAs of the close."""

    macd = TechnicalIndicators.calculate_macd(full_frame)

    close = full_frame['close']
    fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = close.ewm(span=26, adjust=False, min_periods=26).mean()

    assert macd['macd'].iloc[-1] == pytest.approx((fast - slow).iloc[-1])


def test_macd_signal_is_an_ema_of_the_macd_line(full_frame):
    """The signal line smooths the MACD line, not the price."""

    macd = TechnicalIndicators.calculate_macd(full_frame)

    expected_signal = macd['macd'].ewm(span=9, adjust=False, min_periods=9).mean()

    assert macd['signal'].iloc[-1] == pytest.approx(expected_signal.iloc[-1])


def test_macd_histogram_is_macd_minus_signal(full_frame):
    """The histogram is the gap between the two lines."""

    macd = TechnicalIndicators.calculate_macd(full_frame)

    assert macd['histogram'].iloc[-1] == pytest.approx(
        macd['macd'].iloc[-1] - macd['signal'].iloc[-1]
    )


# --------------------------------------------------------------------------
# Bollinger bands
# --------------------------------------------------------------------------


def test_bollinger_middle_band_is_the_sma(full_frame):
    """The middle band is the plain moving average."""

    bands = TechnicalIndicators.calculate_bollinger_bands(full_frame, period=20)
    sma = TechnicalIndicators.calculate_sma(full_frame, 20)

    assert bands['middle'].iloc[-1] == pytest.approx(sma.iloc[-1])


def test_bollinger_bands_use_population_std_not_sample_std(full_frame):
    """Width is two POPULATION standard deviations (ddof=0)."""

    bands = TechnicalIndicators.calculate_bollinger_bands(full_frame, period=20, std_dev=2)

    close = full_frame['close']
    population = close.rolling(20).std(ddof=0).iloc[-1]
    sample = close.rolling(20).std(ddof=1).iloc[-1]
    middle = bands['middle'].iloc[-1]

    assert population != pytest.approx(sample)  # the two really do differ here
    assert bands['upper'].iloc[-1] == pytest.approx(middle + 2 * population)
    assert bands['lower'].iloc[-1] == pytest.approx(middle - 2 * population)


def test_bollinger_bands_collapse_on_a_flat_series():
    """A motionless market has zero deviation, so the bands meet the mean."""

    bands = TechnicalIndicators.calculate_bollinger_bands(
        close_only_frame([50.0] * 30), period=20
    )

    assert bands['upper'].iloc[-1] == pytest.approx(50.0)
    assert bands['lower'].iloc[-1] == pytest.approx(50.0)


# --------------------------------------------------------------------------
# ATR
# --------------------------------------------------------------------------


def test_atr_true_range_includes_the_previous_close():
    """A gap between candles counts, so ATR exceeds the high to low span."""

    # Candle 1 gaps from a close of 10 up to a 19..20 range. Its high to low
    # span is only 1, but the true range measured from the previous close is 10.
    df = pd.DataFrame({
        'open': [9.5, 19.5, 20.5],
        'high': [10.0, 20.0, 21.0],
        'low': [9.0, 19.0, 20.0],
        'close': [10.0, 20.0, 21.0],
        'volume': [1.0, 1.0, 1.0]
    })

    atr = TechnicalIndicators.calculate_atr(df, period=2)

    # Warm-up row has no reading at all.
    assert pd.isna(atr.iloc[0])
    # First reading is the mean of true ranges 1.0 and 10.0.
    assert atr.iloc[1] == pytest.approx(5.5)
    # Ignoring the previous close would have given 1.0 here, not 5.5.
    assert atr.iloc[1] > (df['high'] - df['low']).max()
    # Wilder smoothing: (5.5 * (2 - 1) + 1.0) / 2.
    assert atr.iloc[2] == pytest.approx(3.25)


def test_atr_of_a_steady_range_equals_that_range():
    """A market with a constant true range has an ATR equal to it."""

    df = pd.DataFrame({
        'open': [100.0] * 20,
        'high': [101.0] * 20,
        'low': [99.0] * 20,
        'close': [100.0] * 20,
        'volume': [1.0] * 20
    })

    atr = TechnicalIndicators.calculate_atr(df, period=14)

    assert atr.iloc[-1] == pytest.approx(2.0)


def test_atr_warm_up_is_nan_never_zero(full_frame):
    """The rows before the first reading must be NaN, not a fabricated 0.0."""

    atr = TechnicalIndicators.calculate_atr(full_frame, period=14)

    assert atr.iloc[:13].isna().all()
    assert not (atr.iloc[:13] == 0.0).any()
    assert not pd.isna(atr.iloc[13])


# --------------------------------------------------------------------------
# Short and empty frames
# --------------------------------------------------------------------------


def test_calculate_all_does_not_raise_on_a_short_frame(short_frame):
    """Ten candles used to raise IndexError from the 14 period ATR."""

    result = TechnicalIndicators.calculate_all(short_frame)

    assert len(result) == len(short_frame)
    assert 'atr' in result.columns


def test_short_frame_reports_nan_atr_rather_than_zero(short_frame):
    """The distinction that matters: unknown volatility is not zero volatility."""

    result = TechnicalIndicators.calculate_all(short_frame)

    assert result['atr'].isna().all()
    assert not (result['atr'] == 0.0).any()


def test_short_frame_summary_reports_none_for_unavailable_readings(short_frame):
    """`get_market_summary` must not present a warm-up zero as a reading."""

    summary = TechnicalIndicators.get_market_summary(short_frame)

    assert summary['volatility_atr'] is None
    assert summary['rsi'] is None
    assert summary['trend'] == 'unknown'
    assert summary['current_price'] == pytest.approx(short_frame['close'].iloc[-1])


def test_calculate_all_computes_what_a_short_frame_does_support(short_frame):
    """Windows the frame can satisfy are still filled in."""

    result = TechnicalIndicators.calculate_all(short_frame)

    # The 12 period EMA is defined from the first row, so it is available.
    assert not pd.isna(result['ema_12'].iloc[-1])
    # The 20 period SMA is not.
    assert result['sma_20'].isna().all()


def test_calculate_all_survives_a_single_candle():
    """One candle is the extreme case and must still come back as a frame."""

    df = close_only_frame([100.0])

    result = TechnicalIndicators.calculate_all(df)

    assert len(result) == 1
    assert pd.isna(result['atr'].iloc[0])


def test_calculate_all_survives_an_empty_frame():
    """An empty frame yields an empty frame carrying the indicator columns."""

    df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'], dtype='float64')

    result = TechnicalIndicators.calculate_all(df)

    assert result.empty
    for column in ('sma_20', 'rsi', 'macd', 'bb_upper', 'atr', 'volume_sma'):
        assert column in result.columns


def test_get_market_summary_of_an_empty_frame_is_empty():
    """Nothing to summarise, and nothing invented."""

    df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'], dtype='float64')

    assert TechnicalIndicators.get_market_summary(df) == {}


# --------------------------------------------------------------------------
# Trend and summary on a full frame
# --------------------------------------------------------------------------


def test_identify_trend_reads_uptrend_on_the_rising_fixture(full_frame):
    """Price above the fast average, fast average above the slow one."""

    assert TechnicalIndicators.identify_trend(full_frame) == 'uptrend'


def test_identify_trend_reads_downtrend_on_a_falling_series():
    """The mirror case."""

    df = close_only_frame([float(300 - i) for i in range(100)])

    assert TechnicalIndicators.identify_trend(df) == 'downtrend'


def test_identify_trend_is_unknown_when_the_slow_average_is_undefined(short_frame):
    """Ten candles cannot support a 50 period average, so the trend is unknown."""

    assert TechnicalIndicators.identify_trend(short_frame) == 'unknown'


def test_full_frame_summary_reports_real_readings(full_frame):
    """On a frame long enough for everything, nothing comes back as None."""

    summary = TechnicalIndicators.get_market_summary(full_frame)

    assert summary['trend'] == 'uptrend'
    assert summary['rsi'] == pytest.approx(100.0)
    assert summary['volatility_atr'] is not None
    assert summary['volatility_atr'] > 0
    assert summary['volume_trend'] == 'increasing'
    assert summary['price_change_24h'] > 0


def test_calculate_all_agrees_with_the_individual_indicators(full_frame):
    """The bundled frame is not a separate implementation."""

    result = TechnicalIndicators.calculate_all(full_frame)

    assert result['sma_20'].iloc[-1] == pytest.approx(
        TechnicalIndicators.calculate_sma(full_frame, 20).iloc[-1]
    )
    assert result['ema_26'].iloc[-1] == pytest.approx(
        TechnicalIndicators.calculate_ema(full_frame, 26).iloc[-1]
    )
    assert result['atr'].iloc[-1] == pytest.approx(
        TechnicalIndicators.calculate_atr(full_frame).iloc[-1]
    )


def test_calculate_all_leaves_the_input_frame_untouched(full_frame):
    """Indicators are added to a copy; the caller's frame is not mutated."""

    original_columns = list(full_frame.columns)

    TechnicalIndicators.calculate_all(full_frame)

    assert list(full_frame.columns) == original_columns
