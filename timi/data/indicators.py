"""Technical indicators for market analysis.

Every indicator here obeys two rules, because the callers are trading systems
and a wrong number is worse than no number:

1. An indicator whose window is longer than the frame returns NaN, never 0.0.
   A short frame (a newly listed pair, a truncated response) must be visibly
   unknown rather than quietly reported as a real reading of zero.
2. No indicator raises on a short frame. `calculate_all` computes whatever the
   data supports and leaves the rest as NaN.
"""

from typing import Optional

import numpy as np
import pandas as pd
import ta  # Technical Analysis library


def _nan_series(index: pd.Index, name: Optional[str] = None) -> pd.Series:
    """Build an all NaN float series aligned to an index.

    Args:
        index: Index to align to
        name: Optional series name

    Returns:
        Series of NaN with dtype float64
    """
    return pd.Series(np.nan, index=index, name=name, dtype='float64')


class TechnicalIndicators:
    """Technical indicators for macro analysis agent."""

    @staticmethod
    def calculate_sma(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """Calculate Simple Moving Average.

        The first `period - 1` rows are NaN: an average over fewer samples than
        requested is a different statistic and is not returned as if it were an
        SMA of `period`.

        Args:
            df: DataFrame with OHLCV data
            period: SMA period
            column: Column to calculate SMA on

        Returns:
            SMA series, NaN where the window is not yet full
        """
        if df.empty or column not in df.columns:
            return _nan_series(df.index, column)

        return df[column].rolling(window=period).mean()

    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """Calculate Exponential Moving Average.

        Recursive form with smoothing factor alpha = 2 / (period + 1) and no
        rebalancing of the weights (`adjust=False`), seeded with the first
        observation. Defined from the first row onwards, so unlike the SMA it
        has no NaN warm-up.

        Args:
            df: DataFrame with OHLCV data
            period: EMA period
            column: Column to calculate EMA on

        Returns:
            EMA series
        """
        if df.empty or column not in df.columns:
            return _nan_series(df.index, column)

        return df[column].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """Calculate Relative Strength Index.

        Bounded 0 to 100. A window with no down moves at all gives 100.

        Args:
            df: DataFrame with OHLCV data
            period: RSI period
            column: Column to calculate RSI on

        Returns:
            RSI series, NaN where the frame is shorter than the window
        """
        if df.empty or column not in df.columns or len(df) < period + 1:
            return _nan_series(df.index, 'rsi')

        return ta.momentum.RSIIndicator(df[column], window=period).rsi()

    @staticmethod
    def calculate_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = 'close'
    ) -> pd.DataFrame:
        """Calculate MACD indicator.

        MACD line is EMA(fast) - EMA(slow); the signal line is an EMA of the
        MACD line over `signal` periods; the histogram is their difference.

        Args:
            df: DataFrame with OHLCV data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            column: Column to calculate MACD on

        Returns:
            DataFrame with MACD, signal, and histogram, all NaN where the frame
            is shorter than the slow window
        """
        if df.empty or column not in df.columns or len(df) < slow:
            return pd.DataFrame({
                'macd': _nan_series(df.index, 'macd'),
                'signal': _nan_series(df.index, 'signal'),
                'histogram': _nan_series(df.index, 'histogram')
            })

        macd_indicator = ta.trend.MACD(
            df[column],
            window_fast=fast,
            window_slow=slow,
            window_sign=signal
        )

        return pd.DataFrame({
            'macd': macd_indicator.macd(),
            'signal': macd_indicator.macd_signal(),
            'histogram': macd_indicator.macd_diff()
        })

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: int = 2,
        column: str = 'close'
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands.

        Middle band is the SMA over `period`; the outer bands sit `std_dev`
        population standard deviations (ddof=0) away from it.

        Args:
            df: DataFrame with OHLCV data
            period: Moving average period
            std_dev: Standard deviation multiplier
            column: Column to calculate on

        Returns:
            DataFrame with upper, middle, and lower bands, all NaN where the
            frame is shorter than the window
        """
        if df.empty or column not in df.columns or len(df) < period:
            return pd.DataFrame({
                'upper': _nan_series(df.index, 'upper'),
                'middle': _nan_series(df.index, 'middle'),
                'lower': _nan_series(df.index, 'lower')
            })

        bb_indicator = ta.volatility.BollingerBands(
            df[column],
            window=period,
            window_dev=std_dev
        )

        return pd.DataFrame({
            'upper': bb_indicator.bollinger_hband(),
            'middle': bb_indicator.bollinger_mavg(),
            'lower': bb_indicator.bollinger_lband()
        })

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range.

        True range for a candle is the greatest of the high to low span, the
        gap from the previous close to the high, and the gap from the previous
        close to the low, so an overnight gap is counted. The first candle has
        no previous close and falls back to its own high to low span.

        Smoothing is Wilder's: the reading at index `period - 1` is the mean of
        the first `period` true ranges, and each later reading is
        (previous * (period - 1) + true_range) / period.

        The warm-up rows are NaN, not 0.0. The library implementation fills
        them with zeros, which is indistinguishable from a genuinely flat
        market and defeats every `pd.isna` guard downstream.

        Args:
            df: DataFrame with OHLCV data
            period: ATR period

        Returns:
            ATR series, NaN for the warm-up rows and throughout if the frame is
            shorter than `period`
        """
        required = {'high', 'low', 'close'}
        if df.empty or not required.issubset(df.columns) or len(df) < period:
            return _nan_series(df.index, 'atr')

        high = df['high']
        low = df['low']
        prev_close = df['close'].shift(1)

        true_range = pd.DataFrame({
            'high_low': high - low,
            'high_prev_close': (high - prev_close).abs(),
            'low_prev_close': (low - prev_close).abs()
        }).max(axis=1)

        values = np.full(len(df), np.nan)
        values[period - 1] = true_range.iloc[0:period].mean()
        for i in range(period, len(values)):
            values[i] = (
                values[i - 1] * (period - 1) + true_range.iloc[i]
            ) / float(period)

        return pd.Series(values, index=df.index, name='atr')

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all common technical indicators.

        Safe on a frame of any length, including empty. Indicators whose window
        the frame cannot satisfy come back as NaN columns rather than raising or
        reporting zero, so a newly listed pair still yields a usable frame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all indicators added
        """
        result = df.copy()

        if df.empty:
            for column in (
                'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi',
                'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'volume_sma'
            ):
                result[column] = _nan_series(df.index, column)
            return result

        # Moving Averages
        result['sma_20'] = TechnicalIndicators.calculate_sma(df, 20)
        result['sma_50'] = TechnicalIndicators.calculate_sma(df, 50)
        result['ema_12'] = TechnicalIndicators.calculate_ema(df, 12)
        result['ema_26'] = TechnicalIndicators.calculate_ema(df, 26)

        # RSI
        result['rsi'] = TechnicalIndicators.calculate_rsi(df)

        # MACD
        macd_data = TechnicalIndicators.calculate_macd(df)
        result['macd'] = macd_data['macd']
        result['macd_signal'] = macd_data['signal']
        result['macd_histogram'] = macd_data['histogram']

        # Bollinger Bands
        bb_data = TechnicalIndicators.calculate_bollinger_bands(df)
        result['bb_upper'] = bb_data['upper']
        result['bb_middle'] = bb_data['middle']
        result['bb_lower'] = bb_data['lower']

        # ATR
        result['atr'] = TechnicalIndicators.calculate_atr(df)

        # Volume indicators
        if 'volume' in df.columns:
            result['volume_sma'] = df['volume'].rolling(window=20).mean()
        else:
            result['volume_sma'] = _nan_series(df.index, 'volume_sma')

        return result

    @staticmethod
    def identify_trend(df: pd.DataFrame) -> str:
        """Identify market trend.

        Args:
            df: DataFrame with OHLCV data and indicators

        Returns:
            Trend string: 'uptrend', 'downtrend', 'sideways', or 'unknown' when
            the frame is too short for the 50 period SMA
        """
        if df.empty:
            return 'unknown'

        if 'sma_20' not in df.columns or 'sma_50' not in df.columns:
            df = TechnicalIndicators.calculate_all(df)

        recent_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]

        if pd.isna(sma_20) or pd.isna(sma_50):
            return 'unknown'

        # Strong uptrend
        if recent_price > sma_20 > sma_50:
            return 'uptrend'

        # Strong downtrend
        elif recent_price < sma_20 < sma_50:
            return 'downtrend'

        # Sideways/consolidation
        else:
            return 'sideways'

    @staticmethod
    def get_market_summary(df: pd.DataFrame) -> dict:
        """Get comprehensive market summary.

        Any reading the frame cannot support is reported as None rather than as
        a number, so a caller can tell "not enough data" from "zero".

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with market summary statistics, or an empty dictionary
            if the frame has no rows
        """
        if df.empty:
            return {}

        df_with_indicators = TechnicalIndicators.calculate_all(df)

        rsi = df_with_indicators['rsi'].iloc[-1]
        atr = df_with_indicators['atr'].iloc[-1]
        first_close = df['close'].iloc[0]

        return {
            'current_price': df['close'].iloc[-1],
            'trend': TechnicalIndicators.identify_trend(df_with_indicators),
            'rsi': None if pd.isna(rsi) else float(rsi),
            'volume_trend': (
                'increasing'
                if df['volume'].iloc[-5:].mean() > df['volume'].iloc[-20:].mean()
                else 'decreasing'
            ),
            'volatility_atr': None if pd.isna(atr) else float(atr),
            'price_change_24h': (
                ((df['close'].iloc[-1] - first_close) / first_close * 100)
                if first_close else 0
            )
        }
