"""Technical indicators for market analysis."""

import pandas as pd
import ta  # Technical Analysis library


class TechnicalIndicators:
    """Technical indicators for macro analysis agent."""

    @staticmethod
    def calculate_sma(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """Calculate Simple Moving Average.

        Args:
            df: DataFrame with OHLCV data
            period: SMA period
            column: Column to calculate SMA on

        Returns:
            SMA series
        """
        return df[column].rolling(window=period).mean()

    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """Calculate Exponential Moving Average.

        Args:
            df: DataFrame with OHLCV data
            period: EMA period
            column: Column to calculate EMA on

        Returns:
            EMA series
        """
        return df[column].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """Calculate Relative Strength Index.

        Args:
            df: DataFrame with OHLCV data
            period: RSI period
            column: Column to calculate RSI on

        Returns:
            RSI series
        """
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

        Args:
            df: DataFrame with OHLCV data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            column: Column to calculate MACD on

        Returns:
            DataFrame with MACD, signal, and histogram
        """
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

        Args:
            df: DataFrame with OHLCV data
            period: Moving average period
            std_dev: Standard deviation multiplier
            column: Column to calculate on

        Returns:
            DataFrame with upper, middle, and lower bands
        """
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

        Args:
            df: DataFrame with OHLCV data
            period: ATR period

        Returns:
            ATR series
        """
        return ta.volatility.AverageTrueRange(
            df['high'],
            df['low'],
            df['close'],
            window=period
        ).average_true_range()

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all common technical indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all indicators added
        """
        result = df.copy()

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
        result['volume_sma'] = df['volume'].rolling(window=20).mean()

        return result

    @staticmethod
    def identify_trend(df: pd.DataFrame) -> str:
        """Identify market trend.

        Args:
            df: DataFrame with OHLCV data and indicators

        Returns:
            Trend string: 'uptrend', 'downtrend', or 'sideways'
        """
        if 'sma_20' not in df.columns:
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

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with market summary statistics
        """
        if df.empty:
            return {}

        df_with_indicators = TechnicalIndicators.calculate_all(df)

        return {
            'current_price': df['close'].iloc[-1],
            'trend': TechnicalIndicators.identify_trend(df_with_indicators),
            'rsi': df_with_indicators['rsi'].iloc[-1] if not pd.isna(df_with_indicators['rsi'].iloc[-1]) else None,
            'volume_trend': 'increasing' if df['volume'].iloc[-5:].mean() > df['volume'].iloc[-20:].mean() else 'decreasing',
            'volatility_atr': df_with_indicators['atr'].iloc[-1] if not pd.isna(df_with_indicators['atr'].iloc[-1]) else None,
            'price_change_24h': ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100) if len(df) > 0 else 0
        }
