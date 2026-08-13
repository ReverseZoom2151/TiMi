"""Market data management for TiMi system."""

import re
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from ..exchange.base import BaseExchange, Ticker
from ..utils.logging import get_logger


logger = get_logger(__name__)


# Columns an OHLCV frame always carries, so an empty response still produces a
# frame the rest of the system can address by name instead of raising.
OHLCV_COLUMNS = ('open', 'high', 'low', 'close', 'volume')

# Units of the suffix in a timeframe string such as '1m', '4h', '1d'.
_TIMEFRAME_UNITS = {
    's': 'seconds',
    'm': 'minutes',
    'h': 'hours',
    'd': 'days',
    'w': 'weeks'
}

# Φ feeds `(1 - Φ) ** exponent` in the grid engine. At Φ >= 1 that expression
# turns negative or complex and the resulting order prices are nonsense, so a
# reading at or above this ceiling is clamped and logged rather than passed on.
MAX_VOLATILITY = 0.99


class InsufficientDataError(RuntimeError):
    """Raised when a statistic cannot be computed from the data available.

    Distinct from a value of zero. Zero means the market did not move; this
    means the market was not observed.
    """


@dataclass
class MarketStats:
    """Market statistics for a trading pair."""
    pair: str
    volume_24h: float
    volatility: float  # Φ from paper, a fraction of price (0.005 is 0.5%)
    price: float
    funding_rate: Optional[float] = None
    market_cap: Optional[float] = None
    timestamp: datetime = None

    def meets_requirements(
        self,
        min_volume: float,
        min_volatility: float
    ) -> bool:
        """Check if pair meets trading requirements.

        Args:
            min_volume: Minimum volume requirement
            min_volatility: Minimum volatility requirement (Φreq from paper),
                as a fraction of price on the same scale as `volatility`, so
                0.005 means 0.5% and 0.5 would mean 50%

        Returns:
            True if requirements met
        """
        return (
            self.volume_24h >= min_volume and
            self.volatility >= min_volatility
        )


class MarketDataManager:
    """Manager for market data retrieval and processing."""

    def __init__(
        self,
        exchange: BaseExchange,
        cache_ttl: timedelta = timedelta(seconds=15)
    ):
        """Initialize market data manager.

        Args:
            exchange: Exchange connector instance
            cache_ttl: How long a fetched frame stays servable from cache. This
                is measured from the moment of the fetch, not from the time of
                the last candle in the frame
        """
        self.exchange = exchange
        self.cache_ttl = cache_ttl
        # Keyed by (pair, timeframe, limit); the value carries the moment of
        # the fetch so freshness is a property of the request, not the data.
        self.cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
        self.logger = get_logger(__name__)

    @staticmethod
    def cache_key(pair: str, timeframe: str, limit: int) -> str:
        """Build the cache key for a historical data request.

        `limit` is part of the key. Without it a 100 candle request and a 60
        candle request for the same pair and timeframe collide, and since Φ is
        a range statistic that grows with the window, serving the longer frame
        to the shorter request inflates every grid width downstream.

        Args:
            pair: Trading pair
            timeframe: Candle timeframe
            limit: Number of candles requested

        Returns:
            Cache key string
        """
        return f"{pair}_{timeframe}_{limit}"

    @staticmethod
    def timeframe_to_timedelta(timeframe: str) -> Optional[timedelta]:
        """Convert a timeframe string to its candle interval.

        Args:
            timeframe: Timeframe such as '1m', '15m', '4h', '1d'

        Returns:
            The interval, or None if the string is not recognised
        """
        match = re.fullmatch(r'(\d+)([smhdw])', timeframe.strip().lower())
        if not match:
            return None

        amount, unit = match.groups()
        return timedelta(**{_TIMEFRAME_UNITS[unit]: int(amount)})

    @staticmethod
    def exclude_unclosed_candle(
        df: pd.DataFrame,
        timeframe: str,
        now: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Drop the final candle while it is still forming.

        Exchanges return the in-progress candle as the last row. Its open, high,
        low and close are all still moving, so any statistic taken from it, and
        any grid built on it, is provisional. A candle whose index timestamp is
        its open time `t` is complete only once `t + interval` has passed.

        Args:
            df: OHLCV frame indexed by candle open time
            timeframe: Timeframe the frame was fetched at
            now: Current time, injectable for testing. Defaults to `datetime.now()`

        Returns:
            The frame without its trailing in-progress candle. The frame is
            returned untouched if it is empty, if the timeframe is unrecognised,
            or if the index is not made of timestamps
        """
        if df.empty:
            return df

        interval = MarketDataManager.timeframe_to_timedelta(timeframe)
        if interval is None:
            return df

        reference = now if now is not None else datetime.now()
        last_open = df.index[-1]

        try:
            still_forming = last_open + interval > reference
        except TypeError:
            # Index is not a timestamp index; nothing sensible to decide.
            return df

        return df.iloc[:-1] if still_forming else df

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        """Build an empty OHLCV frame with the usual columns and index.

        Returns:
            Empty DataFrame indexed by 'timestamp' with the OHLCV columns
        """
        df = pd.DataFrame(columns=list(OHLCV_COLUMNS), dtype='float64')
        df.index.name = 'timestamp'
        return df

    async def get_ticker(self, pair: str) -> Ticker:
        """Get current ticker for a pair.

        Args:
            pair: Trading pair

        Returns:
            Ticker data
        """
        return await self.exchange.get_ticker(pair)

    async def get_historical_data(
        self,
        pair: str,
        timeframe: str = '1m',
        limit: int = 100,
        use_cache: bool = True,
        include_unclosed: bool = False
    ) -> pd.DataFrame:
        """Get historical OHLCV data.

        The trailing in-progress candle is excluded by default, so callers work
        from settled prices. The cache is keyed on the limit as well as the pair
        and timeframe, and its freshness is measured from the time of the fetch.

        Args:
            pair: Trading pair
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            limit: Number of candles
            use_cache: Whether to use cached data
            include_unclosed: Keep the trailing in-progress candle

        Returns:
            DataFrame with OHLCV data, indexed by candle open time. Empty, with
            the OHLCV columns present, if the exchange returned no candles
        """
        key = self.cache_key(pair, timeframe, limit)

        # Check cache. Age is fetch age, not candle age: a 1h frame is not
        # stale merely because its last candle opened an hour ago.
        if use_cache and key in self.cache:
            fetched_at, cached_df = self.cache[key]
            if datetime.now() - fetched_at < self.cache_ttl:
                if include_unclosed:
                    return cached_df
                return self.exclude_unclosed_candle(cached_df, timeframe)

        # Fetch fresh data
        ohlcv_data = await self.exchange.get_ohlcv(pair, timeframe, limit)

        # Convert to DataFrame. An empty response has no columns to index on,
        # so it is handled before touching set_index.
        if not ohlcv_data:
            self.logger.warning(
                "Exchange returned no candles",
                pair=pair,
                timeframe=timeframe,
                limit=limit
            )
            return self._empty_frame()

        df = pd.DataFrame([ohlcv.to_dict() for ohlcv in ohlcv_data])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # Cache the frame as fetched, together with the moment it arrived.
        self.cache[key] = (datetime.now(), df)

        if include_unclosed:
            return df

        return self.exclude_unclosed_candle(df, timeframe)

    async def calculate_volatility(
        self,
        pair: str,
        lookback_period: int = 60,
        timeframe: str = '1m'
    ) -> float:
        """Calculate volatility (Φ) as defined in Algorithm 1.

        Φ = (max(O, C) - min(O, C)) / C_recent

        Precisely what this is, since the whole strategy is calibrated to it:

        * A **range** statistic, not a standard deviation and not a return
          series. It is the peak-to-trough spread of the window expressed as a
          fraction of the most recent close.
        * A **fraction, not a percentage**. 0.005 is 0.5%. `min_volatility` in
          the configuration is compared against this directly and must be
          written in the same units.
        * **Scaled by the window.** Φ grows with `lookback_period`, because a
          longer window can only contain a wider range. Two calls with different
          lookbacks are not comparable, which is why the cache keys on limit.
        * Computed from **opens and closes only**, ignoring wicks, so it
          understates the true range of the window. This understatement is
          deliberate and left in place: the grid widths and profit targets are
          calibrated to it.
        * Computed from **closed candles only**. The in-progress candle is
          excluded by `get_historical_data`.

        Errors are not swallowed. A timeout, a rate limit or an empty response
        raises, because returning 0.0 in those cases is indistinguishable from a
        genuinely motionless market, and a Φ of zero collapses every grid level
        onto the spot price.

        Args:
            pair: Trading pair
            lookback_period: Number of periods to look back (T2 from paper)
            timeframe: Timeframe for candles

        Returns:
            Volatility as a fraction of the recent close (e.g., 0.05 for 5%),
            clamped to at most MAX_VOLATILITY. A genuine 0.0 means the window
            was flat

        Raises:
            InsufficientDataError: No candles, or a zero recent close, so Φ is
                undefined
            Exception: Whatever the exchange raised, propagated unchanged
        """
        df = await self.get_historical_data(pair, timeframe, lookback_period)

        if df.empty:
            raise InsufficientDataError(
                f"No candles for {pair} at {timeframe}; volatility is undefined"
            )

        # Calculate max and min prices over lookback period
        max_price = max(df['open'].max(), df['close'].max())
        min_price = min(df['open'].min(), df['close'].min())

        # Recent closing price
        recent_close = df['close'].iloc[-1]

        if not recent_close or pd.isna(recent_close):
            raise InsufficientDataError(
                f"Recent close for {pair} is {recent_close}; volatility is undefined"
            )

        # Calculate volatility
        volatility = (max_price - min_price) / recent_close

        if pd.isna(volatility):
            raise InsufficientDataError(
                f"Volatility for {pair} evaluated to NaN"
            )

        if volatility >= MAX_VOLATILITY:
            self.logger.warning(
                "Volatility clamped to ceiling",
                pair=pair,
                timeframe=timeframe,
                raw_volatility=f"{volatility:.4f}",
                ceiling=MAX_VOLATILITY
            )
            volatility = MAX_VOLATILITY

        return float(volatility)

    async def get_market_stats(
        self,
        pair: str,
        lookback_period: int = 60
    ) -> MarketStats:
        """Get comprehensive market statistics for a pair.

        Propagates rather than substituting a default, matching
        `calculate_volatility`: a pair whose data could not be read is skipped
        by the caller, not traded on a fabricated statistic.

        Args:
            pair: Trading pair
            lookback_period: Lookback period for volatility calculation

        Returns:
            Market statistics

        Raises:
            InsufficientDataError: Volatility could not be computed
            Exception: Whatever the exchange raised, propagated unchanged
        """
        try:
            # Get ticker for volume and price
            ticker = await self.get_ticker(pair)

            # Calculate volatility
            volatility = await self.calculate_volatility(pair, lookback_period)

            stats = MarketStats(
                pair=pair,
                volume_24h=ticker.volume_24h,
                volatility=volatility,
                price=ticker.last,
                timestamp=datetime.now()
            )

            self.logger.info(
                "Market stats calculated",
                pair=pair,
                volume=stats.volume_24h,
                volatility=f"{stats.volatility:.4f}"
            )

            return stats

        except Exception as e:
            self.logger.error("Error getting market stats", pair=pair, error=str(e))
            raise

    async def qualify_trading_pairs(
        self,
        pairs: List[str],
        min_volume: float,
        min_volatility: float,
        lookback_period: int = 60
    ) -> List[str]:
        """Qualify trading pairs based on volume and volatility requirements.

        Implements pair qualification logic from Algorithm 1:
        P = {p | V_p >= V_req ∧ Φ_p >= Φ_req}

        Args:
            pairs: List of candidate trading pairs
            min_volume: Minimum volume requirement (V_req)
            min_volatility: Minimum volatility requirement (Φ_req)
            lookback_period: Lookback period for volatility

        Returns:
            List of qualified trading pairs
        """
        qualified_pairs = []

        for pair in pairs:
            try:
                stats = await self.get_market_stats(pair, lookback_period)

                if stats.meets_requirements(min_volume, min_volatility):
                    qualified_pairs.append(pair)
                    self.logger.info(
                        "Pair qualified",
                        pair=pair,
                        volume=stats.volume_24h,
                        volatility=f"{stats.volatility:.4f}"
                    )
                else:
                    self.logger.debug(
                        "Pair did not qualify",
                        pair=pair,
                        volume=stats.volume_24h,
                        volatility=f"{stats.volatility:.4f}"
                    )

            except Exception as e:
                self.logger.error("Error qualifying pair", pair=pair, error=str(e))
                continue

        self.logger.info(
            "Pair qualification complete",
            total_pairs=len(pairs),
            qualified=len(qualified_pairs)
        )

        return qualified_pairs

    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        self.logger.info("Market data cache cleared")
