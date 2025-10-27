"""Market data management for TiMi system."""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

from ..exchange.base import BaseExchange, Ticker
from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class MarketStats:
    """Market statistics for a trading pair."""
    pair: str
    volume_24h: float
    volatility: float  # Φ from paper
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
            min_volatility: Minimum volatility requirement (Φreq from paper)

        Returns:
            True if requirements met
        """
        return (
            self.volume_24h >= min_volume and
            self.volatility >= min_volatility
        )


class MarketDataManager:
    """Manager for market data retrieval and processing."""

    def __init__(self, exchange: BaseExchange):
        """Initialize market data manager.

        Args:
            exchange: Exchange connector instance
        """
        self.exchange = exchange
        self.cache: Dict[str, pd.DataFrame] = {}
        self.logger = get_logger(__name__)

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
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get historical OHLCV data.

        Args:
            pair: Trading pair
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            limit: Number of candles
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{pair}_{timeframe}"

        # Check cache
        if use_cache and cache_key in self.cache:
            cached_df = self.cache[cache_key]
            # Return if cache is recent (within last minute)
            if not cached_df.empty:
                last_time = cached_df.index[-1]
                if datetime.now() - last_time < timedelta(minutes=1):
                    return cached_df

        # Fetch fresh data
        ohlcv_data = await self.exchange.get_ohlcv(pair, timeframe, limit)

        # Convert to DataFrame
        df = pd.DataFrame([ohlcv.to_dict() for ohlcv in ohlcv_data])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # Cache the data
        self.cache[cache_key] = df

        return df

    async def calculate_volatility(
        self,
        pair: str,
        lookback_period: int = 60,
        timeframe: str = '1m'
    ) -> float:
        """Calculate volatility (Φ) as defined in Algorithm 1.

        Φ = (max(O, C) - min(O, C)) / C_recent

        Args:
            pair: Trading pair
            lookback_period: Number of periods to look back (T2 from paper)
            timeframe: Timeframe for candles

        Returns:
            Volatility as a decimal (e.g., 0.05 for 5%)
        """
        try:
            df = await self.get_historical_data(pair, timeframe, lookback_period)

            if df.empty:
                return 0.0

            # Calculate max and min prices over lookback period
            max_price = max(df['open'].max(), df['close'].max())
            min_price = min(df['open'].min(), df['close'].min())

            # Recent closing price
            recent_close = df['close'].iloc[-1]

            if recent_close == 0:
                return 0.0

            # Calculate volatility
            volatility = (max_price - min_price) / recent_close

            return volatility

        except Exception as e:
            self.logger.error("Error calculating volatility", pair=pair, error=str(e))
            return 0.0

    async def get_market_stats(
        self,
        pair: str,
        lookback_period: int = 60
    ) -> MarketStats:
        """Get comprehensive market statistics for a pair.

        Args:
            pair: Trading pair
            lookback_period: Lookback period for volatility calculation

        Returns:
            Market statistics
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
