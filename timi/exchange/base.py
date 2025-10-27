"""Base exchange interface for TiMi system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order data structure."""
    id: str
    pair: str
    type: OrderType
    side: OrderSide
    price: float
    quantity: float
    filled_quantity: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    timestamp: Optional[datetime] = None
    filled_timestamp: Optional[datetime] = None
    fee: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_open(self) -> bool:
        """Check if order is still open."""
        return self.status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]

    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to be filled."""
        return self.quantity - self.filled_quantity


@dataclass
class Position:
    """Position data structure."""
    pair: str
    size: float  # Positive for long, negative for short
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.size < 0

    @property
    def market_value(self) -> float:
        """Get current market value of position."""
        return abs(self.size) * self.current_price

    @property
    def pnl_percentage(self) -> float:
        """Get PnL as percentage of entry value."""
        entry_value = abs(self.size) * self.entry_price
        if entry_value == 0:
            return 0.0
        return (self.unrealized_pnl / entry_value) * 100


@dataclass
class Ticker:
    """Market ticker data."""
    pair: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    timestamp: datetime

    @property
    def mid_price(self) -> float:
        """Get mid price between bid and ask."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Get bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_percentage(self) -> float:
        """Get spread as percentage."""
        if self.mid_price == 0:
            return 0.0
        return (self.spread / self.mid_price) * 100


@dataclass
class OHLCV:
    """OHLCV candlestick data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


class BaseExchange(ABC):
    """Base class for exchange connectors."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        **kwargs
    ):
        """Initialize exchange connector.

        Args:
            api_key: API key
            api_secret: API secret
            testnet: Whether to use testnet
            **kwargs: Additional exchange-specific parameters
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.kwargs = kwargs

    @abstractmethod
    async def get_ticker(self, pair: str) -> Ticker:
        """Get current ticker for a trading pair.

        Args:
            pair: Trading pair (e.g., 'BTC/USDT')

        Returns:
            Ticker data
        """
        pass

    @abstractmethod
    async def get_ohlcv(
        self,
        pair: str,
        timeframe: str = '1m',
        limit: int = 100
    ) -> List[OHLCV]:
        """Get OHLCV candlestick data.

        Args:
            pair: Trading pair
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            limit: Number of candles

        Returns:
            List of OHLCV data
        """
        pass

    @abstractmethod
    async def get_balance(self, currency: str) -> Dict[str, float]:
        """Get balance for a currency.

        Args:
            currency: Currency symbol (e.g., 'USDT')

        Returns:
            Dict with 'free', 'used', and 'total' balance
        """
        pass

    @abstractmethod
    async def create_order(
        self,
        pair: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Create a new order.

        Args:
            pair: Trading pair
            order_type: Order type
            side: Buy or sell
            quantity: Order quantity
            price: Limit price (required for limit orders)
            **kwargs: Additional order parameters

        Returns:
            Created order
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, pair: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order ID
            pair: Trading pair

        Returns:
            True if canceled successfully
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: str, pair: str) -> Order:
        """Get order information.

        Args:
            order_id: Order ID
            pair: Trading pair

        Returns:
            Order data
        """
        pass

    @abstractmethod
    async def get_open_orders(self, pair: Optional[str] = None) -> List[Order]:
        """Get all open orders.

        Args:
            pair: Trading pair (None for all pairs)

        Returns:
            List of open orders
        """
        pass

    @abstractmethod
    async def get_positions(self, pair: Optional[str] = None) -> List[Position]:
        """Get current positions.

        Args:
            pair: Trading pair (None for all pairs)

        Returns:
            List of positions
        """
        pass

    @abstractmethod
    async def get_trading_fees(self, pair: str) -> Dict[str, float]:
        """Get trading fees for a pair.

        Args:
            pair: Trading pair

        Returns:
            Dict with 'maker' and 'taker' fee rates
        """
        pass

    @abstractmethod
    async def get_market_info(self, pair: str) -> Dict[str, Any]:
        """Get market information for a pair.

        Args:
            pair: Trading pair

        Returns:
            Market info including limits, precision, etc.
        """
        pass


class ExchangeError(Exception):
    """Base exception for exchange-related errors."""
    pass


class InsufficientBalanceError(ExchangeError):
    """Insufficient balance error."""
    pass


class OrderError(ExchangeError):
    """Order creation/management error."""
    pass


class RateLimitError(ExchangeError):
    """Rate limit exceeded error."""
    pass


class APIError(ExchangeError):
    """General API error."""
    pass
