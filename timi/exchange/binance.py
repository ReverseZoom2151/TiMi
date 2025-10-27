"""Binance exchange connector for TiMi system."""

import ccxt.async_support as ccxt
from datetime import datetime
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import (
    BaseExchange,
    Order,
    Position,
    Ticker,
    OHLCV,
    OrderType,
    OrderSide,
    OrderStatus,
    InsufficientBalanceError,
    OrderError,
    RateLimitError,
    APIError
)
from ..utils.logging import get_logger


logger = get_logger(__name__)


class BinanceExchange(BaseExchange):
    """Binance exchange connector using CCXT."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        **kwargs
    ):
        """Initialize Binance connector.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Whether to use testnet (default True for safety)
            **kwargs: Additional parameters
        """
        super().__init__(api_key, api_secret, testnet, **kwargs)

        # Initialize CCXT Binance client
        options = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Use futures by default
            }
        }

        if testnet:
            self.exchange = ccxt.binance({
                **options,
                'urls': {
                    'api': {
                        'public': 'https://testnet.binancefuture.com/fapi/v1',
                        'private': 'https://testnet.binancefuture.com/fapi/v1',
                    }
                }
            })
            logger.info("Binance connector initialized in TESTNET mode")
        else:
            self.exchange = ccxt.binance(options)
            logger.warning("Binance connector initialized in LIVE mode")

    async def close(self):
        """Close exchange connection."""
        await self.exchange.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_ticker(self, pair: str) -> Ticker:
        """Get current ticker for a trading pair."""
        try:
            ticker = await self.exchange.fetch_ticker(pair)

            return Ticker(
                pair=pair,
                bid=float(ticker['bid']) if ticker['bid'] else 0.0,
                ask=float(ticker['ask']) if ticker['ask'] else 0.0,
                last=float(ticker['last']),
                volume_24h=float(ticker['quoteVolume']) if ticker['quoteVolume'] else 0.0,
                timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000) if ticker['timestamp'] else datetime.now()
            )

        except ccxt.RateLimitExceeded as e:
            logger.error("Rate limit exceeded", pair=pair, error=str(e))
            raise RateLimitError(str(e))
        except Exception as e:
            logger.error("Error fetching ticker", pair=pair, error=str(e))
            raise APIError(f"Failed to fetch ticker: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_ohlcv(
        self,
        pair: str,
        timeframe: str = '1m',
        limit: int = 100
    ) -> List[OHLCV]:
        """Get OHLCV candlestick data."""
        try:
            ohlcv_data = await self.exchange.fetch_ohlcv(
                pair,
                timeframe=timeframe,
                limit=limit
            )

            return [
                OHLCV(
                    timestamp=datetime.fromtimestamp(candle[0] / 1000),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5])
                )
                for candle in ohlcv_data
            ]

        except Exception as e:
            logger.error("Error fetching OHLCV", pair=pair, error=str(e))
            raise APIError(f"Failed to fetch OHLCV: {str(e)}")

    async def get_balance(self, currency: str) -> Dict[str, float]:
        """Get balance for a currency."""
        try:
            balance = await self.exchange.fetch_balance()

            if currency in balance:
                return {
                    'free': float(balance[currency]['free']),
                    'used': float(balance[currency]['used']),
                    'total': float(balance[currency]['total'])
                }
            else:
                return {'free': 0.0, 'used': 0.0, 'total': 0.0}

        except Exception as e:
            logger.error("Error fetching balance", currency=currency, error=str(e))
            raise APIError(f"Failed to fetch balance: {str(e)}")

    async def create_order(
        self,
        pair: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Create a new order."""
        try:
            # Convert order type
            ccxt_type = self._convert_order_type(order_type)
            ccxt_side = 'buy' if side == OrderSide.BUY else 'sell'

            # Create order parameters
            params = kwargs.copy()

            # Create order
            if order_type == OrderType.MARKET:
                order_result = await self.exchange.create_market_order(
                    pair,
                    ccxt_side,
                    quantity,
                    params
                )
            else:
                if price is None:
                    raise OrderError("Price required for limit orders")

                order_result = await self.exchange.create_limit_order(
                    pair,
                    ccxt_side,
                    quantity,
                    price,
                    params
                )

            logger.info(
                "Order created",
                order_id=order_result['id'],
                pair=pair,
                type=order_type.value,
                side=side.value,
                quantity=quantity,
                price=price
            )

            return self._parse_order(order_result)

        except ccxt.InsufficientFunds as e:
            logger.error("Insufficient funds", pair=pair, error=str(e))
            raise InsufficientBalanceError(str(e))
        except Exception as e:
            logger.error("Error creating order", pair=pair, error=str(e))
            raise OrderError(f"Failed to create order: {str(e)}")

    async def cancel_order(self, order_id: str, pair: str) -> bool:
        """Cancel an open order."""
        try:
            await self.exchange.cancel_order(order_id, pair)
            logger.info("Order canceled", order_id=order_id, pair=pair)
            return True

        except Exception as e:
            logger.error("Error canceling order", order_id=order_id, error=str(e))
            raise OrderError(f"Failed to cancel order: {str(e)}")

    async def get_order(self, order_id: str, pair: str) -> Order:
        """Get order information."""
        try:
            order_result = await self.exchange.fetch_order(order_id, pair)
            return self._parse_order(order_result)

        except Exception as e:
            logger.error("Error fetching order", order_id=order_id, error=str(e))
            raise APIError(f"Failed to fetch order: {str(e)}")

    async def get_open_orders(self, pair: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        try:
            orders = await self.exchange.fetch_open_orders(pair)
            return [self._parse_order(order) for order in orders]

        except Exception as e:
            logger.error("Error fetching open orders", error=str(e))
            raise APIError(f"Failed to fetch open orders: {str(e)}")

    async def get_positions(self, pair: Optional[str] = None) -> List[Position]:
        """Get current positions."""
        try:
            positions = await self.exchange.fetch_positions(symbols=[pair] if pair else None)

            active_positions = []
            for pos in positions:
                # Only include positions with non-zero size
                size = float(pos.get('contracts', 0))
                if size != 0:
                    active_positions.append(
                        Position(
                            pair=pos['symbol'],
                            size=size if pos['side'] == 'long' else -size,
                            entry_price=float(pos['entryPrice']),
                            current_price=float(pos['markPrice']),
                            unrealized_pnl=float(pos['unrealizedPnl']),
                            realized_pnl=float(pos.get('realizedPnl', 0)),
                            timestamp=datetime.fromtimestamp(pos['timestamp'] / 1000) if pos.get('timestamp') else datetime.now(),
                            metadata=pos
                        )
                    )

            return active_positions

        except Exception as e:
            logger.error("Error fetching positions", error=str(e))
            raise APIError(f"Failed to fetch positions: {str(e)}")

    async def get_trading_fees(self, pair: str) -> Dict[str, float]:
        """Get trading fees for a pair."""
        try:
            fees = await self.exchange.fetch_trading_fees()

            if pair in fees:
                return {
                    'maker': float(fees[pair]['maker']),
                    'taker': float(fees[pair]['taker'])
                }
            else:
                # Return default fees
                return {'maker': 0.0002, 'taker': 0.0004}

        except Exception as e:
            logger.warning("Error fetching fees, using defaults", error=str(e))
            return {'maker': 0.0002, 'taker': 0.0004}

    async def get_market_info(self, pair: str) -> Dict[str, Any]:
        """Get market information for a pair."""
        try:
            markets = await self.exchange.load_markets()

            if pair in markets:
                market = markets[pair]
                return {
                    'symbol': market['symbol'],
                    'base': market['base'],
                    'quote': market['quote'],
                    'active': market['active'],
                    'precision': {
                        'price': market['precision']['price'],
                        'amount': market['precision']['amount']
                    },
                    'limits': {
                        'amount': {
                            'min': market['limits']['amount']['min'],
                            'max': market['limits']['amount']['max']
                        },
                        'price': {
                            'min': market['limits']['price']['min'],
                            'max': market['limits']['price']['max']
                        },
                        'cost': {
                            'min': market['limits']['cost']['min'],
                            'max': market['limits']['cost']['max']
                        }
                    }
                }
            else:
                raise APIError(f"Market {pair} not found")

        except Exception as e:
            logger.error("Error fetching market info", pair=pair, error=str(e))
            raise APIError(f"Failed to fetch market info: {str(e)}")

    def _parse_order(self, order_data: Dict[str, Any]) -> Order:
        """Parse CCXT order data to Order object."""
        return Order(
            id=order_data['id'],
            pair=order_data['symbol'],
            type=self._parse_order_type(order_data['type']),
            side=OrderSide.BUY if order_data['side'] == 'buy' else OrderSide.SELL,
            price=float(order_data['price']) if order_data['price'] else 0.0,
            quantity=float(order_data['amount']),
            filled_quantity=float(order_data['filled']),
            status=self._parse_order_status(order_data['status']),
            timestamp=datetime.fromtimestamp(order_data['timestamp'] / 1000) if order_data.get('timestamp') else None,
            fee=float(order_data['fee']['cost']) if order_data.get('fee') else 0.0,
            metadata=order_data
        )

    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert OrderType to CCXT order type."""
        mapping = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP_LOSS: 'stop_loss',
            OrderType.TAKE_PROFIT: 'take_profit'
        }
        return mapping.get(order_type, 'limit')

    def _parse_order_type(self, ccxt_type: str) -> OrderType:
        """Parse CCXT order type to OrderType."""
        mapping = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop_loss': OrderType.STOP_LOSS,
            'stop_loss_limit': OrderType.STOP_LOSS,
            'take_profit': OrderType.TAKE_PROFIT,
            'take_profit_limit': OrderType.TAKE_PROFIT
        }
        return mapping.get(ccxt_type.lower(), OrderType.LIMIT)

    def _parse_order_status(self, ccxt_status: str) -> OrderStatus:
        """Parse CCXT order status to OrderStatus."""
        mapping = {
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELED,
            'expired': OrderStatus.CANCELED,
            'rejected': OrderStatus.REJECTED,
            'pending': OrderStatus.PENDING
        }
        return mapping.get(ccxt_status.lower(), OrderStatus.PENDING)
