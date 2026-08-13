"""Binance exchange connector for TiMi system.

This connector trades the SPOT market. Symbols are written in the plain
`BASE/QUOTE` form (`BTC/USDT`), which resolves to a spot market. The USDT
perpetual would be `BTC/USDT:USDT` and is deliberately not used here: there is
no leverage, no funding and no liquidation anywhere in this system.
"""

import hashlib
import time
import ccxt.async_support as ccxt
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential
)

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
    MinimumNotionalError,
    OrderError,
    OrderStatusUnknown,
    RateLimitError,
    APIError
)
from ..utils.logging import get_logger


logger = get_logger(__name__)


#: Failures worth retrying: the request never got a verdict from the exchange,
#: or the exchange asked us to come back later. `RequestTimeout`,
#: `ExchangeNotAvailable`, `DDoSProtection` and `RateLimitExceeded` all derive
#: from `NetworkError`, but they are named for the reader.
TRANSIENT_ERRORS: Tuple[type, ...] = (
    ccxt.NetworkError,
    ccxt.RequestTimeout,
    ccxt.ExchangeNotAvailable,
    ccxt.DDoSProtection,
    ccxt.RateLimitExceeded,
)

#: Failures that will fail again no matter how often they are repeated.
PERMANENT_ERRORS: Tuple[type, ...] = (
    ccxt.AuthenticationError,
    ccxt.PermissionDenied,
    ccxt.BadSymbol,
    ccxt.BadRequest,
    ccxt.InvalidOrder,
    ccxt.InsufficientFunds,
    ccxt.OrderNotFound,
)

#: Prefix for every client order id this system generates.
CLIENT_ORDER_ID_PREFIX = "timi"


class BinanceExchange(BaseExchange):
    """Binance SPOT exchange connector using CCXT."""

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
            **kwargs: Additional parameters. `retry_attempts`,
                `retry_wait_min` and `retry_wait_max` tune the transient
                retry policy.
        """
        super().__init__(api_key, api_secret, testnet, **kwargs)

        self._retry_attempts = int(kwargs.get('retry_attempts', 3))
        self._retry_wait_min = float(kwargs.get('retry_wait_min', 1.0))
        self._retry_wait_max = float(kwargs.get('retry_wait_max', 8.0))

        # No `defaultType` is set. The default is spot, which is what every
        # configured pair (BTC/USDT and friends) actually resolves to.
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

        # The only mechanism that moves every endpoint. Overriding
        # `urls.api.public`/`private` by hand leaves the rest behind, which is
        # how test orders end up at the live venue.
        self.exchange.set_sandbox_mode(bool(testnet))

        if testnet:
            logger.info(
                "Binance connector initialized in TESTNET mode",
                endpoint=self.api_endpoint
            )
        else:
            logger.warning(
                "Binance connector initialized in LIVE mode",
                endpoint=self.api_endpoint
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def api_endpoint(self) -> str:
        """Return the private REST endpoint the client will actually call.

        Exposed so that start-up code can assert the resolved endpoint agrees
        with the configured mode, rather than trusting a log line.
        """
        try:
            return str(self.exchange.urls['api']['private'])
        except (KeyError, TypeError):  # pragma: no cover - defensive
            return ''

    #: Alias, for callers that think in terms of the private URL.
    private_endpoint = api_endpoint

    @property
    def is_sandbox(self) -> bool:
        """Return True when the resolved endpoint is a Binance test endpoint."""
        endpoint = self.api_endpoint.lower()
        return 'testnet' in endpoint or 'vision' in endpoint

    async def close(self):
        """Close exchange connection."""
        await self.exchange.close()

    # ------------------------------------------------------------------
    # Retry policy
    # ------------------------------------------------------------------

    async def _retrying(self, func, *args, **kwargs):
        """Call `func`, retrying transient failures only.

        Permanent failures (bad symbol, bad credentials, rejected order) are
        raised on the first attempt: repeating them only burns rate limit.

        Args:
            func: Coroutine function to invoke
            *args: Positional arguments for `func`
            **kwargs: Keyword arguments for `func`

        Returns:
            Whatever `func` returns
        """
        retryer = AsyncRetrying(
            retry=retry_if_exception_type(TRANSIENT_ERRORS),
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_exponential(
                multiplier=1,
                min=self._retry_wait_min,
                max=self._retry_wait_max
            ),
            reraise=True,
        )
        return await retryer(func, *args, **kwargs)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_ticker(self, pair: str) -> Ticker:
        """Get current ticker for a trading pair."""
        try:
            ticker = await self._retrying(self.exchange.fetch_ticker, pair)

            return Ticker(
                pair=pair,
                bid=float(ticker['bid']) if ticker['bid'] else 0.0,
                ask=float(ticker['ask']) if ticker['ask'] else 0.0,
                last=float(ticker['last']),
                volume_24h=float(ticker['quoteVolume']) if ticker['quoteVolume'] else 0.0,
                timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000) if ticker['timestamp'] else datetime.now()
            )

        except ccxt.RateLimitExceeded as e:
            logger.error("Rate limit exceeded", pair=pair, error=self.redact(str(e)))
            raise RateLimitError(self.redact(str(e)))
        except Exception as e:
            logger.error("Error fetching ticker", pair=pair, error=self.redact(str(e)))
            raise APIError(f"Failed to fetch ticker: {self.redact(str(e))}")

    async def get_ohlcv(
        self,
        pair: str,
        timeframe: str = '1m',
        limit: int = 100
    ) -> List[OHLCV]:
        """Get OHLCV candlestick data."""
        try:
            ohlcv_data = await self._retrying(
                self.exchange.fetch_ohlcv,
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
            logger.error("Error fetching OHLCV", pair=pair, error=self.redact(str(e)))
            raise APIError(f"Failed to fetch OHLCV: {self.redact(str(e))}")

    async def get_balance(self, currency: str) -> Dict[str, float]:
        """Get balance for a currency."""
        try:
            balance = await self._retrying(self.exchange.fetch_balance)

            if currency in balance:
                return {
                    'free': float(balance[currency]['free']),
                    'used': float(balance[currency]['used']),
                    'total': float(balance[currency]['total'])
                }
            else:
                return {'free': 0.0, 'used': 0.0, 'total': 0.0}

        except Exception as e:
            logger.error("Error fetching balance", currency=currency, error=self.redact(str(e)))
            raise APIError(f"Failed to fetch balance: {self.redact(str(e))}")

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    async def create_order(
        self,
        pair: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Create a new order on the spot market.

        Quantity and price are rounded to the market's precision, and the
        order is checked against the market's minimum amount and minimum
        notional before anything is sent.

        This method is deliberately NOT retried. A repeated POST /order can
        open the position twice; a request whose outcome is unknown raises
        `OrderStatusUnknown` instead, carrying the client order id needed to
        reconcile.

        Args:
            pair: Trading pair, e.g. 'BTC/USDT'
            order_type: Order type
            side: Buy or sell
            quantity: Order quantity in base currency
            price: Limit price. Required for everything but MARKET, and worth
                passing for MARKET too, as the reference used to check the
                minimum notional.
            **kwargs: Extra ccxt params. `stop_price` (or `stopPrice` /
                `triggerPrice`) supplies the trigger for stop and take profit
                orders. `clientOrderId` overrides the generated id.

        Returns:
            Created order

        Raises:
            MinimumNotionalError: Order is below the market's minimum
            OrderError: Order was refused, or cannot be expressed
            OrderStatusUnknown: Outcome undetermined, reconcile before retrying
            InsufficientBalanceError: Not enough funds
        """
        market = await self.get_market_info(pair)
        ccxt_side = 'buy' if side == OrderSide.BUY else 'sell'
        params: Dict[str, Any] = dict(kwargs)

        trigger_price = params.pop('stop_price', None)
        trigger_price = params.pop('stopPrice', trigger_price)
        trigger_price = params.pop('triggerPrice', trigger_price)

        # Defect D: the converted type is what gets sent, so a stop loss stays
        # a stop loss.
        ccxt_type = self._convert_order_type(order_type)

        if order_type == OrderType.MARKET:
            send_price = None
        else:
            if price is None:
                raise OrderError(
                    f"Price required for {order_type.value} orders on {pair}"
                )
            send_price = self._round_price(pair, price)

        amount = self._round_amount(pair, quantity)

        if order_type in (OrderType.STOP_LOSS, OrderType.TAKE_PROFIT):
            ccxt_type, params = self._prepare_trigger_order(
                pair, order_type, ccxt_type, trigger_price, send_price, params
            )
        elif trigger_price is not None:
            raise OrderError(
                f"A trigger price was supplied for a {order_type.value} order "
                f"on {pair}; use OrderType.STOP_LOSS or OrderType.TAKE_PROFIT"
            )

        reference_price = send_price if send_price is not None else price
        self._check_limits(pair, market, amount, reference_price)

        client_order_id = params.get('clientOrderId') or self._client_order_id(
            pair, order_type, side, amount, send_price
        )
        params['clientOrderId'] = client_order_id

        try:
            order_result = await self.exchange.create_order(
                pair,
                ccxt_type,
                ccxt_side,
                amount,
                send_price,
                params
            )

        except TRANSIENT_ERRORS as e:
            # Defect E: a timeout on POST /order is not a rejection. The order
            # may be live on the exchange right now.
            logger.error(
                "Order outcome unknown, reconcile before retrying",
                pair=pair,
                client_order_id=client_order_id,
                error=self.redact(str(e))
            )
            raise OrderStatusUnknown(
                f"Order request for {pair} did not return a verdict "
                f"({type(e).__name__}: {self.redact(str(e))}). The order may or "
                f"may not exist. Reconcile using clientOrderId "
                f"'{client_order_id}' via get_open_orders before retrying.",
                pair=pair,
                client_order_id=client_order_id
            )
        except ccxt.InsufficientFunds as e:
            logger.error("Insufficient funds", pair=pair, error=self.redact(str(e)))
            raise InsufficientBalanceError(self.redact(str(e)))
        except Exception as e:
            logger.error("Error creating order", pair=pair, error=self.redact(str(e)))
            raise OrderError(f"Failed to create order: {self.redact(str(e))}")

        logger.info(
            "Order created",
            order_id=order_result['id'],
            client_order_id=client_order_id,
            pair=pair,
            type=ccxt_type,
            side=side.value,
            quantity=amount,
            price=send_price
        )

        return self._parse_order(order_result)

    async def cancel_order(self, order_id: str, pair: str) -> bool:
        """Cancel an open order.

        Cancellation is safe to repeat, so transient failures are retried.
        """
        try:
            await self._retrying(self.exchange.cancel_order, order_id, pair)
            logger.info("Order canceled", order_id=order_id, pair=pair)
            return True

        except ccxt.OrderNotFound:
            # Already gone: filled, or cancelled by an earlier attempt.
            logger.info("Order already absent", order_id=order_id, pair=pair)
            return True
        except Exception as e:
            logger.error(
                "Error canceling order",
                order_id=order_id,
                error=self.redact(str(e))
            )
            raise OrderError(f"Failed to cancel order: {self.redact(str(e))}")

    async def get_order(self, order_id: str, pair: str) -> Order:
        """Get order information."""
        try:
            order_result = await self._retrying(
                self.exchange.fetch_order, order_id, pair
            )
            return self._parse_order(order_result)

        except Exception as e:
            logger.error(
                "Error fetching order",
                order_id=order_id,
                error=self.redact(str(e))
            )
            raise APIError(f"Failed to fetch order: {self.redact(str(e))}")

    async def get_open_orders(self, pair: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        try:
            orders = await self._retrying(self.exchange.fetch_open_orders, pair)
            return [self._parse_order(order) for order in orders]

        except Exception as e:
            logger.error("Error fetching open orders", error=self.redact(str(e)))
            raise APIError(f"Failed to fetch open orders: {self.redact(str(e))}")

    async def get_positions(self, pair: Optional[str] = None) -> List[Position]:
        """Get spot inventory expressed as positions.

        SPOT HAS NO POSITIONS. There is no `fetch_positions` to call: a
        "position" here is simply held base-asset inventory, bought outright.
        There is no leverage, no margin, no funding, no mark price and no
        liquidation price, and nothing can be short.

        Consequences for callers:

        * `size` is the total base balance and is always positive, so
          `is_short` is never True.
        * `entry_price` is 0.0. A balance carries no cost basis; the exchange
          does not report what was paid for it. Anything deriving a target
          price from `entry_price` will get a meaningless answer, and must
          source the entry from the order history instead.
        * `current_price` is 0.0 and `unrealized_pnl` is 0.0 for the same
          reason. Fetch a ticker if a mark is needed.
        * `metadata` carries the raw balance entry plus `spot: True`.

        Args:
            pair: Trading pair to report on. When None, every non-zero base
                asset that has a market against the quote currency is
                reported.

        Returns:
            List of positions, one per held base asset
        """
        try:
            balance = await self._retrying(self.exchange.fetch_balance)
            markets = await self._retrying(self.exchange.load_markets)

            symbols = [pair] if pair else self._symbols_for_balances(balance, markets)

            positions: List[Position] = []
            for symbol in symbols:
                market = markets.get(symbol)
                if market is None:
                    continue

                base, quote = self._base_quote(symbol, market)
                holding = balance.get(base) or {}
                total = float(holding.get('total') or 0.0)

                if total <= 0:
                    continue

                positions.append(
                    Position(
                        pair=symbol,
                        size=total,
                        entry_price=0.0,
                        current_price=0.0,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        timestamp=datetime.now(),
                        metadata={
                            'spot': True,
                            'base': base,
                            'quote': quote,
                            'free': float(holding.get('free') or 0.0),
                            'used': float(holding.get('used') or 0.0),
                            'total': total,
                            'entry_price_known': False,
                        }
                    )
                )

            return positions

        except Exception as e:
            logger.error("Error fetching spot inventory", error=self.redact(str(e)))
            raise APIError(f"Failed to fetch spot inventory: {self.redact(str(e))}")

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
                # Return default spot fees
                return {'maker': 0.001, 'taker': 0.001}

        except Exception as e:
            logger.warning(
                "Error fetching fees, using defaults",
                error=self.redact(str(e))
            )
            return {'maker': 0.001, 'taker': 0.001}

    async def get_market_info(self, pair: str) -> Dict[str, Any]:
        """Get market information for a pair."""
        try:
            markets = await self._retrying(self.exchange.load_markets)

            if pair not in markets:
                raise APIError(f"Market {pair} not found")

            market = markets[pair]
            base, quote = self._base_quote(pair, market)
            return {
                'symbol': market['symbol'],
                'base': base,
                'quote': quote,
                'active': market.get('active', True),
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

        except APIError:
            raise
        except Exception as e:
            logger.error(
                "Error fetching market info",
                pair=pair,
                error=self.redact(str(e))
            )
            raise APIError(f"Failed to fetch market info: {self.redact(str(e))}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _base_quote(symbol: str, market: Dict[str, Any]) -> Tuple[str, str]:
        """Return the base and quote currencies of a market.

        Falls back to splitting the symbol, so a market description that omits
        the fields is still usable.

        Args:
            symbol: Market symbol, e.g. 'BTC/USDT'
            market: Market description

        Returns:
            Tuple of base and quote currency codes
        """
        head, _, tail = symbol.partition('/')
        quote = tail.split(':')[0] if tail else ''
        return market.get('base') or head, market.get('quote') or quote

    def _round_amount(self, pair: str, quantity: float) -> float:
        """Round a quantity down to the market's lot size.

        Args:
            pair: Trading pair
            quantity: Requested quantity

        Returns:
            Quantity the exchange will accept
        """
        return float(self.exchange.amount_to_precision(pair, quantity))

    def _round_price(self, pair: str, price: float) -> float:
        """Round a price to the market's tick size.

        Args:
            pair: Trading pair
            price: Requested price

        Returns:
            Price the exchange will accept
        """
        return float(self.exchange.price_to_precision(pair, price))

    def _check_limits(
        self,
        pair: str,
        market: Dict[str, Any],
        amount: float,
        reference_price: Optional[float]
    ) -> None:
        """Reject an order that the market would reject.

        Args:
            pair: Trading pair
            market: Market info from `get_market_info`
            amount: Rounded quantity
            reference_price: Price used to value the order, if known

        Raises:
            MinimumNotionalError: Below the minimum amount or notional
        """
        limits = market['limits']

        min_amount = limits['amount']['min']
        if min_amount is not None and amount < float(min_amount):
            raise MinimumNotionalError(
                f"Quantity {amount} for {pair} is below the market minimum "
                f"amount of {min_amount}"
            )

        if amount <= 0:
            raise MinimumNotionalError(
                f"Quantity {amount} for {pair} rounds to zero at the market's "
                f"lot size"
            )

        min_cost = limits['cost']['min']
        if min_cost is None:
            return

        if reference_price is None:
            # A market order placed without a reference price cannot be valued
            # here. Pass `price` to have the notional checked.
            logger.warning(
                "Minimum notional not checked, no reference price",
                pair=pair,
                quantity=amount,
                min_notional=min_cost
            )
            return

        notional = amount * reference_price
        if notional < float(min_cost):
            raise MinimumNotionalError(
                f"Order value {notional} {market['quote']} for {pair} "
                f"({amount} at {reference_price}) is below the market minimum "
                f"notional of {min_cost} {market['quote']}"
            )

    def _prepare_trigger_order(
        self,
        pair: str,
        order_type: OrderType,
        ccxt_type: str,
        trigger_price: Optional[float],
        send_price: Optional[float],
        params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Build the type and params for a stop loss or take profit order.

        Args:
            pair: Trading pair
            order_type: STOP_LOSS or TAKE_PROFIT
            ccxt_type: Converted base type
            trigger_price: Price at which the order activates
            send_price: Limit price, if any
            params: Params accumulated so far

        Returns:
            Tuple of the type to send and the updated params

        Raises:
            OrderError: No trigger price, or the exchange cannot express it
        """
        if trigger_price is None:
            raise OrderError(
                f"A trigger price is required for a {order_type.value} order "
                f"on {pair}; pass stop_price=..."
            )

        capabilities = getattr(self.exchange, 'has', None) or {}
        if capabilities.get('createStopOrder') is False and \
                capabilities.get('createTriggerOrder') is False:
            raise OrderError(
                f"This exchange cannot express a {order_type.value} order for "
                f"{pair}; refusing to downgrade it to a plain order"
            )

        # A limit price makes it the "_limit" variant; without one it triggers
        # into a market order.
        if send_price is not None:
            ccxt_type = f"{ccxt_type}_limit"

        params['triggerPrice'] = self._round_price(pair, trigger_price)
        return ccxt_type, params

    def _client_order_id(
        self,
        pair: str,
        order_type: OrderType,
        side: OrderSide,
        amount: float,
        price: Optional[float]
    ) -> str:
        """Build a deterministic client order id.

        The same order, described the same way within the same second, yields
        the same id. That makes an order recoverable after a timeout: the id
        can be recomputed and looked up rather than guessed at.

        Args:
            pair: Trading pair
            order_type: Order type
            side: Buy or sell
            amount: Rounded quantity
            price: Rounded price, if any

        Returns:
            Client order id, within Binance's 36 character limit
        """
        payload = "|".join([
            pair,
            order_type.value,
            side.value,
            f"{amount}",
            f"{price}",
            f"{int(time.time())}",
        ])
        digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]
        return f"{CLIENT_ORDER_ID_PREFIX}-{digest}"

    def _symbols_for_balances(
        self,
        balance: Dict[str, Any],
        markets: Dict[str, Any]
    ) -> List[str]:
        """Find the markets matching every non-zero base-asset holding.

        Args:
            balance: ccxt balance structure
            markets: Loaded markets

        Returns:
            List of market symbols
        """
        totals = balance.get('total') or {}
        held = {
            asset for asset, value in totals.items()
            if value and float(value) > 0
        }

        symbols = []
        for symbol, market in markets.items():
            base, quote = self._base_quote(symbol, market)
            if base in held and quote in totals:
                symbols.append(symbol)
        return symbols

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
