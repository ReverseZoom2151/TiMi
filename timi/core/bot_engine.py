"""Trading Bot Execution Engine - Implementation of Algorithm 1 from paper.

Implements the deployment stage with minute-level trading dynamics.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..exchange.base import BaseExchange, Order, OrderType, OrderSide
from ..data.market_data import MarketDataManager
from ..utils.config import Config
from ..utils.logging import TradingLogger


@dataclass
class BotConfig:
    """Bot configuration parameters (Θ from paper)."""
    # T1: Execution intervals (minutes)
    execution_interval: int = 1

    # T2: Look-back period (minutes)
    lookback_period: int = 60

    # Vreq: Minimum volume requirement
    min_volume: float = 1_000_000

    # Φreq: Minimum volatility requirement
    min_volatility: float = 0.005

    # A: Capital allocation per pair
    capital_per_pair: float = 100

    # MP: Price distribution matrix (exponents for grid levels)
    price_distribution: List[float] = None

    # MQ: Quantity distribution matrix (proportions)
    quantity_distribution: List[float] = None

    # H: Profit/loss thresholds
    profit_loss_thresholds: List[float] = None

    # Scaling coefficients
    market_cap_coefficient: float = 1.0
    funding_rate_coefficient: float = 1.0
    entry_coefficient: float = 0.8

    # λ: Position size divisor
    position_size_divisor: int = 10

    def __post_init__(self):
        """Set defaults for list parameters."""
        if self.price_distribution is None:
            self.price_distribution = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
        if self.quantity_distribution is None:
            self.quantity_distribution = [0.2, 0.2, 0.2, 0.15, 0.15, 0.05, 0.05]
        if self.profit_loss_thresholds is None:
            self.profit_loss_thresholds = [1.5, 2.0, 3.0, 5.0]


class TradingBot:
    """Trading bot implementing Algorithm 1 from paper.

    Executes minute-level grid trading with dynamic parameter adjustment.
    """

    def __init__(
        self,
        pair: str,
        exchange: BaseExchange,
        market_data: MarketDataManager,
        config: BotConfig,
        paper_mode: bool = True
    ):
        """Initialize trading bot.

        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            exchange: Exchange connector
            market_data: Market data manager
            config: Bot configuration parameters
            paper_mode: If True, simulate orders without real execution
        """
        self.pair = pair
        self.exchange = exchange
        self.market_data = market_data
        self.config = config
        self.paper_mode = paper_mode

        self.logger = TradingLogger(f"bot.{pair}")
        self.active_orders: List[Order] = []
        self.positions: Dict[str, Any] = {}
        self.running = False

        # Statistics
        self.stats = {
            'trades_executed': 0,
            'total_pnl': 0.0,
            'win_count': 0,
            'loss_count': 0
        }

    async def start(self):
        """Start the trading bot."""
        self.running = True
        self.logger.logger.info("Bot started", pair=self.pair, paper_mode=self.paper_mode)

        while self.running:
            try:
                await self._execute_cycle()
                await asyncio.sleep(self.config.execution_interval * 60)
            except Exception as e:
                self.logger.log_error(e, {"pair": self.pair})
                await asyncio.sleep(60)  # Wait before retry

    async def stop(self):
        """Stop the trading bot."""
        self.running = False
        self.logger.logger.info("Bot stopped", pair=self.pair, stats=self.stats)

    async def _execute_cycle(self):
        """Execute one trading cycle (Algorithm 1, lines 5-16)."""
        # Line 6-7: Retrieve market and select pairs
        market_stats = await self.market_data.get_market_stats(
            self.pair,
            self.config.lookback_period
        )

        # Check if pair qualifies
        if not market_stats.meets_requirements(
            self.config.min_volume,
            self.config.min_volatility
        ):
            self.logger.logger.debug(
                "Pair does not meet requirements",
                pair=self.pair,
                volume=market_stats.volume_24h,
                volatility=market_stats.volatility
            )
            return

        # Lines 8-12: Place orders for each level
        current_price = market_stats.price
        volatility = market_stats.volatility

        orders_to_place = self._calculate_orders(current_price, volatility)

        for order_spec in orders_to_place:
            await self._place_order(order_spec)

        # Lines 13-16: Monitor positions and close when profitable
        await self._monitor_positions(current_price, volatility)

    def _calculate_orders(
        self,
        current_price: float,
        volatility: float
    ) -> List[Dict[str, Any]]:
        """Calculate order specifications (Algorithm 1, lines 9-12).

        Pi = Precent × (1 ± Φ)^MP[i]
        Qi = A × MQ[i] × cm × cf

        Args:
            current_price: Current market price
            volatility: Calculated volatility (Φ)

        Returns:
            List of order specifications
        """
        orders = []

        # Calculate buy orders (below current price)
        for i, (price_exp, qty_prop) in enumerate(
            zip(self.config.price_distribution, self.config.quantity_distribution)
        ):
            # Buy order: price below current
            buy_price = current_price * ((1 - volatility) ** price_exp)
            buy_quantity = (
                self.config.capital_per_pair *
                qty_prop *
                self.config.market_cap_coefficient *
                self.config.funding_rate_coefficient
            ) / buy_price  # Convert USD to quantity

            orders.append({
                'side': OrderSide.BUY,
                'price': buy_price,
                'quantity': buy_quantity,
                'level': i
            })

            # Sell order: price above current
            sell_price = current_price * ((1 + volatility) ** price_exp)
            # For sell orders, we need existing position
            # This is simplified - in production, check actual positions
            sell_quantity = buy_quantity

            orders.append({
                'side': OrderSide.SELL,
                'price': sell_price,
                'quantity': sell_quantity,
                'level': i
            })

        return orders

    async def _place_order(self, order_spec: Dict[str, Any]):
        """Place a limit order.

        Args:
            order_spec: Order specification
        """
        if self.paper_mode:
            # Simulate order in paper mode
            order = Order(
                id=f"paper_{datetime.now().timestamp()}",
                pair=self.pair,
                type=OrderType.LIMIT,
                side=order_spec['side'],
                price=order_spec['price'],
                quantity=order_spec['quantity']
            )
            self.logger.log_trade(
                action=f"PAPER_{order_spec['side'].value}",
                pair=self.pair,
                price=order_spec['price'],
                quantity=order_spec['quantity'],
                order_id=order.id
            )
        else:
            # Real order placement
            try:
                order = await self.exchange.create_order(
                    pair=self.pair,
                    order_type=OrderType.LIMIT,
                    side=order_spec['side'],
                    quantity=order_spec['quantity'],
                    price=order_spec['price']
                )

                self.logger.log_trade(
                    action=order_spec['side'].value,
                    pair=self.pair,
                    price=order_spec['price'],
                    quantity=order_spec['quantity'],
                    order_id=order.id
                )

                self.active_orders.append(order)
                self.stats['trades_executed'] += 1

            except Exception as e:
                self.logger.log_error(e, {
                    "action": "place_order",
                    "order_spec": order_spec
                })

    async def _monitor_positions(self, current_price: float, volatility: float):
        """Monitor positions and close when profit conditions met (Algorithm 1, lines 14-16).

        Close when: Pentry × (1 ± Φ)^H[i]
        Or when: Pentry × Q < A/λ (small profitable positions)

        Args:
            current_price: Current price
            volatility: Volatility
        """
        if self.paper_mode:
            return  # Skip position monitoring in paper mode for now

        try:
            positions = await self.exchange.get_positions(self.pair)

            for position in positions:
                entry_price = position.entry_price
                position_size = position.size
                current_pnl = position.unrealized_pnl

                # Check profit/loss thresholds
                for threshold in self.config.profit_loss_thresholds:
                    if position.is_long:
                        target_price = entry_price * ((1 + volatility) ** threshold)
                        if current_price >= target_price:
                            await self._close_position(position, current_price)
                            break
                    else:
                        target_price = entry_price * ((1 - volatility) ** threshold)
                        if current_price <= target_price:
                            await self._close_position(position, current_price)
                            break

                # Check for small profitable positions
                position_value = abs(position_size) * entry_price
                min_position_value = self.config.capital_per_pair / self.config.position_size_divisor

                if position_value < min_position_value and current_pnl > 0:
                    await self._close_position(position, current_price)

        except Exception as e:
            self.logger.log_error(e, {"action": "monitor_positions"})

    async def _close_position(self, position: Any, current_price: float):
        """Close a position.

        Args:
            position: Position to close
            current_price: Current market price
        """
        try:
            # Place market order to close
            side = OrderSide.SELL if position.is_long else OrderSide.BUY
            quantity = abs(position.size)

            order = await self.exchange.create_order(
                pair=self.pair,
                order_type=OrderType.MARKET,
                side=side,
                quantity=quantity
            )

            pnl = position.unrealized_pnl
            self.stats['total_pnl'] += pnl

            if pnl > 0:
                self.stats['win_count'] += 1
            else:
                self.stats['loss_count'] += 1

            self.logger.log_trade(
                action=f"CLOSE_{side.value}",
                pair=self.pair,
                price=current_price,
                quantity=quantity,
                order_id=order.id,
                pnl=pnl
            )

        except Exception as e:
            self.logger.log_error(e, {"action": "close_position"})

    def get_stats(self) -> Dict[str, Any]:
        """Get bot statistics.

        Returns:
            Statistics dictionary
        """
        win_rate = 0
        if self.stats['win_count'] + self.stats['loss_count'] > 0:
            win_rate = self.stats['win_count'] / (self.stats['win_count'] + self.stats['loss_count'])

        return {
            **self.stats,
            'win_rate': win_rate,
            'pair': self.pair
        }


class BotEngine:
    """Engine for managing multiple trading bots."""

    def __init__(
        self,
        exchange: BaseExchange,
        market_data: MarketDataManager,
        config: Config
    ):
        """Initialize bot engine.

        Args:
            exchange: Exchange connector
            market_data: Market data manager
            config: System configuration
        """
        self.exchange = exchange
        self.market_data = market_data
        self.config = config
        self.logger = TradingLogger("bot_engine")

        self.bots: Dict[str, TradingBot] = {}
        self.running = False

    async def add_bot(self, pair: str, bot_config: Optional[BotConfig] = None):
        """Add a trading bot for a pair.

        Args:
            pair: Trading pair
            bot_config: Bot configuration (uses default if None)
        """
        if pair in self.bots:
            self.logger.logger.warning("Bot already exists", pair=pair)
            return

        if bot_config is None:
            bot_config = BotConfig(
                execution_interval=self.config.strategy.execution_interval,
                lookback_period=self.config.strategy.lookback_period,
                min_volume=self.config.strategy.min_volume,
                min_volatility=self.config.strategy.min_volatility,
                capital_per_pair=self.config.strategy.capital_per_pair,
                price_distribution=self.config.strategy.price_distribution,
                quantity_distribution=self.config.strategy.quantity_distribution,
                profit_loss_thresholds=self.config.strategy.profit_loss_thresholds
            )

        bot = TradingBot(
            pair=pair,
            exchange=self.exchange,
            market_data=self.market_data,
            config=bot_config,
            paper_mode=self.config.is_paper_trading()
        )

        self.bots[pair] = bot
        self.logger.logger.info("Bot added", pair=pair)

    async def start_all(self):
        """Start all trading bots."""
        self.running = True
        self.logger.logger.info("Starting all bots", count=len(self.bots))

        # Start all bots concurrently
        tasks = [bot.start() for bot in self.bots.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self):
        """Stop all trading bots."""
        self.running = False
        self.logger.logger.info("Stopping all bots")

        for bot in self.bots.values():
            await bot.stop()

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all bots.

        Returns:
            Dictionary mapping pair to statistics
        """
        return {
            pair: bot.get_stats()
            for pair, bot in self.bots.items()
        }
