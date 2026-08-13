"""Trading Bot Execution Engine - Implementation of Algorithm 1 from paper.

Implements the deployment stage with minute-level trading dynamics, against a
SPOT venue. Three consequences of that run through everything below.

* Nothing can be short. A sell is only ever a disposal of base currency already
  held, so every sell is gated on free inventory and is never larger than what
  is held. There is no leverage and no liquidation.
* The exchange knows the QUANTITY held and nothing about what was paid for it.
  `get_positions()` reports an entry price of 0.0. Cost basis therefore comes
  from `PositionManager`, which accumulates it from fills, and the exchange is
  used only to reconcile the quantity. A holding with no known basis is never
  profit-taken: with an entry of 0.0 every profit target evaluates to 0.0 and
  every price on earth clears it, which would close the whole book on the first
  monitoring tick.
* The grid is diffed, not rebuilt. Levels that are still wanted are left
  resting, levels that are not are cancelled, and only missing levels are
  placed. Re-placing the whole ladder every minute without cancelling anything
  produced hundreds of live orders an hour against a hundred dollar allocation.
"""

import asyncio
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..exchange.base import (
    BaseExchange,
    MinimumNotionalError,
    Order,
    OrderStatus,
    OrderStatusUnknown,
    OrderType,
    OrderSide,
)
from ..data.market_data import MarketDataManager
from ..utils.config import Config
from ..utils.logging import TradingLogger
from .order_manager import OrderManager
from .position_manager import PositionManager


#: Prices are compared at this resolution when deciding whether a resting order
#: already covers a wanted grid level.
GRID_PRICE_DP = 8

#: Quantities below this are treated as nothing.
DUST_SIZE = 1e-12

#: Identifies a grid level: side and price.
GridKey = Tuple[str, float]


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

    # Stop loss per holding, as a percentage of the average entry price.
    stop_loss_pct: float = 5.0

    # Circuit breaker. Consecutive failed cycles or order placements before
    # the bot stops rather than looping on the same error.
    max_consecutive_failures: int = 5

    # Seconds to back off after a failure, multiplied by the failure count.
    failure_backoff_seconds: float = 5.0

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
        paper_mode: bool = True,
        position_manager: Optional[PositionManager] = None,
        risk_manager: Optional[Any] = None,
        order_manager: Optional[OrderManager] = None
    ):
        """Initialize trading bot.

        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            exchange: Exchange connector
            market_data: Market data manager
            config: Bot configuration parameters
            paper_mode: If True, orders are simulated rather than transmitted
            position_manager: Shared position book, the source of truth for
                cost basis. One is created if not supplied
            risk_manager: Risk gate. Every order is checked through it. A bot
                without one logs that it is running unchecked
            order_manager: Shared order book. One is created if not supplied
        """
        self.pair = pair
        self.exchange = exchange
        self.market_data = market_data
        self.config = config
        self.paper_mode = paper_mode

        self.logger = TradingLogger(f"bot.{pair}")
        self.position_manager = position_manager or PositionManager()
        self.order_manager = order_manager or OrderManager()
        self.risk_manager = risk_manager

        if self.risk_manager is None:
            self.logger.logger.warning(
                "no_risk_manager",
                pair=pair,
                message="Bot is running without a risk gate on its orders"
            )

        #: Resting orders this bot placed, keyed by order id.
        self.active_orders: Dict[str, Order] = {}
        #: Grid level to order id, so the grid can be diffed.
        self._grid: Dict[GridKey, str] = {}
        #: Simulated resting orders in paper mode, keyed by order id.
        self._paper_orders: Dict[str, Order] = {}
        self._paper_sequence = 0

        #: Free base balance last reported by the exchange, when known.
        self._free_base: Optional[float] = None

        self.running = False
        self._stop_event = asyncio.Event()
        self.consecutive_failures = 0
        self.halted_reason: Optional[str] = None

        # Statistics
        self.stats = {
            'trades_executed': 0,
            'total_pnl': 0.0,
            'win_count': 0,
            'loss_count': 0,
            'orders_placed': 0,
            'orders_cancelled': 0,
            'orders_rejected': 0,
            'fills': 0,
            'cycles': 0
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the trading bot.

        The wait between cycles is interruptible: `stop()` sets an event rather
        than leaving the task asleep for a whole interval.
        """
        self.running = True
        self._stop_event.clear()
        self.logger.logger.info("Bot started", pair=self.pair, paper_mode=self.paper_mode)

        try:
            while self.running:
                delay = self.config.execution_interval * 60

                try:
                    await self._execute_cycle()
                    self._record_success()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.log_error(e, {"pair": self.pair, "action": "cycle"})
                    delay = self._record_failure("cycle")

                if not self.running:
                    break

                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    continue
                else:
                    break
        finally:
            self.running = False
            await self._cancel_all_orders()

    async def stop(self) -> None:
        """Stop the trading bot and leave nothing resting behind.

        Cancels every order this bot placed and wakes the cycle task, rather
        than setting a flag and waiting out the remaining sleep. Open holdings
        are deliberately NOT liquidated: dumping inventory at market on
        shutdown is its own risk, and the operator is told what is still held.
        """
        self.running = False
        self._stop_event.set()

        await self._cancel_all_orders()

        held = self.position_manager.get_position(self.pair)
        self.logger.logger.info(
            "Bot stopped",
            pair=self.pair,
            stats=self.stats,
            open_size=held.size if held else 0.0,
            halted_reason=self.halted_reason
        )

    def halt(self, reason: str) -> None:
        """Stop trading because something is wrong, not because we are done.

        Args:
            reason: Why the bot is halting
        """
        self.halted_reason = reason
        self.running = False
        self._stop_event.set()
        self.logger.logger.critical(
            "bot_halted",
            pair=self.pair,
            reason=reason,
            consecutive_failures=self.consecutive_failures
        )
        if self.risk_manager is not None:
            self.risk_manager.halt(f"{self.pair}: {reason}")

    def _record_success(self) -> None:
        """Reset the circuit breaker after a clean cycle."""
        self.consecutive_failures = 0

    def _record_failure(self, action: str) -> float:
        """Count a failure, back off, and halt once they stack up.

        Args:
            action: What failed, for the log

        Returns:
            Seconds to wait before trying again
        """
        self.consecutive_failures += 1
        self.logger.logger.warning(
            "operation_failed",
            pair=self.pair,
            action=action,
            consecutive_failures=self.consecutive_failures,
            limit=self.config.max_consecutive_failures
        )

        if self.consecutive_failures >= self.config.max_consecutive_failures:
            self.halt(
                f"{self.consecutive_failures} consecutive failures at {action}"
            )
            return 0.0

        return self.config.failure_backoff_seconds * self.consecutive_failures

    # ------------------------------------------------------------------
    # Trading cycle
    # ------------------------------------------------------------------

    async def _execute_cycle(self) -> None:
        """Execute one trading cycle (Algorithm 1, lines 5-16)."""
        if self.risk_manager is not None and self.risk_manager.emergency_stop:
            self.logger.logger.critical(
                "emergency_stop_active",
                pair=self.pair,
                message="Risk manager has halted trading; cancelling and stopping"
            )
            await self.stop()
            return

        self.stats['cycles'] += 1

        # Line 6-7: Retrieve market and select pairs
        market_stats = await self.market_data.get_market_stats(
            self.pair,
            self.config.lookback_period
        )

        current_price = market_stats.price
        volatility = market_stats.volatility

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
            # A pair that no longer qualifies must not keep a live ladder.
            await self._cancel_all_orders()
            return

        if not self._volatility_usable(volatility):
            await self._cancel_all_orders()
            return

        # Resolve what happened to resting orders FIRST. A buy that filled
        # since the last cycle is inventory this system paid a known price
        # for; reconciling before booking it would see it as surplus with no
        # cost basis and throw the basis away.
        if self.paper_mode:
            self._simulate_fills(current_price)
        else:
            await self._refresh_resting_orders()

        # Quantity truth from the exchange, cost basis truth from our own book.
        # Paper mode does not reconcile: the balance on the account belongs to
        # whatever else is trading it, and folding real inventory into a
        # simulated book would make the paper statistics a fiction about money
        # the simulation never spent.
        if not self.paper_mode:
            await self._reconcile_inventory()

        # Lines 8-12: bring the grid to the shape the current market implies.
        desired = self._calculate_orders(current_price, volatility)
        await self._reconcile_grid(desired, current_price, volatility)

        # Lines 13-16: Monitor holdings and close when conditions are met.
        await self._monitor_positions(current_price, volatility)

        if self.risk_manager is not None:
            self.risk_manager.update_capital()

    def _volatility_usable(self, volatility: float) -> bool:
        """Whether Φ can be used to build a grid.

        The data layer clamps at 0.99, but this check does not assume that. At
        Φ <= 0 every grid level collapses onto the spot price and the ladder is
        meaningless; at Φ >= 1 the buy side goes to zero or negative and
        `(1 - Φ) ** exponent` stops being a price at all.

        Args:
            volatility: Φ as a fraction of price

        Returns:
            True when a grid can be built from it
        """
        if not isinstance(volatility, (int, float)) or isinstance(volatility, bool):
            usable = False
        elif not math.isfinite(volatility):
            usable = False
        else:
            usable = 0 < volatility < 1

        if not usable:
            self.logger.logger.warning(
                "volatility_unusable",
                pair=self.pair,
                volatility=volatility,
                message=(
                    "Volatility must be strictly between 0 and 1 to build a "
                    "grid; skipping this cycle"
                )
            )
        return usable

    def _distributions_usable(self) -> bool:
        """Validate the price and quantity distributions against each other.

        `zip` silently truncates to the shorter of the two, so a price
        distribution with more levels than the quantity distribution quietly
        loses its deepest levels. Proportions that do not sum to one mean the
        allocation is not what the configuration says it is.

        Returns:
            True when the distributions can be used
        """
        prices = self.config.price_distribution or []
        quantities = self.config.quantity_distribution or []

        if not prices or not quantities:
            self.logger.logger.error(
                "distribution_empty",
                pair=self.pair,
                price_levels=len(prices),
                quantity_levels=len(quantities)
            )
            return False

        if len(prices) != len(quantities):
            self.logger.logger.error(
                "distribution_length_mismatch",
                pair=self.pair,
                price_levels=len(prices),
                quantity_levels=len(quantities),
                message=(
                    "Price and quantity distributions must have the same "
                    "number of levels; refusing to truncate silently"
                )
            )
            return False

        if any(not math.isfinite(p) for p in prices):
            self.logger.logger.error("price_distribution_not_finite", pair=self.pair)
            return False

        if any((not math.isfinite(q)) or q < 0 for q in quantities):
            self.logger.logger.error(
                "quantity_distribution_invalid", pair=self.pair
            )
            return False

        total = sum(quantities)
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-9):
            self.logger.logger.error(
                "quantity_distribution_sum",
                pair=self.pair,
                total=total,
                message="Quantity proportions must sum to 1.0"
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _calculate_orders(
        self,
        current_price: float,
        volatility: float
    ) -> List[Dict[str, Any]]:
        """Calculate order specifications (Algorithm 1, lines 9-12).

        Pi = Precent × (1 ± Φ)^MP[i]
        Qi = A × MQ[i] × cm × cf

        The sell side is bounded by inventory. On spot a sell that is not
        covered by held base currency simply fails at the exchange, so the sell
        ladder distributes whatever is actually sellable across the levels and
        places nothing when nothing is held.

        Args:
            current_price: Current market price
            volatility: Calculated volatility (Φ)

        Returns:
            List of order specifications
        """
        if not self._distributions_usable():
            return []

        if current_price <= 0 or not math.isfinite(current_price):
            self.logger.logger.error(
                "price_unusable", pair=self.pair, price=current_price
            )
            return []

        orders: List[Dict[str, Any]] = []
        sellable = self._sellable_inventory()

        levels = list(zip(
            self.config.price_distribution,
            self.config.quantity_distribution
        ))

        for i, (price_exp, qty_prop) in enumerate(levels):
            # Buy order: price below current
            buy_price = current_price * ((1 - volatility) ** price_exp)
            if buy_price > 0 and math.isfinite(buy_price):
                buy_quantity = (
                    self.config.capital_per_pair *
                    qty_prop *
                    self.config.market_cap_coefficient *
                    self.config.funding_rate_coefficient
                ) / buy_price  # Convert quote value to base quantity

                if buy_quantity > DUST_SIZE:
                    orders.append({
                        'side': OrderSide.BUY,
                        'price': buy_price,
                        'quantity': buy_quantity,
                        'level': i
                    })

            # Sell order: price above current, funded by held inventory only.
            sell_quantity = sellable * qty_prop
            if sell_quantity <= DUST_SIZE:
                continue

            sell_price = current_price * ((1 + volatility) ** price_exp)
            if sell_price <= 0 or not math.isfinite(sell_price):
                continue

            orders.append({
                'side': OrderSide.SELL,
                'price': sell_price,
                'quantity': sell_quantity,
                'level': i
            })

        return orders

    def _sellable_inventory(self) -> float:
        """Base currency that may be offered for sale right now.

        The floor of what the position book says is held and what the exchange
        says is free, less anything already committed to resting sell orders.
        Never negative, so the sell ladder cannot be built on a phantom.

        Returns:
            Quantity available to sell
        """
        position = self.position_manager.get_position(self.pair)
        held = position.size if position else 0.0

        if self._free_base is not None:
            held = min(held, self._free_base)

        committed = sum(
            order.remaining_quantity
            for order in self.active_orders.values()
            if order.side == OrderSide.SELL
        )

        return max(held - committed, 0.0)

    def _grid_price_tolerance(self, volatility: float) -> float:
        """Deviation a resting grid level is allowed from the market price.

        The configured deviation limit describes an order meant to execute now.
        A grid level is meant to rest away from the market, and the deepest one
        sits at `(1 ± Φ) ** max(MP)`. The bound returned here is that distance
        plus the configured slack, so the check still rejects a price built
        from a corrupted Φ or a stale reference without rejecting the strategy.

        Args:
            volatility: Φ as a fraction of price

        Returns:
            Allowed deviation as a fraction
        """
        exponents = self.config.price_distribution or [1.0]
        deepest = max(exponents)
        below = 1 - ((1 - volatility) ** deepest)
        above = ((1 + volatility) ** deepest) - 1

        slack = 0.0
        if self.risk_manager is not None:
            slack = getattr(self.risk_manager, 'max_price_deviation', 0.0)

        return max(below, above) + slack

    @staticmethod
    def _grid_key(side: OrderSide, price: float) -> GridKey:
        """Identify a grid level by side and price.

        Args:
            side: Order side
            price: Level price

        Returns:
            Key used to diff wanted levels against resting orders
        """
        return (side.value, round(float(price), GRID_PRICE_DP))

    async def _reconcile_grid(
        self,
        desired: List[Dict[str, Any]],
        current_price: float,
        volatility: float
    ) -> None:
        """Cancel levels that are no longer wanted, place the ones that are.

        Args:
            desired: Order specifications for this cycle
            current_price: Current market price
            volatility: Φ, used to bound the price sanity check
        """
        wanted: Dict[GridKey, Dict[str, Any]] = {}
        for spec in desired:
            wanted[self._grid_key(spec['side'], spec['price'])] = spec

        # Cancel first, so inventory and capital committed to stale levels is
        # released before anything new is asked for.
        for key, order_id in list(self._grid.items()):
            if key not in wanted:
                await self._cancel_order(order_id)

        tolerance = self._grid_price_tolerance(volatility)

        for key, spec in wanted.items():
            if key in self._grid:
                continue
            await self._place_order(spec, current_price, tolerance)

    # ------------------------------------------------------------------
    # Order placement and cancellation
    # ------------------------------------------------------------------

    async def _place_order(
        self,
        order_spec: Dict[str, Any],
        market_price: float,
        price_tolerance: Optional[float] = None
    ) -> Optional[Order]:
        """Place a limit order, after the risk gate has approved it.

        Args:
            order_spec: Order specification
            market_price: Current market price, the reference for risk checks
            price_tolerance: Deviation allowance for this order

        Returns:
            The resting order, or None when nothing was placed
        """
        side: OrderSide = order_spec['side']
        price = float(order_spec['price'])
        quantity = float(order_spec['quantity'])
        key = self._grid_key(side, price)

        if quantity <= DUST_SIZE or price <= 0:
            return None

        # A sell can never exceed what is held, whatever the caller asked for.
        if side == OrderSide.SELL:
            available = self._sellable_inventory()
            if quantity > available:
                self.logger.logger.warning(
                    "sell_clamped_to_inventory",
                    pair=self.pair,
                    requested=quantity,
                    available=available
                )
                quantity = available
            if quantity <= DUST_SIZE:
                return None

        order_value = quantity * price
        reserve_key = f"{self.pair}:{key[0]}:{key[1]}"

        if self.risk_manager is not None:
            approved = self.risk_manager.check_order_risk(
                pair=self.pair,
                order_value=order_value,
                side=side.value,
                order_price=price,
                market_price=market_price,
                price_tolerance=price_tolerance,
                reserve_key=reserve_key if side == OrderSide.BUY else None
            )
            if not approved:
                self.stats['orders_rejected'] += 1
                return None

        if self.paper_mode:
            return self._place_paper_order(side, price, quantity, key)

        try:
            order = await self.exchange.create_order(
                pair=self.pair,
                order_type=OrderType.LIMIT,
                side=side,
                quantity=quantity,
                price=price
            )

        except OrderStatusUnknown as e:
            # The request never got a verdict, so the order may be live right
            # now. Treating it as a failure and re-placing would double it.
            order = await self._reconcile_unknown_order(e)
            if order is None:
                self._release_reserved(reserve_key)
                self._record_failure("place_order")
                return None

        except MinimumNotionalError as e:
            # Deterministic and benign: this level is simply too small for the
            # market. It is not a fault worth tripping the circuit breaker.
            self._release_reserved(reserve_key)
            self.logger.logger.warning(
                "order_below_minimum",
                pair=self.pair,
                side=side.value,
                price=price,
                quantity=quantity,
                error=str(e)
            )
            self.stats['orders_rejected'] += 1
            return None

        except Exception as e:
            self._release_reserved(reserve_key)
            self.logger.log_error(e, {
                "action": "place_order",
                "pair": self.pair,
                "side": side.value,
                "price": price,
                "quantity": quantity
            })
            self.stats['orders_rejected'] += 1
            self._record_failure("place_order")
            return None

        self._track_order(order, key)

        self.logger.log_trade(
            action=side.value,
            pair=self.pair,
            price=price,
            quantity=quantity,
            order_id=order.id
        )
        return order

    def _place_paper_order(
        self,
        side: OrderSide,
        price: float,
        quantity: float,
        key: GridKey
    ) -> Order:
        """Record a simulated resting order.

        Paper mode is a simulation, not a dry run that does nothing: the order
        rests, and `_simulate_fills` fills it when the price trades through it.
        That is what makes paper statistics non-zero, and it is why the README
        gate of validating in paper mode means something.

        Args:
            side: Order side
            price: Limit price
            quantity: Order quantity
            key: Grid level key

        Returns:
            The simulated order
        """
        self._paper_sequence += 1
        order = Order(
            id=f"paper-{self.pair}-{self._paper_sequence}",
            pair=self.pair,
            type=OrderType.LIMIT,
            side=side,
            price=price,
            quantity=quantity,
            status=OrderStatus.OPEN,
            timestamp=datetime.now(),
            metadata={'simulated': True}
        )

        self._paper_orders[order.id] = order
        self._track_order(order, key)

        self.logger.log_trade(
            action=f"PAPER_{side.value}",
            pair=self.pair,
            price=price,
            quantity=quantity,
            order_id=order.id
        )
        return order

    def _track_order(self, order: Order, key: GridKey) -> None:
        """Start tracking a resting order against its grid level.

        Args:
            order: Resting order
            key: Grid level key
        """
        self.active_orders[order.id] = order
        self._grid[key] = order.id
        self.order_manager.add_order(order)
        self.stats['orders_placed'] += 1

    def _forget_order(self, order_id: str) -> Optional[GridKey]:
        """Stop tracking an order and free its grid level.

        Args:
            order_id: Order ID

        Returns:
            The grid level the order occupied, if any
        """
        self.active_orders.pop(order_id, None)
        self._paper_orders.pop(order_id, None)

        for key, tracked_id in list(self._grid.items()):
            if tracked_id == order_id:
                del self._grid[key]
                self._release_reserved(f"{self.pair}:{key[0]}:{key[1]}")
                return key
        return None

    def _release_reserved(self, reserve_key: str) -> None:
        """Release reserved exposure, if a risk manager is holding any.

        Args:
            reserve_key: Reservation key
        """
        if self.risk_manager is not None:
            self.risk_manager.release_exposure(reserve_key)

    async def _cancel_order(self, order_id: str) -> bool:
        """Cancel one resting order.

        Args:
            order_id: Order ID

        Returns:
            True when the order is no longer resting
        """
        order = self.active_orders.get(order_id)

        if order is not None and order.metadata and order.metadata.get('simulated'):
            self._forget_order(order_id)
            self.order_manager.remove_order(order_id, OrderStatus.CANCELED)
            self.stats['orders_cancelled'] += 1
            return True

        try:
            await self.exchange.cancel_order(order_id, self.pair)
        except Exception as e:
            self.logger.log_error(e, {
                "action": "cancel_order",
                "pair": self.pair,
                "order_id": order_id
            })
            self._record_failure("cancel_order")
            return False

        self._forget_order(order_id)
        self.order_manager.remove_order(order_id, OrderStatus.CANCELED)
        self.stats['orders_cancelled'] += 1
        return True

    async def _cancel_all_orders(self) -> None:
        """Cancel every order this bot has resting."""
        for order_id in list(self.active_orders.keys()):
            await self._cancel_order(order_id)

        # Anything left is an order we failed to cancel. It stays tracked so a
        # later cycle or a human can deal with it, and it is named here.
        if self.active_orders:
            self.logger.logger.error(
                "orders_still_resting",
                pair=self.pair,
                order_ids=list(self.active_orders.keys()),
                message="Failed to cancel these orders; they may still be live"
            )

    async def _reconcile_unknown_order(
        self,
        error: OrderStatusUnknown
    ) -> Optional[Order]:
        """Find out whether an order of unknown outcome actually exists.

        Args:
            error: The raised unknown-status error, carrying the client order id

        Returns:
            The order if it is resting on the exchange, otherwise None
        """
        client_order_id = error.client_order_id
        self.logger.logger.warning(
            "order_outcome_unknown",
            pair=self.pair,
            client_order_id=client_order_id,
            message="Reconciling against open orders before doing anything else"
        )

        if not client_order_id:
            return None

        try:
            open_orders = await self.exchange.get_open_orders(self.pair)
        except Exception as e:
            self.logger.log_error(e, {
                "action": "reconcile_unknown_order",
                "pair": self.pair,
                "client_order_id": client_order_id
            })
            return None

        for order in open_orders:
            if OrderManager.client_order_id_of(order) == client_order_id:
                self.logger.logger.info(
                    "unknown_order_found_resting",
                    pair=self.pair,
                    order_id=order.id,
                    client_order_id=client_order_id
                )
                return order

        self.logger.logger.info(
            "unknown_order_not_found",
            pair=self.pair,
            client_order_id=client_order_id,
            message="No resting order carries this id; treating it as not placed"
        )
        return None

    async def _refresh_resting_orders(self) -> None:
        """Find out what happened to the orders this bot has resting.

        An order that has left the open list is fetched individually rather
        than assumed filled, because a cancellation from elsewhere looks
        identical from the open list alone.
        """
        if not self.active_orders:
            return

        try:
            open_orders = await self.exchange.get_open_orders(self.pair)
        except Exception as e:
            self.logger.log_error(e, {
                "action": "refresh_orders",
                "pair": self.pair
            })
            self._record_failure("refresh_orders")
            return

        still_open = {order.id for order in open_orders}

        for order_id in list(self.active_orders.keys()):
            if order_id in still_open:
                continue

            try:
                resolved = await self.exchange.get_order(order_id, self.pair)
            except Exception as e:
                self.logger.log_error(e, {
                    "action": "resolve_order",
                    "pair": self.pair,
                    "order_id": order_id
                })
                continue

            self._resolve_order(resolved)

    def _resolve_order(self, order: Order) -> None:
        """Apply a terminal order state to the books.

        Args:
            order: Order as the exchange now reports it
        """
        tracked = self.active_orders.get(order.id)
        if tracked is None:
            return

        if order.status == OrderStatus.FILLED:
            filled = order.filled_quantity or order.quantity
            price = order.price or tracked.price
            self._record_fill(order.side, filled, price, order.id)
            self._forget_order(order.id)
            self.order_manager.update_order(order)
            return

        if order.status in (OrderStatus.CANCELED, OrderStatus.REJECTED):
            self._forget_order(order.id)
            self.order_manager.update_order(order)
            return

        if order.status == OrderStatus.PARTIALLY_FILLED:
            newly_filled = order.filled_quantity - tracked.filled_quantity
            if newly_filled > DUST_SIZE:
                self._record_fill(
                    order.side, newly_filled, order.price or tracked.price, order.id
                )
            tracked.filled_quantity = order.filled_quantity

    def _record_fill(
        self,
        side: OrderSide,
        quantity: float,
        price: float,
        order_id: str
    ) -> None:
        """Feed a fill into the position book and the statistics.

        This is where cost basis comes from. A buy adds inventory at the price
        paid and moves the average entry; a sell reduces inventory and realises
        P&L against that average.

        Args:
            side: Order side
            quantity: Quantity filled
            price: Fill price
            order_id: Order the fill belongs to
        """
        if quantity <= DUST_SIZE or price <= 0:
            return

        self.stats['fills'] += 1
        self.stats['trades_executed'] += 1

        if self.risk_manager is not None:
            self.risk_manager.record_trade()

        if side == OrderSide.BUY:
            self.position_manager.add_position(
                pair=self.pair,
                size=quantity,
                entry_price=price
            )
            self.logger.log_trade(
                action="FILL_BUY",
                pair=self.pair,
                price=price,
                quantity=quantity,
                order_id=order_id
            )
            return

        before = self.position_manager.get_position(self.pair)
        realised_before = before.realized_pnl if before else 0.0

        after = self.position_manager.close_position(
            pair=self.pair,
            exit_price=price,
            size=quantity
        )

        realised = 0.0
        if after is not None:
            realised = after.realized_pnl - realised_before

        self.stats['total_pnl'] += realised
        if realised > 0:
            self.stats['win_count'] += 1
        elif realised < 0:
            self.stats['loss_count'] += 1

        if self.risk_manager is not None:
            self.risk_manager.check_trade_loss(self.pair, realised)

        self.logger.log_trade(
            action="FILL_SELL",
            pair=self.pair,
            price=price,
            quantity=quantity,
            order_id=order_id,
            pnl=realised
        )

    def _simulate_fills(self, current_price: float) -> None:
        """Fill simulated orders the market has traded through.

        SIMULATED, not real. The model is deliberately simple and optimistic:
        a resting buy fills in full at its limit price as soon as the observed
        price is at or below it, and a resting sell likewise at or above. There
        is no queue position, no partial fill, no fee and no slippage, so paper
        results are an upper bound on what the same orders would have achieved.

        Args:
            current_price: Current market price
        """
        for order_id, order in list(self._paper_orders.items()):
            if order.side == OrderSide.BUY:
                filled = current_price <= order.price
            else:
                filled = current_price >= order.price

            if not filled:
                continue

            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_timestamp = datetime.now()

            self._record_fill(order.side, order.quantity, order.price, order.id)
            self._forget_order(order_id)
            self.order_manager.update_order(order)

    # ------------------------------------------------------------------
    # Inventory and holdings
    # ------------------------------------------------------------------

    async def _reconcile_inventory(self) -> None:
        """Align the position book's quantity with the exchange's.

        The exchange is authoritative for how much is held and knows nothing
        about what it cost. Only the quantity is taken from it.
        """
        try:
            positions = await self.exchange.get_positions(self.pair)
        except Exception as e:
            self.logger.log_error(e, {
                "action": "reconcile_inventory",
                "pair": self.pair
            })
            return

        held = 0.0
        free: Optional[float] = None

        for position in positions:
            if position.pair != self.pair:
                continue
            held += abs(position.size)
            metadata = position.metadata or {}
            if 'free' in metadata:
                free = float(metadata['free'])

        self._free_base = free

        tracked = self.position_manager.get_position(self.pair)
        tracked_size = tracked.size if tracked else 0.0

        if abs(held - tracked_size) <= DUST_SIZE:
            return

        self.position_manager.reconcile_size(
            pair=self.pair,
            exchange_size=held,
            metadata={'spot': True, 'source': 'exchange_balance'}
        )

    async def _monitor_positions(
        self,
        current_price: float,
        volatility: float
    ) -> None:
        """Monitor holdings and close when conditions are met.

        Each holding is acted on at most once per pass. The checks are ordered
        by severity and the loop moves to the next holding as soon as one of
        them acts, so a holding cannot be closed by the profit ladder and then
        closed again by the small-position rule against a stale object.

        Order of decisions:

        1. No cost basis, no action. Reported, not traded on.
        2. Stop loss, at `entry × (1 - stop_loss_pct)`.
        3. Profit ladder. The HIGHEST satisfied threshold wins. The thresholds
           ascend and `(1 + Φ) ** 1.5 < (1 + Φ) ** 2.0 < ...`, so taking the
           first satisfied one made every threshold above the smallest dead.
        4. Small profitable holding, worth less than A/λ.

        Args:
            current_price: Current price
            volatility: Volatility
        """
        holdings = [
            position for position in self.position_manager.get_all_positions()
            if position.pair == self.pair
        ]

        for position in holdings:
            self.position_manager.update_position_price(self.pair, current_price)

            if not position.entry_price_known:
                self.logger.logger.info(
                    "cost_basis_unknown",
                    pair=self.pair,
                    size=position.size,
                    message=(
                        "Holding has no recorded cost basis; skipping profit "
                        "taking and stop loss rather than treating it as free"
                    )
                )
                continue

            entry_price = position.entry_price

            # 2. Stop loss.
            stop_price = entry_price * (1 - self.config.stop_loss_pct / 100)
            if current_price <= stop_price:
                await self._close_position(
                    position, current_price, reason='stop_loss'
                )
                continue

            # 3. Profit ladder, highest satisfied threshold first.
            hit = None
            for threshold in sorted(self.config.profit_loss_thresholds, reverse=True):
                target_price = entry_price * ((1 + volatility) ** threshold)
                if current_price >= target_price:
                    hit = threshold
                    break

            if hit is not None:
                await self._close_position(
                    position, current_price, reason=f'take_profit_{hit}'
                )
                continue

            # 4. Small profitable holding.
            divisor = self.config.position_size_divisor
            if not divisor or divisor <= 0:
                self.logger.logger.warning(
                    "position_size_divisor_invalid",
                    pair=self.pair,
                    divisor=divisor,
                    message="Skipping the small position rule"
                )
                continue

            minimum_value = self.config.capital_per_pair / divisor
            if position.notional < minimum_value and position.unrealized_pnl > 0:
                await self._close_position(
                    position, current_price, reason='small_position'
                )
                continue

    async def _close_position(
        self,
        position: Any,
        current_price: float,
        reason: str,
        quantity: Optional[float] = None
    ) -> bool:
        """Sell held inventory at market.

        Args:
            position: Holding to close
            current_price: Current market price
            reason: Why the holding is being closed, for the log
            quantity: Quantity to sell. None sells the whole holding

        Returns:
            True when the sale was recorded
        """
        requested = position.size if quantity is None else abs(quantity)
        sellable = position.size
        if self._free_base is not None:
            sellable = min(sellable, self._free_base)

        sell_quantity = min(requested, sellable)

        if sell_quantity <= DUST_SIZE:
            self.logger.logger.warning(
                "close_skipped_no_free_inventory",
                pair=self.pair,
                reason=reason,
                held=position.size,
                free=self._free_base
            )
            return False

        if self.paper_mode:
            # Simulated market sale at the observed price, no slippage.
            self._record_fill(
                OrderSide.SELL, sell_quantity, current_price, f"paper-close-{reason}"
            )
            self.logger.logger.info(
                "position_closed",
                pair=self.pair,
                reason=reason,
                quantity=sell_quantity,
                price=current_price,
                simulated=True
            )
            return True

        if self.risk_manager is not None:
            approved = self.risk_manager.check_order_risk(
                pair=self.pair,
                order_value=sell_quantity * current_price,
                side=OrderSide.SELL.value,
                order_price=current_price,
                market_price=current_price
            )
            if not approved:
                self.logger.logger.error(
                    "close_refused_by_risk",
                    pair=self.pair,
                    reason=reason,
                    message="Risk gate refused the closing order"
                )
                return False

        try:
            order = await self.exchange.create_order(
                pair=self.pair,
                order_type=OrderType.MARKET,
                side=OrderSide.SELL,
                quantity=sell_quantity,
                price=current_price
            )
        except OrderStatusUnknown as e:
            # The sale may have gone through. Selling again could dispose of
            # inventory twice, so the books are left alone and the next
            # reconciliation will discover the truth.
            self.logger.logger.error(
                "close_outcome_unknown",
                pair=self.pair,
                reason=reason,
                client_order_id=e.client_order_id,
                message="Not retrying; the next cycle reconciles the balance"
            )
            self._record_failure("close_position")
            return False
        except Exception as e:
            self.logger.log_error(e, {
                "action": "close_position",
                "pair": self.pair,
                "reason": reason
            })
            self._record_failure("close_position")
            return False

        fill_price = order.price or current_price
        filled = order.filled_quantity or sell_quantity
        self._record_fill(OrderSide.SELL, filled, fill_price, order.id)

        self.logger.logger.info(
            "position_closed",
            pair=self.pair,
            reason=reason,
            quantity=filled,
            price=fill_price,
            order_id=order.id
        )
        return True

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get bot statistics.

        Returns:
            Statistics dictionary
        """
        win_rate = 0
        decided = self.stats['win_count'] + self.stats['loss_count']
        if decided > 0:
            win_rate = self.stats['win_count'] / decided

        return {
            **self.stats,
            'win_rate': win_rate,
            'pair': self.pair,
            'resting_orders': len(self.active_orders),
            'halted_reason': self.halted_reason
        }


class BotEngine:
    """Engine for managing multiple trading bots."""

    def __init__(
        self,
        exchange: BaseExchange,
        market_data: MarketDataManager,
        config: Config,
        position_manager: Optional[PositionManager] = None,
        risk_manager: Optional[Any] = None,
        order_manager: Optional[OrderManager] = None
    ):
        """Initialize bot engine.

        Args:
            exchange: Exchange connector
            market_data: Market data manager
            config: System configuration
            position_manager: Shared position book. Bots hold cost basis here,
                and the risk manager reads exposure from the same object
            risk_manager: Shared risk gate, applied to every order
            order_manager: Shared order book
        """
        self.exchange = exchange
        self.market_data = market_data
        self.config = config
        self.logger = TradingLogger("bot_engine")

        self.position_manager = position_manager or PositionManager()
        self.order_manager = order_manager or OrderManager()
        self.risk_manager = risk_manager

        self.bots: Dict[str, TradingBot] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self.running = False

    def build_bot_config(self, **overrides: Any) -> BotConfig:
        """Build a bot configuration from the system configuration.

        Every scaling coefficient and the position size divisor are read from
        configuration here. They were previously left at their defaults because
        nothing passed them, which made two of the three multipliers
        permanently 1.0 whatever the file said.

        Args:
            **overrides: Values that beat the configured ones

        Returns:
            Bot configuration
        """
        divisor = self.config.get('risk.position_size_divisor', 10)
        try:
            divisor = int(divisor)
        except (TypeError, ValueError):
            divisor = 0

        if divisor <= 0:
            self.logger.logger.warning(
                "position_size_divisor_invalid",
                configured=self.config.get('risk.position_size_divisor', None),
                message="Falling back to 10; a divisor of zero cannot be used"
            )
            divisor = 10

        values: Dict[str, Any] = {
            'execution_interval': self.config.strategy.execution_interval,
            'lookback_period': self.config.strategy.lookback_period,
            'min_volume': self.config.strategy.min_volume,
            'min_volatility': self.config.strategy.min_volatility,
            'capital_per_pair': self.config.strategy.capital_per_pair,
            'price_distribution': self.config.strategy.price_distribution,
            'quantity_distribution': self.config.strategy.quantity_distribution,
            'profit_loss_thresholds': self.config.strategy.profit_loss_thresholds,
            'market_cap_coefficient': self.config.strategy.market_cap_coefficient,
            'funding_rate_coefficient': self.config.strategy.funding_rate_coefficient,
            'entry_coefficient': self.config.strategy.entry_coefficient,
            'position_size_divisor': divisor,
            'stop_loss_pct': self.config.risk.stop_loss_pct,
        }
        values.update({k: v for k, v in overrides.items() if v is not None})

        return BotConfig(**values)

    async def add_bot(
        self,
        pair: str,
        bot_config: Optional[BotConfig] = None
    ) -> Optional[TradingBot]:
        """Add a trading bot for a pair.

        Args:
            pair: Trading pair
            bot_config: Bot configuration (built from config if None)

        Returns:
            The bot that was added, or None when one already existed
        """
        if pair in self.bots:
            self.logger.logger.warning("Bot already exists", pair=pair)
            return None

        if bot_config is None:
            bot_config = self.build_bot_config()

        # Last chance to bound an allocation that arrived from outside.
        if self.risk_manager is not None:
            try:
                bot_config.capital_per_pair = self.risk_manager.validate_capital_allocation(
                    bot_config.capital_per_pair
                )
            except Exception as e:
                self.logger.logger.error(
                    "bot_rejected",
                    pair=pair,
                    reason=str(e),
                    message="Capital allocation could not be made safe"
                )
                return None

        bot = TradingBot(
            pair=pair,
            exchange=self.exchange,
            market_data=self.market_data,
            config=bot_config,
            paper_mode=self.config.is_paper_trading(),
            position_manager=self.position_manager,
            risk_manager=self.risk_manager,
            order_manager=self.order_manager
        )

        self.bots[pair] = bot
        self.logger.logger.info(
            "Bot added",
            pair=pair,
            capital_per_pair=bot_config.capital_per_pair
        )
        return bot

    async def start_all(self) -> None:
        """Start all trading bots."""
        self.running = True
        self.logger.logger.info("Starting all bots", count=len(self.bots))

        self._tasks = {
            pair: asyncio.ensure_future(bot.start())
            for pair, bot in self.bots.items()
        }

        try:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        finally:
            self.running = False

    async def stop_all(self) -> None:
        """Stop all trading bots and cancel anything they left resting."""
        self.running = False
        self.logger.logger.info("Stopping all bots")

        for bot in self.bots.values():
            try:
                await bot.stop()
            except Exception as e:
                self.logger.log_error(e, {"action": "stop_bot", "pair": bot.pair})

        for pair, task in list(self._tasks.items()):
            if task.done():
                continue
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()

        if self.risk_manager is not None:
            self.risk_manager.release_all_exposure()

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all bots.

        Returns:
            Dictionary mapping pair to statistics
        """
        return {
            pair: bot.get_stats()
            for pair, bot in self.bots.items()
        }
