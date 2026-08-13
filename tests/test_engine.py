"""Tests for the trading engine.

The engine trades a spot venue, and the defects these tests pin down all came
from treating it as though it did not:

* Cost basis was read from `Position.entry_price`, which a spot connector
  reports as 0.0 because a balance has no cost. Every profit target then
  evaluated to 0.0 and every holding was closed on the first monitoring tick.
  `test_holding_without_cost_basis_is_not_closed_as_profitable` is the
  regression test for that.
* Sell orders were sized to match buy orders, which on spot means offering
  base currency that is not held.
* The whole ladder was re-placed every cycle and nothing was ever cancelled.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

from timi.core.bot_engine import BotConfig, TradingBot
from timi.core.position_manager import PositionManager
from timi.data.market_data import MarketStats
from timi.exchange.base import (
    MinimumNotionalError,
    Order,
    OrderSide,
    OrderStatus,
    OrderStatusUnknown,
    OrderType,
    Position,
)


# --------------------------------------------------------------------------
# Doubles. Neither touches the network.
# --------------------------------------------------------------------------


class StubMarketData:
    """Returns whatever market statistics a test asks for."""

    def __init__(
        self,
        price: float = 100.0,
        volatility: float = 0.02,
        volume: float = 10_000_000.0
    ):
        """Build the double.

        Args:
            price: Price reported to the engine
            volatility: Φ reported to the engine
            volume: 24h volume reported to the engine
        """
        self.price = price
        self.volatility = volatility
        self.volume = volume

    async def get_market_stats(
        self,
        pair: str,
        lookback_period: int = 60
    ) -> MarketStats:
        """Report the configured statistics.

        Args:
            pair: Trading pair
            lookback_period: Ignored

        Returns:
            Market statistics
        """
        return MarketStats(
            pair=pair,
            volume_24h=self.volume,
            volatility=self.volatility,
            price=self.price,
            timestamp=datetime.now()
        )


class StubExchange:
    """Records orders rather than transmitting them.

    Deliberately not a mock: the assertions are about exactly what would have
    been sent, so unexpected calls should be visible rather than absorbed.
    """

    def __init__(
        self,
        inventory: Optional[Dict[str, float]] = None,
        free: Optional[Dict[str, float]] = None
    ):
        """Build the double.

        Args:
            inventory: Base-asset quantities the account holds, by pair
            free: Unencumbered part of each holding. Defaults to all of it
        """
        self.created: List[Dict[str, Any]] = []
        self.cancelled: List[str] = []
        self.open_orders: Dict[str, Order] = {}
        self.all_orders: Dict[str, Order] = {}
        self.inventory = inventory or {}
        self.free = free or {}
        self.fail_next: Optional[Exception] = None
        self._sequence = 0

    async def create_order(
        self,
        pair: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        **kwargs: Any
    ) -> Order:
        """Record an order and return it as resting.

        Args:
            pair: Trading pair
            order_type: Order type
            side: Order side
            quantity: Order quantity
            price: Order price
            **kwargs: Ignored

        Returns:
            The recorded order
        """
        if self.fail_next is not None:
            error, self.fail_next = self.fail_next, None
            raise error

        self._sequence += 1
        order = Order(
            id=f"ex-{self._sequence}",
            pair=pair,
            type=order_type,
            side=side,
            price=price or 0.0,
            quantity=quantity,
            status=OrderStatus.OPEN,
            timestamp=datetime.now(),
            metadata={'clientOrderId': f"timi-{self._sequence}"}
        )

        self.created.append({
            'pair': pair,
            'type': order_type,
            'side': side,
            'quantity': quantity,
            'price': price,
            'order': order
        })

        self.all_orders[order.id] = order
        if order_type == OrderType.LIMIT:
            self.open_orders[order.id] = order
        return order

    async def cancel_order(self, order_id: str, pair: str) -> bool:
        """Record a cancellation.

        Args:
            order_id: Order ID
            pair: Trading pair

        Returns:
            True
        """
        self.cancelled.append(order_id)
        self.open_orders.pop(order_id, None)
        return True

    async def get_open_orders(self, pair: Optional[str] = None) -> List[Order]:
        """Return the orders still resting.

        Args:
            pair: Ignored

        Returns:
            List of resting orders
        """
        return list(self.open_orders.values())

    async def get_order(self, order_id: str, pair: str) -> Order:
        """Return a resting order, or a cancelled shell for a vanished one.

        Args:
            order_id: Order ID
            pair: Trading pair

        Returns:
            The order as the exchange now sees it
        """
        order = self.all_orders.get(order_id)
        if order is not None:
            return order
        return Order(
            id=order_id,
            pair=pair,
            type=OrderType.LIMIT,
            side=OrderSide.BUY,
            price=0.0,
            quantity=0.0,
            status=OrderStatus.CANCELED
        )

    async def get_positions(self, pair: Optional[str] = None) -> List[Position]:
        """Report held inventory the way a spot connector does.

        Every priced field is 0.0, exactly as the real connector reports, since
        a balance carries no cost basis.

        Args:
            pair: Trading pair to report on

        Returns:
            List of holdings
        """
        positions = []
        for symbol, size in self.inventory.items():
            if pair is not None and symbol != pair:
                continue
            if size <= 0:
                continue
            base = symbol.split('/')[0]
            positions.append(Position(
                pair=symbol,
                size=size,
                entry_price=0.0,
                current_price=0.0,
                unrealized_pnl=0.0,
                timestamp=datetime.now(),
                metadata={
                    'spot': True,
                    'base': base,
                    'quote': symbol.split('/')[-1],
                    'free': self.free.get(symbol, size),
                    'used': 0.0,
                    'total': size,
                    'entry_price_known': False,
                }
            ))
        return positions


def build_bot(
    exchange: Optional[StubExchange] = None,
    market_data: Optional[StubMarketData] = None,
    paper_mode: bool = False,
    book: Optional[PositionManager] = None,
    **config_overrides: Any
) -> TradingBot:
    """Build a bot wired to doubles.

    Args:
        exchange: Exchange double
        market_data: Market data double
        paper_mode: Whether orders are simulated
        book: Position book to share
        **config_overrides: Overrides for the bot configuration

    Returns:
        A trading bot
    """
    settings: Dict[str, Any] = {
        'execution_interval': 1,
        'min_volume': 0.0,
        'min_volatility': 0.0,
        'capital_per_pair': 100.0,
        'stop_loss_pct': 5.0,
    }
    settings.update(config_overrides)

    return TradingBot(
        pair="BTC/USDT",
        exchange=exchange or StubExchange(),
        market_data=market_data or StubMarketData(),
        config=BotConfig(**settings),
        paper_mode=paper_mode,
        position_manager=book or PositionManager(),
        risk_manager=None
    )


# --------------------------------------------------------------------------
# JOB ONE: cost basis
# --------------------------------------------------------------------------


async def test_holding_without_cost_basis_is_not_closed_as_profitable():
    """The regression test for the defect that closed the whole book.

    A spot holding arrives with `entry_price` of 0.0. The old monitor computed
    `target = entry * (1 + Φ) ** threshold`, which is 0.0, and then asked
    whether `current_price >= 0.0`, which is always true. Every holding was
    sold on the first monitoring tick. The value check was equally broken:
    `size * entry` was 0, which also tripped the small-position branch.
    """
    exchange = StubExchange(inventory={"BTC/USDT": 0.5})
    bot = build_bot(exchange=exchange)

    await bot._reconcile_inventory()

    held = bot.position_manager.get_position("BTC/USDT")
    assert held is not None
    assert held.size == pytest.approx(0.5)
    assert held.entry_price_known is False

    await bot._monitor_positions(current_price=60_000.0, volatility=0.02)

    assert exchange.created == [], "A holding with no cost basis was sold"
    assert bot.position_manager.get_position("BTC/USDT") is not None


async def test_cost_basis_comes_from_fills_not_from_the_exchange():
    """The position book, fed from fills, is what sets the entry price."""
    exchange = StubExchange(inventory={"BTC/USDT": 1.0})
    bot = build_bot(exchange=exchange)

    bot._record_fill(OrderSide.BUY, 1.0, 50_000.0, "ex-1")
    await bot._reconcile_inventory()

    held = bot.position_manager.get_position("BTC/USDT")
    assert held.entry_price == pytest.approx(50_000.0)
    assert held.entry_price_known is True


async def test_a_fill_is_booked_before_the_balance_is_reconciled():
    """Order of operations inside a cycle, and it matters.

    A buy that filled since the last cycle is inventory bought at a known
    price. Reconciling the balance first would see it as surplus with no cost
    basis and discard the basis the fill just established.
    """
    exchange = StubExchange(inventory={"BTC/USDT": 1.0})
    market = StubMarketData(price=100.0, volatility=0.02)
    bot = build_bot(exchange=exchange, market_data=market)

    await bot._place_order(
        {'side': OrderSide.BUY, 'price': 98.0, 'quantity': 1.0, 'level': 0},
        market_price=100.0,
        price_tolerance=1.0
    )
    order_id = next(iter(bot.active_orders))
    resting = exchange.open_orders.pop(order_id)
    resting.status = OrderStatus.FILLED
    resting.filled_quantity = resting.quantity

    await bot._execute_cycle()

    held = bot.position_manager.get_position("BTC/USDT")
    assert held is not None
    assert held.size == pytest.approx(1.0)
    assert held.entry_price_known is True
    assert held.entry_price == pytest.approx(98.0)


async def test_a_basisless_holding_is_not_stopped_out_either():
    """Without an entry price there is no reference for a stop, so no stop."""
    exchange = StubExchange(inventory={"BTC/USDT": 0.5})
    bot = build_bot(exchange=exchange)

    await bot._reconcile_inventory()
    await bot._monitor_positions(current_price=0.01, volatility=0.02)

    assert exchange.created == []


# --------------------------------------------------------------------------
# Grid lifecycle
# --------------------------------------------------------------------------


async def test_the_grid_cancels_before_replacing():
    """Nothing was ever cancelled, and the ladder was re-placed every cycle."""
    exchange = StubExchange()
    market = StubMarketData(price=100.0, volatility=0.02)
    bot = build_bot(exchange=exchange, market_data=market)

    await bot._execute_cycle()
    first_cycle_ids = set(bot.active_orders)
    assert len(first_cycle_ids) == len(bot.config.price_distribution)
    assert exchange.cancelled == []

    # The market moves, so every level moves with it.
    market.price = 120.0
    await bot._execute_cycle()

    assert set(exchange.cancelled) == first_cycle_ids
    assert len(bot.active_orders) == len(bot.config.price_distribution)
    assert bot.active_orders.keys().isdisjoint(first_cycle_ids)


async def test_an_unchanged_grid_is_left_alone():
    """A level that is still wanted is not cancelled and re-placed."""
    exchange = StubExchange()
    bot = build_bot(exchange=exchange, market_data=StubMarketData(price=100.0))

    await bot._execute_cycle()
    placed = len(exchange.created)

    await bot._execute_cycle()

    assert exchange.cancelled == []
    assert len(exchange.created) == placed


async def test_resting_orders_do_not_accumulate_over_many_cycles():
    """The old engine left roughly 840 orders an hour resting per symbol."""
    exchange = StubExchange()
    market = StubMarketData(price=100.0, volatility=0.02)
    bot = build_bot(exchange=exchange, market_data=market)

    for step in range(10):
        market.price = 100.0 + step
        await bot._execute_cycle()

    assert len(bot.active_orders) == len(bot.config.price_distribution)
    assert len(exchange.open_orders) == len(bot.config.price_distribution)


async def test_stopping_cancels_every_resting_order():
    """`stop()` used to set a flag and log, leaving the ladder live."""
    exchange = StubExchange()
    bot = build_bot(exchange=exchange)

    await bot._execute_cycle()
    resting = set(bot.active_orders)
    assert resting

    await bot.stop()

    assert set(exchange.cancelled) == resting
    assert bot.active_orders == {}
    assert bot._stop_event.is_set()


# --------------------------------------------------------------------------
# Selling only what is held
# --------------------------------------------------------------------------


async def test_no_sell_is_placed_without_inventory():
    """A sell you cannot cover simply fails on spot, so it is never sent."""
    exchange = StubExchange()
    bot = build_bot(exchange=exchange)

    await bot._execute_cycle()

    sides = {record['side'] for record in exchange.created}
    assert OrderSide.SELL not in sides


async def test_sell_ladder_never_exceeds_held_inventory():
    """The old sell size was simply the buy size, which opens a short."""
    exchange = StubExchange(inventory={"BTC/USDT": 0.1})
    bot = build_bot(exchange=exchange)
    bot.position_manager.add_position("BTC/USDT", size=0.1, entry_price=100.0)

    await bot._execute_cycle()

    sold = sum(
        record['quantity'] for record in exchange.created
        if record['side'] == OrderSide.SELL
    )
    assert sold <= 0.1 + 1e-12
    assert sold == pytest.approx(0.1)


async def test_sell_ladder_respects_the_free_balance():
    """Inventory locked in other orders is not sellable, however much is held."""
    exchange = StubExchange(
        inventory={"BTC/USDT": 1.0}, free={"BTC/USDT": 0.25}
    )
    bot = build_bot(exchange=exchange)
    bot.position_manager.add_position("BTC/USDT", size=1.0, entry_price=100.0)

    await bot._execute_cycle()

    sold = sum(
        record['quantity'] for record in exchange.created
        if record['side'] == OrderSide.SELL
    )
    assert sold == pytest.approx(0.25)


async def test_a_single_sell_is_clamped_to_available_inventory():
    """The clamp holds even when a caller asks for more directly."""
    exchange = StubExchange()
    bot = build_bot(exchange=exchange)
    bot.position_manager.add_position("BTC/USDT", size=0.05, entry_price=100.0)

    await bot._place_order(
        {'side': OrderSide.SELL, 'price': 110.0, 'quantity': 10.0, 'level': 0},
        market_price=100.0,
        price_tolerance=1.0
    )

    assert len(exchange.created) == 1
    assert exchange.created[0]['quantity'] == pytest.approx(0.05)


# --------------------------------------------------------------------------
# Monitoring: one action per holding per pass
# --------------------------------------------------------------------------


async def test_a_holding_is_acted_on_at_most_once_per_pass():
    """The old `break` left only the inner loop, then the small-position
    branch re-evaluated the same stale object and closed it a second time.

    This holding satisfies both the profit ladder and the small-position rule
    at once, so a second action would show up as a second order.
    """
    exchange = StubExchange()
    bot = build_bot(exchange=exchange, capital_per_pair=100.0)
    # Notional of 5, under the A/λ threshold of 10, and far past every target.
    bot.position_manager.add_position("BTC/USDT", size=0.05, entry_price=100.0)

    await bot._monitor_positions(current_price=1_000.0, volatility=0.02)

    assert len(exchange.created) == 1
    assert exchange.created[0]['side'] == OrderSide.SELL
    assert bot.position_manager.get_position("BTC/USDT") is None


async def test_stop_loss_triggers():
    """`risk.stop_loss_pct` was configured and nothing read it."""
    exchange = StubExchange()
    bot = build_bot(exchange=exchange, stop_loss_pct=5.0)
    bot.position_manager.add_position("BTC/USDT", size=1.0, entry_price=100.0)

    await bot._monitor_positions(current_price=94.0, volatility=0.02)

    assert len(exchange.created) == 1
    assert exchange.created[0]['side'] == OrderSide.SELL
    assert exchange.created[0]['type'] == OrderType.MARKET
    assert bot.position_manager.get_position("BTC/USDT") is None


async def test_stop_loss_does_not_trigger_just_inside_the_limit():
    """Four percent down on a five percent stop is not a breach."""
    exchange = StubExchange()
    bot = build_bot(exchange=exchange, stop_loss_pct=5.0)
    bot.position_manager.add_position("BTC/USDT", size=1.0, entry_price=100.0)

    await bot._monitor_positions(current_price=96.0, volatility=0.02)

    assert exchange.created == []


async def test_the_highest_satisfied_profit_threshold_wins():
    """Ascending thresholds with a break on the first match made 2.0, 3.0 and
    5.0 unreachable, because `(1 + Φ) ** 1.5` is always the easiest to clear.
    """
    bot = build_bot(profit_loss_thresholds=[1.5, 2.0, 3.0, 5.0])
    bot.position_manager.add_position("BTC/USDT", size=1.0, entry_price=100.0)

    closed: List[str] = []

    async def record(position, price, reason, quantity=None):
        closed.append(reason)
        return True

    bot._close_position = record

    # Φ of 0.1: the 5.0 threshold sits at 100 * 1.1 ** 5, about 161.
    await bot._monitor_positions(current_price=200.0, volatility=0.1)

    assert closed == ['take_profit_5.0']


async def test_a_lower_threshold_is_used_when_the_higher_ones_are_unmet():
    """The ladder still works from the bottom when only the first is cleared."""
    bot = build_bot(profit_loss_thresholds=[1.5, 2.0, 3.0, 5.0])
    bot.position_manager.add_position("BTC/USDT", size=1.0, entry_price=100.0)

    closed: List[str] = []

    async def record(position, price, reason, quantity=None):
        closed.append(reason)
        return True

    bot._close_position = record

    # 100 * 1.1 ** 1.5 is about 115.4; 100 * 1.1 ** 2.0 is 121.
    await bot._monitor_positions(current_price=118.0, volatility=0.1)

    assert closed == ['take_profit_1.5']


async def test_a_healthy_holding_is_left_alone():
    """No stop, no target, not small: nothing happens."""
    exchange = StubExchange()
    bot = build_bot(exchange=exchange, capital_per_pair=100.0)
    bot.position_manager.add_position("BTC/USDT", size=10.0, entry_price=100.0)

    await bot._monitor_positions(current_price=101.0, volatility=0.02)

    assert exchange.created == []


async def test_a_zero_position_size_divisor_does_not_divide_by_zero():
    """The divisor was used with no guard against zero."""
    exchange = StubExchange()
    bot = build_bot(exchange=exchange, position_size_divisor=0)
    bot.position_manager.add_position("BTC/USDT", size=0.01, entry_price=100.0)

    await bot._monitor_positions(current_price=101.0, volatility=0.02)

    assert exchange.created == []


# --------------------------------------------------------------------------
# Volatility and distributions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "volatility", [0.0, 1.0, -0.5, 1.5, float("nan"), float("inf")]
)
async def test_unusable_volatility_places_nothing(volatility):
    """At Φ of 0 every level collapses onto the spot price; at Φ of 1 the buy
    side reaches zero. The data layer clamps at 0.99, and this does not rely
    on that.
    """
    exchange = StubExchange()
    market = StubMarketData(price=100.0, volatility=volatility)
    bot = build_bot(exchange=exchange, market_data=market)

    await bot._execute_cycle()

    assert exchange.created == []
    assert bot.active_orders == {}


async def test_unusable_volatility_cancels_an_existing_grid():
    """A ladder must not be left resting on statistics that stopped working."""
    exchange = StubExchange()
    market = StubMarketData(price=100.0, volatility=0.02)
    bot = build_bot(exchange=exchange, market_data=market)

    await bot._execute_cycle()
    resting = set(bot.active_orders)
    assert resting

    market.volatility = 0.0
    await bot._execute_cycle()

    assert set(exchange.cancelled) == resting
    assert bot.active_orders == {}


async def test_usable_volatility_at_the_edges_is_accepted():
    """Strictly between 0 and 1 is the rule, and it is applied strictly."""
    bot = build_bot()

    assert bot._volatility_usable(0.0001) is True
    assert bot._volatility_usable(0.99) is True
    assert bot._volatility_usable(0.0) is False
    assert bot._volatility_usable(1.0) is False


async def test_mismatched_distributions_place_nothing():
    """`zip` truncated the price distribution against the quantity one."""
    exchange = StubExchange()
    bot = build_bot(
        exchange=exchange,
        price_distribution=[0.5, 1.0, 1.5, 2.0],
        quantity_distribution=[0.5, 0.5]
    )

    await bot._execute_cycle()

    assert exchange.created == []


async def test_the_neutral_entry_coefficient_is_a_no_op():
    """1.0 must reproduce the ungeared ladder exactly, not approximately.

    `entry_coefficient` used to be plumbed into the bot configuration and read
    by nothing. Giving it a meaning is only safe if the neutral value leaves
    today's geometry untouched, so this compares against the arithmetic the
    engine performed before the coefficient existed.
    """
    bot = build_bot(entry_coefficient=1.0)
    volatility = 0.02

    orders = bot._calculate_orders(current_price=100.0, volatility=volatility)
    buys = [o for o in orders if o['side'] == OrderSide.BUY]

    assert len(buys) == len(bot.config.price_distribution)
    for order, exponent in zip(buys, bot.config.price_distribution):
        assert order['price'] == 100.0 * ((1 - volatility) ** exponent)

    assert bot._entry_depth(volatility) == volatility


async def test_a_low_entry_coefficient_tightens_the_buy_ladder():
    """Below 1.0 the buy levels sit closer to spot, and only the buy levels."""
    volatility = 0.02
    neutral = build_bot(entry_coefficient=1.0)
    tight = build_bot(entry_coefficient=0.5)
    neutral.position_manager.add_position("BTC/USDT", size=1.0, entry_price=100.0)
    tight.position_manager.add_position("BTC/USDT", size=1.0, entry_price=100.0)

    base = neutral._calculate_orders(100.0, volatility)
    geared = tight._calculate_orders(100.0, volatility)

    base_buys = [o['price'] for o in base if o['side'] == OrderSide.BUY]
    geared_buys = [o['price'] for o in geared if o['side'] == OrderSide.BUY]
    assert all(g > b for g, b in zip(geared_buys, base_buys))
    assert all(price < 100.0 for price in geared_buys)

    # The sell ladder is built from Φ alone and must not have moved.
    base_sells = [o['price'] for o in base if o['side'] == OrderSide.SELL]
    geared_sells = [o['price'] for o in geared if o['side'] == OrderSide.SELL]
    assert base_sells == geared_sells
    assert base_sells


async def test_a_high_entry_coefficient_widens_the_buy_ladder():
    """Above 1.0 the buy levels sit further below spot."""
    volatility = 0.02
    neutral = build_bot(entry_coefficient=1.0)
    wide = build_bot(entry_coefficient=2.0)

    base_buys = [
        o['price'] for o in neutral._calculate_orders(100.0, volatility)
        if o['side'] == OrderSide.BUY
    ]
    wide_buys = [
        o['price'] for o in wide._calculate_orders(100.0, volatility)
        if o['side'] == OrderSide.BUY
    ]

    assert all(w < b for w, b in zip(wide_buys, base_buys))
    assert wide._entry_depth(volatility) == pytest.approx(0.04)


async def test_a_widened_ladder_is_not_then_refused_by_the_price_check():
    """The deviation bound has to follow the ladder it is bounding."""
    wide = build_bot(entry_coefficient=2.0)
    volatility = 0.02

    tolerance = wide._grid_price_tolerance(volatility)
    deepest = min(
        o['price'] for o in wide._calculate_orders(100.0, volatility)
        if o['side'] == OrderSide.BUY
    )

    assert (100.0 - deepest) / 100.0 <= tolerance + 1e-12


@pytest.mark.parametrize(
    "coefficient",
    [0.0, -1.0, 100.0, 0.01, float("nan"), float("inf"), None, "0.8"]
)
async def test_an_unusable_entry_coefficient_falls_back_to_neutral(coefficient):
    """A knob set to nonsense must not silently reshape the grid."""
    bot = build_bot(entry_coefficient=coefficient)

    assert bot._entry_coefficient() == 1.0
    assert bot._entry_depth(0.02) == 0.02


async def test_the_entry_depth_is_clamped_below_one():
    """`(1 - depth) ** exponent` is not a price once depth reaches 1.0."""
    bot = build_bot(entry_coefficient=5.0)

    assert bot._entry_depth(0.5) == pytest.approx(0.99)

    orders = bot._calculate_orders(current_price=100.0, volatility=0.5)
    buys = [o for o in orders if o['side'] == OrderSide.BUY]
    assert buys
    assert all(o['price'] > 0 for o in buys)


async def test_quantity_proportions_must_sum_to_one():
    """Nothing validated the proportions, so the allocation was not the
    allocation the configuration described."""
    exchange = StubExchange()
    bot = build_bot(
        exchange=exchange,
        price_distribution=[0.5, 1.0],
        quantity_distribution=[0.5, 0.9]
    )

    await bot._execute_cycle()

    assert exchange.created == []


# --------------------------------------------------------------------------
# Paper mode
# --------------------------------------------------------------------------


async def test_paper_mode_produces_non_zero_statistics():
    """Paper mode never monitored or recorded anything, so `get_stats()`
    always returned zeros and the advice to validate in paper mode was empty.
    """
    market = StubMarketData(price=100.0, volatility=0.02)
    bot = build_bot(market_data=market, paper_mode=True)

    await bot._execute_cycle()
    assert bot.get_stats()['orders_placed'] > 0

    # The price dips through the shallowest resting buy level, which fills.
    market.price = 98.9
    await bot._execute_cycle()

    stats = bot.get_stats()
    assert stats['fills'] > 0
    assert stats['trades_executed'] > 0

    held = bot.position_manager.get_position("BTC/USDT")
    assert held is not None
    assert held.entry_price_known is True
    assert bot.position_manager.get_statistics()['open_positions'] == 1


async def test_paper_mode_records_a_realised_loss_on_a_stop():
    """Simulated fills feed the same position book, so P&L is real arithmetic."""
    market = StubMarketData(price=100.0, volatility=0.02)
    bot = build_bot(market_data=market, paper_mode=True, stop_loss_pct=5.0)

    await bot._execute_cycle()
    market.price = 50.0
    await bot._execute_cycle()

    # Bought around 100, marked at 50: the stop fires on the next pass.
    market.price = 49.0
    await bot._execute_cycle()

    stats = bot.get_stats()
    assert stats['loss_count'] > 0
    assert stats['total_pnl'] < 0


async def test_paper_mode_transmits_nothing():
    """A simulated run must not reach the exchange, even to cancel."""
    exchange = StubExchange()
    market = StubMarketData(price=100.0, volatility=0.02)
    bot = build_bot(exchange=exchange, market_data=market, paper_mode=True)

    await bot._execute_cycle()
    market.price = 130.0
    await bot._execute_cycle()
    await bot.stop()

    assert exchange.created == []
    assert exchange.cancelled == []
    assert bot.stats['orders_cancelled'] > 0


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------


async def test_repeated_failures_halt_the_bot():
    """Every failure used to be logged and ignored, and the loop went on."""
    exchange = StubExchange()
    bot = build_bot(exchange=exchange, max_consecutive_failures=3)

    for _ in range(3):
        bot._record_failure("place_order")

    assert bot.running is False
    assert bot.halted_reason is not None
    assert bot._stop_event.is_set()


async def test_a_clean_cycle_resets_the_failure_counter():
    """The breaker counts consecutive failures, not lifetime ones."""
    bot = build_bot(max_consecutive_failures=3)

    bot._record_failure("place_order")
    bot._record_failure("place_order")
    bot._record_success()

    assert bot.consecutive_failures == 0
    assert bot.running is False or bot.halted_reason is None


async def test_backoff_grows_with_consecutive_failures():
    """A retry loop with no backoff is how a rate limit becomes an outage."""
    bot = build_bot(max_consecutive_failures=5, failure_backoff_seconds=2.0)

    assert bot._record_failure("cycle") == pytest.approx(2.0)
    assert bot._record_failure("cycle") == pytest.approx(4.0)


async def test_an_unknown_order_outcome_is_reconciled_not_assumed():
    """`OrderStatusUnknown` means the order MAY exist. Assuming it failed and
    re-placing is how a position gets doubled.
    """
    exchange = StubExchange()
    bot = build_bot(exchange=exchange)

    # An order that is already resting under the id the failed request used.
    resting = Order(
        id="ex-99",
        pair="BTC/USDT",
        type=OrderType.LIMIT,
        side=OrderSide.BUY,
        price=98.0,
        quantity=1.0,
        status=OrderStatus.OPEN,
        metadata={'clientOrderId': "timi-abc"}
    )
    exchange.open_orders[resting.id] = resting
    exchange.fail_next = OrderStatusUnknown(
        "timed out", pair="BTC/USDT", client_order_id="timi-abc"
    )

    placed = await bot._place_order(
        {'side': OrderSide.BUY, 'price': 98.0, 'quantity': 1.0, 'level': 0},
        market_price=100.0,
        price_tolerance=1.0
    )

    assert placed is resting
    assert placed.id in bot.active_orders
    assert bot.consecutive_failures == 0


async def test_an_unknown_order_that_does_not_exist_counts_as_a_failure():
    """No resting order carries the id, so nothing was placed and it counts."""
    exchange = StubExchange()
    bot = build_bot(exchange=exchange)
    exchange.fail_next = OrderStatusUnknown(
        "timed out", pair="BTC/USDT", client_order_id="timi-missing"
    )

    placed = await bot._place_order(
        {'side': OrderSide.BUY, 'price': 98.0, 'quantity': 1.0, 'level': 0},
        market_price=100.0,
        price_tolerance=1.0
    )

    assert placed is None
    assert bot.active_orders == {}
    assert bot.consecutive_failures == 1


async def test_a_level_below_the_minimum_notional_does_not_trip_the_breaker():
    """It is deterministic and benign, not a fault worth halting for."""
    exchange = StubExchange()
    bot = build_bot(exchange=exchange, max_consecutive_failures=2)
    exchange.fail_next = MinimumNotionalError("too small")

    placed = await bot._place_order(
        {'side': OrderSide.BUY, 'price': 98.0, 'quantity': 1e-9, 'level': 0},
        market_price=100.0,
        price_tolerance=1.0
    )

    assert placed is None
    assert bot.consecutive_failures == 0
    assert bot.stats['orders_rejected'] == 1


async def test_a_filled_order_becomes_cost_basis():
    """A resting order that resolves as filled feeds the position book."""
    exchange = StubExchange()
    bot = build_bot(exchange=exchange)

    await bot._place_order(
        {'side': OrderSide.BUY, 'price': 98.0, 'quantity': 1.0, 'level': 0},
        market_price=100.0,
        price_tolerance=1.0
    )
    order_id = next(iter(bot.active_orders))

    # The exchange fills it: it leaves the open list and reports as filled.
    filled = exchange.open_orders.pop(order_id)
    filled.status = OrderStatus.FILLED
    filled.filled_quantity = filled.quantity

    await bot._refresh_resting_orders()

    held = bot.position_manager.get_position("BTC/USDT")
    assert held is not None
    assert held.entry_price == pytest.approx(98.0)
    assert bot.active_orders == {}


# --------------------------------------------------------------------------
# Risk gate
# --------------------------------------------------------------------------


class RefusingRiskManager:
    """A risk gate that refuses everything, to prove orders go through it."""

    def __init__(self):
        """Record what was asked of it."""
        self.emergency_stop = False
        self.max_price_deviation = 0.02
        self.checked: List[Dict[str, Any]] = []

    def check_order_risk(self, **kwargs: Any) -> bool:
        """Refuse every order.

        Args:
            **kwargs: Order details

        Returns:
            False, always
        """
        self.checked.append(kwargs)
        return False

    def release_exposure(self, key: str) -> None:
        """No reservations are ever made."""

    def update_capital(self, current_capital=None) -> bool:
        """Nothing to mark."""
        return True

    def halt(self, reason: str) -> None:
        """Record a halt."""
        self.emergency_stop = True


async def test_every_order_passes_through_the_risk_gate():
    """The risk layer had zero call sites; now nothing is placed without it."""
    exchange = StubExchange()
    bot = build_bot(exchange=exchange)
    bot.risk_manager = RefusingRiskManager()

    await bot._execute_cycle()

    assert exchange.created == []
    assert len(bot.risk_manager.checked) == len(bot.config.price_distribution)
    assert bot.stats['orders_rejected'] > 0


async def test_an_emergency_stop_stops_the_bot():
    """The engine halts when the risk manager says trading is over."""
    exchange = StubExchange()
    bot = build_bot(exchange=exchange)
    bot.risk_manager = RefusingRiskManager()
    bot.running = True

    await bot._execute_cycle()
    assert exchange.created == []

    bot.risk_manager.emergency_stop = True
    await bot._execute_cycle()

    assert bot.running is False
    assert bot._stop_event.is_set()
