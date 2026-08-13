"""Position management for tracking and monitoring open positions.

This is the system's source of truth for cost basis. The venue is SPOT, and
`get_positions()` there reports a balance: a balance records how much base
currency is held, never what was paid for it, so the exchange reports an entry
price of 0.0. Anything that derives a profit target, a stop or a P&L figure has
to read the average entry price from here instead, accumulated from the fills
this system actually made.

Sign convention, stated once because the old code did not have one:

* `size` is the quantity of base currency held. It is never negative. Nothing
  can be short on spot, so `is_short` never returns True; it is kept only so
  callers written against the exchange dataclass keep working.
* A reduction is expressed either as a negative `size` passed to
  `add_position`, or as a positive `size` passed to `close_position`. Both mean
  the same thing: inventory left the book.
* Reducing a holding NEVER moves the average entry price. Selling half a
  holding at a profit does not make the remaining half cheaper. Only a purchase
  changes the average cost.

A holding whose cost basis is unknown is marked as such rather than being given
a placeholder price of 0.0 and treated as free inventory. `entry_price_known`
is the flag; callers must check it before computing a target or a stop.
"""

import math
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.logging import TradingLogger


#: Sizes below this are treated as dust and rounded away, so floating point
#: residue from repeated partial closes cannot leave a phantom holding behind.
DUST_SIZE = 1e-12


@dataclass
class ManagedPosition:
    """A held spot inventory line with its accumulated cost basis."""
    pair: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        """Check if position is long.

        True for any non-empty holding, since spot inventory is always long.
        """
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short.

        Always False on spot. Retained for interface compatibility only.
        """
        return self.size < 0

    @property
    def entry_price_known(self) -> bool:
        """Whether this holding has a usable cost basis.

        False for inventory that was already on the account when the system
        started, or that arrived by any route other than a fill this system
        recorded. A holding with no basis must not be profit-taken: with an
        entry price of 0.0 every target price is 0.0 and every current price
        clears it, which closes the holding instantly.

        Returns:
            True when `entry_price` may be used for targets and P&L
        """
        known = self.metadata.get('entry_price_known')
        if known is None:
            return self.entry_price > 0
        return bool(known) and self.entry_price > 0

    @property
    def notional(self) -> float:
        """Cost value of the holding at its average entry price."""
        return abs(self.size) * self.entry_price

    @property
    def pnl_percentage(self) -> float:
        """Get PnL as a percentage of the entry notional.

        Returns 0.0 rather than raising when the notional is zero or not
        finite, which covers a fully closed holding, a holding with no known
        basis, and any value that arithmetic has corrupted.

        Returns:
            P&L as a percentage, or 0.0 when it is undefined
        """
        notional = self.notional
        if not math.isfinite(notional) or abs(notional) < DUST_SIZE:
            return 0.0
        if not math.isfinite(self.unrealized_pnl):
            return 0.0
        return (self.unrealized_pnl / notional) * 100

    def update_price(self, current_price: float) -> None:
        """Update the mark price and recalculate unrealised P&L.

        A holding with no known cost basis keeps an unrealised P&L of 0.0,
        because the true figure is not knowable from a balance.

        Args:
            current_price: Current market price
        """
        self.current_price = current_price
        self.last_update = datetime.now()

        if not self.entry_price_known:
            self.unrealized_pnl = 0.0
            return

        # Long only: inventory gains when the mark rises above the average cost.
        self.unrealized_pnl = (current_price - self.entry_price) * self.size


class PositionManager:
    """Manager for tracking and monitoring positions."""

    def __init__(self):
        """Initialize position manager."""
        self.positions: Dict[str, ManagedPosition] = {}
        self.logger = TradingLogger("position_manager")
        self.closed_positions: List[ManagedPosition] = []
        # Realised P&L is accumulated here as well as on the position, so a
        # partial close is not lost from the totals until the line is fully
        # closed out.
        self.realized_pnl_total: float = 0.0

    # ------------------------------------------------------------------
    # Recording fills
    # ------------------------------------------------------------------

    def add_position(
        self,
        pair: str,
        size: float,
        entry_price: float,
        metadata: Optional[Dict] = None
    ) -> Optional[ManagedPosition]:
        """Record a fill against a holding.

        A positive `size` is a purchase and moves the average entry price
        towards the price paid. A negative `size` is a reduction and leaves the
        average entry price exactly where it was: inventory that leaves the
        book does not change what the remaining inventory cost.

        Args:
            pair: Trading pair
            size: Quantity filled. Positive to buy, negative to reduce
            entry_price: Price of the fill
            metadata: Additional metadata, stored on a newly created holding

        Returns:
            The holding after the fill, or None when nothing is held
        """
        if not math.isfinite(size) or abs(size) < DUST_SIZE:
            return self.positions.get(pair)

        existing = self.positions.get(pair)

        if existing is None:
            if size < 0:
                # Spot cannot go short. A sell with nothing held is a bug in
                # the caller, not a position to open.
                self.logger.logger.warning(
                    "reduce_with_no_holding",
                    pair=pair,
                    size=size,
                    message="Refusing to record a reduction with nothing held"
                )
                return None
            return self._open(pair, size, entry_price, metadata)

        if size > 0:
            self._increase(existing, size, entry_price)
        else:
            self._reduce(existing, abs(size))

        if existing.size <= DUST_SIZE:
            return self._retire(pair)

        return existing

    def _open(
        self,
        pair: str,
        size: float,
        entry_price: float,
        metadata: Optional[Dict]
    ) -> ManagedPosition:
        """Create a new holding from a first fill.

        Args:
            pair: Trading pair
            size: Quantity bought, positive
            entry_price: Price paid
            metadata: Additional metadata

        Returns:
            The newly created holding
        """
        meta = dict(metadata or {})
        meta.setdefault('entry_price_known', entry_price > 0)

        position = ManagedPosition(
            pair=pair,
            size=size,
            entry_price=entry_price,
            current_price=entry_price,
            unrealized_pnl=0.0,
            metadata=meta
        )
        self.positions[pair] = position

        self._log_position(position)
        return position

    def _increase(
        self,
        position: ManagedPosition,
        size: float,
        entry_price: float
    ) -> None:
        """Add to a holding and re-average its cost basis.

        When either the existing basis or the incoming price is unknown, the
        blended average is unknowable too, and the holding is marked as having
        no basis rather than being given a number that looks authoritative.

        Args:
            position: Holding to add to
            size: Quantity bought, positive
            entry_price: Price paid
        """
        if not position.entry_price_known or entry_price <= 0:
            position.size += size
            position.entry_price = 0.0
            position.metadata['entry_price_known'] = False
            self.logger.logger.warning(
                "cost_basis_unknown_after_purchase",
                pair=position.pair,
                size=position.size,
                message=(
                    "Bought into a holding with no known cost basis; the "
                    "blended average cannot be derived"
                )
            )
            return

        total_cost = position.entry_price * position.size + entry_price * size
        position.size += size
        position.entry_price = total_cost / position.size
        position.metadata['entry_price_known'] = True

        self._log_position(position)

    def _reduce(self, position: ManagedPosition, quantity: float) -> None:
        """Remove inventory from a holding, leaving the cost basis untouched.

        Args:
            position: Holding to reduce
            quantity: Quantity sold, positive
        """
        if quantity > position.size:
            self.logger.logger.warning(
                "reduce_exceeds_holding",
                pair=position.pair,
                requested=quantity,
                held=position.size,
                message="Clamping the reduction to the quantity actually held"
            )
            quantity = position.size

        position.size -= quantity
        position.last_update = datetime.now()

    def _retire(self, pair: str) -> None:
        """Move a fully closed holding out of the open book.

        Args:
            pair: Trading pair

        Returns:
            None, since nothing is held any more
        """
        closed = self.positions.pop(pair, None)
        if closed is not None:
            closed.size = 0.0
            closed.unrealized_pnl = 0.0
            self.closed_positions.append(closed)
        return None

    def reconcile_size(
        self,
        pair: str,
        exchange_size: float,
        metadata: Optional[Dict] = None
    ) -> Optional[ManagedPosition]:
        """Align the tracked quantity with the inventory the exchange reports.

        The exchange is authoritative for QUANTITY and knows nothing about
        cost. Surplus inventory therefore arrives with no basis, and since the
        average cost of a mixture that includes an unpriced part is not
        derivable, the whole holding is marked as having no basis. That is
        deliberately loud: it stops profit-taking on that pair until a fresh
        cycle of buy and full sell re-establishes a basis, which is the safe
        direction to fail.

        A shortfall means inventory left by some route this system did not see.
        The quantity is reduced and the basis is kept, as with any other sale.

        Args:
            pair: Trading pair
            exchange_size: Base-currency quantity the exchange reports held
            metadata: Metadata for a holding created here

        Returns:
            The reconciled holding, or None when nothing is held
        """
        if not math.isfinite(exchange_size) or exchange_size < 0:
            self.logger.logger.warning(
                "unusable_exchange_size",
                pair=pair,
                exchange_size=exchange_size
            )
            return self.positions.get(pair)

        existing = self.positions.get(pair)

        if exchange_size <= DUST_SIZE:
            if existing is not None:
                self.logger.logger.info(
                    "inventory_gone_from_exchange",
                    pair=pair,
                    tracked_size=existing.size
                )
                self._retire(pair)
            return None

        if existing is None:
            meta = dict(metadata or {})
            meta['entry_price_known'] = False
            position = ManagedPosition(
                pair=pair,
                size=exchange_size,
                entry_price=0.0,
                current_price=0.0,
                unrealized_pnl=0.0,
                metadata=meta
            )
            self.positions[pair] = position
            self.logger.logger.info(
                "holding_adopted_without_basis",
                pair=pair,
                size=exchange_size,
                message=(
                    "Inventory found on the account with no recorded fills; "
                    "profit taking is disabled for it"
                )
            )
            return position

        difference = exchange_size - existing.size

        if abs(difference) <= DUST_SIZE:
            return existing

        if difference > 0:
            existing.size = exchange_size
            existing.entry_price = 0.0
            existing.metadata['entry_price_known'] = False
            self.logger.logger.warning(
                "surplus_inventory_without_basis",
                pair=pair,
                surplus=difference,
                size=exchange_size,
                message=(
                    "More inventory held than this system bought; the average "
                    "cost of the holding is no longer derivable"
                )
            )
            return existing

        self.logger.logger.warning(
            "inventory_shortfall",
            pair=pair,
            shortfall=abs(difference),
            size=exchange_size,
            message="Less inventory held than tracked; reducing the holding"
        )
        existing.size = exchange_size
        existing.last_update = datetime.now()
        return existing

    # ------------------------------------------------------------------
    # Marking and closing
    # ------------------------------------------------------------------

    def update_position_price(self, pair: str, current_price: float) -> None:
        """Update a holding with the current price.

        Args:
            pair: Trading pair
            current_price: Current market price
        """
        position = self.positions.get(pair)
        if position is None:
            return

        position.update_price(current_price)
        self._log_position(position)

    def close_position(
        self,
        pair: str,
        exit_price: float,
        size: Optional[float] = None
    ) -> Optional[ManagedPosition]:
        """Close a holding, fully or partially, and realise the P&L.

        The quantity is taken as a magnitude, so a caller that passes a
        negative size gets the same result as one that passes a positive one.
        A partial close leaves the average entry price where it was.

        Args:
            pair: Trading pair
            exit_price: Price achieved on the sale
            size: Quantity to close. None closes the whole holding

        Returns:
            The holding after the close, or None when nothing was held
        """
        position = self.positions.get(pair)
        if position is None:
            return None

        requested = position.size if size is None else abs(size)
        close_size = min(requested, position.size)

        if close_size <= DUST_SIZE:
            return position

        if position.entry_price_known:
            realized_pnl = (exit_price - position.entry_price) * close_size
        else:
            # No basis means no honest P&L figure. Recording zero is not a
            # claim that the trade broke even, it is a refusal to invent one.
            realized_pnl = 0.0
            self.logger.logger.warning(
                "realised_pnl_unknown",
                pair=pair,
                quantity=close_size,
                message="Closed a holding with no cost basis; P&L not derivable"
            )

        position.realized_pnl += realized_pnl
        self.realized_pnl_total += realized_pnl

        self.logger.log_trade(
            action="CLOSE",
            pair=pair,
            price=exit_price,
            quantity=close_size,
            pnl=realized_pnl
        )

        full_close = close_size >= position.size - DUST_SIZE
        if full_close:
            closed = self.positions.pop(pair)
            closed.size = 0.0
            closed.unrealized_pnl = 0.0
            self.closed_positions.append(closed)
            return closed

        position.size -= close_size
        position.last_update = datetime.now()
        position.update_price(exit_price)
        return position

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_position(self, pair: str) -> Optional[ManagedPosition]:
        """Get position for a pair.

        Args:
            pair: Trading pair

        Returns:
            Position or None if not found
        """
        return self.positions.get(pair)

    def get_all_positions(self) -> List[ManagedPosition]:
        """Get all open positions.

        Returns:
            List of positions
        """
        return list(self.positions.values())

    def get_total_pnl(self) -> float:
        """Get total unrealized PnL across all positions.

        Returns:
            Total unrealized PnL
        """
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def get_total_realized_pnl(self) -> float:
        """Get total realised PnL, including partial closes still open.

        Returns:
            Total realized PnL
        """
        return self.realized_pnl_total

    def get_statistics(self) -> Dict:
        """Get position statistics.

        Returns:
            Statistics dictionary
        """
        open_positions = len(self.positions)
        total_unrealized_pnl = self.get_total_pnl()
        total_realized_pnl = self.get_total_realized_pnl()

        winning_closed = sum(1 for pos in self.closed_positions if pos.realized_pnl > 0)
        total_closed = len(self.closed_positions)
        win_rate = winning_closed / total_closed if total_closed > 0 else 0

        return {
            'open_positions': open_positions,
            'closed_positions': total_closed,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'win_rate': win_rate,
            'winning_trades': winning_closed,
            'losing_trades': total_closed - winning_closed
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_position(self, position: ManagedPosition) -> None:
        """Emit a position log line, but only when it can be computed.

        `TradingLogger.log_position` divides by `entry_price * size` whenever
        the size is positive, so calling it for a holding with no cost basis
        raises ZeroDivisionError. Such holdings are logged plainly instead.

        Args:
            position: Holding to log
        """
        if position.entry_price_known and position.size > 0:
            self.logger.log_position(
                pair=position.pair,
                size=position.size,
                entry_price=position.entry_price,
                current_price=position.current_price,
                pnl=position.unrealized_pnl
            )
            return

        self.logger.logger.info(
            "position_update_without_basis",
            pair=position.pair,
            size=position.size,
            current_price=position.current_price
        )
