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

Tranche accounting
------------------

A holding is a queue of TRANCHES rather than a single blended number. A tranche
is a quantity of base currency that either carries the price paid for it, taken
from a fill this system recorded, or is explicitly UNPRICED because it arrived
by a route that carries no cost information (found on the account at start-up,
deposited by hand, or bought before the bot ran).

* The average entry price is the quantity-weighted average of the PRICED
  tranches only. Unpriced quantity is never folded in at 0.0, which would drag
  the average towards free inventory.
* `priced_size` and `unpriced_size` report how much of the holding each kind
  accounts for, so a holding can be partly basis-known.
* Tranches are consumed FIRST IN, FIRST OUT. FIFO is the usual convention for
  spot inventory, it needs no extra state beyond acquisition order, and it is
  the easiest to reason about when auditing a sale against the fills that
  preceded it. Realised P&L is computed against the tranche or tranches the
  sale actually consumed, not against the blended average.
* A reduction never RE-PRICES the tranches that remain: a sale price is never
  blended into the cost of inventory that is still held. Selling half a holding
  at a profit does not make the remaining half cheaper. The reported average
  can still move after a sale, but only because whole tranches left the book.

A holding is only reported as having a usable cost basis when EVERY tranche in
it is priced. `entry_price_known` is that flag, and callers must check it before
computing a target or a stop. Mixed holdings are deliberately reported as
basis-unknown: the callers that act on this book close a whole holding at a
time, so letting them act on a mixed holding would apply a priced decision to
unpriced quantity. What tranche accounting buys is that the priced part stays
priced. Once the unpriced quantity leaves, the holding recovers its real
average immediately instead of needing a full flat-and-rebuy cycle.
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
class Tranche:
    """A quantity of base currency acquired in one go.

    A tranche is either priced, meaning `price` is the price actually paid on a
    fill this system recorded, or unpriced, meaning the quantity was adopted
    from the exchange and nothing is known about what it cost. Unpriced is
    represented by `price is None`, never by 0.0, so that no arithmetic can
    mistake it for free inventory.
    """
    quantity: float
    price: Optional[float] = None
    acquired_at: datetime = field(default_factory=datetime.now)

    @property
    def is_priced(self) -> bool:
        """Whether this tranche carries a usable acquisition price.

        Returns:
            True when `price` may be used for a basis or a P&L figure
        """
        return self.price is not None and self.price > 0

    @property
    def cost(self) -> float:
        """Cost of the tranche, or 0.0 when it is unpriced."""
        return self.quantity * self.price if self.is_priced else 0.0


@dataclass
class ManagedPosition:
    """A held spot inventory line with its accumulated cost basis.

    `size` and `entry_price` are kept in step with `tranches` by `_sync`, so
    callers written against the flat fields keep working unchanged. A position
    built directly with no tranches, as some tests and adapters do, falls back
    to reading those flat fields.
    """
    pair: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    tranches: List[Tranche] = field(default_factory=list)

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

        False as well for a holding that is only PARTLY priced. Callers close a
        whole holding at a time, so a True here would apply a decision derived
        from the priced tranches to the unpriced ones too.

        Returns:
            True when `entry_price` may be used for targets and P&L
        """
        known = self.metadata.get('entry_price_known')
        if known is None:
            return self.entry_price > 0
        return bool(known) and self.entry_price > 0

    @property
    def priced_size(self) -> float:
        """Quantity held whose acquisition price is known.

        Returns:
            The summed quantity of the priced tranches
        """
        if not self.tranches:
            return self.size if self.entry_price_known else 0.0
        return sum(t.quantity for t in self.tranches if t.is_priced)

    @property
    def unpriced_size(self) -> float:
        """Quantity held that carries no acquisition price.

        This is the quantity that must be excluded from any decision needing a
        cost basis. It is never valued at 0.0 and called cheap.

        Returns:
            The summed quantity of the unpriced tranches
        """
        return max(self.size - self.priced_size, 0.0)

    @property
    def is_fully_priced(self) -> bool:
        """Whether every unit held has a known acquisition price."""
        return self.priced_size > DUST_SIZE and self.unpriced_size <= DUST_SIZE

    @property
    def notional(self) -> float:
        """Cost value of the holding at its average entry price."""
        return abs(self.size) * self.entry_price

    @property
    def priced_notional(self) -> float:
        """Cost value of the priced tranches only."""
        return self.priced_size * self.entry_price

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
        because the true figure is not knowable from a balance. That includes a
        partly priced holding, whose total is not knowable either.

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

    # ------------------------------------------------------------------
    # Tranche bookkeeping
    # ------------------------------------------------------------------

    def add_tranche(
        self,
        quantity: float,
        price: Optional[float] = None
    ) -> Tranche:
        """Append a tranche to the back of the queue.

        A price of None, of zero or of anything negative is recorded as
        unpriced rather than as a cost of 0.0.

        Args:
            quantity: Quantity acquired, positive
            price: Price paid, or None when the quantity was adopted

        Returns:
            The tranche that was appended
        """
        usable_price = price if (price is not None and price > 0) else None
        tranche = Tranche(quantity=quantity, price=usable_price)
        self.tranches.append(tranche)
        self._sync()
        return tranche

    def consume(self, quantity: float) -> List[Tranche]:
        """Remove `quantity` from the holding, oldest tranche first.

        The FIFO order is the documented convention for this book. A tranche
        that is only partly consumed is split: the consumed part is returned
        and the remainder stays at the front of the queue at its original
        price, because a sale never re-prices what is still held.

        Args:
            quantity: Quantity to remove, positive. Clamped to the quantity
                actually held, since spot cannot sell what it does not have

        Returns:
            The tranches consumed, in the order they were consumed
        """
        taken: List[Tranche] = []
        remaining = min(quantity, self.size)

        while remaining > DUST_SIZE and self.tranches:
            head = self.tranches[0]
            if head.quantity - remaining <= DUST_SIZE:
                taken.append(head)
                remaining -= head.quantity
                self.tranches.pop(0)
                continue

            taken.append(
                Tranche(
                    quantity=remaining,
                    price=head.price,
                    acquired_at=head.acquired_at
                )
            )
            head.quantity -= remaining
            remaining = 0.0

        self._sync()
        return taken

    def _sync(self) -> None:
        """Recompute the flat fields from the tranche queue.

        `size` becomes the total quantity held, `entry_price` the
        quantity-weighted average of the PRICED tranches only, and
        `entry_price_known` is True only when nothing unpriced is left.
        """
        self.tranches = [t for t in self.tranches if t.quantity > DUST_SIZE]

        if not self.tranches:
            self.size = 0.0
            self.entry_price = 0.0
            self.metadata['entry_price_known'] = False
            self.last_update = datetime.now()
            return

        self.size = sum(t.quantity for t in self.tranches)
        priced_quantity = sum(t.quantity for t in self.tranches if t.is_priced)

        if priced_quantity > DUST_SIZE:
            total_cost = sum(t.cost for t in self.tranches)
            self.entry_price = total_cost / priced_quantity
        else:
            self.entry_price = 0.0

        self.metadata['entry_price_known'] = (
            priced_quantity > DUST_SIZE
            and (self.size - priced_quantity) <= DUST_SIZE
        )
        self.last_update = datetime.now()


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

        A positive `size` is a purchase and appends a tranche at the price
        paid, which moves the average entry price towards it. A negative `size`
        is a reduction and consumes tranches FIFO without re-pricing anything
        that remains: inventory that leaves the book does not change what the
        remaining inventory cost.

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

        A caller may state in the metadata that the price is not to be trusted
        by passing `entry_price_known=False`, in which case the opening tranche
        is recorded as unpriced.

        Args:
            pair: Trading pair
            size: Quantity bought, positive
            entry_price: Price paid
            metadata: Additional metadata

        Returns:
            The newly created holding
        """
        meta = dict(metadata or {})
        declared = meta.get('entry_price_known')
        price = entry_price if declared is not False else None

        position = ManagedPosition(
            pair=pair,
            size=0.0,
            entry_price=0.0,
            current_price=entry_price if entry_price > 0 else 0.0,
            unrealized_pnl=0.0,
            metadata=meta
        )
        position.add_tranche(size, price)
        self.positions[pair] = position

        self._log_position(position)
        return position

    def _increase(
        self,
        position: ManagedPosition,
        size: float,
        entry_price: float
    ) -> None:
        """Add to a holding as a new tranche and re-average the priced part.

        A purchase with no usable price is recorded as an unpriced tranche. It
        does not destroy the basis of the tranches already held, but it does
        leave the holding only partly priced, which is reported as having no
        usable basis until that quantity leaves.

        Args:
            position: Holding to add to
            size: Quantity bought, positive
            entry_price: Price paid
        """
        position.add_tranche(size, entry_price)

        if entry_price <= 0:
            self.logger.logger.warning(
                "cost_basis_unknown_after_purchase",
                pair=position.pair,
                size=position.size,
                unpriced=position.unpriced_size,
                message=(
                    "Bought into a holding at no usable price; that quantity "
                    "is recorded as unpriced"
                )
            )
        elif not position.entry_price_known:
            self.logger.logger.info(
                "holding_partly_unpriced",
                pair=position.pair,
                size=position.size,
                priced=position.priced_size,
                unpriced=position.unpriced_size,
                message=(
                    "Purchase recorded, but the holding still contains "
                    "unpriced quantity; no basis is reported for it"
                )
            )

        self._log_position(position)

    def _reduce(self, position: ManagedPosition, quantity: float) -> None:
        """Remove inventory from a holding, consuming tranches FIFO.

        The tranches that remain keep the prices they were acquired at, so the
        sale price is never blended into the cost of what is still held.

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

        position.consume(quantity)

    def _retire(self, pair: str) -> None:
        """Move a fully closed holding out of the open book.

        Args:
            pair: Trading pair

        Returns:
            None, since nothing is held any more
        """
        closed = self.positions.pop(pair, None)
        if closed is not None:
            closed.tranches = []
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
        cost. Surplus inventory therefore arrives as an UNPRICED tranche. The
        tranches already recorded from fills keep their prices, so only the
        adopted quantity is basis-unknown; the holding as a whole is still
        reported as having no usable basis while that quantity is present,
        which stops profit-taking on the pair. That is the safe direction to
        fail, and the priced part no longer has to be rebuilt from scratch once
        the adopted quantity is gone.

        A shortfall means inventory left by some route this system did not see.
        The quantity is reduced FIFO and the remaining tranches keep their
        prices, as with any other sale.

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
                size=0.0,
                entry_price=0.0,
                current_price=0.0,
                unrealized_pnl=0.0,
                metadata=meta
            )
            position.add_tranche(exchange_size, None)
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
            existing.add_tranche(difference, None)
            self.logger.logger.warning(
                "surplus_inventory_without_basis",
                pair=pair,
                surplus=difference,
                size=existing.size,
                priced=existing.priced_size,
                unpriced=existing.unpriced_size,
                message=(
                    "More inventory held than this system bought; the surplus "
                    "is adopted as an unpriced tranche and no basis is "
                    "reported until it leaves"
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
        existing.consume(abs(difference))
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
        A close larger than the holding is clamped: spot cannot sell what it
        does not hold.

        Tranches are consumed FIFO and the realised P&L is computed against the
        tranches actually consumed, not against the blended average. A sale
        that runs past the priced tranches into unpriced quantity realises
        nothing for that part: no cost was ever recorded for it, so no honest
        gain or loss can be attributed to it, and it is reported instead.

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

        consumed = position.consume(close_size)

        realized_pnl = 0.0
        unpriced_quantity = 0.0
        for tranche in consumed:
            if tranche.is_priced:
                realized_pnl += (exit_price - tranche.price) * tranche.quantity
            else:
                unpriced_quantity += tranche.quantity

        if unpriced_quantity > DUST_SIZE:
            # No basis means no honest P&L figure. Recording zero for that part
            # is not a claim that it broke even, it is a refusal to invent one.
            self.logger.logger.warning(
                "realised_pnl_unknown",
                pair=pair,
                quantity=unpriced_quantity,
                message=(
                    "Sold quantity with no cost basis; P&L not derivable for "
                    "that part of the sale"
                )
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

        if position.size <= DUST_SIZE:
            closed = self.positions.pop(pair)
            closed.tranches = []
            closed.size = 0.0
            closed.unrealized_pnl = 0.0
            self.closed_positions.append(closed)
            return closed

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
        raises ZeroDivisionError. Such holdings are logged plainly instead,
        with the split between priced and unpriced quantity.

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
            priced_size=position.priced_size,
            unpriced_size=position.unpriced_size,
            current_price=position.current_price
        )
