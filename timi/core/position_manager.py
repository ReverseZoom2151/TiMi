"""Position management for tracking and monitoring open positions."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..exchange.base import Position
from ..utils.logging import TradingLogger


@dataclass
class ManagedPosition:
    """Extended position with tracking information."""
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
        """Check if position is long."""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.size < 0

    @property
    def pnl_percentage(self) -> float:
        """Get PnL as percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.unrealized_pnl / (abs(self.size) * self.entry_price)) * 100

    def update_price(self, current_price: float):
        """Update current price and recalculate PnL."""
        self.current_price = current_price
        self.last_update = datetime.now()

        # Calculate unrealized PnL
        if self.is_long:
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * abs(self.size)


class PositionManager:
    """Manager for tracking and monitoring positions."""

    def __init__(self):
        """Initialize position manager."""
        self.positions: Dict[str, ManagedPosition] = {}
        self.logger = TradingLogger("position_manager")
        self.closed_positions: List[ManagedPosition] = []

    def add_position(
        self,
        pair: str,
        size: float,
        entry_price: float,
        metadata: Optional[Dict] = None
    ) -> ManagedPosition:
        """Add or update a position.

        Args:
            pair: Trading pair
            size: Position size (positive for long, negative for short)
            entry_price: Entry price
            metadata: Additional metadata

        Returns:
            Managed position
        """
        if pair in self.positions:
            # Update existing position
            existing = self.positions[pair]
            existing.size += size
            # Recalculate average entry price
            total_cost = (existing.entry_price * (existing.size - size)) + (entry_price * size)
            existing.entry_price = total_cost / existing.size if existing.size != 0 else entry_price
        else:
            # Create new position
            position = ManagedPosition(
                pair=pair,
                size=size,
                entry_price=entry_price,
                current_price=entry_price,
                unrealized_pnl=0.0,
                metadata=metadata or {}
            )
            self.positions[pair] = position

            self.logger.log_position(
                pair=pair,
                size=size,
                entry_price=entry_price,
                current_price=entry_price,
                pnl=0.0
            )

        return self.positions[pair]

    def update_position_price(self, pair: str, current_price: float):
        """Update position with current price.

        Args:
            pair: Trading pair
            current_price: Current market price
        """
        if pair in self.positions:
            self.positions[pair].update_price(current_price)

            self.logger.log_position(
                pair=pair,
                size=self.positions[pair].size,
                entry_price=self.positions[pair].entry_price,
                current_price=current_price,
                pnl=self.positions[pair].unrealized_pnl
            )

    def close_position(
        self,
        pair: str,
        exit_price: float,
        size: Optional[float] = None
    ) -> Optional[ManagedPosition]:
        """Close a position (fully or partially).

        Args:
            pair: Trading pair
            exit_price: Exit price
            size: Size to close (None for full close)

        Returns:
            Closed position or None if not found
        """
        if pair not in self.positions:
            return None

        position = self.positions[pair]

        # Calculate realized PnL
        close_size = size if size is not None else position.size
        if position.is_long:
            realized_pnl = (exit_price - position.entry_price) * close_size
        else:
            realized_pnl = (position.entry_price - exit_price) * abs(close_size)

        position.realized_pnl += realized_pnl

        self.logger.log_trade(
            action="CLOSE",
            pair=pair,
            price=exit_price,
            quantity=abs(close_size),
            pnl=realized_pnl
        )

        # Update or remove position
        if size is None or abs(size) >= abs(position.size):
            # Full close
            closed = self.positions.pop(pair)
            self.closed_positions.append(closed)
            return closed
        else:
            # Partial close
            position.size -= close_size
            return position

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
        """Get total realized PnL from closed positions.

        Returns:
            Total realized PnL
        """
        return sum(pos.realized_pnl for pos in self.closed_positions)

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
