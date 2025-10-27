"""Order management for tracking and managing orders."""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

from ..exchange.base import Order, OrderStatus, OrderSide, OrderType
from ..utils.logging import TradingLogger


class OrderManager:
    """Manager for tracking and managing orders."""

    def __init__(self):
        """Initialize order manager."""
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: List[Order] = []
        self.logger = TradingLogger("order_manager")

    def add_order(self, order: Order):
        """Add an order to tracking.

        Args:
            order: Order to track
        """
        self.active_orders[order.id] = order

        self.logger.log_trade(
            action=f"ORDER_{order.side.value}",
            pair=order.pair,
            price=order.price,
            quantity=order.quantity,
            order_id=order.id
        )

    def update_order(self, order: Order):
        """Update order status.

        Args:
            order: Updated order
        """
        if order.id in self.active_orders:
            self.active_orders[order.id] = order

            # Move to completed if filled or canceled
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED]:
                self.completed_orders.append(order)
                del self.active_orders[order.id]

                self.logger.logger.info(
                    "order_completed",
                    order_id=order.id,
                    status=order.status.value,
                    filled=order.filled_quantity
                )

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order or None if not found
        """
        return self.active_orders.get(order_id)

    def get_orders_by_pair(self, pair: str) -> List[Order]:
        """Get all active orders for a pair.

        Args:
            pair: Trading pair

        Returns:
            List of orders
        """
        return [
            order for order in self.active_orders.values()
            if order.pair == pair
        ]

    def get_all_active_orders(self) -> List[Order]:
        """Get all active orders.

        Returns:
            List of active orders
        """
        return list(self.active_orders.values())

    def get_statistics(self) -> Dict:
        """Get order statistics.

        Returns:
            Statistics dictionary
        """
        active_count = len(self.active_orders)
        completed_count = len(self.completed_orders)

        filled_orders = sum(
            1 for order in self.completed_orders
            if order.status == OrderStatus.FILLED
        )

        canceled_orders = sum(
            1 for order in self.completed_orders
            if order.status == OrderStatus.CANCELED
        )

        return {
            'active_orders': active_count,
            'completed_orders': completed_count,
            'filled_orders': filled_orders,
            'canceled_orders': canceled_orders,
            'fill_rate': filled_orders / completed_count if completed_count > 0 else 0
        }
