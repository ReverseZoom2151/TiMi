"""Order management for tracking and managing orders.

The engine needs three things from this class that the previous version did not
offer: a way to find an order again by its client order id, so an order whose
outcome was unknown can be reconciled rather than assumed away; a way to remove
an order that was cancelled; and a record of which grid level an order belongs
to, so the grid can be diffed instead of being torn down and rebuilt.
"""

from typing import Dict, List, Optional

from ..exchange.base import Order, OrderStatus
from ..utils.logging import TradingLogger


class OrderManager:
    """Manager for tracking and managing orders."""

    def __init__(self):
        """Initialize order manager."""
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: List[Order] = []
        self.logger = TradingLogger("order_manager")

    def add_order(self, order: Order) -> None:
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

    def update_order(self, order: Order) -> None:
        """Update order status.

        Args:
            order: Updated order
        """
        if order.id in self.active_orders:
            self.active_orders[order.id] = order

            # Move to completed if filled or canceled
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED,
                                OrderStatus.REJECTED]:
                self.completed_orders.append(order)
                del self.active_orders[order.id]

                self.logger.logger.info(
                    "order_completed",
                    order_id=order.id,
                    status=order.status.value,
                    filled=order.filled_quantity
                )

    def remove_order(
        self,
        order_id: str,
        status: OrderStatus = OrderStatus.CANCELED
    ) -> Optional[Order]:
        """Stop tracking an order and record how it ended.

        Args:
            order_id: Order ID
            status: Terminal status to record

        Returns:
            The order that was removed, or None if it was not tracked
        """
        order = self.active_orders.pop(order_id, None)
        if order is None:
            return None

        order.status = status
        self.completed_orders.append(order)

        self.logger.logger.info(
            "order_completed",
            order_id=order.id,
            status=status.value,
            filled=order.filled_quantity
        )
        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order or None if not found
        """
        return self.active_orders.get(order_id)

    def get_by_client_order_id(self, client_order_id: str) -> Optional[Order]:
        """Find a tracked order by the client order id sent with it.

        Args:
            client_order_id: Deterministic id supplied at placement

        Returns:
            Order or None if no tracked order carries that id
        """
        if not client_order_id:
            return None

        for order in self.active_orders.values():
            if self.client_order_id_of(order) == client_order_id:
                return order
        return None

    @staticmethod
    def client_order_id_of(order: Order) -> Optional[str]:
        """Read the client order id off an order, if it carries one.

        Args:
            order: Order to inspect

        Returns:
            The client order id, or None
        """
        metadata = order.metadata or {}
        return (
            metadata.get('clientOrderId')
            or metadata.get('client_order_id')
            or (metadata.get('info') or {}).get('clientOrderId')
        )

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
