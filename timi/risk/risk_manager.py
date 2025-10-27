"""Risk management system with safety controls."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

from ..core.position_manager import PositionManager
from ..utils.config import Config
from ..utils.logging import TradingLogger


class RiskLevel(Enum):
    """Risk severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class RiskViolation:
    """Risk violation record."""
    type: str
    level: RiskLevel
    message: str
    value: float
    limit: float
    timestamp: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'type': self.type,
            'level': self.level.value,
            'message': self.message,
            'value': self.value,
            'limit': self.limit,
            'timestamp': self.timestamp
        }


class RiskManager:
    """Comprehensive risk management system."""

    def __init__(
        self,
        config: Config,
        position_manager: PositionManager
    ):
        """Initialize risk manager.

        Args:
            config: System configuration
            position_manager: Position manager instance
        """
        self.config = config
        self.position_manager = position_manager
        self.logger = TradingLogger("risk_manager")

        # Risk limits
        self.max_drawdown = config.risk.max_drawdown / 100  # Convert % to decimal
        self.max_position_pct = config.risk.max_position_pct / 100
        self.max_concurrent_positions = config.risk.max_concurrent_positions
        self.stop_loss_pct = config.risk.stop_loss_pct / 100
        self.max_price_deviation = config.risk.max_price_deviation / 100

        # State tracking
        self.initial_capital: Optional[float] = None
        self.peak_capital: Optional[float] = None
        self.violations: List[RiskViolation] = []
        self.emergency_stop = False

    def initialize_capital(self, capital: float):
        """Initialize capital tracking.

        Args:
            capital: Initial capital
        """
        self.initial_capital = capital
        self.peak_capital = capital
        self.logger.logger.info("Risk manager initialized", initial_capital=capital)

    def check_order_risk(
        self,
        pair: str,
        order_value: float,
        side: str
    ) -> bool:
        """Check if an order passes risk checks.

        Args:
            pair: Trading pair
            order_value: Order value in quote currency
            side: Order side (buy/sell)

        Returns:
            True if order is allowed
        """
        if self.emergency_stop:
            self._log_violation(
                "emergency_stop",
                RiskLevel.EMERGENCY,
                "Emergency stop activated - no trading allowed",
                1.0,
                0.0
            )
            return False

        # Check position count limit
        open_positions = len(self.position_manager.get_all_positions())
        if open_positions >= self.max_concurrent_positions:
            self._log_violation(
                "max_positions",
                RiskLevel.WARNING,
                f"Maximum concurrent positions reached: {open_positions}",
                open_positions,
                self.max_concurrent_positions
            )
            return False

        # Check position size limit
        if self.initial_capital:
            position_pct = order_value / self.initial_capital
            if position_pct > self.max_position_pct:
                self._log_violation(
                    "position_size",
                    RiskLevel.WARNING,
                    f"Position size exceeds limit: {position_pct:.2%}",
                    position_pct,
                    self.max_position_pct
                )
                return False

        return True

    def check_drawdown(self, current_capital: float) -> bool:
        """Check drawdown limits.

        Args:
            current_capital: Current total capital

        Returns:
            True if within limits
        """
        if not self.initial_capital or not self.peak_capital:
            return True

        # Update peak
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital

        # Calculate drawdown
        drawdown = (self.peak_capital - current_capital) / self.peak_capital

        if drawdown > self.max_drawdown:
            self._log_violation(
                "max_drawdown",
                RiskLevel.CRITICAL,
                f"Maximum drawdown exceeded: {drawdown:.2%}",
                drawdown,
                self.max_drawdown
            )

            # Trigger emergency stop
            self.emergency_stop = True
            self.logger.log_risk_event(
                event_type="emergency_stop",
                severity="critical",
                message=f"Emergency stop triggered due to drawdown: {drawdown:.2%}"
            )

            return False

        # Warning at 75% of limit
        if drawdown > self.max_drawdown * 0.75:
            self._log_violation(
                "drawdown_warning",
                RiskLevel.WARNING,
                f"Approaching drawdown limit: {drawdown:.2%}",
                drawdown,
                self.max_drawdown
            )

        return True

    def check_position_risk(
        self,
        pair: str,
        entry_price: float,
        current_price: float,
        position_size: float
    ) -> bool:
        """Check position-specific risk.

        Args:
            pair: Trading pair
            entry_price: Entry price
            current_price: Current price
            position_size: Position size

        Returns:
            True if within risk limits
        """
        # Calculate position PnL percentage
        if entry_price == 0:
            return True

        pnl_pct = abs((current_price - entry_price) / entry_price)

        # Check stop loss
        if position_size > 0:  # Long position
            if current_price < entry_price * (1 - self.stop_loss_pct):
                self._log_violation(
                    "stop_loss",
                    RiskLevel.CRITICAL,
                    f"Stop loss triggered for {pair}: {pnl_pct:.2%}",
                    pnl_pct,
                    self.stop_loss_pct
                )
                return False
        else:  # Short position
            if current_price > entry_price * (1 + self.stop_loss_pct):
                self._log_violation(
                    "stop_loss",
                    RiskLevel.CRITICAL,
                    f"Stop loss triggered for {pair}: {pnl_pct:.2%}",
                    pnl_pct,
                    self.stop_loss_pct
                )
                return False

        return True

    def check_price_deviation(
        self,
        pair: str,
        order_price: float,
        market_price: float
    ) -> bool:
        """Check if order price deviates too much from market price.

        Args:
            pair: Trading pair
            order_price: Intended order price
            market_price: Current market price

        Returns:
            True if deviation is acceptable
        """
        if market_price == 0:
            return True

        deviation = abs((order_price - market_price) / market_price)

        if deviation > self.max_price_deviation:
            self._log_violation(
                "price_deviation",
                RiskLevel.WARNING,
                f"Price deviation too high for {pair}: {deviation:.2%}",
                deviation,
                self.max_price_deviation
            )
            return False

        return True

    def get_risk_report(self) -> Dict:
        """Generate risk report.

        Returns:
            Risk report dictionary
        """
        current_capital = self.initial_capital
        if current_capital and self.position_manager:
            current_capital += self.position_manager.get_total_pnl()

        drawdown = 0
        if self.peak_capital and current_capital:
            drawdown = (self.peak_capital - current_capital) / self.peak_capital

        return {
            'initial_capital': self.initial_capital,
            'current_capital': current_capital,
            'peak_capital': self.peak_capital,
            'current_drawdown': drawdown,
            'max_drawdown_limit': self.max_drawdown,
            'open_positions': len(self.position_manager.get_all_positions()),
            'max_positions_limit': self.max_concurrent_positions,
            'emergency_stop': self.emergency_stop,
            'violations_count': len(self.violations),
            'recent_violations': [v.to_dict() for v in self.violations[-10:]]
        }

    def _log_violation(
        self,
        violation_type: str,
        level: RiskLevel,
        message: str,
        value: float,
        limit: float
    ):
        """Log a risk violation.

        Args:
            violation_type: Type of violation
            level: Risk level
            message: Violation message
            value: Actual value
            limit: Limit value
        """
        from datetime import datetime

        violation = RiskViolation(
            type=violation_type,
            level=level,
            message=message,
            value=value,
            limit=limit,
            timestamp=datetime.now().isoformat()
        )

        self.violations.append(violation)

        self.logger.log_risk_event(
            event_type=violation_type,
            severity=level.value,
            message=message,
            value=value,
            limit=limit
        )

    def reset_emergency_stop(self):
        """Reset emergency stop (use with caution)."""
        self.logger.logger.warning("Emergency stop reset manually")
        self.emergency_stop = False

    def get_recent_violations(self, count: int = 10) -> List[Dict]:
        """Get recent risk violations.

        Args:
            count: Number of violations to return

        Returns:
            List of violation dictionaries
        """
        return [v.to_dict() for v in self.violations[-count:]]
