"""Risk management system with safety controls.

Nothing in here protects anything unless it is called, and the previous version
was never called from the order path. Every check below is now reachable from
`RiskManager.check_order_risk`, which the engine invokes before any order is
transmitted, and the engine refuses to trade at all while `emergency_stop` is
set.

Three properties are worth stating because they were the defects:

* The emergency stop is honoured from configuration, so `EMERGENCY_STOP=true`
  in the environment actually stops trading. It is also raised by a drawdown
  breach, which requires the drawdown check to be called; `update_capital`
  does that on every cycle.
* `peak_capital` moves. It is updated whenever capital is marked, not only
  inside a method nothing called, so drawdown is measured against a real peak.
* Exposure is aggregated. Individual orders that each sit under the
  per-position limit can add up past it, and a grid places seven levels at a
  time, so approved-but-unfilled exposure is reserved and counted.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum

from ..core.position_manager import PositionManager
from ..utils.config import Config
from ..utils.logging import TradingLogger
from .constraints import RiskConstraints, RiskConstraintError


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

        # Aggregate exposure ceiling, as a fraction of initial capital. The
        # default is what the per-position limit and the position count imply,
        # capped at the whole account: without it, five orders at ten percent
        # each pass individually and put half the account at risk together.
        implied = self.max_position_pct * max(self.max_concurrent_positions, 1)
        configured = config.get('risk.max_total_exposure_pct', None)
        if configured is None:
            self.max_total_exposure_pct = min(implied, 1.0)
        else:
            self.max_total_exposure_pct = float(configured) / 100

        self.constraints = RiskConstraints.from_config(config)

        # State tracking
        self.initial_capital: Optional[float] = None
        self.peak_capital: Optional[float] = None
        self.violations: List[RiskViolation] = []

        # Honour a stop that was requested before the process started. The
        # environment override lands in the config as `emergency_stop`.
        self.emergency_stop = bool(config.get('emergency_stop', False))
        if self.emergency_stop:
            self.logger.log_risk_event(
                event_type="emergency_stop",
                severity="critical",
                message="Emergency stop is set in configuration; trading is halted"
            )

        # Approved but not yet resolved order value, keyed by reservation.
        self._reserved_exposure: Dict[str, float] = {}

        # Timestamps of executed trades, for the frequency limits.
        self._trade_times: List[datetime] = []

    # ------------------------------------------------------------------
    # Capital tracking
    # ------------------------------------------------------------------

    def initialize_capital(self, capital: float) -> None:
        """Initialize capital tracking.

        Args:
            capital: Initial capital
        """
        self.initial_capital = capital
        self.peak_capital = capital
        self.logger.logger.info("Risk manager initialized", initial_capital=capital)

    def update_capital(self, current_capital: Optional[float] = None) -> bool:
        """Mark capital, move the peak, and test the drawdown limit.

        This is the call that makes the drawdown limit real. It is expected on
        every trading cycle.

        Args:
            current_capital: Current total capital. When None it is derived
                from the initial capital plus tracked P&L

        Returns:
            True if trading may continue
        """
        if self.initial_capital is None:
            return True

        if current_capital is None:
            current_capital = self.current_capital()

        if not math.isfinite(current_capital):
            self._log_violation(
                "capital_not_finite",
                RiskLevel.CRITICAL,
                f"Capital marked as {current_capital}; refusing to trade on it",
                0.0,
                0.0
            )
            self.halt("capital could not be marked")
            return False

        if self.peak_capital is None or current_capital > self.peak_capital:
            self.peak_capital = current_capital

        return self.check_drawdown(current_capital)

    def current_capital(self) -> float:
        """Best estimate of current capital from tracked P&L.

        Returns:
            Initial capital adjusted by realised and unrealised P&L
        """
        base = self.initial_capital or 0.0
        return (
            base
            + self.position_manager.get_total_pnl()
            + self.position_manager.get_total_realized_pnl()
        )

    # ------------------------------------------------------------------
    # Order gate
    # ------------------------------------------------------------------

    def check_order_risk(
        self,
        pair: str,
        order_value: float,
        side: str,
        order_price: Optional[float] = None,
        market_price: Optional[float] = None,
        price_tolerance: Optional[float] = None,
        reserve_key: Optional[str] = None
    ) -> bool:
        """Check whether an order may be transmitted.

        Every order goes through here. A buy consumes capital and is checked
        against the position count, the per-order size limit and the aggregate
        exposure ceiling. A sell releases inventory rather than committing
        capital, so the sizing limits do not apply to it, but the emergency
        stop, the frequency limits and the price sanity check do.

        Args:
            pair: Trading pair
            order_value: Order value in quote currency
            side: Order side, 'buy' or 'sell'
            order_price: Intended order price, when there is one
            market_price: Reference market price, when known
            price_tolerance: Override for the price deviation limit, as a
                fraction. Resting grid orders sit deliberately far from the
                market, so they pass their own wider bound
            reserve_key: When given and the order is approved, the value is
                reserved under this key so it counts towards aggregate
                exposure until released

        Returns:
            True if the order is allowed
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

        if not math.isfinite(order_value) or order_value < 0:
            self._log_violation(
                "order_value_invalid",
                RiskLevel.CRITICAL,
                f"Order value {order_value!r} for {pair} is not a usable number",
                0.0,
                0.0
            )
            return False

        if not self.check_trade_frequency():
            return False

        if order_price is not None and market_price is not None:
            if not self.check_price_deviation(
                pair, order_price, market_price, tolerance=price_tolerance
            ):
                return False

        is_buy = str(side).lower().endswith('buy')

        if not is_buy:
            # Selling held inventory reduces exposure. Nothing below applies.
            return True

        # Position count limit. A pair already held does not consume a new slot.
        open_pairs = {pos.pair for pos in self.position_manager.get_all_positions()}
        if pair not in open_pairs and len(open_pairs) >= self.max_concurrent_positions:
            self._log_violation(
                "max_positions",
                RiskLevel.WARNING,
                f"Maximum concurrent positions reached: {len(open_pairs)}",
                len(open_pairs),
                self.max_concurrent_positions
            )
            return False

        # Per-order size limit. Capital that is unset or zero fails closed:
        # a size limit cannot be evaluated against nothing, and an unevaluated
        # limit must not read as a pass.
        if self.initial_capital is None:
            self._log_violation(
                "capital_uninitialised",
                RiskLevel.CRITICAL,
                "Capital has not been initialised; cannot size an order",
                0.0,
                0.0
            )
            return False

        if self.initial_capital <= 0:
            self._log_violation(
                "capital_exhausted",
                RiskLevel.CRITICAL,
                f"Capital is {self.initial_capital}; no order can be sized",
                self.initial_capital,
                0.0
            )
            return False

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

        # Aggregate exposure across everything already approved or held.
        if not self._check_aggregate_exposure(pair, order_value):
            return False

        if reserve_key is not None:
            self.reserve_exposure(reserve_key, order_value)

        return True

    def _check_aggregate_exposure(self, pair: str, order_value: float) -> bool:
        """Check total committed value against the aggregate ceiling.

        Args:
            pair: Trading pair
            order_value: Value of the order under consideration

        Returns:
            True if the order keeps total exposure within limits
        """
        projected = self.total_exposure() + order_value
        limit = min(
            self.initial_capital * self.max_total_exposure_pct,
            self.constraints.max_total_position_value
        )

        if projected > limit:
            self._log_violation(
                "aggregate_exposure",
                RiskLevel.WARNING,
                f"Total exposure {projected:.2f} for {pair} would exceed "
                f"the ceiling of {limit:.2f}",
                projected,
                limit
            )
            return False

        return True

    def total_exposure(self) -> float:
        """Total committed value: held inventory plus reserved order value.

        Returns:
            Exposure in quote currency
        """
        held = sum(
            pos.notional for pos in self.position_manager.get_all_positions()
        )
        reserved = sum(self._reserved_exposure.values())
        return held + reserved

    def reserve_exposure(self, key: str, value: float) -> None:
        """Reserve order value against the aggregate ceiling.

        Args:
            key: Reservation key, normally an order or grid level identifier
            value: Value in quote currency
        """
        self._reserved_exposure[key] = value

    def release_exposure(self, key: str) -> None:
        """Release a reservation once the order is filled or cancelled.

        A filled order becomes held inventory, which `total_exposure` counts
        through the position manager, so the reservation must go either way.

        Args:
            key: Reservation key
        """
        self._reserved_exposure.pop(key, None)

    def release_all_exposure(self) -> None:
        """Drop every outstanding reservation, as on shutdown."""
        self._reserved_exposure.clear()

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_drawdown(self, current_capital: float) -> bool:
        """Check drawdown limits and raise the emergency stop on a breach.

        Args:
            current_capital: Current total capital

        Returns:
            True if within limits
        """
        if self.initial_capital is None or not self.peak_capital:
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
            self.halt(f"drawdown of {drawdown:.2%}")
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
        """Check position-specific risk, principally the stop loss.

        A holding whose entry price is unknown cannot be stopped out, because
        there is no reference to measure the loss from. Such a holding returns
        True here and is reported by the engine instead.

        Args:
            pair: Trading pair
            entry_price: Entry price
            current_price: Current price
            position_size: Position size

        Returns:
            True if within risk limits, False when the stop is breached
        """
        if entry_price <= 0 or not math.isfinite(entry_price):
            return True
        if not math.isfinite(current_price) or current_price <= 0:
            return True

        pnl_pct = abs((current_price - entry_price) / entry_price)

        # Spot inventory is long. The short branch is unreachable here and is
        # kept only so a negative size cannot slip past unchecked.
        if position_size >= 0:
            breached = current_price < entry_price * (1 - self.stop_loss_pct)
        else:
            breached = current_price > entry_price * (1 + self.stop_loss_pct)

        if breached:
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
        market_price: float,
        tolerance: Optional[float] = None
    ) -> bool:
        """Check that an order price is not absurd relative to the market.

        The default tolerance is the configured deviation limit, which suits an
        order intended to execute now. A resting grid level is deliberately far
        from the market and passes its own wider bound; the check still catches
        a price built from a corrupted volatility or a stale reference, which
        is the failure it exists for.

        Args:
            pair: Trading pair
            order_price: Intended order price
            market_price: Current market price
            tolerance: Override deviation limit, as a fraction

        Returns:
            True if deviation is acceptable
        """
        if market_price == 0 or not math.isfinite(market_price):
            return True

        if not math.isfinite(order_price) or order_price <= 0:
            self._log_violation(
                "price_invalid",
                RiskLevel.CRITICAL,
                f"Order price {order_price!r} for {pair} is not usable",
                0.0,
                0.0
            )
            return False

        limit = self.max_price_deviation if tolerance is None else tolerance
        deviation = abs((order_price - market_price) / market_price)

        if deviation > limit:
            self._log_violation(
                "price_deviation",
                RiskLevel.WARNING,
                f"Price deviation too high for {pair}: {deviation:.2%}",
                deviation,
                limit
            )
            return False

        return True

    def check_trade_frequency(self, now: Optional[datetime] = None) -> bool:
        """Check the hourly and daily trade counts.

        Args:
            now: Current time, injectable for testing

        Returns:
            True if another trade is allowed
        """
        reference = now or datetime.now()
        day_ago = reference - timedelta(days=1)
        hour_ago = reference - timedelta(hours=1)

        # Trim anything older than the longest window while we are here.
        self._trade_times = [t for t in self._trade_times if t > day_ago]

        in_hour = sum(1 for t in self._trade_times if t > hour_ago)
        if in_hour >= self.constraints.max_trades_per_hour:
            self._log_violation(
                "trades_per_hour",
                RiskLevel.WARNING,
                f"Hourly trade limit reached: {in_hour}",
                in_hour,
                self.constraints.max_trades_per_hour
            )
            return False

        in_day = len(self._trade_times)
        if in_day >= self.constraints.max_trades_per_day:
            self._log_violation(
                "trades_per_day",
                RiskLevel.WARNING,
                f"Daily trade limit reached: {in_day}",
                in_day,
                self.constraints.max_trades_per_day
            )
            return False

        return True

    def record_trade(self, now: Optional[datetime] = None) -> None:
        """Record an executed trade against the frequency limits.

        Args:
            now: Time of the trade, injectable for testing
        """
        self._trade_times.append(now or datetime.now())

    def check_trade_loss(self, pair: str, loss: float) -> bool:
        """Check a realised loss against the per-trade limit.

        Args:
            pair: Trading pair
            loss: Realised P&L. Negative values are losses

        Returns:
            True if the loss is within the configured limit
        """
        if not math.isfinite(loss) or loss >= 0:
            return True

        magnitude = abs(loss)
        if magnitude > self.constraints.max_loss_per_trade:
            self._log_violation(
                "max_loss_per_trade",
                RiskLevel.CRITICAL,
                f"Loss of {magnitude:.2f} on {pair} exceeds the per-trade "
                f"limit of {self.constraints.max_loss_per_trade}",
                magnitude,
                self.constraints.max_loss_per_trade
            )
            return False

        return True

    def validate_capital_allocation(self, value) -> float:
        """Bound a capital allocation supplied from outside the system.

        Args:
            value: Requested allocation

        Returns:
            An allocation within the configured band

        Raises:
            RiskConstraintError: The value cannot be made safe
        """
        try:
            return self.constraints.validate_capital_allocation(value)
        except RiskConstraintError as e:
            self._log_violation(
                "capital_allocation",
                RiskLevel.CRITICAL,
                str(e),
                0.0,
                self.constraints.max_single_position_value
            )
            raise

    # ------------------------------------------------------------------
    # Emergency stop
    # ------------------------------------------------------------------

    def halt(self, reason: str) -> None:
        """Raise the emergency stop.

        Args:
            reason: Why trading is being halted
        """
        if self.emergency_stop:
            return

        self.emergency_stop = True
        self.logger.log_risk_event(
            event_type="emergency_stop",
            severity="critical",
            message=f"Emergency stop triggered: {reason}"
        )

    def reset_emergency_stop(self) -> None:
        """Reset emergency stop (use with caution)."""
        self.logger.logger.warning("Emergency stop reset manually")
        self.emergency_stop = False

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_risk_report(self) -> Dict:
        """Generate risk report.

        Returns:
            Risk report dictionary
        """
        current_capital = None
        if self.initial_capital is not None:
            current_capital = self.current_capital()

        drawdown = 0
        if self.peak_capital and current_capital is not None:
            drawdown = (self.peak_capital - current_capital) / self.peak_capital

        return {
            'initial_capital': self.initial_capital,
            'current_capital': current_capital,
            'peak_capital': self.peak_capital,
            'current_drawdown': drawdown,
            'max_drawdown_limit': self.max_drawdown,
            'open_positions': len(self.position_manager.get_all_positions()),
            'max_positions_limit': self.max_concurrent_positions,
            'total_exposure': self.total_exposure(),
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
    ) -> None:
        """Log a risk violation.

        Args:
            violation_type: Type of violation
            level: Risk level
            message: Violation message
            value: Actual value
            limit: Limit value
        """
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

    def get_recent_violations(self, count: int = 10) -> List[Dict]:
        """Get recent risk violations.

        Args:
            count: Number of violations to return

        Returns:
            List of violation dictionaries
        """
        return [v.to_dict() for v in self.violations[-count:]]
