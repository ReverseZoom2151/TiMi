"""Risk constraints and validation.

Every field declared here is evaluated somewhere. A limit that is declared and
never checked is worse than no limit at all, because it reads like protection.

The sizing figures matter more than they look. `capital_allocation` reaches
this module from a language model's JSON, travels through the strategy
adaptation agent and lands in `BotConfig.capital_per_pair`, where it multiplies
every order quantity in the grid. Nothing upstream bounds it. This module is
the boundary where it gets bounded, and it is the last one before an order is
sized.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List


#: A requested allocation more than this multiple of the single-position limit
#: is not clamped, it is rejected. Clamping 10 million dollars down to a
#: thousand would hide the fact that something upstream produced nonsense.
ABSURDITY_FACTOR = 10.0

#: Spot has no leverage. The limit exists so a strategy that asks for leverage
#: is refused rather than silently granted it.
SPOT_MAX_LEVERAGE = 1.0


class RiskConstraintError(ValueError):
    """Raised when a parameter cannot be made safe by clamping."""


@dataclass
class RiskConstraints:
    """Risk constraints for parameter optimization.

    Implements the constraint system from the paper: A(R)Θ <= b(R)
    """
    # Position size constraints, in quote currency
    max_total_position_value: float = 10000
    max_single_position_value: float = 1000
    min_single_position_value: float = 1.0

    # Risk limits. `max_leverage` is fixed at the spot value of 1.0: this
    # system holds inventory outright, so any request above 1.0 is a category
    # error and is reported as a violation rather than quietly honoured.
    max_leverage: float = SPOT_MAX_LEVERAGE
    max_loss_per_trade: float = 100

    # Frequency limits. Enforced by RiskManager, which owns the clock.
    max_trades_per_hour: int = 100
    max_trades_per_day: int = 1000

    # Structural limits
    max_grid_levels: int = 20

    def validate_capital_allocation(self, value: Any) -> float:
        """Bound a requested capital allocation before it can size an order.

        Values inside the sane band pass through. Values above the band but
        still plausible are clamped down to the single-position limit. Values
        that are not numbers, not finite, not positive, or absurdly large are
        rejected outright, because those indicate a broken producer rather than
        an aggressive one.

        Args:
            value: Requested allocation in quote currency, from any source

        Returns:
            An allocation guaranteed to sit within the configured band

        Raises:
            RiskConstraintError: The value cannot be made safe by clamping
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RiskConstraintError(
                f"Capital allocation {value!r} is not a number"
            )

        allocation = float(value)

        if not math.isfinite(allocation):
            raise RiskConstraintError(
                f"Capital allocation {allocation!r} is not finite"
            )

        if allocation <= 0:
            raise RiskConstraintError(
                f"Capital allocation {allocation} is not positive"
            )

        ceiling = self.max_single_position_value * ABSURDITY_FACTOR
        if allocation > ceiling:
            raise RiskConstraintError(
                f"Capital allocation {allocation} exceeds {ceiling}, which is "
                f"{ABSURDITY_FACTOR:g} times the single position limit of "
                f"{self.max_single_position_value}. Refusing to clamp a value "
                f"this far out of range"
            )

        if allocation > self.max_single_position_value:
            return self.max_single_position_value

        if allocation < self.min_single_position_value:
            return self.min_single_position_value

        return allocation

    def validate_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Validate parameters against every declared constraint.

        Args:
            parameters: Trading parameters

        Returns:
            List of constraint violations (empty if valid)
        """
        violations: List[str] = []

        # Capital allocation. An absent allocation is a violation: defaulting
        # it to zero made a config that omitted it pass silently.
        if 'capital_allocation' not in parameters:
            violations.append("Missing required parameter: capital_allocation")
        else:
            try:
                self.validate_capital_allocation(parameters['capital_allocation'])
            except RiskConstraintError as e:
                violations.append(str(e))

        # Grid levels
        grid_levels = parameters.get('grid_levels')
        if grid_levels is not None:
            if not self._is_finite_number(grid_levels) or grid_levels < 1:
                violations.append(f"Invalid grid level count: {grid_levels!r}")
            elif grid_levels > self.max_grid_levels:
                violations.append(
                    f"Too many grid levels: {grid_levels} "
                    f"(max {self.max_grid_levels})"
                )

        # Leverage. Spot only.
        leverage = parameters.get('leverage')
        if leverage is not None:
            if not self._is_finite_number(leverage):
                violations.append(f"Invalid leverage: {leverage!r}")
            elif leverage > self.max_leverage:
                violations.append(
                    f"Leverage {leverage} exceeds limit {self.max_leverage}. "
                    f"This system trades spot and cannot borrow"
                )

        # Loss per trade
        max_loss = parameters.get('max_loss_per_trade')
        if max_loss is not None:
            if not self._is_finite_number(max_loss) or max_loss < 0:
                violations.append(f"Invalid max loss per trade: {max_loss!r}")
            elif max_loss > self.max_loss_per_trade:
                violations.append(
                    f"Loss per trade {max_loss} exceeds limit "
                    f"{self.max_loss_per_trade}"
                )

        # Aggregate position value
        total_value = parameters.get('total_position_value')
        if total_value is not None:
            if not self._is_finite_number(total_value) or total_value < 0:
                violations.append(f"Invalid total position value: {total_value!r}")
            elif total_value > self.max_total_position_value:
                violations.append(
                    f"Total position value {total_value} exceeds limit "
                    f"{self.max_total_position_value}"
                )

        # Frequency
        violations.extend(
            self._check_frequency(parameters, 'trades_per_hour', self.max_trades_per_hour)
        )
        violations.extend(
            self._check_frequency(parameters, 'trades_per_day', self.max_trades_per_day)
        )

        return violations

    def _check_frequency(
        self,
        parameters: Dict[str, Any],
        key: str,
        limit: int
    ) -> List[str]:
        """Check one frequency parameter against its limit.

        Args:
            parameters: Trading parameters
            key: Parameter name
            limit: Configured maximum

        Returns:
            List of violations for this parameter
        """
        value = parameters.get(key)
        if value is None:
            return []
        if not self._is_finite_number(value) or value < 0:
            return [f"Invalid {key}: {value!r}"]
        if value > limit:
            return [f"{key} {value} exceeds limit {limit}"]
        return []

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        """Whether a value is a finite, non-boolean number.

        Args:
            value: Candidate value

        Returns:
            True when the value can be compared against a numeric limit
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return math.isfinite(float(value))

    @classmethod
    def from_config(cls, config: Any) -> 'RiskConstraints':
        """Build constraints from the system configuration.

        Missing keys fall back to the dataclass defaults. `max_leverage` is
        never read from configuration: on spot it is fixed.

        Args:
            config: System configuration exposing a dotted `get`

        Returns:
            Constraints for this run
        """
        get = getattr(config, 'get', None)

        def _read(key: str, default: Any) -> Any:
            if get is None:
                return default
            value = get(f'risk.constraints.{key}', None)
            return default if value is None else value

        return cls(
            max_total_position_value=float(_read('max_total_position_value', 10000)),
            max_single_position_value=float(_read('max_single_position_value', 1000)),
            min_single_position_value=float(_read('min_single_position_value', 1.0)),
            max_leverage=SPOT_MAX_LEVERAGE,
            max_loss_per_trade=float(_read('max_loss_per_trade', 100)),
            max_trades_per_hour=int(_read('max_trades_per_hour', 100)),
            max_trades_per_day=int(_read('max_trades_per_day', 1000)),
            max_grid_levels=int(_read('max_grid_levels', 20)),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Constraint dictionary
        """
        return {
            'max_total_position_value': self.max_total_position_value,
            'max_single_position_value': self.max_single_position_value,
            'min_single_position_value': self.min_single_position_value,
            'max_leverage': self.max_leverage,
            'max_loss_per_trade': self.max_loss_per_trade,
            'max_trades_per_hour': self.max_trades_per_hour,
            'max_trades_per_day': self.max_trades_per_day,
            'max_grid_levels': self.max_grid_levels
        }
