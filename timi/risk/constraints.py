"""Risk constraints and validation."""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class RiskConstraints:
    """Risk constraints for parameter optimization.

    Implements constraint system from paper: A(R)Θ ≤ b(R)
    """
    # Position size constraints
    max_total_position_value: float = 10000
    max_single_position_value: float = 1000

    # Risk limits
    max_leverage: float = 3.0
    max_loss_per_trade: float = 100

    # Frequency limits
    max_trades_per_hour: int = 100
    max_trades_per_day: int = 1000

    def validate_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Validate parameters against constraints.

        Args:
            parameters: Trading parameters

        Returns:
            List of constraint violations (empty if valid)
        """
        violations = []

        # Check capital allocation
        capital = parameters.get('capital_allocation', 0)
        if capital > self.max_single_position_value:
            violations.append(
                f"Capital allocation {capital} exceeds limit {self.max_single_position_value}"
            )

        # Check grid levels
        grid_levels = parameters.get('grid_levels', 0)
        if grid_levels > 20:
            violations.append(f"Too many grid levels: {grid_levels} (max 20)")

        return violations

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Constraint dictionary
        """
        return {
            'max_total_position_value': self.max_total_position_value,
            'max_single_position_value': self.max_single_position_value,
            'max_leverage': self.max_leverage,
            'max_loss_per_trade': self.max_loss_per_trade,
            'max_trades_per_hour': self.max_trades_per_hour,
            'max_trades_per_day': self.max_trades_per_day
        }
