"""Tests for the risk layer.

The whole layer used to be unreachable: `check_order_risk`, `check_drawdown`,
`check_position_risk` and `check_price_deviation` had no call sites anywhere,
which meant the emergency stop could not fire, the peak capital never moved,
and an order could reach the exchange without passing a single check. These
tests assert that each check rejects what it should, and that the stop both
fires and is honoured.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from timi.core.position_manager import PositionManager
from timi.risk.constraints import RiskConstraintError, RiskConstraints
from timi.risk.risk_manager import RiskManager
from timi.utils.config import RiskConfig


class StubConfig:
    """Minimal stand-in for `Config`, so a test can vary one setting.

    The real `Config` is a singleton loaded from the shipped YAML file, and
    mutating it would leak between tests.
    """

    def __init__(self, risk: dict | None = None, values: dict | None = None):
        """Build a configuration double.

        Args:
            risk: Overrides for the risk section
            values: Values returned by dotted `get` lookups
        """
        self._risk = RiskConfig(**(risk or {}))
        self._values = values or {}

    @property
    def risk(self) -> RiskConfig:
        """Risk configuration section."""
        return self._risk

    def get(self, key: str, default=None):
        """Dotted configuration lookup."""
        return self._values.get(key, default)


def build_manager(
    risk: dict | None = None,
    values: dict | None = None,
    capital: float | None = 1000.0
) -> RiskManager:
    """Build a risk manager over an empty position book.

    Args:
        risk: Overrides for the risk section
        values: Dotted configuration values
        capital: Initial capital, or None to leave it uninitialised

    Returns:
        A configured risk manager
    """
    manager = RiskManager(StubConfig(risk, values), PositionManager())
    if capital is not None:
        manager.initialize_capital(capital)
    return manager


# --------------------------------------------------------------------------
# Emergency stop
# --------------------------------------------------------------------------


def test_emergency_stop_is_honoured_from_configuration():
    """EMERGENCY_STOP in the environment lands in the config as this key.

    The previous version hardcoded the flag to False and never read the
    setting, so setting it did nothing at all.
    """
    manager = build_manager(values={'emergency_stop': True})

    assert manager.emergency_stop is True
    assert manager.check_order_risk("BTC/USDT", 10.0, "buy") is False


def test_emergency_stop_defaults_to_off():
    """Absent configuration must not halt a healthy system."""
    manager = build_manager()

    assert manager.emergency_stop is False
    assert manager.check_order_risk("BTC/USDT", 10.0, "buy") is True


def test_drawdown_breach_fires_the_emergency_stop_and_halts_trading():
    """The stop is reachable, and once set nothing else may be traded."""
    manager = build_manager(risk={'max_drawdown': 20.0}, capital=1000.0)

    assert manager.check_drawdown(700.0) is False
    assert manager.emergency_stop is True
    assert manager.check_order_risk("BTC/USDT", 10.0, "buy") is False
    assert manager.check_order_risk("BTC/USDT", 10.0, "sell") is False


def test_drawdown_within_the_limit_does_not_halt():
    """A drawdown under the limit is a warning at most."""
    manager = build_manager(risk={'max_drawdown': 20.0}, capital=1000.0)

    assert manager.check_drawdown(900.0) is True
    assert manager.emergency_stop is False


def test_peak_capital_moves_with_the_account():
    """The peak was frozen because only the dead method updated it."""
    manager = build_manager(risk={'max_drawdown': 20.0}, capital=1000.0)

    manager.update_capital(2000.0)
    assert manager.peak_capital == pytest.approx(2000.0)

    # 1500 is above the initial capital but 25% below the new peak.
    assert manager.update_capital(1500.0) is False
    assert manager.emergency_stop is True


def test_update_capital_rejects_a_non_finite_mark():
    """Capital that cannot be marked is not a reason to keep trading."""
    manager = build_manager(capital=1000.0)

    assert manager.update_capital(float("nan")) is False
    assert manager.emergency_stop is True


def test_halt_is_idempotent_and_reversible_only_deliberately():
    """A halt stays halted until someone resets it on purpose."""
    manager = build_manager()

    manager.halt("testing")
    manager.halt("testing again")
    assert manager.emergency_stop is True

    manager.reset_emergency_stop()
    assert manager.emergency_stop is False


# --------------------------------------------------------------------------
# Order sizing
# --------------------------------------------------------------------------


def test_order_over_the_position_size_limit_is_rejected():
    """Ten percent of a thousand is a hundred, and 150 is not allowed."""
    manager = build_manager(risk={'max_position_pct': 10.0}, capital=1000.0)

    assert manager.check_order_risk("BTC/USDT", 150.0, "buy") is False
    assert manager.check_order_risk("BTC/USDT", 90.0, "buy") is True


def test_zero_capital_fails_closed():
    """`if self.initial_capital:` skipped the size check when capital was 0.0.

    An unevaluated limit must never read as a pass.
    """
    manager = build_manager(capital=0.0)

    assert manager.check_order_risk("BTC/USDT", 1_000_000.0, "buy") is False


def test_uninitialised_capital_fails_closed():
    """No capital figure means no order can be sized, so none is allowed."""
    manager = build_manager(capital=None)

    assert manager.check_order_risk("BTC/USDT", 1.0, "buy") is False


def test_order_value_that_is_not_a_number_is_rejected():
    """A corrupted size never reaches the exchange."""
    manager = build_manager()

    assert manager.check_order_risk("BTC/USDT", float("nan"), "buy") is False
    assert manager.check_order_risk("BTC/USDT", -5.0, "buy") is False


def test_concurrent_position_limit_is_enforced():
    """A new pair beyond the limit is refused; an existing one is not."""
    manager = build_manager(risk={'max_concurrent_positions': 2}, capital=10_000.0)

    manager.position_manager.add_position("BTC/USDT", 1.0, 10.0)
    manager.position_manager.add_position("ETH/USDT", 1.0, 10.0)

    assert manager.check_order_risk("SOL/USDT", 10.0, "buy") is False
    # Adding to a pair already held does not consume a new slot.
    assert manager.check_order_risk("BTC/USDT", 10.0, "buy") is True


# --------------------------------------------------------------------------
# Aggregate exposure
# --------------------------------------------------------------------------


def test_aggregate_exposure_is_enforced_across_multiple_orders():
    """Five orders at ten percent each pass individually and must not together.

    A grid places seven levels per cycle, so this is the ordinary case rather
    than an edge one. The ceiling defaults to the per-position limit times the
    position count: ten percent of a thousand, five positions, so 500.
    """
    manager = build_manager(
        risk={'max_position_pct': 10.0, 'max_concurrent_positions': 5},
        capital=1000.0
    )

    approved = 0
    for level in range(6):
        if manager.check_order_risk(
            "BTC/USDT", 100.0, "buy", reserve_key=f"level-{level}"
        ):
            approved += 1

    assert approved == 5
    assert manager.total_exposure() == pytest.approx(500.0)
    assert manager.check_order_risk("BTC/USDT", 100.0, "buy") is False


def test_releasing_a_reservation_frees_the_exposure():
    """A cancelled order must not keep occupying the ceiling forever."""
    manager = build_manager(
        risk={'max_position_pct': 10.0, 'max_concurrent_positions': 5},
        capital=1000.0
    )

    for level in range(5):
        manager.check_order_risk(
            "BTC/USDT", 100.0, "buy", reserve_key=f"level-{level}"
        )

    assert manager.check_order_risk("BTC/USDT", 100.0, "buy") is False

    manager.release_exposure("level-0")
    assert manager.check_order_risk("BTC/USDT", 100.0, "buy") is True


def test_held_inventory_counts_towards_exposure():
    """Exposure is held value plus reserved value, not reservations alone."""
    manager = build_manager(
        risk={'max_position_pct': 50.0, 'max_concurrent_positions': 1},
        capital=1000.0
    )

    manager.position_manager.add_position("BTC/USDT", 10.0, 40.0)

    assert manager.total_exposure() == pytest.approx(400.0)
    # 400 held plus 200 requested is over the 500 ceiling.
    assert manager.check_order_risk("BTC/USDT", 200.0, "buy") is False


def test_a_sell_is_not_charged_against_the_exposure_ceiling():
    """Disposing of inventory reduces exposure, so sizing limits do not apply."""
    manager = build_manager(
        risk={'max_position_pct': 10.0, 'max_concurrent_positions': 5},
        capital=1000.0
    )

    for level in range(5):
        manager.check_order_risk(
            "BTC/USDT", 100.0, "buy", reserve_key=f"level-{level}"
        )

    assert manager.check_order_risk("BTC/USDT", 100.0, "buy") is False
    assert manager.check_order_risk("BTC/USDT", 100.0, "sell") is True


# --------------------------------------------------------------------------
# Prices and stops
# --------------------------------------------------------------------------


def test_price_deviation_beyond_the_limit_is_rejected():
    """A price far from the market is refused by default."""
    manager = build_manager(risk={'max_price_deviation': 2.0})

    assert manager.check_price_deviation("BTC/USDT", 100.0, 100.0) is True
    assert manager.check_price_deviation("BTC/USDT", 110.0, 100.0) is False


def test_price_deviation_accepts_a_wider_bound_for_resting_orders():
    """A grid level rests deliberately far out and passes its own bound."""
    manager = build_manager(risk={'max_price_deviation': 2.0})

    assert manager.check_price_deviation(
        "BTC/USDT", 110.0, 100.0, tolerance=0.25
    ) is True


def test_an_unusable_order_price_is_rejected_at_any_tolerance():
    """A wider bound is not permission to send a negative price."""
    manager = build_manager()

    assert manager.check_price_deviation(
        "BTC/USDT", -1.0, 100.0, tolerance=10.0
    ) is False
    assert manager.check_price_deviation(
        "BTC/USDT", float("nan"), 100.0, tolerance=10.0
    ) is False


def test_order_risk_applies_the_price_check_when_a_market_price_is_given():
    """The deviation check is reachable from the order path, not just directly."""
    manager = build_manager(risk={'max_price_deviation': 2.0}, capital=10_000.0)

    assert manager.check_order_risk(
        "BTC/USDT", 10.0, "buy", order_price=150.0, market_price=100.0
    ) is False


def test_stop_loss_check_rejects_a_breached_holding():
    """Five percent below the average entry is a breach."""
    manager = build_manager(risk={'stop_loss_pct': 5.0})

    assert manager.check_position_risk("BTC/USDT", 100.0, 96.0, 1.0) is True
    assert manager.check_position_risk("BTC/USDT", 100.0, 94.0, 1.0) is False


def test_stop_loss_check_is_inert_without_a_cost_basis():
    """No entry price means no reference to measure a loss from."""
    manager = build_manager(risk={'stop_loss_pct': 5.0})

    assert manager.check_position_risk("BTC/USDT", 0.0, 1.0, 1.0) is True


# --------------------------------------------------------------------------
# Frequency and per-trade loss
# --------------------------------------------------------------------------


def test_hourly_trade_limit_is_enforced():
    """A declared frequency limit that is never evaluated is not a limit."""
    manager = build_manager(values={'risk.constraints.max_trades_per_hour': 3})

    for _ in range(3):
        assert manager.check_trade_frequency() is True
        manager.record_trade()

    assert manager.check_trade_frequency() is False
    assert manager.check_order_risk("BTC/USDT", 10.0, "buy") is False


def test_trades_outside_the_window_stop_counting():
    """The hourly window slides."""
    manager = build_manager(values={'risk.constraints.max_trades_per_hour': 2})

    stale = datetime.now() - timedelta(hours=2)
    manager.record_trade(stale)
    manager.record_trade(stale)

    assert manager.check_trade_frequency() is True


def test_per_trade_loss_limit_is_enforced():
    """`max_loss_per_trade` was declared and never evaluated."""
    manager = build_manager(values={'risk.constraints.max_loss_per_trade': 50})

    assert manager.check_trade_loss("BTC/USDT", -10.0) is True
    assert manager.check_trade_loss("BTC/USDT", -80.0) is False
    assert manager.check_trade_loss("BTC/USDT", 500.0) is True


# --------------------------------------------------------------------------
# Bounding externally supplied sizing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        -5.0,
        0.0,
        1e12,
        "100",
        None,
        True,
    ],
)
def test_absurd_capital_allocation_is_rejected(value):
    """Sizing arrives as free-form JSON and multiplies every grid quantity.

    Nothing upstream bounds it, so the risk boundary rejects anything that
    cannot be made safe rather than clamping nonsense into range.
    """
    manager = build_manager()

    with pytest.raises(RiskConstraintError):
        manager.validate_capital_allocation(value)


def test_plausible_capital_allocation_is_clamped_into_range():
    """Aggressive but plausible values are clamped, not refused."""
    constraints = RiskConstraints()

    assert constraints.validate_capital_allocation(500.0) == pytest.approx(500.0)
    assert constraints.validate_capital_allocation(5_000.0) == pytest.approx(
        constraints.max_single_position_value
    )
    assert constraints.validate_capital_allocation(0.01) == pytest.approx(
        constraints.min_single_position_value
    )


def test_rejected_allocation_is_recorded_as_a_violation():
    """A refusal is visible in the risk report, not only in a traceback."""
    manager = build_manager()

    with pytest.raises(RiskConstraintError):
        manager.validate_capital_allocation(-1.0)

    assert manager.get_risk_report()['violations_count'] == 1


# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------


def test_missing_capital_allocation_is_a_violation():
    """It defaulted to zero, so a config that omitted it passed silently."""
    violations = RiskConstraints().validate_parameters({'grid_levels': 5})

    assert any('capital_allocation' in v for v in violations)


def test_every_declared_constraint_is_evaluated():
    """Each field is checked, rather than declared and ignored."""
    constraints = RiskConstraints()
    violations = constraints.validate_parameters({
        'capital_allocation': 100.0,
        'grid_levels': 50,
        'leverage': 3.0,
        'max_loss_per_trade': 1_000.0,
        'total_position_value': 1_000_000.0,
        'trades_per_hour': 5_000,
        'trades_per_day': 50_000,
    })

    joined = " ".join(violations)
    assert 'grid levels' in joined.lower()
    assert 'leverage' in joined.lower()
    assert 'loss per trade' in joined.lower()
    assert 'total position value' in joined.lower()
    assert 'trades_per_hour' in joined
    assert 'trades_per_day' in joined


def test_leverage_above_one_is_refused_on_spot():
    """There is no borrowing here, so any leverage request is a category error."""
    constraints = RiskConstraints()

    assert constraints.max_leverage == 1.0
    assert constraints.validate_parameters(
        {'capital_allocation': 100.0, 'leverage': 1.0}
    ) == []
    assert constraints.validate_parameters(
        {'capital_allocation': 100.0, 'leverage': 2.0}
    ) != []


def test_valid_parameters_produce_no_violations():
    """The happy path stays clean."""
    violations = RiskConstraints().validate_parameters({
        'capital_allocation': 100.0,
        'grid_levels': 7,
        'leverage': 1.0,
        'max_loss_per_trade': 10.0,
        'total_position_value': 500.0,
        'trades_per_hour': 10,
        'trades_per_day': 100,
    })

    assert violations == []


def test_risk_report_is_finite_and_complete():
    """The report is what an operator reads, so it must always compute."""
    manager = build_manager(capital=1000.0)
    report = manager.get_risk_report()

    assert math.isfinite(report['current_drawdown'])
    assert report['emergency_stop'] is False
    assert report['total_exposure'] == pytest.approx(0.0)
