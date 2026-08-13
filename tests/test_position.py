"""Tests for the position book, which is the only source of cost basis.

The venue is spot, so `get_positions()` reports a balance and a balance has no
cost basis. Everything downstream that computes a target price, a stop or a
P&L figure reads its entry price from `PositionManager`, which makes the
arithmetic in this module load bearing rather than incidental.
"""

from __future__ import annotations

import math

import pytest

from timi.core.position_manager import ManagedPosition, PositionManager


@pytest.fixture
def book() -> PositionManager:
    """A fresh, empty position book."""
    return PositionManager()


# --------------------------------------------------------------------------
# Average entry price
# --------------------------------------------------------------------------


def test_reducing_a_holding_does_not_move_the_average_entry(book):
    """Selling half a holding at a profit must not make the rest cheaper.

    The regression: `size` was mutated before the weighted average was
    computed, and a reduce was passed as a negative size, so the sale was
    treated as a purchase at a negative cost. Long 1 BTC at 100k, reduce 0.5 at
    110k, and the recorded entry fell to 90k, inventing profit that was then
    used to set every target and stop.
    """
    book.add_position("BTC/USDT", size=1.0, entry_price=100_000.0)
    book.add_position("BTC/USDT", size=-0.5, entry_price=110_000.0)

    position = book.get_position("BTC/USDT")

    assert position is not None
    assert position.size == pytest.approx(0.5)
    assert position.entry_price == pytest.approx(100_000.0)
    assert position.entry_price != pytest.approx(90_000.0)


def test_partial_close_does_not_move_the_average_entry(book):
    """The same guarantee through `close_position`, which also realises P&L."""
    book.add_position("BTC/USDT", size=1.0, entry_price=100_000.0)

    remaining = book.close_position("BTC/USDT", exit_price=110_000.0, size=0.5)

    assert remaining is not None
    assert remaining.entry_price == pytest.approx(100_000.0)
    assert remaining.size == pytest.approx(0.5)
    assert remaining.realized_pnl == pytest.approx(5_000.0)


def test_buying_more_moves_the_average_entry_by_weight(book):
    """A purchase, and only a purchase, re-averages the cost basis."""
    book.add_position("ETH/USDT", size=1.0, entry_price=2_000.0)
    book.add_position("ETH/USDT", size=3.0, entry_price=1_000.0)

    position = book.get_position("ETH/USDT")

    # (1 * 2000 + 3 * 1000) / 4
    assert position.entry_price == pytest.approx(1_250.0)
    assert position.size == pytest.approx(4.0)


# --------------------------------------------------------------------------
# Closing
# --------------------------------------------------------------------------


def test_full_close_realises_the_right_pnl(book):
    """A full close realises the whole gain against the average entry."""
    book.add_position("BTC/USDT", size=2.0, entry_price=50_000.0)

    closed = book.close_position("BTC/USDT", exit_price=55_000.0)

    assert closed is not None
    assert closed.realized_pnl == pytest.approx(10_000.0)
    assert book.get_position("BTC/USDT") is None
    assert book.get_total_realized_pnl() == pytest.approx(10_000.0)
    assert book.get_statistics()['winning_trades'] == 1


def test_full_close_at_a_loss_is_recorded_as_a_loss(book):
    """Losses are realised with the sign intact."""
    book.add_position("BTC/USDT", size=1.0, entry_price=50_000.0)

    closed = book.close_position("BTC/USDT", exit_price=45_000.0)

    assert closed.realized_pnl == pytest.approx(-5_000.0)
    assert book.get_statistics()['losing_trades'] == 1


def test_partial_closes_accumulate_into_the_realised_total(book):
    """Realised P&L from a partial close is not lost before the full close."""
    book.add_position("BTC/USDT", size=2.0, entry_price=100.0)

    book.close_position("BTC/USDT", exit_price=110.0, size=1.0)
    assert book.get_total_realized_pnl() == pytest.approx(10.0)

    book.close_position("BTC/USDT", exit_price=120.0, size=1.0)
    assert book.get_total_realized_pnl() == pytest.approx(30.0)
    assert book.get_position("BTC/USDT") is None


def test_close_larger_than_held_is_clamped(book):
    """Spot cannot sell what it does not hold, so an oversized close clamps."""
    book.add_position("BTC/USDT", size=0.5, entry_price=100.0)

    closed = book.close_position("BTC/USDT", exit_price=110.0, size=5.0)

    assert closed is not None
    assert book.get_position("BTC/USDT") is None
    # Only the 0.5 actually held was realised.
    assert closed.realized_pnl == pytest.approx(5.0)


def test_close_accepts_a_negative_size_as_a_magnitude(book):
    """The sign convention is explicit: a close size is a magnitude."""
    book.add_position("BTC/USDT", size=1.0, entry_price=100.0)

    remaining = book.close_position("BTC/USDT", exit_price=110.0, size=-0.25)

    assert remaining.size == pytest.approx(0.75)
    assert remaining.realized_pnl == pytest.approx(2.5)


def test_closing_an_absent_pair_returns_none(book):
    """Nothing held means nothing to close, and no exception."""
    assert book.close_position("DOGE/USDT", exit_price=1.0) is None


# --------------------------------------------------------------------------
# Degenerate values
# --------------------------------------------------------------------------


def test_pnl_percentage_of_a_zero_size_position_does_not_raise():
    """A zero size divides by zero in the naive form. It must return 0.0."""
    position = ManagedPosition(
        pair="BTC/USDT",
        size=0.0,
        entry_price=100_000.0,
        current_price=110_000.0,
        unrealized_pnl=0.0
    )

    assert position.pnl_percentage == 0.0


def test_pnl_percentage_of_a_zero_entry_position_does_not_raise():
    """A holding with no cost basis has no percentage, not an exception."""
    position = ManagedPosition(
        pair="BTC/USDT",
        size=5.0,
        entry_price=0.0,
        current_price=110_000.0,
        unrealized_pnl=0.0
    )

    assert position.pnl_percentage == 0.0
    assert position.entry_price_known is False


def test_pnl_percentage_survives_a_non_finite_pnl():
    """Corrupted arithmetic upstream must not propagate as an exception."""
    position = ManagedPosition(
        pair="BTC/USDT",
        size=1.0,
        entry_price=100.0,
        current_price=100.0,
        unrealized_pnl=float("nan")
    )

    assert position.pnl_percentage == 0.0


def test_updating_price_on_a_basisless_holding_keeps_pnl_at_zero(book):
    """No basis, no P&L. Inventing one is what closed the whole book."""
    book.reconcile_size("BTC/USDT", exchange_size=0.5)

    book.update_position_price("BTC/USDT", 60_000.0)
    position = book.get_position("BTC/USDT")

    assert position.unrealized_pnl == 0.0
    assert position.entry_price_known is False


def test_zero_and_non_finite_fill_sizes_are_ignored(book):
    """A fill of nothing, or of NaN, changes nothing and raises nothing."""
    book.add_position("BTC/USDT", size=1.0, entry_price=100.0)

    book.add_position("BTC/USDT", size=0.0, entry_price=999.0)
    book.add_position("BTC/USDT", size=float("nan"), entry_price=999.0)

    position = book.get_position("BTC/USDT")
    assert position.size == pytest.approx(1.0)
    assert position.entry_price == pytest.approx(100.0)


def test_reducing_with_nothing_held_is_refused(book):
    """Spot cannot go short, so a sell with no holding opens nothing."""
    assert book.add_position("BTC/USDT", size=-1.0, entry_price=100.0) is None
    assert book.get_position("BTC/USDT") is None


def test_a_holding_is_never_short(book):
    """The sign convention: size is a held quantity and stays non-negative."""
    book.add_position("BTC/USDT", size=1.0, entry_price=100.0)
    book.add_position("BTC/USDT", size=-5.0, entry_price=100.0)

    assert book.get_position("BTC/USDT") is None
    assert all(not p.is_short for p in book.closed_positions)


# --------------------------------------------------------------------------
# Reconciliation against exchange inventory
# --------------------------------------------------------------------------


def test_inventory_found_on_the_account_has_no_cost_basis(book):
    """A balance carries no record of what was paid, and must say so."""
    position = book.reconcile_size("BTC/USDT", exchange_size=0.5)

    assert position is not None
    assert position.size == pytest.approx(0.5)
    assert position.entry_price == 0.0
    assert position.entry_price_known is False


def test_surplus_inventory_invalidates_a_known_basis(book):
    """Mixing priced and unpriced inventory leaves no derivable average."""
    book.add_position("BTC/USDT", size=1.0, entry_price=100.0)

    book.reconcile_size("BTC/USDT", exchange_size=1.5)
    position = book.get_position("BTC/USDT")

    assert position.size == pytest.approx(1.5)
    assert position.entry_price_known is False


def test_shortfall_reduces_the_quantity_and_keeps_the_basis(book):
    """Inventory leaving by another route is a sale, and sales keep the cost."""
    book.add_position("BTC/USDT", size=2.0, entry_price=100.0)

    book.reconcile_size("BTC/USDT", exchange_size=1.2)
    position = book.get_position("BTC/USDT")

    assert position.size == pytest.approx(1.2)
    assert position.entry_price == pytest.approx(100.0)
    assert position.entry_price_known is True


def test_reconciling_to_zero_retires_the_holding(book):
    """Nothing on the account means nothing tracked."""
    book.add_position("BTC/USDT", size=1.0, entry_price=100.0)

    assert book.reconcile_size("BTC/USDT", exchange_size=0.0) is None
    assert book.get_position("BTC/USDT") is None


def test_reconciling_to_a_non_finite_size_changes_nothing(book):
    """An unusable reading is not an instruction to liquidate."""
    book.add_position("BTC/USDT", size=1.0, entry_price=100.0)

    book.reconcile_size("BTC/USDT", exchange_size=float("inf"))

    position = book.get_position("BTC/USDT")
    assert position.size == pytest.approx(1.0)
    assert position.entry_price == pytest.approx(100.0)


def test_buying_into_a_basisless_holding_stays_basisless(book):
    """A blended average that includes an unpriced part is not derivable."""
    book.reconcile_size("BTC/USDT", exchange_size=1.0)
    book.add_position("BTC/USDT", size=1.0, entry_price=50_000.0)

    position = book.get_position("BTC/USDT")

    assert position.size == pytest.approx(2.0)
    assert position.entry_price_known is False


def test_closing_a_basisless_holding_realises_no_invented_pnl(book):
    """Recording zero is a refusal to guess, not a claim of breaking even."""
    book.reconcile_size("BTC/USDT", exchange_size=1.0)

    closed = book.close_position("BTC/USDT", exit_price=60_000.0)

    assert closed.realized_pnl == 0.0
    assert book.get_total_realized_pnl() == 0.0


def test_statistics_are_finite_on_an_empty_book(book):
    """A fresh book reports zeros rather than dividing by nothing."""
    stats = book.get_statistics()

    assert stats['open_positions'] == 0
    assert stats['win_rate'] == 0
    assert math.isfinite(stats['total_pnl'])


# --------------------------------------------------------------------------
# Tranche accounting
# --------------------------------------------------------------------------


def decide(position) -> dict:
    """Reproduce the profit-taking and stop decisions made downstream.

    The engine reads `entry_price_known` first and skips the holding entirely
    when it is False, precisely so that a holding with no basis is never given
    a target of `0 * anything`. This mirror lets the tests assert on what a
    decision would see without importing the engine.

    Args:
        position: Holding to decide about

    Returns:
        A dict with `acted`, and the stop and target prices when it acted
    """
    if not position.entry_price_known:
        return {'acted': False, 'stop': None, 'target': None}

    return {
        'acted': True,
        'stop': position.entry_price * 0.98,
        'target': position.entry_price * 1.02
    }


def test_a_reduce_leaves_the_remaining_tranche_at_its_own_price(book):
    """The sale price is never blended into inventory that is still held."""
    book.add_position("BTC/USDT", size=1.0, entry_price=100_000.0)
    book.add_position("BTC/USDT", size=-0.5, entry_price=110_000.0)

    position = book.get_position("BTC/USDT")

    assert position.entry_price == pytest.approx(100_000.0)
    assert len(position.tranches) == 1
    assert position.tranches[0].price == pytest.approx(100_000.0)
    assert position.tranches[0].quantity == pytest.approx(0.5)


def test_a_partial_sale_realises_against_the_oldest_tranche_first(book):
    """FIFO is the convention, so the first tranche bought is the first sold.

    Buy 1 at 100 then 1 at 200, sell 1 at 150. Against the blended average of
    150 the sale would look flat. Against the tranche actually consumed it is a
    gain of 50, and what remains is the 200 tranche.
    """
    book.add_position("ETH/USDT", size=1.0, entry_price=100.0)
    book.add_position("ETH/USDT", size=1.0, entry_price=200.0)

    remaining = book.close_position("ETH/USDT", exit_price=150.0, size=1.0)

    assert remaining.realized_pnl == pytest.approx(50.0)
    assert book.get_total_realized_pnl() == pytest.approx(50.0)
    assert remaining.size == pytest.approx(1.0)
    assert remaining.entry_price == pytest.approx(200.0)
    assert remaining.entry_price_known is True


def test_selling_the_rest_realises_against_the_second_tranche(book):
    """The tail of a FIFO queue is priced at its own cost, not the average."""
    book.add_position("ETH/USDT", size=1.0, entry_price=100.0)
    book.add_position("ETH/USDT", size=1.0, entry_price=200.0)

    book.close_position("ETH/USDT", exit_price=150.0, size=1.0)
    closed = book.close_position("ETH/USDT", exit_price=150.0)

    # +50 against the 100 tranche, then -50 against the 200 tranche.
    assert closed.realized_pnl == pytest.approx(0.0)
    assert book.get_total_realized_pnl() == pytest.approx(0.0)
    assert book.get_position("ETH/USDT") is None


def test_surplus_inventory_leaves_the_bought_quantity_still_priced(book):
    """Adopting surplus must not throw away the basis of what was bought.

    This is the point of tranche accounting: the coin the system bought keeps
    the price it was bought at, and only the coin that appeared from nowhere is
    unpriced.
    """
    book.add_position("BTC/USDT", size=1.0, entry_price=100.0)

    book.reconcile_size("BTC/USDT", exchange_size=1.5)
    position = book.get_position("BTC/USDT")

    assert position.size == pytest.approx(1.5)
    assert position.priced_size == pytest.approx(1.0)
    assert position.unpriced_size == pytest.approx(0.5)
    assert position.tranches[0].price == pytest.approx(100.0)
    assert position.tranches[1].price is None
    # The whole holding is still reported as basis-unknown, because callers
    # act on a whole holding at a time.
    assert position.entry_price_known is False
    assert position.is_fully_priced is False


def test_a_partly_unpriced_holding_reports_both_quantities(book):
    """A holding can be partly basis-known, and says so in both directions."""
    book.reconcile_size("BTC/USDT", exchange_size=2.0)
    book.add_position("BTC/USDT", size=3.0, entry_price=100.0)

    position = book.get_position("BTC/USDT")

    assert position.size == pytest.approx(5.0)
    assert position.priced_size == pytest.approx(3.0)
    assert position.unpriced_size == pytest.approx(2.0)
    assert position.priced_size + position.unpriced_size == pytest.approx(
        position.size
    )
    assert position.is_fully_priced is False


def test_the_basis_recovers_once_the_unpriced_quantity_is_gone(book):
    """The priced tranches survive, so no flat-and-rebuy cycle is needed.

    Under FIFO the adopted quantity sits at the back of the queue, so it leaves
    when the holding is drawn down to it or when the exchange reports the
    shortfall. Either way the priced tranches were never destroyed.
    """
    book.reconcile_size("BTC/USDT", exchange_size=0.5)
    book.add_position("BTC/USDT", size=1.0, entry_price=100.0)

    assert book.get_position("BTC/USDT").entry_price_known is False

    # The adopted 0.5 leaves first, being the oldest tranche.
    book.close_position("BTC/USDT", exit_price=120.0, size=0.5)
    position = book.get_position("BTC/USDT")

    assert position.size == pytest.approx(1.0)
    assert position.unpriced_size == pytest.approx(0.0)
    assert position.entry_price == pytest.approx(100.0)
    assert position.entry_price_known is True
    assert position.is_fully_priced is True
    # Nothing was invented for the unpriced quantity that was sold.
    assert book.get_total_realized_pnl() == pytest.approx(0.0)


def test_a_decision_never_sees_a_basis_of_zero_for_unpriced_quantity(book):
    """Unpriced quantity is skipped, never valued at zero and taken as free."""
    book.reconcile_size("BTC/USDT", exchange_size=1.0)
    unpriced = book.get_position("BTC/USDT")

    assert unpriced.entry_price == 0.0
    assert decide(unpriced) == {'acted': False, 'stop': None, 'target': None}

    # Mixed: a priced tranche and an adopted one.
    book.add_position("BTC/USDT", size=1.0, entry_price=100.0)
    mixed = book.get_position("BTC/USDT")

    assert mixed.unpriced_size == pytest.approx(1.0)
    assert decide(mixed)['acted'] is False
    # And the mark stays flat, because the total is not knowable either.
    book.update_position_price("BTC/USDT", 500.0)
    assert mixed.unrealized_pnl == 0.0
    assert mixed.pnl_percentage == 0.0


def test_a_fully_priced_holding_is_still_acted_on(book):
    """The safety rule must not become a blanket refusal to trade."""
    book.add_position("BTC/USDT", size=1.0, entry_price=100.0)

    decision = decide(book.get_position("BTC/USDT"))

    assert decision['acted'] is True
    assert decision['stop'] == pytest.approx(98.0)
    assert decision['target'] == pytest.approx(102.0)


def test_selling_past_the_priced_quantity_realises_only_the_priced_part(book):
    """A sale that runs into unpriced quantity invents no P&L for it.

    Selling more than the priced quantity is allowed, since the inventory is
    genuinely held and the exchange will let it go. What it must not do is
    report a gain for the part that never had a cost recorded.
    """
    book.add_position("BTC/USDT", size=1.0, entry_price=100.0)
    book.reconcile_size("BTC/USDT", exchange_size=2.0)

    remaining = book.close_position("BTC/USDT", exit_price=150.0, size=1.5)

    # 1.0 priced at 100 sold at 150, plus 0.5 unpriced worth nothing knowable.
    assert remaining.realized_pnl == pytest.approx(50.0)
    assert book.get_total_realized_pnl() == pytest.approx(50.0)
    assert remaining.size == pytest.approx(0.5)
    assert remaining.priced_size == pytest.approx(0.0)
    assert remaining.unpriced_size == pytest.approx(0.5)
    assert remaining.entry_price == 0.0
    assert remaining.entry_price_known is False


def test_selling_everything_including_unpriced_quantity_closes_cleanly(book):
    """An oversized close still clamps, and still realises only the priced part."""
    book.add_position("BTC/USDT", size=1.0, entry_price=100.0)
    book.reconcile_size("BTC/USDT", exchange_size=3.0)

    closed = book.close_position("BTC/USDT", exit_price=150.0, size=99.0)

    assert closed is not None
    assert closed.realized_pnl == pytest.approx(50.0)
    assert closed.size == 0.0
    assert closed.tranches == []
    assert book.get_position("BTC/USDT") is None


def test_buying_at_no_usable_price_only_makes_that_quantity_unpriced(book):
    """A fill with no usable price does not destroy the tranches already held."""
    book.add_position("BTC/USDT", size=1.0, entry_price=100.0)
    book.add_position("BTC/USDT", size=1.0, entry_price=0.0)

    position = book.get_position("BTC/USDT")

    assert position.size == pytest.approx(2.0)
    assert position.priced_size == pytest.approx(1.0)
    assert position.unpriced_size == pytest.approx(1.0)
    assert position.entry_price_known is False


def test_a_shortfall_consumes_the_oldest_tranche_first(book):
    """Reconciliation downwards follows the same FIFO convention as a sale."""
    book.add_position("BTC/USDT", size=1.0, entry_price=100.0)
    book.add_position("BTC/USDT", size=1.0, entry_price=300.0)

    book.reconcile_size("BTC/USDT", exchange_size=1.0)
    position = book.get_position("BTC/USDT")

    assert position.size == pytest.approx(1.0)
    assert position.entry_price == pytest.approx(300.0)
    assert position.entry_price_known is True


def test_the_average_is_taken_over_priced_tranches_only(book):
    """Unpriced quantity is excluded from the average, not counted as free."""
    book.add_position("ETH/USDT", size=1.0, entry_price=1_000.0)
    book.add_position("ETH/USDT", size=1.0, entry_price=3_000.0)
    book.reconcile_size("ETH/USDT", exchange_size=6.0)

    position = book.get_position("ETH/USDT")

    # (1000 + 3000) / 2, not (1000 + 3000) / 6.
    assert position.entry_price == pytest.approx(2_000.0)
    assert position.priced_size == pytest.approx(2.0)
    assert position.unpriced_size == pytest.approx(4.0)


def test_a_position_built_without_tranches_still_reports_its_split():
    """Adapters that build a holding directly are not left without an answer."""
    priced = ManagedPosition(
        pair="BTC/USDT",
        size=2.0,
        entry_price=100.0,
        current_price=100.0,
        unrealized_pnl=0.0
    )

    assert priced.priced_size == pytest.approx(2.0)
    assert priced.unpriced_size == pytest.approx(0.0)
    assert priced.is_fully_priced is True

    basisless = ManagedPosition(
        pair="BTC/USDT",
        size=2.0,
        entry_price=0.0,
        current_price=100.0,
        unrealized_pnl=0.0
    )

    assert basisless.priced_size == pytest.approx(0.0)
    assert basisless.unpriced_size == pytest.approx(2.0)
    assert basisless.is_fully_priced is False
