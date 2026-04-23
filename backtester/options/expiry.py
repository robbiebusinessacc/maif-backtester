"""
Expiration and exercise handling for options positions.

Processes legs expiring on a given date, determines whether each leg
is exercised, assigned, or expires worthless, and computes the
resulting cash impact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List

from backtester.options.positions import OptionsPosition


# ─────────────────────────────────────────────────────────────
# Exercise Event
# ─────────────────────────────────────────────────────────────

@dataclass
class ExerciseEvent:
    """Record of an option exercise/assignment/expiration."""

    position_id: str
    leg_index: int
    event_type: str            # "exercise", "assignment", "expired_worthless"
    strike: float
    underlying_price: float
    quantity: int
    cash_impact: float         # positive = received cash, negative = paid


# ─────────────────────────────────────────────────────────────
# Expiration Handler
# ─────────────────────────────────────────────────────────────

class ExpirationHandler:
    """Process option expirations."""

    def __init__(self, exercise_threshold: float = 0.01):
        """
        Parameters
        ----------
        exercise_threshold : float
            ITM amount per share required for auto-exercise.
            Options ITM by less than this amount expire worthless.
        """
        self.exercise_threshold = exercise_threshold

    def process_expiring_positions(
        self,
        positions: List[OptionsPosition],
        expiry_date: date,
        underlying_price: float,
    ) -> List[ExerciseEvent]:
        """
        Process all positions with legs expiring on expiry_date.

        For each expiring leg:
        - If ITM by more than exercise_threshold:
            - Long legs are exercised
            - Short legs are assigned
        - Otherwise: expires worthless

        Cash impact for exercised/assigned options:
        - Call exercise: buy 100 shares at strike => cash = -(strike * 100 * |qty|)
        - Call assignment: sell 100 shares at strike => cash = +(strike * 100 * |qty|)
        - Put exercise: sell 100 shares at strike => cash = +(strike * 100 * |qty|)
        - Put assignment: buy 100 shares at strike => cash = -(strike * 100 * |qty|)

        Returns
        -------
        List of ExerciseEvent objects.
        """
        events: List[ExerciseEvent] = []

        for position in positions:
            if position.is_closed:
                continue

            for leg_idx, leg in enumerate(position.legs):
                if leg.expiration != expiry_date:
                    continue

                itm_amount = leg.intrinsic_value(underlying_price)
                abs_qty = abs(leg.quantity)

                if itm_amount >= self.exercise_threshold:
                    # ITM: exercise (long) or assignment (short)
                    is_long = leg.quantity > 0
                    event_type = "exercise" if is_long else "assignment"

                    if leg.option_type == "C":
                        if is_long:
                            # Exercise call: buy shares at strike
                            cash = -(leg.strike * 100 * abs_qty)
                        else:
                            # Assigned call: sell shares at strike
                            cash = leg.strike * 100 * abs_qty
                    else:  # Put
                        if is_long:
                            # Exercise put: sell shares at strike
                            cash = leg.strike * 100 * abs_qty
                        else:
                            # Assigned put: buy shares at strike
                            cash = -(leg.strike * 100 * abs_qty)

                    events.append(ExerciseEvent(
                        position_id=position.position_id,
                        leg_index=leg_idx,
                        event_type=event_type,
                        strike=leg.strike,
                        underlying_price=underlying_price,
                        quantity=leg.quantity,
                        cash_impact=cash,
                    ))
                else:
                    # OTM or barely ITM: expires worthless
                    events.append(ExerciseEvent(
                        position_id=position.position_id,
                        leg_index=leg_idx,
                        event_type="expired_worthless",
                        strike=leg.strike,
                        underlying_price=underlying_price,
                        quantity=leg.quantity,
                        cash_impact=0.0,
                    ))

        return events


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from backtester.options.positions import OptionLeg, OptionsPosition, PositionFactory

    today = date(2026, 4, 1)
    exp = date(2026, 5, 15)
    handler = ExpirationHandler(exercise_threshold=0.01)

    # --- Bull call spread: long 100C, short 110C ---
    spread = PositionFactory.vertical_spread(
        symbol="AAPL", expiration=exp,
        long_strike=100, short_strike=110,
        long_price=5.0, short_price=2.0,
        option_type="C", entry_date=today,
    )

    # Case 1: Spot = 115 (both ITM)
    events = handler.process_expiring_positions([spread], exp, 115.0)
    assert len(events) == 2
    # Long 100C exercised: cash = -(100 * 100 * 1) = -10000
    ex_event = [e for e in events if e.event_type == "exercise"][0]
    assert ex_event.cash_impact == -10000.0, f"Got {ex_event.cash_impact}"
    # Short 110C assigned: cash = +(110 * 100 * 1) = 11000
    as_event = [e for e in events if e.event_type == "assignment"][0]
    assert as_event.cash_impact == 11000.0, f"Got {as_event.cash_impact}"
    print(f"Both ITM: exercise cash={ex_event.cash_impact}, assignment cash={as_event.cash_impact}")

    # Case 2: Spot = 90 (both OTM)
    events_otm = handler.process_expiring_positions([spread], exp, 90.0)
    assert len(events_otm) == 2
    assert all(e.event_type == "expired_worthless" for e in events_otm)
    assert all(e.cash_impact == 0.0 for e in events_otm)
    print("Both OTM: correctly expired worthless")

    # Case 3: Spot = 105 (long ITM, short OTM)
    events_mixed = handler.process_expiring_positions([spread], exp, 105.0)
    assert len(events_mixed) == 2
    itm_ev = [e for e in events_mixed if e.strike == 100][0]
    otm_ev = [e for e in events_mixed if e.strike == 110][0]
    assert itm_ev.event_type == "exercise"
    assert otm_ev.event_type == "expired_worthless"
    print(f"Mixed: 100C exercised, 110C expired worthless")

    # --- Put exercise ---
    put_leg = OptionLeg(
        expiration=exp, strike=100, option_type="P",
        quantity=1, entry_price=3.0, entry_date=today,
    )
    put_pos = OptionsPosition(
        position_id="put_test", strategy_name="long_put",
        underlying_symbol="AAPL", legs=[put_leg], entry_date=today,
    )
    events_put = handler.process_expiring_positions([put_pos], exp, 90.0)
    assert len(events_put) == 1
    assert events_put[0].event_type == "exercise"
    # Put exercise: sell at strike => cash = 100 * 100 * 1 = 10000
    assert events_put[0].cash_impact == 10000.0, f"Got {events_put[0].cash_impact}"
    print(f"Long put exercise: cash={events_put[0].cash_impact}")

    # --- Closed position should be skipped ---
    spread.is_closed = True
    events_closed = handler.process_expiring_positions([spread], exp, 115.0)
    assert len(events_closed) == 0
    print("Closed position: correctly skipped")

    # --- Non-expiring legs should be skipped ---
    spread.is_closed = False
    events_wrong_date = handler.process_expiring_positions([spread], date(2026, 6, 1), 115.0)
    assert len(events_wrong_date) == 0
    print("Wrong expiry date: correctly skipped")

    print("\nAll expiry tests passed.")
