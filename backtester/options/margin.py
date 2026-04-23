"""
Margin requirement calculations for options positions.

Implements simplified Reg-T margin rules and a basic portfolio margin
estimate. Handles long options, naked shorts, vertical spreads, and
iron condors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from backtester.options.positions import OptionsPosition, OptionLeg


# ─────────────────────────────────────────────────────────────
# Margin Requirement
# ─────────────────────────────────────────────────────────────

@dataclass
class MarginRequirement:
    """Margin requirement for a position."""

    initial_margin: float
    maintenance_margin: float
    model: str  # "reg_t" or "portfolio"


# ─────────────────────────────────────────────────────────────
# Margin Calculator
# ─────────────────────────────────────────────────────────────

class MarginCalculator:
    """Compute margin requirements for options positions."""

    def __init__(self, model: str = "reg_t"):
        if model not in ("reg_t", "portfolio"):
            raise ValueError(f"Unknown margin model: {model}")
        self.model = model

    def calculate(
        self,
        position: OptionsPosition,
        underlying_price: float,
    ) -> MarginRequirement:
        """
        Compute margin for a position.

        Reg-T rules (simplified):
        - Long options: premium paid (no additional margin)
        - Naked short call: 20% of underlying + premium - OTM amount
        - Naked short put: 20% of underlying + premium - OTM amount
        - Vertical spread: max loss between strikes
        - Iron condor: max loss of wider wing
        """
        if self.model == "reg_t":
            return self._reg_t(position, underlying_price)
        return self._portfolio(position, underlying_price)

    def _reg_t(
        self,
        position: OptionsPosition,
        underlying_price: float,
    ) -> MarginRequirement:
        """Simplified Reg-T margin calculation."""
        strategy = position.strategy_name
        legs = position.legs

        if strategy == "vertical_spread":
            margin = self._vertical_spread_margin(legs)
        elif strategy == "iron_condor":
            margin = self._iron_condor_margin(legs)
        else:
            # Generic: sum individual leg margins
            margin = self._sum_leg_margins(legs, underlying_price)

        return MarginRequirement(
            initial_margin=margin,
            maintenance_margin=margin,  # Reg-T: initial == maintenance for options
            model="reg_t",
        )

    def _sum_leg_margins(
        self,
        legs: List[OptionLeg],
        underlying_price: float,
    ) -> float:
        """Sum margin for individual legs (handles long and naked short)."""
        total = 0.0
        for leg in legs:
            if leg.quantity > 0:
                # Long option: margin = premium paid
                total += leg.entry_price * abs(leg.quantity) * 100
            else:
                # Naked short option: 20% of underlying + premium - OTM amount
                total += self._naked_short_margin(leg, underlying_price)
        return total

    @staticmethod
    def _naked_short_margin(leg: OptionLeg, underlying_price: float) -> float:
        """
        Naked short margin per Reg-T:
        max(20% of underlying + premium - OTM amount,
            10% of underlying + premium)
        All scaled by 100 shares per contract and abs(quantity).
        """
        abs_qty = abs(leg.quantity)

        if leg.option_type == "C":
            otm_amount = max(leg.strike - underlying_price, 0)
        else:
            otm_amount = max(underlying_price - leg.strike, 0)

        premium = leg.entry_price

        # Standard formula
        method_a = 0.20 * underlying_price + premium - otm_amount
        # Minimum formula
        method_b = 0.10 * underlying_price + premium

        margin_per_share = max(method_a, method_b)
        return margin_per_share * 100 * abs_qty

    @staticmethod
    def _vertical_spread_margin(legs: List[OptionLeg]) -> float:
        """Vertical spread margin = max loss = width of strikes * 100 * quantity."""
        strikes = sorted(leg.strike for leg in legs)
        width = strikes[-1] - strikes[0]

        # Net premium paid or received
        net_premium = sum(leg.entry_price * leg.quantity * 100 for leg in legs)

        # Max loss is width * 100 - net credit, or net debit
        # For debit spread: max loss = net premium paid
        # For credit spread: max loss = width * 100 - net credit received
        abs_qty = max(abs(leg.quantity) for leg in legs)

        if net_premium > 0:
            # Debit spread: max loss = premium paid
            return net_premium
        else:
            # Credit spread: max loss = (width * 100 * qty) - |net credit|
            return width * 100 * abs_qty + net_premium  # net_premium is negative

    @staticmethod
    def _iron_condor_margin(legs: List[OptionLeg]) -> float:
        """
        Iron condor margin = max loss of the wider wing.

        Splits legs into put wing and call wing, computes max loss of each,
        and returns the larger value.
        """
        put_legs = [l for l in legs if l.option_type == "P"]
        call_legs = [l for l in legs if l.option_type == "C"]

        def wing_max_loss(wing_legs: List[OptionLeg]) -> float:
            if len(wing_legs) < 2:
                return 0.0
            strikes = sorted(leg.strike for leg in wing_legs)
            width = strikes[-1] - strikes[0]
            net_premium = sum(leg.entry_price * leg.quantity * 100 for leg in wing_legs)
            abs_qty = max(abs(leg.quantity) for leg in wing_legs)
            if net_premium > 0:
                return net_premium
            return width * 100 * abs_qty + net_premium

        put_loss = wing_max_loss(put_legs)
        call_loss = wing_max_loss(call_legs)

        return max(put_loss, call_loss)

    def _portfolio(
        self,
        position: OptionsPosition,
        underlying_price: float,
    ) -> MarginRequirement:
        """
        Simplified portfolio margin: ~15% of notional risk.

        Uses max loss scanning as a proxy for the TIMS/OCC risk arrays.
        """
        max_loss = abs(position.max_loss())
        # Portfolio margin is typically lower than Reg-T
        # Use 60% of Reg-T equivalent as rough estimate
        reg_t = self._reg_t(position, underlying_price)
        portfolio_margin = min(max_loss, reg_t.initial_margin * 0.60)
        # Floor at 15% of notional exposure
        notional = underlying_price * 100 * max(abs(leg.quantity) for leg in position.legs)
        floor = notional * 0.15
        portfolio_margin = max(portfolio_margin, floor)

        return MarginRequirement(
            initial_margin=portfolio_margin,
            maintenance_margin=portfolio_margin * 0.80,
            model="portfolio",
        )


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import date
    from backtester.options.positions import PositionFactory

    today = date(2026, 4, 1)
    exp = date(2026, 5, 15)
    calc = MarginCalculator(model="reg_t")

    # --- Bull call spread (debit): long 100C@5, short 110C@2, net debit = $300 ---
    spread = PositionFactory.vertical_spread(
        symbol="AAPL", expiration=exp,
        long_strike=100, short_strike=110,
        long_price=5.0, short_price=2.0,
        option_type="C", entry_date=today,
    )
    margin = calc.calculate(spread, underlying_price=105.0)
    # Net premium = 5*1*100 + 2*(-1)*100 = 300 (debit)
    assert margin.initial_margin == 300.0, f"Expected 300, got {margin.initial_margin}"
    assert margin.model == "reg_t"
    print(f"Bull call spread margin: ${margin.initial_margin:.2f}")

    # --- Bear put spread (credit): long 90P@1.5, short 100P@3.0, net credit = -$150 ---
    credit_spread = PositionFactory.vertical_spread(
        symbol="AAPL", expiration=exp,
        long_strike=90, short_strike=100,
        long_price=1.50, short_price=3.00,
        option_type="P", entry_date=today,
    )
    credit_margin = calc.calculate(credit_spread, underlying_price=105.0)
    # Width = 10, net premium = 1.5*1*100 + 3.0*(-1)*100 = -150 (credit)
    # Margin = 10*100*1 + (-150) = 850
    assert credit_margin.initial_margin == 850.0, f"Expected 850, got {credit_margin.initial_margin}"
    print(f"Bear put spread (credit) margin: ${credit_margin.initial_margin:.2f}")

    # --- Iron condor ---
    ic = PositionFactory.iron_condor(
        symbol="SPY", expiration=exp,
        put_long_strike=90, put_short_strike=95,
        call_short_strike=105, call_long_strike=110,
        put_long_price=0.50, put_short_price=1.50,
        call_short_price=1.50, call_long_price=0.50,
        entry_date=today,
    )
    ic_margin = calc.calculate(ic, underlying_price=100.0)
    # Each wing: width=5, net credit per wing = 0.50*1*100 + 1.50*(-1)*100 = -100
    # Wing margin = 5*100*1 + (-100) = 400
    # Max of two wings = 400
    assert ic_margin.initial_margin == 400.0, f"Expected 400, got {ic_margin.initial_margin}"
    print(f"Iron condor margin: ${ic_margin.initial_margin:.2f}")

    # --- Naked short call ---
    from backtester.options.positions import OptionLeg, OptionsPosition
    naked_call = OptionsPosition(
        position_id="naked_test",
        strategy_name="naked_call",
        underlying_symbol="AAPL",
        legs=[OptionLeg(
            expiration=exp, strike=110, option_type="C",
            quantity=-1, entry_price=3.0, entry_date=today,
        )],
        entry_date=today,
    )
    naked_margin = calc.calculate(naked_call, underlying_price=105.0)
    # Method A: 0.20*105 + 3.0 - max(110-105,0) = 21+3-5 = 19 => 19*100 = 1900
    # Method B: 0.10*105 + 3.0 = 13.5 => 1350
    # Max = 1900
    assert naked_margin.initial_margin == 1900.0, f"Expected 1900, got {naked_margin.initial_margin}"
    print(f"Naked short call margin: ${naked_margin.initial_margin:.2f}")

    # --- Long option (margin = premium) ---
    long_call = OptionsPosition(
        position_id="long_test",
        strategy_name="long_call",
        underlying_symbol="AAPL",
        legs=[OptionLeg(
            expiration=exp, strike=110, option_type="C",
            quantity=1, entry_price=3.0, entry_date=today,
        )],
        entry_date=today,
    )
    long_margin = calc.calculate(long_call, underlying_price=105.0)
    assert long_margin.initial_margin == 300.0, f"Expected 300, got {long_margin.initial_margin}"
    print(f"Long call margin: ${long_margin.initial_margin:.2f}")

    # --- Portfolio margin model ---
    port_calc = MarginCalculator(model="portfolio")
    port_margin = port_calc.calculate(ic, underlying_price=100.0)
    assert port_margin.model == "portfolio"
    assert port_margin.maintenance_margin < port_margin.initial_margin
    print(f"Portfolio margin (IC): initial=${port_margin.initial_margin:.2f}, "
          f"maintenance=${port_margin.maintenance_margin:.2f}")

    print("\nAll margin tests passed.")
