"""
Multi-leg option position tracking.

Supports arbitrary combinations of calls and puts with signed quantities,
aggregate Greeks, and payoff analysis. Factory methods create common
strategies (covered call, vertical spread, iron condor, straddle).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────
# Option Leg
# ─────────────────────────────────────────────────────────────

@dataclass
class OptionLeg:
    """Single leg of an options position."""

    expiration: date
    strike: float
    option_type: str           # "C" or "P"
    quantity: int              # signed: + = long, - = short
    entry_price: float         # price per contract at entry
    entry_date: date

    # Current Greeks (updated by tracker)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

    def intrinsic_value(self, spot: float) -> float:
        """Intrinsic value per share at a given spot price."""
        if self.option_type == "C":
            return max(spot - self.strike, 0)
        return max(self.strike - spot, 0)

    def payoff_at_expiry(self, spot: float) -> float:
        """P&L per contract at expiration (includes entry premium)."""
        return (self.intrinsic_value(spot) - self.entry_price) * self.quantity * 100


# ─────────────────────────────────────────────────────────────
# Options Position
# ─────────────────────────────────────────────────────────────

@dataclass
class OptionsPosition:
    """Multi-leg options position."""

    position_id: str
    strategy_name: str          # "covered_call", "iron_condor", etc.
    underlying_symbol: str
    legs: List[OptionLeg]
    entry_date: date

    # Aggregate Greeks
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0

    # P&L tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    is_closed: bool = False

    def net_premium(self) -> float:
        """Net premium paid (positive) or received (negative) at entry."""
        return sum(leg.entry_price * leg.quantity * 100 for leg in self.legs)

    def max_profit(self) -> float:
        """Estimate max profit by scanning spot prices at expiry."""
        avg_strike = np.mean([leg.strike for leg in self.legs])
        spots = np.linspace(avg_strike * 0.5, avg_strike * 2.0, 1000)
        payoffs = np.array([self.payoff_at_expiry(s) for s in spots])
        return float(np.max(payoffs))

    def max_loss(self) -> float:
        """Estimate max loss by scanning spot prices at expiry."""
        avg_strike = np.mean([leg.strike for leg in self.legs])
        spots = np.linspace(avg_strike * 0.5, avg_strike * 2.0, 1000)
        payoffs = np.array([self.payoff_at_expiry(s) for s in spots])
        return float(np.min(payoffs))

    def payoff_at_expiry(self, spot: float) -> float:
        """Total P&L across all legs at given spot price."""
        return sum(leg.payoff_at_expiry(spot) for leg in self.legs)

    def update_greeks(self) -> None:
        """Recompute aggregate Greeks from legs."""
        self.net_delta = sum(leg.delta * leg.quantity for leg in self.legs)
        self.net_gamma = sum(leg.gamma * leg.quantity for leg in self.legs)
        self.net_theta = sum(leg.theta * leg.quantity for leg in self.legs)
        self.net_vega = sum(leg.vega * leg.quantity for leg in self.legs)


# ─────────────────────────────────────────────────────────────
# Position Factory
# ─────────────────────────────────────────────────────────────

class PositionFactory:
    """Factory methods for common option strategies."""

    @staticmethod
    def _make_id() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def covered_call(
        symbol: str,
        expiration: date,
        call_strike: float,
        call_price: float,
        entry_date: date,
    ) -> OptionsPosition:
        """Long stock + short OTM call. Caller handles the stock leg separately."""
        leg = OptionLeg(
            expiration=expiration,
            strike=call_strike,
            option_type="C",
            quantity=-1,
            entry_price=call_price,
            entry_date=entry_date,
        )
        return OptionsPosition(
            position_id=PositionFactory._make_id(),
            strategy_name="covered_call",
            underlying_symbol=symbol,
            legs=[leg],
            entry_date=entry_date,
        )

    @staticmethod
    def vertical_spread(
        symbol: str,
        expiration: date,
        long_strike: float,
        short_strike: float,
        long_price: float,
        short_price: float,
        option_type: str,
        entry_date: date,
    ) -> OptionsPosition:
        """Bull/bear spread with two legs of the same option type."""
        long_leg = OptionLeg(
            expiration=expiration,
            strike=long_strike,
            option_type=option_type,
            quantity=1,
            entry_price=long_price,
            entry_date=entry_date,
        )
        short_leg = OptionLeg(
            expiration=expiration,
            strike=short_strike,
            option_type=option_type,
            quantity=-1,
            entry_price=short_price,
            entry_date=entry_date,
        )
        return OptionsPosition(
            position_id=PositionFactory._make_id(),
            strategy_name="vertical_spread",
            underlying_symbol=symbol,
            legs=[long_leg, short_leg],
            entry_date=entry_date,
        )

    @staticmethod
    def iron_condor(
        symbol: str,
        expiration: date,
        put_long_strike: float,
        put_short_strike: float,
        call_short_strike: float,
        call_long_strike: float,
        put_long_price: float,
        put_short_price: float,
        call_short_price: float,
        call_long_price: float,
        entry_date: date,
    ) -> OptionsPosition:
        """Iron condor: short OTM put spread + short OTM call spread."""
        legs = [
            OptionLeg(expiration=expiration, strike=put_long_strike,
                      option_type="P", quantity=1, entry_price=put_long_price,
                      entry_date=entry_date),
            OptionLeg(expiration=expiration, strike=put_short_strike,
                      option_type="P", quantity=-1, entry_price=put_short_price,
                      entry_date=entry_date),
            OptionLeg(expiration=expiration, strike=call_short_strike,
                      option_type="C", quantity=-1, entry_price=call_short_price,
                      entry_date=entry_date),
            OptionLeg(expiration=expiration, strike=call_long_strike,
                      option_type="C", quantity=1, entry_price=call_long_price,
                      entry_date=entry_date),
        ]
        return OptionsPosition(
            position_id=PositionFactory._make_id(),
            strategy_name="iron_condor",
            underlying_symbol=symbol,
            legs=legs,
            entry_date=entry_date,
        )

    @staticmethod
    def straddle(
        symbol: str,
        expiration: date,
        strike: float,
        call_price: float,
        put_price: float,
        entry_date: date,
        long: bool = True,
    ) -> OptionsPosition:
        """Long or short straddle at a single strike."""
        qty = 1 if long else -1
        legs = [
            OptionLeg(expiration=expiration, strike=strike,
                      option_type="C", quantity=qty, entry_price=call_price,
                      entry_date=entry_date),
            OptionLeg(expiration=expiration, strike=strike,
                      option_type="P", quantity=qty, entry_price=put_price,
                      entry_date=entry_date),
        ]
        return OptionsPosition(
            position_id=PositionFactory._make_id(),
            strategy_name="long_straddle" if long else "short_straddle",
            underlying_symbol=symbol,
            legs=legs,
            entry_date=entry_date,
        )


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import date

    today = date(2026, 4, 1)
    exp = date(2026, 5, 15)

    # --- Bull call spread: long 100C @ $5, short 110C @ $2 ---
    spread = PositionFactory.vertical_spread(
        symbol="AAPL", expiration=exp,
        long_strike=100, short_strike=110,
        long_price=5.0, short_price=2.0,
        option_type="C", entry_date=today,
    )

    # Net premium paid = 5*1*100 + 2*(-1)*100 = 300
    premium = spread.net_premium()
    assert premium == 300.0, f"Expected 300, got {premium}"

    # At spot=90 (both OTM): payoff = (0 - 5)*1*100 + (0 - 2)*(-1)*100 = -500 + 200 = -300
    p90 = spread.payoff_at_expiry(90)
    assert p90 == -300.0, f"Expected -300, got {p90}"

    # At spot=115 (both ITM): payoff = (15 - 5)*1*100 + (5 - 2)*(-1)*100 = 1000 - 300 = 700
    p115 = spread.payoff_at_expiry(115)
    assert p115 == 700.0, f"Expected 700, got {p115}"

    # Max profit should be 700
    mp = spread.max_profit()
    assert abs(mp - 700.0) < 1.0, f"Expected ~700, got {mp}"

    # Max loss should be -300
    ml = spread.max_loss()
    assert abs(ml - (-300.0)) < 1.0, f"Expected ~-300, got {ml}"

    # --- Iron condor ---
    ic = PositionFactory.iron_condor(
        symbol="SPY", expiration=exp,
        put_long_strike=90, put_short_strike=95,
        call_short_strike=105, call_long_strike=110,
        put_long_price=0.50, put_short_price=1.50,
        call_short_price=1.50, call_long_price=0.50,
        entry_date=today,
    )

    # Net premium received: 0.50*1*100 + 1.50*(-1)*100 + 1.50*(-1)*100 + 0.50*1*100
    #                     = 50 - 150 - 150 + 50 = -200 (received $200)
    ic_prem = ic.net_premium()
    assert ic_prem == -200.0, f"Expected -200, got {ic_prem}"

    # At spot=100 (all OTM): payoff = -premium costs sum to +200
    p100 = ic.payoff_at_expiry(100)
    assert p100 == 200.0, f"Expected 200, got {p100}"

    # --- Long straddle ---
    strad = PositionFactory.straddle(
        symbol="TSLA", expiration=exp, strike=200,
        call_price=10.0, put_price=10.0,
        entry_date=today, long=True,
    )
    # At spot=200 (ATM): payoff = (0 - 10)*1*100 + (0 - 10)*1*100 = -2000
    p200 = strad.payoff_at_expiry(200)
    assert p200 == -2000.0, f"Expected -2000, got {p200}"

    # At spot=230: call pays (30-10)*100=2000, put pays (0-10)*100=-1000 => 1000
    p230 = strad.payoff_at_expiry(230)
    assert p230 == 1000.0, f"Expected 1000, got {p230}"

    # --- Greeks aggregation ---
    spread.legs[0].delta = 0.60
    spread.legs[0].gamma = 0.03
    spread.legs[1].delta = 0.40
    spread.legs[1].gamma = 0.025
    spread.update_greeks()
    # net_delta = 0.60*1 + 0.40*(-1) = 0.20
    assert abs(spread.net_delta - 0.20) < 1e-9, f"Expected 0.20, got {spread.net_delta}"

    print("All positions tests passed.")
