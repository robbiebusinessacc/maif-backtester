"""
Turtle-style trend following strategy.

Based on the original Turtle Trading rules by Richard Dennis (1983).
Uses Donchian channel breakouts for entries, ATR-based position sizing,
and trailing stops for exits. This is a well-documented, historically
profitable trend-following system.

Key properties:
  - Enters on N-day high breakout (momentum confirmation)
  - Sizes positions inversely to volatility (risk parity)
  - Exits on opposite channel break or trailing ATR stop
  - Long-only (simpler, avoids short-side complexities)
  - Should NOT profit on GBM (no trends in random walks)
  - Should profit in trending regimes, lose in sideways/chop
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.base import Strategy, StrategyState


class TurtleTrend(Strategy):
    """
    Simplified Turtle trend follower.

    Entry: buy when close breaks above the N-day high channel.
    Exit: sell when close breaks below the shorter exit channel,
          OR when price drops below entry - ATR * stop_mult (trailing stop).
    Position size: risk 2% of equity per ATR of movement.
    """

    def __init__(
        self,
        entry_period: int = 55,
        exit_period: int = 20,
        atr_period: int = 20,
        risk_pct: float = 0.02,
        stop_mult: float = 2.0,
    ):
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.risk_pct = risk_pct
        self.stop_mult = stop_mult

    @property
    def name(self) -> str:
        return f"Turtle Trend ({self.entry_period}/{self.exit_period}d, {self.stop_mult}ATR stop)"

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        df["entry_high"] = df["High"].rolling(self.entry_period).max().shift(1)
        df["exit_low"] = df["Low"].rolling(self.exit_period).min().shift(1)

        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.atr_period).mean()
        return df

    def target_position(self, row: pd.Series, state: StrategyState) -> int:
        close = row.get("Close", row.get("close", 0))
        entry_high = row.get("entry_high", np.nan)
        exit_low = row.get("exit_low", np.nan)
        atr = row.get("atr", np.nan)

        if pd.isna(entry_high) or pd.isna(exit_low) or pd.isna(atr) or atr <= 0:
            return state.current_position

        # Currently flat — look for breakout entry
        if state.current_position == 0:
            if close > entry_high:
                # Size: risk risk_pct of equity per ATR
                dollar_risk = state.equity * self.risk_pct
                shares = max(1, int(dollar_risk / (atr * self.stop_mult)))
                # Cap at what we can afford
                max_shares = int(state.cash * 0.95 / max(close, 1))
                return min(shares, max_shares)
            return 0

        # Currently long — check exits
        # Exit 1: price breaks below exit channel
        if close < exit_low:
            return 0

        # Exit 2: trailing ATR stop
        stop_price = state.avg_price - atr * self.stop_mult
        if close < stop_price:
            return 0

        return state.current_position
