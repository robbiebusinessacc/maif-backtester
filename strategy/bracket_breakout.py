"""
Bracket breakout strategy — uses limit/stop orders and SL/TP brackets.

This strategy creates DIVERGENCE between bar-based and event-driven engines
because it uses order types that the bar engine cannot model:
  - STOP entry orders (buy on breakout above N-day high)
  - Bracket exits with SL/TP (OCO stop-loss + take-profit)
  - Limit entries (buy on pullback to support)

The bar engine converts everything to market orders at next-bar-open,
while the event engine fills stops/limits intrabar at the trigger price.
This price difference is the expected divergence.
"""

from __future__ import annotations

from typing import Any, List

import pandas as pd

from strategy.base import Strategy, StrategyState
from backtester.events import OrderEvent


class BracketBreakout(Strategy):
    """
    Breakout with bracket exits.

    Entry: STOP order above N-day high (buy the breakout).
    Exit: OCO bracket — stop-loss at entry - ATR*sl_mult,
          take-profit at entry + ATR*tp_mult.

    When run on the bar engine (which only does market orders at open),
    the entry will fill at next-bar-open instead of the breakout price.
    The SL/TP will use the simplified bar-engine logic instead of
    intrabar OCO execution.
    """

    def __init__(
        self,
        lookback: int = 20,
        atr_period: int = 14,
        sl_mult: float = 1.5,
        tp_mult: float = 3.0,
    ):
        self.lookback = lookback
        self.atr_period = atr_period
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult

    @property
    def name(self) -> str:
        return f"Bracket Breakout ({self.lookback}d, SL={self.sl_mult}ATR, TP={self.tp_mult}ATR)"

    @property
    def mode(self) -> str:
        return "signal"

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        df["highest"] = df["High"].rolling(self.lookback).max()
        df["lowest"] = df["Low"].rolling(self.lookback).min()
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.atr_period).mean()
        return df

    def on_bar_event(
        self,
        event,
        history: pd.DataFrame,
        state: StrategyState,
    ) -> Any:
        """
        Native event-driven hook — returns OrderEvent with stop entries
        and SL/TP brackets that the event engine handles properly.
        """
        if len(history) < self.lookback + 1:
            return []

        row = history.iloc[-1]
        prev = history.iloc[-2]
        atr = row.get("atr", 0)
        if atr <= 0 or pd.isna(atr):
            return []

        orders: List[OrderEvent] = []
        ts = event.timestamp

        # If flat, enter on breakout above previous high channel
        if state.current_position == 0:
            breakout_level = prev.get("highest", 0)
            if breakout_level > 0 and not pd.isna(breakout_level):
                sl_price = breakout_level - atr * self.sl_mult
                tp_price = breakout_level + atr * self.tp_mult
                shares = max(1, int(state.equity * 0.25 / breakout_level))

                orders.append(OrderEvent(
                    timestamp=ts,
                    symbol=event.symbol,
                    side="BUY",
                    quantity=shares,
                    order_type="STOP",
                    stop_price=breakout_level,
                    sl=sl_price,
                    tp=tp_price,
                    reason=f"breakout above {breakout_level:.2f}",
                ))

        return orders

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Fallback for bar engine — uses simple signal mode.
        Buys when close breaks above N-day high, sells when it breaks below.
        This is a SIMPLIFIED version that can't use stop entries or brackets.
        """
        from strategy.base import Signal
        signals = pd.Series(Signal.HOLD, index=df.index)
        if len(df) < self.lookback + 1:
            return signals

        highest = df["High"].rolling(self.lookback).max().shift(1)
        lowest = df["Low"].rolling(self.lookback).min().shift(1)

        for i in range(self.lookback + 1, len(df)):
            if df["Close"].iloc[i] > highest.iloc[i] and signals.iloc[i - 1] != Signal.BUY:
                signals.iloc[i] = Signal.BUY
            elif df["Close"].iloc[i] < lowest.iloc[i] and signals.iloc[i - 1] != Signal.SELL:
                signals.iloc[i] = Signal.SELL

        return signals

    def get_sl_tp(self, entry_price, direction):
        """Bar engine SL/TP — rough fixed percentage since bar engine can't track ATR per trade."""
        is_long = direction == "LONG" if isinstance(direction, str) else direction > 0
        if is_long:
            return entry_price * 0.97, entry_price * 1.06
        return entry_price * 1.03, entry_price * 0.94
