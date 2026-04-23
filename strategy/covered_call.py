"""
Covered call options strategy.

Sells monthly covered calls on the underlying, synthesizing option prices
from realized volatility via Black-Scholes. Works with just OHLCV data --
no historical options data needed.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

from backtester.options.engine import OptionsPosition
from backtester.options.greeks import GreeksEngine
from strategy.options_base import OptionsStrategy


class CoveredCallStrategy(OptionsStrategy):
    """
    Sell monthly covered calls on underlying.

    Logic:
    - When no call is open, sell a call ~30 DTE, delta ~0.30
    - Close at 50% profit or 7 DTE (whichever comes first)
    - Roll if about to expire ITM

    This is a simplified backtest that uses Black-Scholes to generate
    synthetic option prices from the underlying's realized volatility.
    """

    def __init__(
        self,
        target_dte: int = 30,
        target_delta: float = 0.30,
        profit_target_pct: float = 0.50,
        close_dte: int = 7,
        vol_lookback: int = 20,
        num_contracts: int = 1,
    ):
        self.target_dte = target_dte
        self.target_delta = target_delta
        self.profit_target_pct = profit_target_pct
        self.close_dte = close_dte
        self.vol_lookback = vol_lookback
        self.num_contracts = num_contracts

        self.greeks = GreeksEngine()
        self._vol_series: pd.Series = pd.Series(dtype=float)
        self._dates_index = None

        # Tell the engine to buy and hold the underlying stock
        self.holds_underlying = True

    @property
    def name(self) -> str:
        return f"Covered Call ({self.target_delta:.0%} delta, {self.target_dte}DTE)"

    def initialize(self, underlying_df: pd.DataFrame) -> None:
        """Pre-compute realized volatility from the underlying."""
        close = underlying_df["Close"]
        log_ret = np.log(close / close.shift(1))
        self._vol_series = (
            log_ret.rolling(window=self.vol_lookback, min_periods=5)
            .std()
            .mul(math.sqrt(252))
            .fillna(0.20)
        )
        self._dates_index = underlying_df.index

    def on_bar(
        self,
        current_date: date,
        underlying_price: float,
        open_positions: list,
        portfolio_value: float,
        cash: float,
    ) -> Tuple[List[OptionsPosition], List[str]]:
        """
        Decision logic for each bar.

        Uses realized volatility to synthesize option prices via Black-Scholes.
        """
        new_positions: List[OptionsPosition] = []
        close_ids: List[str] = []

        # Look up current realized vol
        current_vol = self._get_vol(current_date)

        # ── Check existing positions for exit signals ─────────
        for pos in open_positions:
            if pos.option_type != "call" or pos.quantity >= 0:
                continue  # only manage our short calls

            dte_days = (pos.expiration - current_date).days
            T = max(dte_days / 365.0, 1e-6)

            # Current theoretical price of the option
            current_opt_price = self.greeks.price(
                underlying_price, pos.strike, T, current_vol, "call"
            )

            # We sold at entry_price, current price is what we'd buy back at.
            # Profit for short position = entry_price - current_price
            profit_pct = (pos.entry_price - current_opt_price) / pos.entry_price \
                if pos.entry_price > 0 else 0.0

            # Exit condition 1: hit profit target
            if profit_pct >= self.profit_target_pct:
                close_ids.append(pos.position_id)
                continue

            # Exit condition 2: close at DTE threshold to avoid assignment risk
            if dte_days <= self.close_dte:
                close_ids.append(pos.position_id)
                continue

        # ── Open new position if no calls are open ────────────
        has_open_call = any(
            p.option_type == "call" and p.quantity < 0
            for p in open_positions
            if p.position_id not in close_ids
        )

        if not has_open_call:
            # Find a strike at target delta
            expiration_date = current_date + timedelta(days=self.target_dte)
            T = self.target_dte / 365.0

            strike = self.greeks.find_strike_for_delta(
                spot=underlying_price,
                tte=T,
                vol=current_vol,
                target_delta=self.target_delta,
                option_type="call",
            )

            # Round strike to nearest dollar
            strike = round(strike)

            new_pos = OptionsPosition(
                option_type="call",
                strike=float(strike),
                expiration=expiration_date,
                quantity=-self.num_contracts,  # negative = short
                tag="covered_call",
            )
            new_positions.append(new_pos)

        return new_positions, close_ids

    def _get_vol(self, current_date: date) -> float:
        """Look up realized vol for the current date."""
        if self._dates_index is None or self._vol_series.empty:
            return 0.20

        # Find the closest date in our index
        if hasattr(current_date, "date"):
            lookup = current_date
        else:
            lookup = pd.Timestamp(current_date)

        try:
            vol = float(self._vol_series.loc[lookup])
        except KeyError:
            # Find nearest date
            idx = self._dates_index.get_indexer([lookup], method="nearest")[0]
            if 0 <= idx < len(self._vol_series):
                vol = float(self._vol_series.iloc[idx])
            else:
                vol = 0.20

        return max(vol, 0.05)
