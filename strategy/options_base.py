"""Abstract base class for options trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import List, Tuple

import pandas as pd


class OptionsStrategy(ABC):
    """
    Base class for options trading strategies.

    Strategies receive daily underlying bars and decide when to open/close
    options positions. The engine handles pricing (via Black-Scholes),
    fills, expirations, and P&L tracking.

    Lifecycle:
    1. initialize() is called once with the full underlying DataFrame
    2. on_bar() is called for each daily bar with current state
    3. Strategy returns new positions to open and existing position IDs to close
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    def initialize(self, underlying_df: pd.DataFrame) -> None:
        """
        Called once before the backtest loop.

        Use this to pre-compute indicators (e.g., realized volatility)
        from the underlying data.

        Args:
            underlying_df: Full OHLCV DataFrame for the underlying.
        """
        pass

    @abstractmethod
    def on_bar(
        self,
        current_date: date,
        underlying_price: float,
        open_positions: list,
        portfolio_value: float,
        cash: float,
    ) -> Tuple[list, List[str]]:
        """
        Called on each bar.

        Args:
            current_date: The current bar date.
            underlying_price: Current underlying close price.
            open_positions: List of OptionsPosition currently open.
            portfolio_value: Total portfolio value (cash + MTM).
            cash: Available cash.

        Returns:
            Tuple of (new_positions, close_position_ids):
            - new_positions: list of OptionsPosition to open
            - close_position_ids: list of position_id strings to close
        """
