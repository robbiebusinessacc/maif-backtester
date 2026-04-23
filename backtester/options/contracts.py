"""
Core option data structures: OptionContract and OptionChain.

OptionContract is a snapshot of a single option at a point in time.
OptionChain is a collection of contracts for one underlying at one timestamp,
with filtering/lookup helpers for strike, expiry, delta, and ATM selection.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from typing import List, Optional

import pandas as pd


@dataclass
class OptionContract:
    """Single option contract snapshot at a point in time."""

    symbol: str              # underlying symbol e.g. "SPY"
    expiration: date         # expiry date
    strike: float
    option_type: str         # "C" or "P"
    bid: float
    ask: float
    last: Optional[float] = None
    open_interest: int = 0
    volume: int = 0
    implied_vol: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    @property
    def mid(self) -> float:
        """Mid-market price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def option_symbol(self) -> str:
        """Human-readable option identifier."""
        return f"{self.symbol} {self.expiration} {self.option_type} {self.strike}"


class OptionChain:
    """
    Collection of option contracts for one underlying at one timestamp.

    Provides filtering by type, expiry, strike proximity, ATM selection,
    and delta-based lookup.
    """

    def __init__(
        self,
        symbol: str,
        timestamp: date,
        underlying_price: float,
        contracts: List[OptionContract],
    ):
        self.symbol = symbol
        self.timestamp = timestamp
        self.underlying_price = underlying_price
        self.contracts = list(contracts)

    def __len__(self) -> int:
        return len(self.contracts)

    def __repr__(self) -> str:
        return (
            f"OptionChain({self.symbol}, {self.timestamp}, "
            f"spot={self.underlying_price}, contracts={len(self.contracts)})"
        )

    # ── Filtering ───────────────────────────────────────────────

    def calls(self) -> List[OptionContract]:
        """Return all call contracts."""
        return [c for c in self.contracts if c.option_type == "C"]

    def puts(self) -> List[OptionContract]:
        """Return all put contracts."""
        return [c for c in self.contracts if c.option_type == "P"]

    def at_expiry(self, expiration: date) -> "OptionChain":
        """Return a new OptionChain containing only contracts at the given expiry."""
        filtered = [c for c in self.contracts if c.expiration == expiration]
        return OptionChain(self.symbol, self.timestamp, self.underlying_price, filtered)

    def near_strike(self, strike: float, tolerance_pct: float = 2.0) -> List[OptionContract]:
        """Return contracts within *tolerance_pct* percent of the given strike."""
        lo = strike * (1 - tolerance_pct / 100)
        hi = strike * (1 + tolerance_pct / 100)
        return [c for c in self.contracts if lo <= c.strike <= hi]

    # ── Selection ───────────────────────────────────────────────

    def atm(self, option_type: str = "C") -> OptionContract:
        """
        Return the at-the-money contract (nearest strike to underlying_price).

        Parameters
        ----------
        option_type : str
            ``"C"`` for call, ``"P"`` for put.

        Raises
        ------
        ValueError
            If no contracts of the requested type exist.
        """
        candidates = [c for c in self.contracts if c.option_type == option_type]
        if not candidates:
            raise ValueError(f"No {option_type} contracts in chain")
        return min(candidates, key=lambda c: abs(c.strike - self.underlying_price))

    def by_delta(self, target_delta: float, option_type: str = "C") -> OptionContract:
        """
        Return the contract whose delta is closest to *target_delta*.

        Parameters
        ----------
        target_delta : float
            Target delta value (e.g. 0.30 for a 30-delta call).
        option_type : str
            ``"C"`` for call, ``"P"`` for put.

        Raises
        ------
        ValueError
            If no contracts of the requested type exist.
        """
        candidates = [c for c in self.contracts if c.option_type == option_type]
        if not candidates:
            raise ValueError(f"No {option_type} contracts in chain")
        return min(candidates, key=lambda c: abs(c.delta - target_delta))

    # ── Export ──────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all contracts to a pandas DataFrame."""
        if not self.contracts:
            return pd.DataFrame()
        field_names = [f.name for f in fields(OptionContract)]
        rows = [{fn: getattr(c, fn) for fn in field_names} for c in self.contracts]
        df = pd.DataFrame(rows)
        # Add computed columns
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df["spread"] = df["ask"] - df["bid"]
        return df
