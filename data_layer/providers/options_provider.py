"""
Historical options chain provider using philippdubach/options-data.

Free dataset covering 104 major US equities/ETFs from Jan 2008 to Dec 2025.
Data includes: strike, expiry, bid, ask, volume, OI, IV, delta, gamma, theta,
vega, rho for every listed contract.

Source: https://github.com/philippdubach/options-data
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from backtester.options.contracts import OptionContract, OptionChain

# URL pattern — lowercase ticker
_BASE_URL = "https://static.philippdubach.com/data/options"

# Default local cache directory
_DEFAULT_CACHE = Path.home() / ".options_data"

# Available tickers (subset — full list at the GitHub repo)
AVAILABLE_TICKERS = [
    "AAPL", "ABNB", "ADBE", "AMD", "AMZN", "BA", "BABA", "BAC", "C",
    "CRM", "COIN", "COST", "CVX", "DIS", "DKNG", "EEM", "F", "FXI",
    "GDX", "GOOG", "GOOGL", "GS", "HYG", "INTC", "IWM", "JNK", "JPM",
    "KO", "LLY", "LQD", "META", "MARA", "MSTR", "MSFT", "MU", "NFLX",
    "NIO", "NKE", "NVDA", "PFE", "PLTR", "PYPL", "QQQ", "RIOT", "ROKU",
    "SHOP", "SLV", "SNAP", "SOFI", "SPY", "SQ", "T", "TGT", "TLT",
    "TSLA", "TSM", "UBER", "UNH", "USO", "V", "VXX", "WFC", "WMT",
    "XLE", "XLF", "XOM", "XSP",
]


class OptionsDataProvider:
    """
    Provides historical options chain data from philippdubach/options-data.

    Data is downloaded once per ticker and cached locally as parquet files.
    Subsequent loads read from cache.

    Usage::

        provider = OptionsDataProvider()

        # Get the full chain for SPY on a specific date
        chain = provider.get_chain("SPY", date(2024, 1, 15))

        # Get chains for a date range (returns dict of date -> OptionChain)
        chains = provider.get_chains("SPY", date(2024, 1, 1), date(2024, 3, 31))

        # Get raw DataFrame for custom queries
        df = provider.get_raw("SPY", date(2024, 1, 1), date(2024, 3, 31))
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: Dict[str, pd.DataFrame] = {}

    # ── Public API ──────────────────────────────────────────────

    def get_chain(self, symbol: str, on_date: date) -> OptionChain:
        """
        Get the full option chain for a symbol on a specific date.

        Returns an OptionChain with all contracts traded that day.
        """
        df = self._load(symbol)
        day = pd.Timestamp(on_date)
        day_df = df[df["date"] == day]

        # If no data or very few rows (holiday stub), find nearest real trading day
        if len(day_df) < 10:
            dates = df["date"].unique()
            idx = abs(dates - day).argmin()
            # Search outward for a date with real data
            for offset in range(10):
                for candidate_idx in [idx + offset, idx - offset]:
                    if 0 <= candidate_idx < len(dates):
                        candidate = dates[candidate_idx]
                        candidate_df = df[df["date"] == candidate]
                        if len(candidate_df) >= 10:
                            day_df = candidate_df
                            break
                if len(day_df) >= 10:
                    break

        actual_date = day_df["date"].iloc[0].date() if not day_df.empty else on_date
        underlying_price = self._get_underlying_price(symbol, actual_date)
        contracts = self._df_to_contracts(day_df, symbol)
        return OptionChain(
            symbol=symbol,
            timestamp=on_date,
            underlying_price=underlying_price,
            contracts=contracts,
        )

    def get_chains(
        self, symbol: str, start: date, end: date
    ) -> Dict[date, OptionChain]:
        """Get option chains for every trading day in a date range."""
        df = self._load(symbol)
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        filtered = df[mask]

        underlying_prices = self._get_underlying_series(symbol, start, end)

        chains = {}
        for day, group in filtered.groupby("date"):
            d = day.date()
            price = underlying_prices.get(d, 0.0)
            contracts = self._df_to_contracts(group, symbol)
            chains[d] = OptionChain(
                symbol=symbol,
                timestamp=d,
                underlying_price=price,
                contracts=contracts,
            )
        return chains

    def get_raw(
        self, symbol: str, start: date, end: date
    ) -> pd.DataFrame:
        """Get raw DataFrame for custom queries. Fastest for bulk analysis."""
        df = self._load(symbol)
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        return df[mask].copy()

    def available_dates(self, symbol: str) -> List[date]:
        """List all dates with data for a symbol."""
        df = self._load(symbol)
        return sorted(df["date"].dt.date.unique())

    def download(self, symbol: str) -> Path:
        """Download a ticker's data to the cache. Returns path to cached file."""
        return self._download(symbol.upper())

    # ── Internal ────────────────────────────────────────────────

    def _load(self, symbol: str) -> pd.DataFrame:
        """Load data for a symbol, downloading if needed."""
        sym = symbol.upper()
        if sym in self._loaded:
            return self._loaded[sym]

        cache_file = self.cache_dir / f"{sym}_options.parquet"
        if not cache_file.exists():
            self._download(sym)

        df = pd.read_parquet(cache_file)
        # Normalise date columns
        df["date"] = pd.to_datetime(df["date"])
        df["expiration"] = pd.to_datetime(df["expiration"])
        self._loaded[sym] = df
        return df

    def _download(self, symbol: str) -> Path:
        """Download parquet from static host (URLs are lowercase)."""
        import urllib.request

        ticker_lower = symbol.lower()
        opts_url = f"{_BASE_URL}/{ticker_lower}/options.parquet"
        underlying_url = f"{_BASE_URL}/{ticker_lower}/underlying.parquet"

        opts_path = self.cache_dir / f"{symbol}_options.parquet"
        underlying_path = self.cache_dir / f"{symbol}_underlying.parquet"

        print(f"Downloading {symbol} options data (may be 50-500 MB)...")
        urllib.request.urlretrieve(opts_url, str(opts_path))
        print(f"  Options: {opts_path.stat().st_size / 1e6:.1f} MB")

        try:
            urllib.request.urlretrieve(underlying_url, str(underlying_path))
            print(f"  Underlying: {underlying_path.stat().st_size / 1e6:.1f} MB")
        except Exception:
            pass  # underlying file is optional

        print(f"  Cached to: {self.cache_dir}")
        return opts_path

    def _get_underlying_price(self, symbol: str, on_date: date) -> float:
        """
        Get underlying close price for a date.

        Tries the underlying parquet first, then estimates from the option chain
        by finding the strike where call and put deltas are closest to ±0.50.
        """
        sym = symbol.upper()
        underlying_path = self.cache_dir / f"{sym}_underlying.parquet"

        if underlying_path.exists():
            udf = pd.read_parquet(underlying_path)
            if len(udf) > 0:
                udf["date"] = pd.to_datetime(udf["date"])
                day = pd.Timestamp(on_date)
                row = udf[udf["date"] == day]
                close_col = "close" if "close" in udf.columns else "Close"
                if not row.empty:
                    return float(row.iloc[0][close_col])
                idx = abs(udf["date"] - day).argmin()
                return float(udf.iloc[idx][close_col])

        # Fallback: estimate from option chain (strike where |delta| ≈ 0.50)
        return self._estimate_underlying_from_chain(sym, on_date)

    def _estimate_underlying_from_chain(self, symbol: str, on_date: date) -> float:
        """Estimate underlying price from ATM call delta ≈ 0.50 on a non-expiring chain."""
        df = self._load(symbol)
        day = pd.Timestamp(on_date)
        day_df = df[df["date"] == day]
        if day_df.empty:
            return 0.0

        calls = day_df[day_df["type"].str.lower().str.startswith("c")].copy()
        if calls.empty:
            return 0.0

        # Filter to expirations > 7 days out to avoid near-expiry delta distortion
        calls["dte"] = (calls["expiration"] - day).dt.days
        mid_term = calls[(calls["dte"] >= 7) & (calls["dte"] <= 60)]
        if mid_term.empty:
            mid_term = calls[calls["dte"] >= 1]
        if mid_term.empty:
            mid_term = calls

        # ATM call has delta closest to 0.50
        mid_term = mid_term.copy()
        mid_term["delta_dist"] = (mid_term["delta"] - 0.50).abs()
        atm = mid_term.loc[mid_term["delta_dist"].idxmin()]
        return float(atm["strike"])

    def _get_underlying_series(
        self, symbol: str, start: date, end: date
    ) -> Dict[date, float]:
        """Get dict of date -> underlying price."""
        sym = symbol.upper()
        underlying_path = self.cache_dir / f"{sym}_underlying.parquet"

        if underlying_path.exists():
            udf = pd.read_parquet(underlying_path)
            if len(udf) > 0:
                udf["date"] = pd.to_datetime(udf["date"])
                mask = (udf["date"] >= pd.Timestamp(start)) & (udf["date"] <= pd.Timestamp(end))
                filtered = udf[mask]
                close_col = "close" if "close" in filtered.columns else "Close"
                return {
                    row["date"].date(): float(row[close_col])
                    for _, row in filtered.iterrows()
                }

        # Fallback: estimate from options data per date
        result = {}
        df = self._load(sym)
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        for day, group in df[mask].groupby("date"):
            calls = group[group["type"].str.lower().str.startswith("c")]
            if not calls.empty:
                calls = calls.copy()
                calls["delta_dist"] = (calls["delta"] - 0.50).abs()
                atm = calls.loc[calls["delta_dist"].idxmin()]
                result[day.date()] = float(atm["strike"])
        return result

    @staticmethod
    def _df_to_contracts(df: pd.DataFrame, symbol: str) -> List[OptionContract]:
        """Convert DataFrame rows to OptionContract objects."""
        contracts = []
        for _, row in df.iterrows():
            opt_type = "C" if str(row.get("type", "call")).lower().startswith("c") else "P"
            contracts.append(OptionContract(
                symbol=symbol,
                expiration=row["expiration"].date() if hasattr(row["expiration"], "date") else row["expiration"],
                strike=float(row["strike"]),
                option_type=opt_type,
                bid=float(row.get("bid", 0)),
                ask=float(row.get("ask", 0)),
                last=float(row["last"]) if pd.notna(row.get("last")) else None,
                open_interest=int(row.get("open_interest", 0)),
                volume=int(row.get("volume", 0)),
                implied_vol=float(row.get("implied_volatility", 0)),
                delta=float(row.get("delta", 0)),
                gamma=float(row.get("gamma", 0)),
                theta=float(row.get("theta", 0)),
                vega=float(row.get("vega", 0)),
                rho=float(row.get("rho", 0)),
            ))
        return contracts


if __name__ == "__main__":
    provider = OptionsDataProvider()

    print("Downloading SPY options data...")
    provider.download("SPY")

    print("\nFetching chain for 2024-01-15...")
    chain = provider.get_chain("SPY", date(2024, 1, 15))
    print(f"  Underlying: ${chain.underlying_price:,.2f}")
    print(f"  Total contracts: {len(chain.contracts)}")
    print(f"  Calls: {len(chain.calls())}")
    print(f"  Puts: {len(chain.puts())}")

    expirations = sorted(set(c.expiration for c in chain.contracts))
    print(f"  Expirations: {len(expirations)} ({expirations[0]} to {expirations[-1]})")

    atm_call = chain.atm("C")
    print(f"\n  ATM Call: K={atm_call.strike}, bid={atm_call.bid:.2f}, "
          f"ask={atm_call.ask:.2f}, IV={atm_call.implied_vol:.2%}, "
          f"Δ={atm_call.delta:.3f}")

    atm_put = chain.atm("P")
    print(f"  ATM Put:  K={atm_put.strike}, bid={atm_put.bid:.2f}, "
          f"ask={atm_put.ask:.2f}, IV={atm_put.implied_vol:.2%}, "
          f"Δ={atm_put.delta:.3f}")

    print("\n  Done.")
