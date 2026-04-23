"""
Implied volatility surface interpolation.

Fits a 2-D IV surface from an :class:`OptionChain` (DTE vs moneyness K/S)
and provides interpolated look-ups.  Falls back from cubic to linear to
nearest-neighbor interpolation when there are insufficient data points.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
from scipy.interpolate import griddata

from backtester.options.contracts import OptionChain


class IVSurface:
    """
    Implied volatility surface interpolation.

    Workflow
    --------
    1. ``surface.fit(chain)`` -- extract IV grid from an option chain.
    2. ``surface.get_iv(dte, moneyness)`` -- interpolate IV at any point.
    """

    def __init__(self):
        self._points: Optional[np.ndarray] = None   # (N, 2) array of [dte, moneyness]
        self._values: Optional[np.ndarray] = None    # (N,)   array of IV values
        self._fitted: bool = False

    def __repr__(self) -> str:
        n = len(self._values) if self._values is not None else 0
        return f"IVSurface(fitted={self._fitted}, points={n})"

    # ── Fitting ─────────────────────────────────────────────────

    def fit(self, chain: OptionChain) -> None:
        """
        Fit the IV surface from an option chain.

        Extracts DTE (days to expiration) and moneyness (K / S) from every
        contract that has a positive implied volatility and stores them for
        later interpolation.

        Parameters
        ----------
        chain : OptionChain
            Option chain with contracts whose ``implied_vol`` field is populated.
        """
        dtes = []
        moneyness = []
        ivs = []

        spot = chain.underlying_price
        if spot <= 0:
            raise ValueError("underlying_price must be positive for IV surface fitting")

        ref_date = chain.timestamp

        for c in chain.contracts:
            if c.implied_vol <= 0:
                continue

            dte = (c.expiration - ref_date).days
            if dte < 0:
                continue

            dtes.append(float(dte))
            moneyness.append(c.strike / spot)
            ivs.append(c.implied_vol)

        if len(ivs) < 1:
            raise ValueError("Need at least 1 contract with positive IV to fit surface")

        self._points = np.column_stack([dtes, moneyness])
        self._values = np.array(ivs)
        self._fitted = True

    # ── Interpolation ───────────────────────────────────────────

    def get_iv(self, dte: float, moneyness: float) -> float:
        """
        Interpolate implied volatility at a given DTE and moneyness.

        Uses cubic interpolation when possible, falls back to linear, then
        nearest-neighbor if insufficient points.

        Parameters
        ----------
        dte : float
            Days to expiration.
        moneyness : float
            Strike / Spot ratio (e.g. 1.0 = ATM).

        Returns
        -------
        float
            Interpolated implied volatility.

        Raises
        ------
        RuntimeError
            If the surface has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError("IVSurface has not been fitted -- call fit() first")

        query = np.array([[dte, moneyness]])

        # If we only have 1 point, return that value
        if len(self._values) == 1:
            return float(self._values[0])

        # Check dimensionality: if all points share the same DTE or same
        # moneyness the 2-D Delaunay triangulation degenerates.  Fall back
        # to 1-D interpolation along the non-degenerate axis, or nearest
        # if both axes are constant.
        unique_dtes = np.unique(self._points[:, 0])
        unique_money = np.unique(self._points[:, 1])

        if len(unique_dtes) == 1 and len(unique_money) == 1:
            # All points identical -- return the single value
            return float(self._values[0])

        if len(unique_dtes) == 1 or len(unique_money) == 1:
            # 1-D case: interpolate along the non-constant axis
            if len(unique_dtes) == 1:
                xs = self._points[:, 1]  # moneyness varies
                xq = moneyness
            else:
                xs = self._points[:, 0]  # dte varies
                xq = dte

            order = np.argsort(xs)
            xs_sorted = xs[order]
            vs_sorted = self._values[order]

            return float(np.interp(xq, xs_sorted, vs_sorted))

        # Full 2-D case: try cubic -> linear -> nearest
        for method in ("cubic", "linear", "nearest"):
            try:
                result = griddata(self._points, self._values, query, method=method)
                val = result[0]
                if not np.isnan(val):
                    return float(val)
            except Exception:
                continue

        # Absolute fallback: return nearest point by Euclidean distance
        dists = np.linalg.norm(self._points - query, axis=1)
        return float(self._values[np.argmin(dists)])

    def get_iv_for_contract(self, spot: float, strike: float, dte: float) -> float:
        """
        Convenience: compute moneyness and look up IV.

        Parameters
        ----------
        spot : float
            Current underlying price.
        strike : float
            Option strike.
        dte : float
            Days to expiration.

        Returns
        -------
        float
            Interpolated implied volatility.
        """
        if spot <= 0:
            raise ValueError("spot must be positive")
        return self.get_iv(dte, strike / spot)
