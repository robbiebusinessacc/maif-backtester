"""
Black-Scholes Greeks computation engine.

Provides pricing, individual Greeks, bulk computation, and a Newton-Raphson
implied-volatility solver.  Uses ``scipy.stats.norm`` for the standard normal
CDF/PDF.

Edge-case handling
------------------
* ``tte <= 0`` : intrinsic value; delta = 1.0 (ITM call) / -1.0 (ITM put) / 0.0 (OTM)
* ``vol <= 0`` : intrinsic value (same as expired)
* ``spot <= 0`` or ``strike <= 0`` : returns 0.0 for all outputs
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
from scipy.stats import norm


@dataclass
class Greeks:
    """Option greeks snapshot."""
    price: float
    delta: float
    gamma: float
    theta: float       # per calendar day
    vega: float        # per 1% vol move
    iv: float = 0.0    # implied vol used


def _normalise_type(option_type: str) -> str:
    """Accept 'C'/'P', 'call'/'put', 'Call'/'Put' etc."""
    t = option_type.strip().upper()
    if t in ("C", "CALL"):
        return "C"
    if t in ("P", "PUT"):
        return "P"
    raise ValueError(f"Invalid option_type: {option_type!r}. Use 'C'/'P' or 'call'/'put'.")


class GreeksEngine:
    """Black-Scholes Greeks computation."""

    def __init__(self, risk_free_rate: float = 0.05, dividend_yield: float = 0.0):
        """
        Parameters
        ----------
        risk_free_rate : float
            Continuous risk-free rate (annualised, e.g. 0.05 for 5 %).
        dividend_yield : float
            Continuous dividend yield (annualised).
        """
        self.r = risk_free_rate
        self.q = dividend_yield

    # ── Internal helpers ────────────────────────────────────────

    def _d1d2(self, S: float, K: float, T: float, sigma: float):
        """Return (d1, d2) for the Black-Scholes formula."""
        d1 = (math.log(S / K) + (self.r - self.q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    def _safe(self, S: float, K: float, T: float, sigma: float) -> bool:
        """Return True if inputs are valid for normal BS computation."""
        return S > 0 and K > 0 and T > 0 and sigma > 0

    @staticmethod
    def _intrinsic(S: float, K: float, option_type: str) -> float:
        if S <= 0 or K <= 0:
            return 0.0
        if option_type == "C":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    @staticmethod
    def _intrinsic_delta(S: float, K: float, option_type: str) -> float:
        """Delta when option is at expiry or vol is zero."""
        if S <= 0 or K <= 0:
            return 0.0
        if option_type == "C":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    # ── Pricing ─────────────────────────────────────────────────

    def price(self, spot: float, strike: float, tte: float, vol: float, option_type: str = "C") -> float:
        """
        Black-Scholes option price.

        Parameters
        ----------
        spot : float
            Current price of the underlying.
        strike : float
            Strike price.
        tte : float
            Time to expiration in years.
        vol : float
            Annualised implied volatility (e.g. 0.20 for 20 %).
        option_type : str
            ``"C"`` for call, ``"P"`` for put.
        """
        option_type = _normalise_type(option_type)
        if not self._safe(spot, strike, tte, vol):
            return self._intrinsic(spot, strike, option_type)

        d1, d2 = self._d1d2(spot, strike, tte, vol)
        df_spot = math.exp(-self.q * tte)
        df_strike = math.exp(-self.r * tte)

        if option_type == "C":
            return spot * df_spot * norm.cdf(d1) - strike * df_strike * norm.cdf(d2)
        return strike * df_strike * norm.cdf(-d2) - spot * df_spot * norm.cdf(-d1)

    # ── Greeks ──────────────────────────────────────────────────

    def delta(self, spot: float, strike: float, tte: float, vol: float, option_type: str = "C") -> float:
        """Option delta (dV/dS)."""
        option_type = _normalise_type(option_type)
        if not self._safe(spot, strike, tte, vol):
            return self._intrinsic_delta(spot, strike, option_type)

        d1, _ = self._d1d2(spot, strike, tte, vol)
        df_spot = math.exp(-self.q * tte)

        if option_type == "C":
            return df_spot * norm.cdf(d1)
        return df_spot * (norm.cdf(d1) - 1)

    def gamma(self, spot: float, strike: float, tte: float, vol: float, option_type: str = "C") -> float:
        """Option gamma (d^2 V / dS^2).  Same for calls and puts."""
        option_type = _normalise_type(option_type)
        if not self._safe(spot, strike, tte, vol):
            return 0.0

        d1, _ = self._d1d2(spot, strike, tte, vol)
        df_spot = math.exp(-self.q * tte)
        return df_spot * norm.pdf(d1) / (spot * vol * math.sqrt(tte))

    def theta(self, spot: float, strike: float, tte: float, vol: float, option_type: str = "C") -> float:
        """
        Option theta (dV/dt), returned as value **per calendar day**.

        Negative for long options (time decay).
        """
        option_type = _normalise_type(option_type)
        if not self._safe(spot, strike, tte, vol):
            return 0.0

        d1, d2 = self._d1d2(spot, strike, tte, vol)
        df_spot = math.exp(-self.q * tte)
        df_strike = math.exp(-self.r * tte)
        sqrt_T = math.sqrt(tte)

        common = -(spot * df_spot * norm.pdf(d1) * vol) / (2 * sqrt_T)

        if option_type == "C":
            annual_theta = (
                common
                + self.q * spot * df_spot * norm.cdf(d1)
                - self.r * strike * df_strike * norm.cdf(d2)
            )
        else:
            annual_theta = (
                common
                - self.q * spot * df_spot * norm.cdf(-d1)
                + self.r * strike * df_strike * norm.cdf(-d2)
            )

        return annual_theta / 365.0

    def vega(self, spot: float, strike: float, tte: float, vol: float, option_type: str = "C") -> float:
        """
        Option vega (dV / d_sigma), scaled to a **1 percentage-point** move in vol.

        Same for calls and puts.
        """
        option_type = _normalise_type(option_type)
        if not self._safe(spot, strike, tte, vol):
            return 0.0

        d1, _ = self._d1d2(spot, strike, tte, vol)
        df_spot = math.exp(-self.q * tte)
        return spot * df_spot * norm.pdf(d1) * math.sqrt(tte) / 100.0

    def rho(self, spot: float, strike: float, tte: float, vol: float, option_type: str = "C") -> float:
        """Option rho (dV / dr), scaled to a 1 percentage-point move in rates."""
        option_type = _normalise_type(option_type)
        if not self._safe(spot, strike, tte, vol):
            return 0.0

        _, d2 = self._d1d2(spot, strike, tte, vol)
        df_strike = math.exp(-self.r * tte)

        if option_type == "C":
            return strike * tte * df_strike * norm.cdf(d2) / 100.0
        return -strike * tte * df_strike * norm.cdf(-d2) / 100.0

    # ── Bulk ────────────────────────────────────────────────────

    def all_greeks(
        self, spot: float, strike: float, tte: float, vol: float, option_type: str = "C",
    ) -> Dict[str, float]:
        """
        Compute all Greeks in one call.

        Returns
        -------
        dict
            Keys: ``price``, ``delta``, ``gamma``, ``theta``, ``vega``, ``rho``.
        """
        option_type = _normalise_type(option_type)
        return {
            "price": self.price(spot, strike, tte, vol, option_type),
            "delta": self.delta(spot, strike, tte, vol, option_type),
            "gamma": self.gamma(spot, strike, tte, vol, option_type),
            "theta": self.theta(spot, strike, tte, vol, option_type),
            "vega": self.vega(spot, strike, tte, vol, option_type),
            "rho": self.rho(spot, strike, tte, vol, option_type),
        }

    def greeks(
        self, spot: float, strike: float, tte: float, vol: float, option_type: str = "C",
    ) -> Greeks:
        """Compute all Greeks and return as a ``Greeks`` dataclass."""
        d = self.all_greeks(spot, strike, tte, vol, option_type)
        return Greeks(
            price=d["price"], delta=d["delta"], gamma=d["gamma"],
            theta=d["theta"], vega=d["vega"], iv=vol,
        )

    def find_strike_for_delta(
        self, spot: float, tte: float, vol: float, target_delta: float,
        option_type: str = "call",
    ) -> float:
        """Find strike that gives approximately ``target_delta`` via bisection."""
        option_type = _normalise_type(option_type)
        lo, hi = spot * 0.5, spot * 1.5
        for _ in range(50):
            mid = (lo + hi) / 2.0
            d = self.delta(spot, mid, tte, vol, option_type)
            if option_type == "C":
                if d > target_delta:
                    lo = mid
                else:
                    hi = mid
            else:
                if d < target_delta:
                    hi = mid
                else:
                    lo = mid
            if hi - lo < 0.01:
                break
        return round((lo + hi) / 2.0, 2)

    # ── Implied volatility ──────────────────────────────────────

    def implied_vol(
        self,
        market_price: float,
        spot: float,
        strike: float,
        tte: float,
        option_type: str,
        initial_guess: float = 0.25,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> float:
        """
        Newton-Raphson implied-volatility solver.

        Parameters
        ----------
        market_price : float
            Observed market price of the option.
        spot, strike, tte : float
            Underlying price, strike, and time to expiration (years).
        option_type : str
            ``"C"`` or ``"P"``.
        initial_guess : float
            Starting volatility for the solver.
        tol : float
            Convergence tolerance on price difference.
        max_iter : int
            Maximum Newton iterations.

        Returns
        -------
        float
            Implied volatility, or ``float('nan')`` if the solver fails to converge.
        """
        option_type = _normalise_type(option_type)
        if spot <= 0 or strike <= 0 or tte <= 0 or market_price <= 0:
            return float("nan")

        # Check that market_price exceeds intrinsic (otherwise no valid IV)
        intrinsic = self._intrinsic(spot, strike, option_type)
        if market_price < intrinsic - tol:
            return float("nan")

        sigma = initial_guess
        for _ in range(max_iter):
            bs_price = self.price(spot, strike, tte, sigma, option_type)
            diff = bs_price - market_price

            if abs(diff) < tol:
                return sigma

            # Vega in absolute terms (not /100 scaled)
            d1, _ = self._d1d2(spot, strike, tte, sigma)
            vega_abs = spot * math.exp(-self.q * tte) * norm.pdf(d1) * math.sqrt(tte)

            if vega_abs < 1e-12:
                break  # vega too small, can't converge

            sigma -= diff / vega_abs

            # Keep sigma in a reasonable range
            if sigma <= 0.001:
                sigma = 0.001
            elif sigma > 10.0:
                sigma = 10.0

        return float("nan")


# ─────────────────────────────────────────────────────────────
# Self-test: put-call parity and known BS values
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = GreeksEngine(risk_free_rate=0.05, dividend_yield=0.0)

    S, K, T, sigma = 100.0, 100.0, 1.0, 0.20
    call_price = engine.price(S, K, T, sigma, "C")
    put_price = engine.price(S, K, T, sigma, "P")

    # Put-call parity: C - P = S * exp(-qT) - K * exp(-rT)
    lhs = call_price - put_price
    rhs = S * math.exp(-engine.q * T) - K * math.exp(-engine.r * T)
    parity_diff = abs(lhs - rhs)

    print("=== Black-Scholes Verification ===\n")
    print(f"Inputs: S={S}, K={K}, T={T}, sigma={sigma}, r={engine.r}, q={engine.q}")
    print(f"Call price:  {call_price:.6f}")
    print(f"Put price:   {put_price:.6f}")
    print(f"Put-call parity LHS (C-P):          {lhs:.6f}")
    print(f"Put-call parity RHS (S - K*exp(-rT)):{rhs:.6f}")
    print(f"Parity difference:                   {parity_diff:.2e}")
    assert parity_diff < 1e-10, f"Put-call parity FAILED: diff={parity_diff}"
    print("PUT-CALL PARITY: PASS\n")

    # Known BS values for ATM call: S=100, K=100, T=1, sigma=0.2, r=0.05
    # Expected call ~ 10.4506, delta ~ 0.6368
    print(f"--- Known-value checks (ATM call) ---")
    expected_call = 10.4506
    assert abs(call_price - expected_call) < 0.01, (
        f"Call price mismatch: got {call_price:.4f}, expected ~{expected_call}"
    )
    print(f"Call price {call_price:.4f} matches expected ~{expected_call}: PASS")

    greeks = engine.all_greeks(S, K, T, sigma, "C")
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.4f}")
    print(f"Theta: {greeks['theta']:.4f} (per day)")
    print(f"Vega:  {greeks['vega']:.4f} (per 1% vol)")
    print(f"Rho:   {greeks['rho']:.4f} (per 1% rate)")

    expected_delta = 0.6368
    assert abs(greeks["delta"] - expected_delta) < 0.01, (
        f"Delta mismatch: got {greeks['delta']:.4f}, expected ~{expected_delta}"
    )
    print(f"Delta {greeks['delta']:.4f} matches expected ~{expected_delta}: PASS\n")

    # Greeks symmetry: call gamma == put gamma, call vega == put vega
    put_greeks = engine.all_greeks(S, K, T, sigma, "P")
    assert abs(greeks["gamma"] - put_greeks["gamma"]) < 1e-10, "Gamma call != put"
    assert abs(greeks["vega"] - put_greeks["vega"]) < 1e-10, "Vega call != put"
    print("Gamma symmetry (call == put): PASS")
    print("Vega symmetry (call == put):  PASS\n")

    # Implied vol round-trip
    iv = engine.implied_vol(call_price, S, K, T, "C")
    print(f"--- Implied Vol Round-trip ---")
    print(f"Input vol:    {sigma:.4f}")
    print(f"Recovered IV: {iv:.4f}")
    assert abs(iv - sigma) < 1e-6, f"IV mismatch: got {iv:.6f}, expected {sigma}"
    print("IV ROUND-TRIP: PASS\n")

    # Edge cases
    print("--- Edge cases ---")
    assert engine.price(100, 100, 0, 0.2, "C") == 0.0, "Expired ATM call should be 0"
    assert engine.price(105, 100, 0, 0.2, "C") == 5.0, "Expired ITM call should be intrinsic"
    assert engine.delta(105, 100, 0, 0.2, "C") == 1.0, "Expired ITM call delta should be 1"
    assert engine.delta(95, 100, 0, 0.2, "C") == 0.0, "Expired OTM call delta should be 0"
    assert engine.price(0, 100, 1, 0.2, "C") == 0.0, "Zero spot should return 0"
    assert engine.price(100, 0, 1, 0.2, "C") == 0.0, "Zero strike should return 0"
    assert engine.price(100, 100, 1, 0, "C") == 0.0, "Zero vol ATM should be 0 (intrinsic)"
    print("All edge cases: PASS\n")

    # OTM put parity check
    S2, K2 = 100.0, 110.0  # OTM call / ITM put
    c2 = engine.price(S2, K2, T, sigma, "C")
    p2 = engine.price(S2, K2, T, sigma, "P")
    lhs2 = c2 - p2
    rhs2 = S2 * math.exp(-engine.q * T) - K2 * math.exp(-engine.r * T)
    assert abs(lhs2 - rhs2) < 1e-10, "OTM put-call parity FAILED"
    print(f"OTM strike ({K2}) put-call parity: PASS")

    print("\n=== ALL TESTS PASSED ===")
