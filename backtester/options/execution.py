"""
Realistic options execution with bid-ask modeling.

Models fill prices based on spread location (mid, ask, natural),
market impact, open interest capacity checks, and per-contract
commissions.
"""

from __future__ import annotations

from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────
# Fill Result
# ─────────────────────────────────────────────────────────────

@dataclass
class OptionsFill:
    """Result of an options order execution."""

    filled: bool
    fill_price: float = 0.0
    quantity: int = 0
    cost: float = 0.0          # total $ cost including spread
    slippage: float = 0.0      # $ lost to spread


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class OptionsExecutionConfig:
    """Configuration for options execution modeling."""

    fill_at: str = "mid"       # "mid", "ask" (for buys), "natural" (cross the spread)
    spread_multiplier: float = 1.0   # scale bid-ask spread (>1 = pessimistic)
    market_impact_bps: float = 5.0
    max_pct_of_oi: float = 0.10     # reject if order > 10% of open interest
    commission_per_contract: float = 0.65  # typical broker fee


# ─────────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────────

class OptionsExecutor:
    """Execute options orders with realistic fill modeling."""

    def __init__(self, config: OptionsExecutionConfig | None = None):
        self.config = config or OptionsExecutionConfig()

    def fill_order(
        self,
        side: str,
        bid: float,
        ask: float,
        quantity: int,
        open_interest: int,
    ) -> OptionsFill:
        """
        Simulate filling an options order.

        Parameters
        ----------
        side : str
            "BUY" or "SELL"
        bid : float
            Current best bid price.
        ask : float
            Current best ask price.
        quantity : int
            Number of contracts (unsigned).
        open_interest : int
            Current open interest for this strike/expiry.

        Returns
        -------
        OptionsFill with fill details.
        """
        cfg = self.config

        # --- Capacity check ---
        if open_interest > 0 and quantity > cfg.max_pct_of_oi * open_interest:
            return OptionsFill(filled=False)

        # --- Compute raw spread ---
        raw_spread = max(ask - bid, 0.0)
        spread = raw_spread * cfg.spread_multiplier
        mid = (bid + ask) / 2.0

        # --- Determine base fill price ---
        if cfg.fill_at == "mid":
            base_price = mid
        elif cfg.fill_at == "ask":
            # Buyers pay ask, sellers receive bid
            base_price = ask if side == "BUY" else bid
        elif cfg.fill_at == "natural":
            # Cross the spread: buyers pay ask, sellers hit bid
            base_price = ask if side == "BUY" else bid
        else:
            base_price = mid

        # --- Market impact ---
        impact = base_price * (cfg.market_impact_bps / 10_000)
        if side == "BUY":
            fill_price = base_price + impact
        else:
            fill_price = base_price - impact

        # Ensure fill price is non-negative
        fill_price = max(fill_price, 0.0)

        # --- Slippage = difference from mid ---
        slippage_per_contract = abs(fill_price - mid) * 100  # per contract in $
        total_slippage = slippage_per_contract * quantity

        # --- Commission ---
        commission = cfg.commission_per_contract * quantity

        # --- Total cost ---
        # For buys: pay fill_price per share * 100 shares * quantity + commission
        # For sells: receive fill_price per share * 100 shares * quantity - commission
        notional = fill_price * 100 * quantity
        if side == "BUY":
            cost = notional + commission
        else:
            cost = -notional + commission  # negative = cash received, + commission

        return OptionsFill(
            filled=True,
            fill_price=fill_price,
            quantity=quantity,
            cost=cost,
            slippage=total_slippage,
        )


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Basic mid-fill ---
    executor = OptionsExecutor()
    fill = executor.fill_order("BUY", bid=2.00, ask=2.20, quantity=10, open_interest=5000)
    assert fill.filled is True
    assert fill.quantity == 10
    # Mid = 2.10, impact = 2.10 * 5/10000 = 0.00105 => fill ~ 2.10105
    assert abs(fill.fill_price - 2.10105) < 0.001, f"Got {fill.fill_price}"
    # Commission = 0.65 * 10 = 6.50
    expected_cost = fill.fill_price * 100 * 10 + 6.50
    assert abs(fill.cost - expected_cost) < 0.01, f"Got {fill.cost}"
    print(f"Mid-fill BUY: price={fill.fill_price:.5f}, cost=${fill.cost:.2f}, slippage=${fill.slippage:.2f}")

    # --- Sell at mid ---
    sell_fill = executor.fill_order("SELL", bid=2.00, ask=2.20, quantity=5, open_interest=5000)
    assert sell_fill.filled is True
    # Mid = 2.10, impact = -0.00105 => fill ~ 2.09895
    assert sell_fill.fill_price < 2.10, f"Got {sell_fill.fill_price}"
    print(f"Mid-fill SELL: price={sell_fill.fill_price:.5f}, cost=${sell_fill.cost:.2f}")

    # --- Capacity rejection ---
    rejected = executor.fill_order("BUY", bid=2.00, ask=2.20, quantity=600, open_interest=5000)
    assert rejected.filled is False, "Should reject order exceeding 10% of OI"
    print("Capacity rejection: correctly rejected")

    # --- Ask fill mode ---
    ask_executor = OptionsExecutor(OptionsExecutionConfig(fill_at="ask"))
    ask_fill = ask_executor.fill_order("BUY", bid=2.00, ask=2.20, quantity=1, open_interest=5000)
    assert ask_fill.fill_price > 2.20, f"Ask fill should be >= ask, got {ask_fill.fill_price}"
    print(f"Ask-fill BUY: price={ask_fill.fill_price:.5f}")

    # --- Pessimistic spread multiplier ---
    pessimistic = OptionsExecutor(OptionsExecutionConfig(spread_multiplier=2.0))
    pess_fill = pessimistic.fill_order("BUY", bid=2.00, ask=2.20, quantity=1, open_interest=5000)
    assert pess_fill.filled is True
    print(f"Pessimistic fill: price={pess_fill.fill_price:.5f}")

    # --- Zero OI treated as unlimited ---
    zero_oi = executor.fill_order("BUY", bid=1.00, ask=1.10, quantity=100, open_interest=0)
    assert zero_oi.filled is True, "Zero OI should not reject"
    print("Zero OI: correctly filled")

    print("\nAll execution tests passed.")
