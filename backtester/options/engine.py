"""
Options backtesting engine.

Provides the main loop and infrastructure for options strategy backtests.
Strategies receive daily underlying bars and can open/close options positions.
Option pricing is done via Black-Scholes using realized volatility, so only
OHLCV data for the underlying is required -- no historical options data needed.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Dict, Optional, Any, Literal

import numpy as np
import pandas as pd

from backtester.options.greeks import GreeksEngine, Greeks


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class OptionsPosition:
    """
    Represents a single options leg (one contract series).

    Quantities are signed: positive = long, negative = short.
    All dollar amounts are per-contract (multiply by abs(quantity) for total).
    """
    position_id: str = ""
    option_type: Literal["call", "put"] = "call"
    strike: float = 0.0
    expiration: date = field(default_factory=date.today)
    quantity: int = 0                    # +N = long, -N = short (in contracts)
    entry_price: float = 0.0            # premium per share at entry
    entry_date: date = field(default_factory=date.today)
    underlying_price_at_entry: float = 0.0
    iv_at_entry: float = 0.0
    tag: str = ""                        # strategy label

    def __post_init__(self):
        if not self.position_id:
            self.position_id = uuid.uuid4().hex[:12]


@dataclass
class OptionsBacktestConfig:
    """All tunable knobs for an options backtest run."""
    initial_capital: float = 100_000.0
    commission_per_contract: float = 0.65
    shares_per_contract: int = 100
    fill_at: str = "mid"                # "mid", "ask", "natural"
    spread_multiplier: float = 1.0      # widen synthetic spread
    max_pct_of_oi: float = 0.10
    market_impact_bps: float = 5.0
    risk_free_rate: float = 0.05
    bars_per_year: int = 252


@dataclass
class OptionsBacktestResult:
    """Results from an options backtest."""
    total_return_pct: float
    total_pnl: float
    num_trades: int                      # number of positions opened
    win_rate: float
    avg_trade_pnl: float
    best_trade_pnl: float
    worst_trade_pnl: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_premium_collected: float       # for selling strategies
    total_premium_paid: float            # for buying strategies
    total_commissions: float
    total_spread_cost: float             # cost from bid-ask spreads
    equity_curve: pd.Series              # daily equity
    benchmark_curve: pd.Series           # buy-and-hold underlying equity
    benchmark_return_pct: float          # underlying buy-and-hold return
    trade_log: List[Dict[str, Any]]      # list of trade records

    def summary(self) -> str:
        lines = [
            f"--- Options Backtest Results ---",
            f"Total return:          {self.total_return_pct:+.2f}%",
            f"Total P&L:             ${self.total_pnl:,.2f}",
            f"Num trades:            {self.num_trades}",
            f"Win rate:              {self.win_rate:.1f}%",
            f"Avg trade P&L:         ${self.avg_trade_pnl:,.2f}",
            f"Best trade P&L:        ${self.best_trade_pnl:,.2f}",
            f"Worst trade P&L:       ${self.worst_trade_pnl:,.2f}",
            f"Max drawdown:          {self.max_drawdown_pct:.2f}%",
            f"Sharpe ratio:          {self.sharpe_ratio:.2f}",
            f"Premium collected:     ${self.total_premium_collected:,.2f}",
            f"Premium paid:          ${self.total_premium_paid:,.2f}",
            f"Total commissions:     ${self.total_commissions:,.2f}",
            f"Total spread cost:     ${self.total_spread_cost:,.2f}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────

class OptionsBacktester:
    """
    Options backtesting engine.

    Simplified flow:
    1. Iterate through daily underlying OHLCV bars
    2. On each bar, strategy receives current state and can:
       - Open new positions (returns list of OptionsPosition to open)
       - Close existing positions (returns list of position_ids to close)
    3. Engine handles fills, expiration, P&L tracking

    Strategy provides synthetic option pricing via Black-Scholes using
    the underlying's realized volatility.
    """

    def __init__(self, config: OptionsBacktestConfig = None):
        self.config = config or OptionsBacktestConfig()
        self.greeks = GreeksEngine(risk_free_rate=self.config.risk_free_rate)

    def run(
        self,
        strategy,
        underlying_df: pd.DataFrame,
        chain_provider=None,
    ) -> OptionsBacktestResult:
        """
        Run options backtest.

        Args:
            strategy: OptionsStrategy instance
            underlying_df: OHLCV DataFrame for the underlying (DatetimeIndex)
            chain_provider: Optional callable(date, underlying_price) -> dict.
                           If None, strategy must provide its own pricing.

        Returns:
            OptionsBacktestResult with equity curve, trade log, and metrics.
        """
        cfg = self.config
        multiplier = cfg.shares_per_contract

        cash = cfg.initial_capital
        open_positions: Dict[str, OptionsPosition] = {}
        closed_trades: List[Dict[str, Any]] = []
        equity_values = []

        total_premium_collected = 0.0
        total_premium_paid = 0.0
        total_commissions = 0.0
        total_spread_cost = 0.0

        # Pre-compute realized vol series (rolling 20-day)
        close_series = underlying_df["Close"]
        log_returns = np.log(close_series / close_series.shift(1))
        realized_vol = log_returns.rolling(window=20, min_periods=5).std() * math.sqrt(252)
        realized_vol = realized_vol.fillna(0.20)  # default 20% if insufficient data

        dates = underlying_df.index
        strategy.initialize(underlying_df)

        # If strategy holds the underlying (e.g., covered call), buy shares at bar 0
        holds_underlying = getattr(strategy, "holds_underlying", False)
        shares_held = 0
        share_cost_basis = 0.0
        if holds_underlying:
            first_price = float(close_series.iloc[0])
            shares_held = int(cash // first_price)
            share_cost_basis = shares_held * first_price
            cash -= share_cost_basis

        for i in range(len(underlying_df)):
            row = underlying_df.iloc[i]
            current_date = dates[i]
            current_price = float(row["Close"])
            current_vol = float(realized_vol.iloc[i])

            # Ensure vol has a reasonable floor
            current_vol = max(current_vol, 0.05)

            # Convert current_date to a date object for DTE calculations
            if hasattr(current_date, "date"):
                bar_date = current_date.date()
            else:
                bar_date = current_date

            # ── 1. Mark-to-market existing positions ──────────────
            for pos in list(open_positions.values()):
                dte_days = (pos.expiration - bar_date).days
                T = max(dte_days / 365.0, 0.0)

                if dte_days <= 0:
                    # Position has expired
                    if pos.option_type == "call":
                        intrinsic = max(current_price - pos.strike, 0.0)
                    else:
                        intrinsic = max(pos.strike - current_price, 0.0)

                    # Settlement P&L
                    exit_price = intrinsic
                    pnl = (exit_price - pos.entry_price) * pos.quantity * multiplier

                    closed_trades.append({
                        "position_id": pos.position_id,
                        "option_type": pos.option_type,
                        "strike": pos.strike,
                        "expiration": pos.expiration,
                        "entry_date": pos.entry_date,
                        "exit_date": bar_date,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "quantity": pos.quantity,
                        "pnl": pnl,
                        "exit_reason": "expiration",
                        "tag": pos.tag,
                    })

                    cash += pnl
                    del open_positions[pos.position_id]

            # ── 2. Call strategy ──────────────────────────────────
            pos_list = list(open_positions.values())
            portfolio_value = cash + self._mark_to_market(
                pos_list, current_price, current_vol, bar_date, multiplier
            )

            new_positions, close_ids = strategy.on_bar(
                current_date=bar_date,
                underlying_price=current_price,
                open_positions=pos_list,
                portfolio_value=portfolio_value,
                cash=cash,
            )

            # ── 3. Close requested positions ──────────────────────
            for pid in close_ids:
                if pid not in open_positions:
                    continue
                pos = open_positions[pid]
                dte_days = (pos.expiration - bar_date).days
                T = max(dte_days / 365.0, 1e-6)

                exit_price = self.greeks.price(
                    current_price, pos.strike, T, current_vol, pos.option_type
                )

                # Spread cost on exit
                spread_cost = self._spread_cost(exit_price, current_vol)
                if pos.quantity > 0:
                    # Long position: sell at bid (lower)
                    exit_price -= spread_cost
                else:
                    # Short position: buy back at ask (higher)
                    exit_price += spread_cost

                exit_price = max(exit_price, 0.0)

                commission = cfg.commission_per_contract * abs(pos.quantity)
                pnl = (exit_price - pos.entry_price) * pos.quantity * multiplier - commission

                total_commissions += commission
                total_spread_cost += spread_cost * abs(pos.quantity) * multiplier

                closed_trades.append({
                    "position_id": pos.position_id,
                    "option_type": pos.option_type,
                    "strike": pos.strike,
                    "expiration": pos.expiration,
                    "entry_date": pos.entry_date,
                    "exit_date": bar_date,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "quantity": pos.quantity,
                    "pnl": pnl,
                    "exit_reason": "strategy_close",
                    "tag": pos.tag,
                })

                cash += pnl
                del open_positions[pid]

            # ── 4. Open new positions ─────────────────────────────
            for new_pos in (new_positions or []):
                dte_days = (new_pos.expiration - bar_date).days
                T = max(dte_days / 365.0, 1e-6)

                theo_price = self.greeks.price(
                    current_price, new_pos.strike, T, current_vol, new_pos.option_type
                )

                # Apply spread cost
                spread_cost = self._spread_cost(theo_price, current_vol)
                if new_pos.quantity > 0:
                    # Buying: pay the ask (higher)
                    fill_price = theo_price + spread_cost
                else:
                    # Selling: receive the bid (lower)
                    fill_price = theo_price - spread_cost

                fill_price = max(fill_price, 0.001)

                commission = cfg.commission_per_contract * abs(new_pos.quantity)
                premium_flow = fill_price * new_pos.quantity * multiplier

                # Premium tracking
                if new_pos.quantity > 0:
                    total_premium_paid += abs(premium_flow)
                else:
                    total_premium_collected += abs(premium_flow)

                total_commissions += commission
                total_spread_cost += spread_cost * abs(new_pos.quantity) * multiplier

                # Cash: buying costs premium, selling receives premium
                # premium_flow is negative when buying (qty>0 * price > 0 = positive,
                # but we pay it), positive when selling (qty<0 * price > 0 = negative,
                # but we receive it). Let's be explicit:
                cash -= premium_flow  # buy: -pos*price => cash decreases; sell: -neg*price => cash increases
                cash -= commission

                new_pos.entry_price = fill_price
                new_pos.entry_date = bar_date
                new_pos.underlying_price_at_entry = current_price
                new_pos.iv_at_entry = current_vol

                open_positions[new_pos.position_id] = new_pos

            # ── 5. Record equity ──────────────────────────────────
            mtm = self._mark_to_market(
                list(open_positions.values()), current_price, current_vol,
                bar_date, multiplier,
            )
            stock_value = shares_held * current_price if holds_underlying else 0.0
            equity_values.append(cash + mtm + stock_value)

        # ── Build results ─────────────────────────────────────────
        equity_curve = pd.Series(equity_values, index=dates, name="Equity")

        return self._build_result(
            equity_curve=equity_curve,
            closed_trades=closed_trades,
            initial_capital=cfg.initial_capital,
            total_premium_collected=total_premium_collected,
            total_premium_paid=total_premium_paid,
            total_commissions=total_commissions,
            total_spread_cost=total_spread_cost,
            bars_per_year=cfg.bars_per_year,
            underlying_prices=close_series,
        )

    # ── Helpers ──────────────────────────────────────────────────

    def _mark_to_market(
        self,
        positions: List[OptionsPosition],
        underlying_price: float,
        vol: float,
        bar_date: date,
        multiplier: int,
    ) -> float:
        """Return total mark-to-market value of open positions."""
        total = 0.0
        for pos in positions:
            dte_days = (pos.expiration - bar_date).days
            T = max(dte_days / 365.0, 1e-6)
            current_price = self.greeks.price(
                underlying_price, pos.strike, T, vol, pos.option_type
            )
            # Value = (current_price - entry_price) * quantity * multiplier
            # But for MTM we want the market value of the position, not P&L.
            # A long position has positive market value; a short has negative.
            total += current_price * pos.quantity * multiplier
        return total

    def _spread_cost(self, theo_price: float, vol: float) -> float:
        """
        Estimate half-spread cost.

        Uses a simple model: spread is proportional to price and volatility.
        Returns the half-spread (distance from mid to bid or ask).
        """
        cfg = self.config
        # Base spread: ~2% of option price, scaled by vol
        base_spread = theo_price * 0.02 * (vol / 0.20)
        # Market impact
        impact = theo_price * (cfg.market_impact_bps / 10_000)
        half_spread = (base_spread + impact) * cfg.spread_multiplier
        return max(half_spread, 0.001)

    @staticmethod
    def _build_result(
        equity_curve: pd.Series,
        closed_trades: List[Dict[str, Any]],
        initial_capital: float,
        total_premium_collected: float,
        total_premium_paid: float,
        total_commissions: float,
        total_spread_cost: float,
        bars_per_year: int,
        underlying_prices: pd.Series = None,
    ) -> OptionsBacktestResult:
        """Compute final metrics and build result object."""
        total_pnl = float(equity_curve.iloc[-1] - initial_capital)
        total_return = total_pnl / initial_capital * 100

        # Trade stats
        num_trades = len(closed_trades)
        pnls = [t["pnl"] for t in closed_trades]

        if num_trades > 0:
            wins = [p for p in pnls if p > 0]
            win_rate = len(wins) / num_trades * 100
            avg_pnl = sum(pnls) / num_trades
            best_pnl = max(pnls)
            worst_pnl = min(pnls)
        else:
            win_rate = 0.0
            avg_pnl = 0.0
            best_pnl = 0.0
            worst_pnl = 0.0

        # Max drawdown
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak * 100
        max_dd = float(drawdown.min())

        # Sharpe ratio
        daily_returns = equity_curve.pct_change().replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = float(
                daily_returns.mean() / daily_returns.std() * math.sqrt(bars_per_year)
            )
        else:
            sharpe = 0.0

        # Benchmark: buy-and-hold the underlying
        if underlying_prices is not None and len(underlying_prices) > 0:
            benchmark_curve = (underlying_prices / underlying_prices.iloc[0]) * initial_capital
            benchmark_curve.name = "Benchmark"
            benchmark_return = (underlying_prices.iloc[-1] / underlying_prices.iloc[0] - 1) * 100
        else:
            benchmark_curve = equity_curve.copy()
            benchmark_curve[:] = initial_capital
            benchmark_return = 0.0

        return OptionsBacktestResult(
            total_return_pct=total_return,
            total_pnl=total_pnl,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_trade_pnl=avg_pnl,
            best_trade_pnl=best_pnl,
            worst_trade_pnl=worst_pnl,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            total_premium_collected=total_premium_collected,
            total_premium_paid=total_premium_paid,
            total_commissions=total_commissions,
            total_spread_cost=total_spread_cost,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            benchmark_return_pct=float(benchmark_return),
            trade_log=closed_trades,
        )


# ─────────────────────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from strategy.covered_call import CoveredCallStrategy

    # Generate synthetic underlying data (~2 years of daily bars)
    np.random.seed(42)
    n_days = 504
    dates = pd.bdate_range(start="2022-01-03", periods=n_days, freq="B")

    # Geometric Brownian Motion: drift=8%, vol=20%
    dt = 1 / 252
    mu = 0.08
    sigma = 0.20
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn(n_days)
    prices = 100.0 * np.exp(np.cumsum(log_returns))

    # Build OHLCV
    high = prices * (1 + np.abs(np.random.randn(n_days)) * 0.005)
    low = prices * (1 - np.abs(np.random.randn(n_days)) * 0.005)
    open_prices = prices * (1 + np.random.randn(n_days) * 0.002)
    volume = np.random.randint(1_000_000, 10_000_000, size=n_days)

    df = pd.DataFrame(
        {
            "Open": open_prices,
            "High": high,
            "Low": low,
            "Close": prices,
            "Volume": volume,
        },
        index=dates,
    )

    # Run covered call backtest
    strategy = CoveredCallStrategy(
        target_dte=30,
        target_delta=0.30,
        profit_target_pct=0.50,
        close_dte=7,
        vol_lookback=20,
    )

    engine = OptionsBacktester(OptionsBacktestConfig(initial_capital=100_000))
    result = engine.run(strategy, df)

    print(result.summary())
    print(f"\nEquity curve: {len(result.equity_curve)} bars")
    print(f"  Start: ${result.equity_curve.iloc[0]:,.2f}")
    print(f"  End:   ${result.equity_curve.iloc[-1]:,.2f}")
    print(f"\nTrade log: {len(result.trade_log)} trades")
    if result.trade_log:
        for t in result.trade_log[:5]:
            print(
                f"  {t['entry_date']} -> {t['exit_date']}  "
                f"{t['option_type'].upper()} K={t['strike']:.0f}  "
                f"qty={t['quantity']}  pnl=${t['pnl']:+,.2f}  "
                f"({t['exit_reason']})"
            )
        if len(result.trade_log) > 5:
            print(f"  ... and {len(result.trade_log) - 5} more trades")
