"""
QUICKSTART: Add your own strategy and get a full scorecard.

This is the minimum code a new club member needs.
Just copy this file, replace MyStrategy with your logic, and run it.

    python3 quickstart.py
"""

from datetime import date
import pandas as pd
from strategy.base import Strategy, Signal


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: Write your strategy
#
# Implement generate_signals() to return BUY / SELL / HOLD
# for each bar. That's it — the framework handles everything
# else (position sizing, execution, costs, reporting).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MyStrategy(Strategy):

    @property
    def name(self) -> str:
        return "My Strategy"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(Signal.HOLD, index=df.index)

        # ── YOUR LOGIC HERE ──────────────────────────────────
        # Example: buy when 10-day SMA crosses above 30-day SMA
        fast = df["Close"].rolling(10).mean()
        slow = df["Close"].rolling(30).mean()

        for i in range(1, len(df)):
            if fast.iloc[i] > slow.iloc[i] and fast.iloc[i-1] <= slow.iloc[i-1]:
                signals.iloc[i] = Signal.BUY
            elif fast.iloc[i] < slow.iloc[i] and fast.iloc[i-1] >= slow.iloc[i-1]:
                signals.iloc[i] = Signal.SELL
        # ─────────────────────────────────────────────────────

        return signals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: Fetch data and generate your scorecard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from data_layer import DataLayer, YahooFinanceProvider
from backtester import BacktestConfig, generate_scorecard, EventDrivenBacktester

# Get SPY data (free, no API key)
dl = DataLayer()
dl.add_provider(YahooFinanceProvider())
df = dl.fetch("SPY", date(2022, 1, 1), date(2025, 12, 31))

# Configure the backtest
config = BacktestConfig(
    initial_capital=100_000,
    commission_per_order=1.00,
    slippage_bps=2.0,
)

# Run event-driven backtest for engine comparison
ed_result = EventDrivenBacktester(config).run(MyStrategy(), df)

# Generate the full 4-page scorecard
generate_scorecard(
    strategy=MyStrategy(),
    df=df,
    config=config,
    output_path="my_scorecard.png",
    event_driven_results={"result": ed_result},
)

# That's it. Check my_scorecard_scorecard.png for your grades.
