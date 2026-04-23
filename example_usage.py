"""
Quick start — run a strategy through the full pipeline.

Just run: python3 example_usage.py
"""

from datetime import date
from data_layer import DataLayer, YahooFinanceProvider
from strategy import SMACrossover
from backtester import (
    Backtester, BacktestConfig, generate_scorecard,
    optimize, plot_heatmap, run_validation_suite,
    generate_distribution_plots,
)

# ═══════════════════════════════════════════════════════════════
# EQUITIES — SMA Crossover on SPY
# ═══════════════════════════════════════════════════════════════

# 1. Get data (Yahoo Finance — no API key needed)
dl = DataLayer()
dl.add_provider(YahooFinanceProvider())

df = dl.fetch("SPY", date(2022, 1, 1), date(2025, 12, 31))
print(f"Data: {len(df)} bars\n")

# 2. Backtest
config = BacktestConfig(
    initial_capital=100_000,
    commission_per_order=1.00,
    slippage_bps=2.0,
)
result = Backtester(config).run(SMACrossover(10, 30), df)
print(result.summary())

# 3. Optimize (Optuna — smart Bayesian search)
print("\n" + "=" * 50)
best, params, results = optimize(
    SMACrossover, df, config,
    maximize="sharpe_ratio",
    method="optuna",
    n_trials=30,
    constraint=lambda p: p["fast_period"] < p["slow_period"],
    fast_period=[5, 10, 15, 20],
    slow_period=[20, 30, 40, 50, 60],
)
print(f"\nOptimal strategy: SMA Crossover ({params['fast_period']}/{params['slow_period']})")
print(f"Sharpe: {best.sharpe_ratio:.2f}  |  Return: {best.total_return_pct:+.1f}%")

# 4. Validate on synthetic data
print("\n" + "=" * 50)
run_validation_suite(SMACrossover(**params), config=config)

# 5. Event-driven backtest (validates bar-based results)
from backtester import EventDrivenBacktester

print("\n" + "=" * 50)
print("Event-Driven Backtest:")
ed_result = EventDrivenBacktester(config).run(SMACrossover(**params), df)
print(f"  Bar return:   {best.total_return_pct:+.2f}%")
print(f"  Event return: {ed_result.total_return_pct:+.2f}%")
print(f"  Divergence:   {abs(best.total_return_pct - ed_result.total_return_pct):.2f}pp")

# 6. Generate scorecard with event-driven results connected
print()
generate_scorecard(
    strategy=SMACrossover(**params),
    df=df,
    config=config,
    output_path="scorecard.png",
    event_driven_results={"result": ed_result},
)


# ═══════════════════════════════════════════════════════════════
# CRYPTO — SMA Crossover on BTC/USD
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("CRYPTO BACKTEST: SMA Crossover on BTC/USD")
print("=" * 60)

from data_layer import CCXTProvider

# Fetch BTC data from Kraken (free, no API key)
crypto_provider = CCXTProvider("kraken")
btc = crypto_provider.fetch_ohlcv("BTC/USD", date(2024, 1, 1), date(2025, 12, 31))
print(f"Data: {len(btc)} daily bars from {btc.index[0].date()} to {btc.index[-1].date()}")
print(f"Price range: ${btc['Close'].min():,.0f} - ${btc['Close'].max():,.0f}\n")

# Use crypto config (365 bars/year for 24/7 markets)
crypto_config = BacktestConfig.for_crypto(
    initial_capital=100_000,
    commission_per_order=5.0,
    slippage_bps=5.0,
)
crypto_result = Backtester(crypto_config).run(SMACrossover(10, 30), btc)
print(crypto_result.summary())

# Crypto scorecard
generate_scorecard(
    strategy=SMACrossover(10, 30),
    df=btc,
    config=crypto_config,
    output_path="crypto_scorecard.png",
    symbol="BTC/USD",
)


# ═══════════════════════════════════════════════════════════════
# OPTIONS — Covered Call on SPY
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("OPTIONS BACKTEST: Covered Call on SPY")
print("=" * 60)

from strategy import CoveredCallStrategy
from backtester import generate_options_scorecard
from backtester.options import OptionsBacktester, OptionsBacktestConfig

# Uses the same SPY data — no options data needed!
# Option prices are synthesized via Black-Scholes from realized vol.
options_config = OptionsBacktestConfig(initial_capital=100_000)
options_engine = OptionsBacktester(options_config)
options_result = options_engine.run(
    CoveredCallStrategy(target_dte=30, target_delta=0.30),
    df,
)

print(f"Total Return:      {options_result.total_return_pct:+.2f}%")
print(f"Total P&L:         ${options_result.total_pnl:,.2f}")
print(f"Trades:            {options_result.num_trades}")
print(f"Win Rate:          {options_result.win_rate:.1f}%")
print(f"Sharpe Ratio:      {options_result.sharpe_ratio:.2f}")
print(f"Max Drawdown:      {options_result.max_drawdown_pct:.2f}%")
print(f"Premium Collected: ${options_result.total_premium_collected:,.2f}")
print(f"Commissions:       ${options_result.total_commissions:,.2f}")
print(f"Spread Costs:      ${options_result.total_spread_cost:,.2f}")
print(f"\nSample trades:")
for t in options_result.trade_log[:5]:
    print(f"  {t['entry_date']} -> {t['exit_date']}  "
          f"{t['option_type'].upper()} K={t['strike']:<6.0f}  "
          f"pnl=${t['pnl']:>+8.2f}  ({t['exit_reason']})")

# Options scorecard
generate_options_scorecard(
    strategy=CoveredCallStrategy(target_dte=30, target_delta=0.30),
    df=df,
    config=options_config,
    output_path="options_scorecard.png",
    symbol="SPY",
)
