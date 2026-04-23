# Trading Strategy Testing Framework

**Team presentation — progress update**

---

## Slide 1 — What problem are we solving?

Most retail traders (and even many quant students) test strategies with a spreadsheet or a one-off Python script. That process has three failure modes:

1. **Lookahead bias** — the script accidentally uses tomorrow's price to decide today's trade.
2. **Overfitting** — the strategy looks great on one historical window and collapses on anything new.
3. **Ignored frictions** — commissions, slippage, and the gap between the price you *see* and the price you actually *fill at* get left out.

**Our framework is an opinionated harness that forces an honest answer to: "would this strategy have actually made money?"**

It does that across three asset classes (equities, crypto, options), against two independent execution engines, and against synthetic market data — so a strategy has to survive more than one way of being wrong before we trust it.

---

## Slide 2 — Where we were 2 weeks ago

- Single asset class: US equities only (Yahoo Finance).
- Single engine: a bar-based backtester (market-order at next open).
- Scorecard existed but graded on absolute return, no benchmark comparison.
- No validation that our engine actually modeled realistic execution.

In short: a working toy, but easy to fool.

---

## Slide 3 — Where we are now

Four pillars, each independently testable:

| Pillar | 2 weeks ago | Today |
|---|---|---|
| **Asset classes** | Equities | Equities + Crypto (100+ exchanges via CCXT) + Options (Black-Scholes + real historical chains) |
| **Execution engines** | Bar-based only | Bar-based **+** Event-driven (stop/limit/OCO intrabar) |
| **Validation** | Historical only | Historical + Monte Carlo synthetic + Engine-vs-engine divergence analysis |
| **Grading** | Absolute thresholds | Benchmark-relative (alpha vs buy-and-hold) |

---

## Slide 4 — Concept: Bar vs Event engines

This is the part most non-quant audiences don't see.

- **Bar engine**: replays daily OHLCV. Every order becomes "buy at tomorrow's open." Fast, simple, but fundamentally cannot model a stop order triggering intraday.
- **Event engine**: replays the same bars, but models each bar as an event with order-book state. Stop orders fire intrabar at the trigger price. Bracket exits (OCO stop-loss + take-profit) are real.

**Why run both?** If a strategy performs identically under both engines, its edge doesn't depend on execution timing (good — robust). If results *diverge*, the strategy's PnL is sensitive to how orders actually fill — which is a flag that live results will differ from backtest.

Our scorecard now reports this divergence as a first-class metric.

---

## Slide 5 — Concept: Options backtesting

Options are hard because you can't just replay a stock price. Each day has a *chain* of hundreds of contracts at different strikes and expirations.

Two free data paths we built:

1. **Synthetic (Black-Scholes from realized vol)** — given a stock price, we compute an option's theoretical price from the underlying's own volatility. Good for strategy prototyping; ignores vol smile.
2. **Historical chains (philippdubach dataset)** — free parquet files of real option chains for 104 tickers back to 2008. Actual bid/ask, actual IV surface.

The engine tracks Greeks (delta, gamma, theta, vega), handles expiration, and correctly marks-to-market short options against the underlying (this caught a bug: our covered call was initially being simulated as a *naked* short call — Sharpe looked amazing for the wrong reason).

---

## Slide 6 — Code walkthrough (3 lines of user code)

```python
from data_layer import DataLayer, YahooFinanceProvider
from strategy import SMACrossover
from backtester import Backtester, BacktestConfig, generate_scorecard

df = DataLayer().add_provider(YahooFinanceProvider()).fetch("SPY", date(2022,1,1), date(2025,12,31))
result = Backtester(BacktestConfig(initial_capital=100_000)).run(SMACrossover(10, 30), df)
generate_scorecard(strategy=SMACrossover(10, 30), df=df, config=config, output_path="scorecard.png")
```

That's it — one scorecard PNG with four pages: summary/grades, historical equity curve vs benchmark, Monte-Carlo distributions, and bar-vs-event divergence.

Swapping to crypto is one line: `BacktestConfig.for_crypto(...)` (handles 365 trading days and 24/7 markets).
Swapping to options is one import: `OptionsBacktester` with `CoveredCallStrategy`.

---

## Slide 7 — Results we can show today

**SPY, SMA Crossover (5/40, optimized), 2022–2025:**
- Bar engine: +46.4% total return, Sharpe 0.91
- Event engine: +46.5% — 0.02pp divergence → engines agree, strategy is execution-robust

**BTC/USD, SMA (10/30), 2024–2025 (Kraken via CCXT):**
- Full scorecard generated, crypto-aware validation (no "weekend gap" false positives)

**SPY Covered Call, 30 DTE, 0.30 delta:**
- Total return +90.1% vs SPY buy-and-hold +51%
- Sharpe 1.41 (realistic — the initial 2.17 was the naked-call bug we caught)

**SPY Bracket Breakout (intentional divergence test):**
- Uses stop entries + OCO brackets that bar engine can't model
- Scorecard flags this as "execution-sensitive," correctly

**GAN stress test (SMA Crossover across regime scenarios, 50 each):**

| Regime | Mean Return | Sharpe | Max Drawdown |
|---|---|---|---|
| Bullish | +29.1% | +4.18 | -2.1% |
| Bearish | -5.5% | -0.62 | -10.5% |
| Crash | -5.2% | -0.11 | -27.1% |
| Mixed | +5.3% | +0.96 | -13.0% |

The strategy survives bullish/mixed but bleeds in bear/crash — exactly what a trend-following system should do.

---

## Slide 8 — GAN-based synthetic data (Sachit)

**Problem:** Our Monte Carlo validation (Page 3 of the scorecard) bootstraps returns from the *same* historical window. That means every synthetic path shares the exact same distribution — no fat tails beyond what the sample already contained, no regime shifts, no volatility clustering.

**Solution:** Train a Conditional Wasserstein GAN (cGAN) on real SPY data from four distinct market regimes, then generate unlimited synthetic paths that preserve the *regime-specific* statistical fingerprint.

**Training data — real SPY, four regimes:**

| Regime | Period | Character |
|---|---|---|
| Bullish | 2017 | Strong uptrend, low vol |
| Bearish | 2007–2009 | Financial crisis drawdown |
| Sideways | 2015–2016 | Consolidation, choppy |
| Crash | Feb–May 2020 | COVID sell-off and recovery |

**Architecture:**
- **Generator**: noise vector (dim 32) + regime embedding → 2-layer LSTM (hidden 256) → synthetic return sequence. Regime label is fed at *every* timestep so the LSTM learns regime-dependent dynamics, not just a shifted mean.
- **Discriminator**: return sequence + regime embedding → 2-layer LSTM → realism score (unbounded). Trained with WGAN-GP (gradient penalty λ=10) for stable convergence without mode collapse.
- Trained for 1000 epochs, 5 critic updates per generator step.

**What it validates that bootstrap can't:**
- **Volatility clustering** — autocorrelation of squared returns matches real data (verified via ACF plots).
- **Fat tails** — excess kurtosis is preserved per regime, not washed out by averaging.
- **Regime conditioning** — ask the model for 1,000 crash scenarios and it generates paths with COVID-like drawdown dynamics, not generic noise.

**Integration with the backtester (live and working):**
A bridge module (`backtester/gan_bridge.py`) loads Sachit's trained checkpoint, generates 30-step windows (matching training), stitches them into full-length price paths, and wraps them as a `GANSource`. Three lines of user code:

```python
from backtester import make_gan_source, run_scenario_suite
crash_source = make_gan_source("crash", n_bars=252)
batch = run_scenario_suite(strategy, crash_source, n_scenarios=1000)
```

This is tested end-to-end — the GAN stress-test results in Slide 7 were produced by this pipeline.

---

## Slide 8b — Heston Monte Carlo (complementary approach)

Alongside the GAN, a Heston stochastic volatility model generates tick-level synthetic data:

- Price follows GBM with time-varying variance.
- Variance follows a mean-reverting Ornstein-Uhlenbeck process (κ=3.0, θ=0.1, ξ=0.05).
- Price and variance shocks are correlated (ρ = −0.7) — drops in price increase volatility, matching the real leverage effect.
- Outputs OHLCV bars aggregated from 10,000 micro-steps.

**GAN vs Heston:** The Heston model is parametric — you *specify* the dynamics and it generates accordingly. The GAN is non-parametric — it *learns* the dynamics from data. Using both gives us model-free and model-based stress tests.

---

## Slide 9 — What's next

- GAN distributions on the scorecard (Page 3 currently shows bootstrap only — add GAN regime overlay).
- Walk-forward optimization (currently single-shot Optuna).
- Portfolio-level backtesting (currently single-symbol).
- Live paper-trading hook via the same event engine.

---

## Slide 10 — Takeaway

We replaced "does this strategy make money on SPY 2022-2025?" with **"does this strategy make money across three asset classes, two execution engines, one benchmark comparison, and N synthetic market regimes — and by how much does each answer disagree?"**

The disagreement is the point. A strategy that only works under one engine, one asset, or one historical window isn't a strategy — it's a fit.
