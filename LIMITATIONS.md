# Known Limitations & Design Decisions

What this framework does well, where it has boundaries, and why certain trade-offs were made.

---

## 1. Data Providers

We support 12 equity providers, crypto via CCXT (110+ exchanges), and options via philippdubach. Yahoo Finance and CCXT are the primary free-tier providers most users will use.

### Yahoo Finance (primary, free, no key)
- Supports daily and intraday (1m, 5m, 15m, 1h)
- Auto-adjusted for splits and dividends
- Intraday limits: 1m data available for ~8 days, 5m/15m for ~60 days, 1h for ~730 days
- Daily data: 20+ years
- No API key required
- Wraps yfinance (unofficial library) — could break if Yahoo changes their site, but yfinance is actively maintained and widely used

### Other Equity Providers
Each has different free-tier limits — documented in the provider docstrings:
- **Tiingo**: 20+ years, 1000 req/day (best depth)
- **Polygon**: ~2 years free, implements pagination + rate limit sleep
- **Alpaca**: 5-6 years, 200 req/min, implements pagination
- **Alpha Vantage**: ~5 months on free compact mode
- **Twelve Data, Finnhub, FMP, MarketStack**: Various limits

### Design decision: no pre-validation of tickers
We let the API return an error rather than maintaining a ticker list that goes stale. The trade-off is slightly less helpful error messages in exchange for zero maintenance.

---

## 2. Crypto (CCXT)

### What works well
- 110+ exchanges (far more than QuantConnect's 6)
- Public OHLCV free without API keys on all exchanges
- All intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w
- Rate limiting with `enableRateLimit=True` + sleep between pages
- Retry logic: 3 retries with exponential backoff
- Pagination with deduplication at chunk boundaries
- `BacktestConfig.for_crypto()` sets 365 bars/year for correct annualization

### Known quirks
- Kraken uses `XBT/USD` internally — pass `BTC/USD` and it works for daily but some intervals may need the native symbol
- Binance is geo-blocked in some regions (US) — use Kraken, Coinbase, or other exchanges
- Historical depth varies by exchange (~3-5 years for most pairs)
- Volume is reported in base currency (e.g., BTC not USD) — standard for crypto

---

## 3. Options

Options backtesting uses **synthetic Black-Scholes pricing** from the underlying's realized volatility. This is a deliberate design choice — it lets you test options strategies with just OHLCV data, no options data feed required.

### What works well
- Greeks (delta, gamma, theta, vega, rho) verified against known values to 1e-10 precision
- Multi-leg positions: verticals, iron condors, straddles, strangles via PositionFactory
- Margin calculation: Reg-T and simplified portfolio margin
- IV surface interpolation with cubic/linear/nearest fallback
- Exercise and assignment handling at expiration
- Bid-ask spread modeling with configurable width and market impact
- Real historical chains available via philippdubach (104 tickers, 2008-2025, free)

### Design decisions and boundaries
- **European-style pricing**: we use Black-Scholes, not binomial trees. For the strategies we're testing (covered calls, spreads), the difference is small. American early exercise matters most for deep-ITM puts near ex-div — an edge case for our use case.
- **Realized vol, not implied vol**: we price from the underlying's own volatility, not from the options market. This means our prices don't capture supply/demand effects (skew, smile). For strategy logic testing this is fine; for P&L estimation it's approximate.
- **No dividend tracking**: the covered call strategy uses `holds_underlying=True` to correctly track stock exposure, but we don't model dividend events or ex-div assignment risk.

---

## 4. Synthetic Data (Monte Carlo + GAN)

We have 6 synthetic data generators — more than any comparable free framework. Each serves a different testing purpose.

### GBM (Geometric Brownian Motion)
- **Purpose**: baseline sanity check. No strategy should profit on pure random walks.
- Normal increments (no fat tails, no vol clustering) — this is by design. GBM is supposed to be the "null hypothesis."
- If your strategy shows positive Sharpe on GBM, it's overfitting.

### Block Bootstrap
- Preserves the real distribution (fat tails, vol clustering) from historical data
- Autocorrelation preserved within blocks (default size 5), breaks at boundaries
- Trade-off: larger blocks preserve more structure but reduce scenario diversity

### Regime Switching (Markov)
- 3 regimes (bull/sideways/bear) with configurable transition matrix
- Regime persistence ~2-3 weeks (realistic timescale)
- Returns are normal within each regime — the regime switching itself creates fat tails in the aggregate distribution

### Noise Injection
- Safest source — inherits all properties from the base data
- Tests graceful degradation: robust strategies degrade smoothly, brittle ones break
- OHLCV consistency maintained via clamping

### cGAN (trained conditional GAN)
- Generates regime-conditioned paths (bullish, bearish, sideways, crash)
- Trained on real SPY data from 4 distinct market periods
- Trained on 30-step windows — we stitch chunks to create longer paths. This means autocorrelation resets every 30 bars. For the purpose of regime stress testing (does the strategy survive a crash scenario?), this is adequate. For testing multi-week momentum strategies, it's less ideal.
- Crash regime has limited training data (73 bars from COVID March 2020) — generated crashes will resemble COVID-style drawdowns specifically
- Could be improved by retraining on longer windows and more crash data

### Heston Monte Carlo
- Stochastic volatility with mean reversion and leverage effect
- More realistic than GBM (captures vol clustering and the leverage effect)
- Parameters are defaults, not calibrated to a specific asset — suitable for general stress testing

---

## 5. Backtester Engines

### Bar engine (fast, simple)
- Market orders only — fills at next bar's open
- Suitable for: signal-based strategies, daily swing trading, indicator testing
- SL/TP checks use the current bar's High/Low — this is standard for bar-based backtesting (same approach as Backtrader, bt, etc.). The event engine provides a more conservative alternative.

### Event engine (realistic)
- Supports market, limit, stop, and OCO bracket orders
- Orders submitted on bar t become active on bar t+1 (no lookahead)
- Stop/limit orders can rest across multiple bars
- Stop-limit orders not supported (limit and stop orders work independently)

### Running both engines on the same strategy
This is one of our unique features. If the bar and event engines agree (typically < 1pp divergence for market-order strategies), the strategy's edge doesn't depend on execution assumptions. If they diverge, the strategy is sensitive to fill timing — important to know before live trading.

### Boundaries (both engines)
- **Single-asset**: one symbol per backtest. Portfolio-level backtesting is a future goal.
- **Integer shares**: no fractional share support. Minimal impact for stocks > $10.
- **Cash-only**: no margin or leverage. Strategies are tested with the capital available.
- **No borrow costs**: short selling works but doesn't model borrow fees. For short-heavy strategies, real costs could be 0.5-50%+ annually depending on the stock.
- **No corporate actions**: Yahoo Finance data is pre-adjusted for splits/dividends, so historical price charts are correct. But we don't track dividend cash flows or model assignment risk around ex-div dates.
- **Fixed BPS slippage**: not volume-aware. Adequate for liquid large-cap equities, may underestimate costs for small-caps or illiquid instruments.

### Metrics
- **Sharpe ratio**: `mean(returns) / std(returns) * sqrt(bars_per_year)` with risk-free = 0%. This is the industry standard formula used by QuantConnect, Zipline, Backtrader, and most platforms. Relative strategy rankings are always correct.
- **Sortino ratio**: same formula using downside deviation (MAR = 0%). Standard implementation.
- **Calmar ratio**: annualized return / max drawdown. Correct.
- **All other metrics** (profit factor, expectancy, win rate, SQN, Kelly, etc.) are standard calculations.

---

## 6. Scorecard

### What works
- 4-page report: bar backtest, Monte Carlo (GBM + GAN + noise), event-driven comparison, grades
- Auto-runs 280+ backtests (50 GBM, 150 GAN across 3 regimes, 150 noise injection)
- Automatic letter grades (A-F) on 12 dimensions
- Dark and light theme support
- Works on daily and intraday data

### Boundaries
- Monte Carlo page adds ~30-60s to scorecard generation (running 280+ scenarios)
- Options scorecard is single-page (no Monte Carlo or engine comparison for options strategies)
- GAN scenarios use SPY-trained model — regime characteristics are SPY-specific

---

## Summary

### Built for
- Strategy validation and comparison (same pipeline, same metrics, fair evaluation)
- Overfitting detection (GBM noise test, GAN crash test, in/out-of-sample)
- Execution robustness testing (bar vs event engine divergence)
- Daily and intraday equity/crypto backtesting
- Options strategy prototyping with synthetic pricing
- Educational use and rapid prototyping

### Outside current scope
- Live/paper trading (no broker integration)
- Multi-asset portfolio allocation (single-symbol per backtest)
- Margin/leverage strategies (cash-only)
- Futures and forex (no data providers for these)
- Tick-level data (minimum resolution is 1-minute)
