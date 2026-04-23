# Known Limitations

Comprehensive audit of every data source, engine, and synthetic generator in the framework.

---

## 1. Equity Data Providers

### Yahoo Finance (free, no key)
- No official API — wraps web scraping via yfinance; can break without notice
- No explicit rate limit in code (~2000 req/hr soft limit from Yahoo)
- Invalid tickers return empty DataFrame silently, then raise generic ValueError
- Auto-adjusted for splits/dividends (good)

### Alpha Vantage (free key)
- **Hardcoded compact mode: only ~100 trading days (~5 months)**
- 5 requests/minute, no throttling in code
- Requests beyond 5 months silently return truncated data

### Finnhub (free key)
- ~1 year of daily history on free tier
- 60 req/min, no throttling or retry
- Older dates silently return empty

### Polygon (free key)
- ~2 years on free tier, 5 req/min
- **Implements chunking + 12s sleep between chunks (good)**
- Catches 403 for historical limit

### Tiingo (free key)
- 20+ years of history (best free provider)
- 1000 req/day
- Silently mixes adjusted/unadjusted prices without indication

### Twelve Data (free key)
- Hardcoded `outputsize=5000` — silently truncates longer requests
- 8 req/min and 800/day, no throttling

### MarketStack (free key)
- **100 requests/month** (extremely tight)
- Free tier uses HTTP only (no HTTPS)
- Pagination can burn monthly quota unknowingly

### Alpaca (free key + secret)
- 5-6 years on free IEX feed
- 200 req/min, implements pagination
- Older requests may silently truncate

### CSV/Stooq (local file)
- No schema validation — assumes standard OHLCV columns exist

### Cross-provider issues
- `cross_validate()` only compares Close prices, ignores OHLCV divergence
- No timezone validation (assumes all providers return UTC)
- No pre-validation of ticker existence on any provider

---

## 2. Crypto (CCXT)

### What works well
- 110+ exchanges, public OHLCV free without API keys
- Rate limiting implemented (`enableRateLimit=True` + sleep between pages)
- Retry logic: 3 retries with exponential backoff
- Pagination with deduplication at boundaries
- Symbol conversion handles both `BTC/USD` and `BTCUSDT` formats

### Limitations
- **Does NOT catch `ccxt.RateLimitExceeded`** — crashes instead of retrying
- **Kraken uses `XBT/USD` not `BTC/USD`** — symbol conversion may fail
- No check for delisted coins — returns incomplete data silently
- Only 8 quote currencies hardcoded (misses GBP, DOGE, etc.)
- No validation that symbol exists on chosen exchange before fetching
- Zero-volume bars not flagged (could indicate delisting)
- `BacktestConfig.for_crypto()` only sets `bars_per_year=365` — doesn't adjust slippage defaults for crypto volatility
- Historical depth varies by exchange (Binance ~5yr, others less) — no documentation or warning
- Volume is in base currency (not quote) — not documented

---

## 3. Options

### Data source
- **Synthetic Black-Scholes from realized vol** (default) — not real market IV
- **philippdubach dataset** (optional) — 104 tickers, Jan 2008-Dec 2025, free parquet files
- If underlying.parquet missing, estimates price from ATM delta ~0.50 (can be off 10-20%)

### Pricing model limitations
- **European-style only** — no American early exercise modeling
- **Constant volatility** per bar — no smile, skew, or term structure
- **No dividends** by default (dividend_yield=0.0)
- **Realized vol lookback** is fixed 20-day — no regime detection
- Risk-free rate defaults to 5%, constant across all tenors

### Engine limitations
- **No early assignment** — covered call assignment risk underestimated
- **No margin call simulation** — positions can go underwater indefinitely
- **No partial fills** — all-or-nothing capacity check (rejects if qty > 10% OI)
- Spread model is symmetric — no adverse selection on fills
- Fill prices are deterministic (no randomness)
- No overnight gap risk modeled
- No holidays in trading calendar

### What works well
- Greeks computation verified against known values (put-call parity to 1e-10)
- Multi-leg positions: verticals, iron condors, straddles, strangles
- Margin calculation: Reg-T and simplified portfolio margin
- IV surface interpolation with cubic/linear/nearest fallback
- Exercise/assignment handling at expiration

### Not suitable for
- Live trading without major modifications
- Dividend-driven strategies
- Earnings strategies (no event detection)

---

## 4. GAN / Monte Carlo Synthetic Data

### GBM (Geometric Brownian Motion)
- **No fat tails** — normal increments, kurtosis = 3 exactly
- **No volatility clustering** — constant vol
- Pure random walk — no trends, no mean reversion
- **Use case: baseline sanity check only** (strategy should NOT profit on GBM)

### Block Bootstrap
- Preserves marginal distribution, fat tails, and vol clustering from historical data
- **Autocorrelation breaks at block boundaries** (default block_size=5)
- Data-dependent: if historical data is only bullish, all scenarios are bullish

### Regime Switching (Markov)
- 3 regimes (bull/sideways/bear) with configurable transition matrix
- Regime persistence matches rough market timescales (~2-3 weeks)
- **Within-regime returns are still normal** — no fat tails
- Fixed transition matrix can't adapt to market conditions
- Always starts in bull regime (initial_regime=0)

### Noise Injection
- Safest source — inherits all properties from base data
- OHLCV consistency maintained via clamping
- Volume can collapse to zero on low-volume bars

### cGAN (Sachit's trained model)
- **Trained on only 30-step windows** — generating 252-bar paths stitches independent chunks, breaking autocorrelation every 30 bars
- **Crash regime has only 23 training windows** — severe overfitting risk, generated crashes are near-replicas
- **Hardcoded global_mu/global_sigma** — same normalization for all regimes, not stored in checkpoint
- Training data is SPY-only from 2007-2020 — no 2021-2024 market structure
- `torch.manual_seed()` sets global seed — non-reproducible if other torch ops run between calls
- No regime transitions learned — each regime generated independently

### Heston Monte Carlo
- Stochastic volatility with mean reversion and leverage effect (good)
- **Parameters not calibrated** to any real asset (hardcoded kappa=3, theta=0.04)
- No jump component — underestimates tail risk
- Correlation rho=-0.7 is equity-specific (wrong for commodities, bonds)

---

## 5. Backtester Engines

### Bar engine
- **Market orders only** — no limit, stop, or conditional orders
- **Lookahead bias**: uses bar[t] High/Low for SL/TP checks before generating bar[t] signal — backtest returns ~2-5% optimistic
- Always liquidates at final bar close

### Event engine
- Supports market, limit, stop, and OCO brackets
- **Stop-limit explicitly unsupported** (raises ValueError)
- No trailing stops, MOC, MOO
- Intrabar fills use OHLC only — assumes monotonic price movement within bar
- Liquidation on finish is configurable

### Both engines
- **No fractional shares** — integer positions only
- **Single-asset only** — no multi-symbol portfolio
- **No margin/leverage** — buying power limited to cash
- **No borrow costs for shorts** — short selling is free
- **No dividends, splits, or corporate actions**
- **Slippage is fixed BPS** — not volume-aware or liquidity-aware
- **Risk-free rate hardcoded to 0%** — Sharpe/Sortino overstated by ~5-10% in current rate environment
- No trading halt handling
- Zero-volume bars can still fill orders

### Metrics
- Sharpe: excess return not computed (uses raw return / std, not return - risk_free)
- Sortino: uses all negative returns, not returns below MAR
- Alpha: uses total return instead of annualized, risk-free = 0%
- Calmar: correct

---

## 6. Scorecard

### What works
- 4-page report: backtest, Monte Carlo, event-driven, grades
- Auto-runs GBM (50 scenarios), GAN (3 regimes x 50), noise injection (5 levels x 30)
- Grades on GBM noise test + GAN crash test + engine divergence
- Dark and light theme support

### Limitations
- Monte Carlo takes ~30-60s per scorecard (runs 280+ backtests)
- GAN grades depend on hardcoded normalization (see section 4)
- Options scorecard is single-page only (no Monte Carlo or engine comparison)

---

## Summary: What This Framework Is and Isn't

### Good for
- Comparative strategy evaluation (same pipeline, same metrics)
- Signal/indicator validation on equities
- Educational backtesting and prototyping
- Stress-testing via multiple synthetic data sources
- Crypto backtesting with proper 24/7 handling

### Not suitable for (without modifications)
- Live/paper trading
- Leveraged or margin strategies
- Short-heavy portfolios (missing borrow costs)
- Long-term backtests > 5 years (missing dividends)
- Sub-hourly bar testing
- Multi-asset portfolio allocation
- Options risk management (synthetic pricing, no early exercise)
- VaR estimation or regulatory risk reporting
