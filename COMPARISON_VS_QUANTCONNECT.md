# Our Framework vs QuantConnect

Side-by-side comparison. Honest about where we lose, clear about where we win.

---

## Quick Summary

| | QuantConnect | Our Framework |
|---|---|---|
| **Best at** | Production backtesting, live trading, multi-asset | Strategy validation, Monte Carlo stress testing, simplicity |
| **Cost** | Free tier + $60+/mo paid | Completely free |
| **Live trading** | Yes (15+ brokerages) | No |
| **Monte Carlo / GAN** | Not built in | Built in (6 generators, auto-scorecard) |
| **Intraday** | Tick to daily, since 1998 | 1m to daily (Yahoo 1m-1h, CCXT crypto all intervals) |

---

## Where QuantConnect Wins

### Data Depth (not resolution — we both do intraday)
- US equities since 1998 at minute resolution (we get 60 days of 1m, 730 days of 1h)
- Real options chains at minute resolution since 2010
- Futures, forex, index data (we don't support these asset classes)
- 40+ alternative data vendors (sentiment, satellite, SEC filings)
- Survivorship-bias-free (includes delisted stocks)
- All data included free for cloud backtesting

### Execution Realism
- 10+ order types including trailing stop, stop-limit, MOC, MOO
- 15+ brokerage-specific fee models (IB, Alpaca, Coinbase, etc.)
- Volume-aware slippage model
- No lookahead bias by design (streaming event engine)

### Corporate Actions
- Dividends, splits, mergers, delistings handled automatically
- Unfilled orders auto-adjusted on splits
- We handle none of this

### Margin / Short Selling
- Full Reg-T margin (2x-4x leverage), margin call simulation
- Short borrow costs with real availability data (10,500 stocks since 2018)
- Locate requirements enforced
- We have no margin and free shorts

### Multi-Asset Portfolios
- Trade equities + options + futures + crypto simultaneously
- Dynamic universe selection (filter top 100 stocks by volume, etc.)
- Portfolio-level risk tracking
- We are single-asset only

### Options
- Real market data (not synthetic Black-Scholes)
- American exercise with automatic hourly assignment checks
- QuantLib-backed pricing (Cox-Ross-Rubinstein, Black-Scholes)
- IV smoothing via call-put pairs

### Live Trading
- Direct integration with 15+ brokerages
- Paper trading environment
- We have no live trading capability

---

## Where We Win

### Intraday Support (tested and working)
- **1-hour bars**: Yahoo Finance, up to 730 days (tested: 1,444 bars SPY, full scorecard generated)
- **15m/5m bars**: Yahoo Finance, last 60 days
- **1-minute bars**: Yahoo Finance, last 8 days per request
- **Crypto intraday**: CCXT supports 1m/5m/15m/30m/1h/4h on 110+ exchanges
- `BacktestConfig.for_intraday("1h")` handles annualization automatically
- Bar and event engines both work on intraday (tested: 0.01pp divergence on 1H SPY)
- QuantConnect has deeper history, but we match on resolution for recent data

### Monte Carlo Stress Testing (QuantConnect has nothing here)
- **GBM noise test**: run strategy on pure random walks — positive Sharpe = overfitting
- **GAN regime scenarios**: trained cGAN generates bullish/bearish/sideways/crash paths
- **Noise injection**: test how gracefully strategy degrades under increasing noise
- **Block bootstrap**: resample historical returns preserving fat tails
- **Regime switching**: Markov-based regime transitions
- **Heston Monte Carlo**: stochastic volatility with leverage effect
- **Automated**: scorecard runs 280+ backtests across all generators automatically
- QuantConnect users must implement all of this from scratch

### Automatic Grading Scorecard
- 4-page report with letter grades (A-F) across 12 dimensions:
  - Performance, Risk, Trade Quality, Robustness
  - GBM noise test grade
  - GAN crash test grade
  - Engine divergence grade
- Dark and light themes
- Works on daily AND intraday data
- QuantConnect has PDF reports with good metrics but no grading system

### Engine Divergence Analysis
- Run the same strategy on bar-based AND event-driven engines
- If results agree: strategy is execution-robust (tested: SMA on 1H SPY, 0.01pp divergence)
- If results diverge: strategy depends on fill assumptions (red flag)
- QuantConnect has one engine — no way to cross-validate

### Crypto Exchange Coverage
- 110+ exchanges via CCXT (QuantConnect has 6)
- Public OHLCV free without API keys on all exchanges
- Intraday crypto data on any interval

### Simplicity
- Write a strategy in 20 lines, get a full scorecard
- No account setup, no cloud, no subscriptions
- `quickstart.py` is copy-paste-modify

```python
# Our framework — complete strategy + scorecard
class MyStrategy(Strategy):
    @property
    def name(self):
        return "My Strategy"
    def generate_signals(self, df):
        signals = pd.Series(Signal.HOLD, index=df.index)
        # your logic here
        return signals

generate_scorecard(MyStrategy(), df, config, "report.png")
```

```python
# QuantConnect — equivalent
class MyAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)
        self.add_equity("SPY", Resolution.DAILY)
    def on_data(self, data):
        # your logic here
        pass
# Then: navigate cloud IDE, wait for backtest, no Monte Carlo, no grades
```

### Cost
- Everything is free: data, backtesting, reporting, GAN scenarios
- QuantConnect free tier has daily backtest limits, 200-project cap
- Tick/second data requires $60+/month
- API/CLI access requires paid plan

---

## Feature Matrix

| Feature | QuantConnect | Us |
|---|---|---|
| Data: equities | Since 1998, survivorship-free | Yahoo (free), Tiingo (20yr) |
| Data: crypto | 6 exchanges | 110+ exchanges via CCXT |
| Data: options | Real chains, minute res | Synthetic BS or philippdubach (104 tickers) |
| Data: futures/forex | Yes | No |
| Data: resolution | Tick to daily (since 1998) | 1m to daily (1m: 8 days, 1h: 730 days, daily: 20yr) |
| Data: crypto intraday | 6 exchanges, all intervals | 110+ exchanges, all intervals |
| Order types | 10+ | Market, limit, stop, OCO |
| Trailing stops | Yes | No |
| Fill model | Volume-aware, per-asset | Fixed BPS |
| Commissions | 15+ brokerage models | Flat + percentage |
| Dividends/splits | Automatic | Not handled |
| Margin/leverage | Yes (Reg-T, PDT) | No |
| Short borrow costs | Yes (real data) | No (shorts are free) |
| Multi-asset portfolio | Yes | Single-asset |
| Options exercise | American, auto-assignment | European only |
| Monte Carlo | Not built in | 6 generators, automatic |
| GAN synthetic data | Not available | cGAN with 5 regimes |
| Automatic grading | No | A-F grades on 12 dimensions |
| Engine cross-validation | No (one engine) | Bar vs event comparison |
| Noise injection testing | No | Built in |
| Live trading | 15+ brokerages | No |
| Languages | Python + C# | Python |
| Cost | Free tier + $60+/mo | Free |
| Setup time | Account + cloud | `pip install` + run |

---

## Verified Test Results

All claims in this document were tested. Here are the actual numbers:

### Intraday (1-hour SPY, Jun 2024 - Apr 2025)
- 1,444 bars fetched from Yahoo Finance
- Bar engine: +3.50%, Event engine: +3.48% (0.01pp divergence)
- Full 4-page scorecard with Monte Carlo + GAN generated successfully

### Daily Regression (SPY 2022-2025)
- SMA Crossover (5/40): +53.11%, Sharpe 1.04, 13 trades
- Turtle Trend (55/20d): +13.60%, Sharpe 0.53, 11 trades
- Mean Reversion: +5.64%, Sharpe 0.26, 56 trades

### Crypto (BTC/USD daily, Kraken)
- SMA (10/30): +7.73%, Sharpe 0.31, 13 trades

---

## When to Use Which

### Use our framework when:
- You want to know if a strategy is overfitting before spending time on QuantConnect
- You need Monte Carlo stress testing or GAN scenario analysis
- You're prototyping equity or crypto strategies (daily or intraday)
- You want standardized, comparable results across the club
- You want automatic grades, not just numbers
- You don't want to pay for anything

### Use QuantConnect when:
- You need minute data going back years (not just 730 days)
- You're trading options and need real IV dynamics
- You need multi-asset portfolio backtesting
- You want realistic margin, short costs, and corporate actions
- You're going to live trade the strategy
- You need futures or forex data

### Best workflow: use both
1. Prototype and validate on our framework (free, fast, graded)
2. If strategy passes Monte Carlo + engine divergence + GAN crash test
3. Then port to QuantConnect for production backtesting with realistic costs
4. Deploy live via QuantConnect broker integration
