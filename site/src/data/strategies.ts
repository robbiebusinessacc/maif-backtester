// Metadata for the five built-in strategies showcased in the gallery.
// All figures read directly from the scorecard PNGs (the source of truth
// for what a visitor actually sees on /scorecards/[id]). Equity scorecards
// display CAGR as the primary return metric; the options scorecard displays
// total return. Periods match the header printed on each PNG.

export type AssetClass = "equities" | "crypto" | "options";

export interface StrategyCard {
  id: string;
  name: string;
  asset: string;
  assetClass: AssetClass;
  period: string;
  oneLiner: string;
  headline: {
    return: string | null;       // "+11.26%"
    returnLabel: string;          // "CAGR" or "Total return"
    sharpe: string | null;
    benchmark?: string | null;    // "SPY B&H +50.91%"
    maxDrawdown?: string | null;  // "-11.7%"
    trades?: string | null;       // "13"
    divergence?: string | null;   // "Bar ≈ Event" or "Notable"
    note?: string | null;
  };
  grade: string;                  // top-line overall grade (from PNG grade wheel)
  thumbnail: string;
  pages: {
    scorecard: string;
    bartest: string;
    eventdriven: string;
    montecarlo: string;
  };
}

export const STRATEGIES: StrategyCard[] = [
  {
    id: "sma-spy",
    name: "SMA Crossover",
    asset: "SPY",
    assetClass: "equities",
    period: "2022-01-03 → 2025-12-31",
    oneLiner: "Dual moving-average crossover (5 / 40) on US equities.",
    headline: {
      return: "+11.26%",
      returnLabel: "CAGR",
      sharpe: "1.04",
      benchmark: "SPY B&H +50.91%",
      maxDrawdown: "-11.7%",
      trades: "13",
      divergence: "Bar ≈ Event",
      note: "Profit factor 4.17, Kelly 40.9%. GBM noise and GAN crash grades both C — suspicious profit on GBM hints at mild overfitting.",
    },
    grade: "B",
    thumbnail: "/scorecards/report_scorecard.png",
    pages: {
      scorecard: "/scorecards/report_scorecard.png",
      bartest: "/scorecards/report_bartest.png",
      eventdriven: "/scorecards/report_eventdriven.png",
      montecarlo: "/scorecards/report_montecarlo.png",
    },
  },
  {
    id: "turtle-spy",
    name: "Turtle Trend",
    asset: "SPY",
    assetClass: "equities",
    period: "2022-01-03 → 2025-12-31",
    oneLiner: "Turtle-style trend follower (55 / 20d, 2.0 ATR stop) with Donchian entries and ATR-based sizing.",
    headline: {
      return: "+3.25%",
      returnLabel: "CAGR",
      sharpe: "0.53",
      benchmark: "SPY B&H +50.91%",
      maxDrawdown: "-11.8%",
      trades: "11",
      divergence: "Bar ≈ Event",
      note: "Profit factor 2.15, expectancy $1,292 per trade. Trend-following behaves as expected — lags in choppy markets.",
    },
    grade: "B",
    thumbnail: "/scorecards/report_turtle_scorecard.png",
    pages: {
      scorecard: "/scorecards/report_turtle_scorecard.png",
      bartest: "/scorecards/report_turtle_bartest.png",
      eventdriven: "/scorecards/report_turtle_eventdriven.png",
      montecarlo: "/scorecards/report_turtle_montecarlo.png",
    },
  },
  {
    id: "bracket-spy",
    name: "Bracket Breakout",
    asset: "SPY",
    assetClass: "equities",
    period: "2022-01-03 → 2025-12-31",
    oneLiner: "Donchian breakout (20d) with OCO stop-loss (1.5 ATR) / take-profit (3.0 ATR) brackets — the canonical engine-divergence test.",
    headline: {
      return: "+6.25%",
      returnLabel: "CAGR",
      sharpe: "1.21",
      benchmark: "SPY B&H +50.91%",
      maxDrawdown: "-12.3%",
      trades: "22",
      divergence: "Notable",
      note: "Intentionally divergent: the bar engine cannot model OCO intrabar fills. Event-driven match grade C confirms execution sensitivity.",
    },
    grade: "B",
    thumbnail: "/scorecards/report_divergence_scorecard.png",
    pages: {
      scorecard: "/scorecards/report_divergence_scorecard.png",
      bartest: "/scorecards/report_divergence_bartest.png",
      eventdriven: "/scorecards/report_divergence_eventdriven.png",
      montecarlo: "/scorecards/report_divergence_montecarlo.png",
    },
  },
  {
    id: "sma-btc",
    name: "SMA Crossover",
    asset: "BTC/USD",
    assetClass: "crypto",
    period: "2024-05-01 → 2025-12-31",
    oneLiner: "Same SMA logic (10 / 30) on Bitcoin via Kraken (CCXT) — crypto-aware 365-day calendar, no weekend-gap false positives.",
    headline: {
      return: "+4.11%",
      returnLabel: "CAGR",
      sharpe: "0.29",
      benchmark: "BTC B&H +47.99%",
      maxDrawdown: "-22.7%",
      trades: "13",
      divergence: "Bar ≈ Event",
      note: "Lags buy-and-hold badly — SMA is a poor fit for BTC's volatility profile. A useful negative case study.",
    },
    grade: "C",
    thumbnail: "/scorecards/crypto_report_scorecard.png",
    pages: {
      scorecard: "/scorecards/crypto_report_scorecard.png",
      bartest: "/scorecards/crypto_report_bartest.png",
      eventdriven: "/scorecards/crypto_report_eventdriven.png",
      montecarlo: "/scorecards/crypto_report_montecarlo.png",
    },
  },
  {
    id: "covered-call-spy",
    name: "Covered Call",
    asset: "SPY",
    assetClass: "options",
    period: "2022-01-03 → 2025-12-31",
    oneLiner: "Monthly 30-DTE, 0.30-delta covered calls on the underlying.",
    headline: {
      return: "+90.08%",
      returnLabel: "Total return",
      sharpe: "1.41",
      benchmark: "SPY B&H +50.91%",
      maxDrawdown: "-11.95%",
      trades: "120",
      divergence: null,
      note: "Beats benchmark by 39.2 pp. Premium collected $46,002, cost efficiency 2 % of PnL. Win rate 74.2 %.",
    },
    grade: "B",
    thumbnail: "/scorecards/options_scorecard.png",
    pages: {
      // Options scorecard is single-page — all slots point at the same file.
      scorecard: "/scorecards/options_scorecard.png",
      bartest: "/scorecards/options_scorecard.png",
      eventdriven: "/scorecards/options_scorecard.png",
      montecarlo: "/scorecards/options_scorecard.png",
    },
  },
];

export const STRATEGY_MAP: Record<string, StrategyCard> = Object.fromEntries(
  STRATEGIES.map((s) => [s.id, s])
);
