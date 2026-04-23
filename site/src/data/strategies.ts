// Metadata for the five built-in strategies showcased in the gallery.
// End values and grades are sourced from PRESENTATION.md Slide 7 and the scorecard PNGs.
// Where an exact number isn't published, the field is left null and a TODO is noted.

export type AssetClass = "equities" | "crypto" | "options";

export interface StrategyCard {
  id: string;
  name: string;
  asset: string;
  assetClass: AssetClass;
  period: string;
  oneLiner: string;
  headline: {
    return: string | null; // e.g. "+46.4%"
    sharpe: string | null;
    benchmark?: string | null;
    divergence?: string | null;
    note?: string | null;
  };
  grade: string; // top-line overall grade
  // image paths (relative to /public)
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
    period: "2022 — 2025",
    oneLiner: "Dual moving-average crossover (5/40, optimized) on US equities.",
    headline: {
      return: "+46.4%",
      sharpe: "0.91",
      benchmark: "SPY B&H",
      divergence: "0.02pp",
      note: "Engines agree — strategy is execution-robust.",
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
    period: "2022 — 2025",
    oneLiner: "Turtle-style trend follower with ATR sizing and Donchian entries.",
    headline: {
      return: null, // TODO: pull exact figures from scorecard PNG
      sharpe: null,
      note: "Trend-following benchmark — full scorecard generated.",
    },
    grade: "B", // TODO: confirm from scorecard PNG
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
    period: "2022 — 2025",
    oneLiner: "Donchian breakout with OCO stop-loss / take-profit brackets — the canonical engine-divergence test.",
    headline: {
      return: null, // TODO
      sharpe: null,
      divergence: "High",
      note: "Intentionally divergent: bar engine cannot model the OCO intrabar fills.",
    },
    grade: "C", // TODO: confirm from scorecard PNG
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
    period: "2024 — 2025",
    oneLiner: "Same SMA logic on Bitcoin via Kraken (CCXT) — crypto-aware 365-day calendar, no weekend-gap false positives.",
    headline: {
      return: null, // TODO
      sharpe: null,
      note: "Full scorecard generated, crypto-aware validation.",
    },
    grade: "B", // TODO
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
    period: "2022 — 2025",
    oneLiner: "Sell monthly 30-DTE, 0.30-delta covered calls on the underlying.",
    headline: {
      return: "+90.1%",
      sharpe: "1.41",
      benchmark: "SPY B&H +51%",
      note: "Initial 2.17 Sharpe was a naked-call bug — now correctly marked as covered.",
    },
    grade: "A",
    thumbnail: "/scorecards/options_scorecard.png",
    pages: {
      // Options scorecard is single-page today.
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
