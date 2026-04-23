// Representative equity curves for the landing-page "race" chart.
//
// End values are anchored to PRESENTATION.md Slide 7 so the final totals are honest.
// The intermediate shape is a smooth, monotonic-ish interpolation — it is
// REPRESENTATIVE only. Replace this file with real series once
// backtester.scorecard.export_scorecard_json() lands and we can load actual
// per-day equity for each strategy.

export interface RaceSeries {
  id: string;
  name: string;
  asset: string;
  color: string;
  finalReturnPct: number; // e.g. 46.4 means +46.4%
  values: number[]; // indexed 0..N, values are portfolio value as a % of start (100 = flat)
}

function curve(finalPct: number, seed: number, volatility = 0.9, n = 120): number[] {
  // Deterministic pseudo-random walk that ends exactly at `finalPct`.
  // Start at 100, end at 100 * (1 + finalPct/100), with plausible wobble.
  const end = 100 * (1 + finalPct / 100);
  const out: number[] = new Array(n);
  let x = seed;
  const rand = () => {
    x = (x * 9301 + 49297) % 233280;
    return x / 233280;
  };
  // Build a wobble around the linear interpolation from 100 → end.
  const wobble: number[] = [];
  for (let i = 0; i < n; i++) {
    const step = (rand() - 0.5) * volatility;
    wobble.push(step);
  }
  // Smooth wobble with a short moving average for financial-chart feel.
  const smoothed: number[] = [];
  const win = 5;
  for (let i = 0; i < n; i++) {
    let sum = 0;
    let c = 0;
    for (let k = -win; k <= win; k++) {
      const j = i + k;
      if (j >= 0 && j < n) {
        sum += wobble[j];
        c++;
      }
    }
    smoothed.push(sum / c);
  }
  // Integrate the smoothed wobble and add the linear trend to the endpoint.
  const drift: number[] = [0];
  for (let i = 1; i < n; i++) {
    drift.push(drift[i - 1] + smoothed[i]);
  }
  // Normalize drift so the endpoint matches the target.
  const driftEnd = drift[n - 1];
  for (let i = 0; i < n; i++) {
    const trend = 100 + ((end - 100) * i) / (n - 1);
    const adj = drift[i] - (driftEnd * i) / (n - 1);
    out[i] = trend + adj;
  }
  out[0] = 100;
  out[n - 1] = end;
  return out;
}

const N = 120;

export const RACE: RaceSeries[] = [
  {
    id: "covered-call",
    name: "Covered Call",
    asset: "SPY",
    color: "#881C1C",
    finalReturnPct: 90.1,
    values: curve(90.1, 1337, 1.15, N),
  },
  {
    id: "sma-spy",
    name: "SMA Crossover",
    asset: "SPY",
    color: "#7B6D5C",
    finalReturnPct: 46.4,
    values: curve(46.4, 4242, 1.0, N),
  },
  {
    id: "spy-bh",
    name: "Buy & Hold",
    asset: "SPY",
    color: "#8B8680",
    finalReturnPct: 51.0,
    values: curve(51.0, 9001, 1.05, N),
  },
];

// X-axis labels — synthetic months across the 2022 → 2025 window.
export const MONTH_LABELS: string[] = (() => {
  const starts = ["2022", "2023", "2024", "2025"];
  const out: string[] = [];
  for (let i = 0; i < N; i++) {
    const frac = i / (N - 1);
    const yearIdx = Math.min(starts.length - 1, Math.floor(frac * starts.length));
    out.push(starts[yearIdx]);
  }
  return out;
})();
