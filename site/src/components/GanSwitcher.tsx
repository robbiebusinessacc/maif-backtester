import { useState } from "react";

const REGIMES = [
  {
    id: "bullish",
    name: "Bullish",
    file: "/gan/Bullish_Output.png",
    period: "trained on SPY 2017",
    note: "Strong uptrend, low volatility.",
  },
  {
    id: "bearish",
    name: "Bearish",
    file: "/gan/Bearish_Output.png",
    period: "trained on SPY 2007 — 2009",
    note: "Financial-crisis drawdown regime.",
  },
  {
    id: "sideways",
    name: "Sideways",
    file: "/gan/Sideways_Output.png",
    period: "trained on SPY 2015 — 2016",
    note: "Consolidation, choppy, low directional drift.",
  },
  {
    id: "crash",
    name: "Crash",
    file: "/gan/Crash_Output.png",
    period: "trained on SPY Feb — May 2020",
    note: "COVID sell-off and recovery.",
  },
];

export default function GanSwitcher() {
  const [active, setActive] = useState(REGIMES[0].id);
  const current = REGIMES.find((r) => r.id === active)!;

  return (
    <div>
      <div className="flex flex-wrap gap-2 mb-5">
        {REGIMES.map((r) => {
          const isActive = r.id === active;
          return (
            <button
              key={r.id}
              onClick={() => setActive(r.id)}
              className={`font-mono text-[0.72rem] uppercase tracking-[0.14em] px-3 py-2 border transition-colors ${
                isActive
                  ? "bg-[#881C1C] text-[#FAF7F2] border-[#881C1C]"
                  : "bg-transparent text-ink border-hair hover:border-[#881C1C]"
              }`}
              aria-pressed={isActive}
            >
              {r.name}
            </button>
          );
        })}
      </div>

      <figure className="border border-hair bg-white">
        <div className="bg-[#FAF7F2] px-4 py-3 border-b border-hair flex items-baseline justify-between gap-4 flex-wrap">
          <div className="font-display text-xl leading-none text-ink">
            {current.name} regime
          </div>
          <div className="font-mono text-[0.68rem] uppercase tracking-[0.18em] text-muted">
            cGAN output &nbsp;·&nbsp; {current.period}
          </div>
        </div>
        <div key={current.id} className="fade-in">
          <img
            src={current.file}
            alt={`cGAN output — ${current.name} regime`}
            className="block w-full h-auto"
          />
        </div>
        <figcaption className="px-4 py-3 border-t border-hair text-sm text-ink">
          {current.note}
        </figcaption>
      </figure>
    </div>
  );
}
