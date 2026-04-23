import { useState } from "react";
import { STRATEGIES, type StrategyCard } from "../data/strategies";

function gradeRank(g: string): number {
  const order = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"];
  const idx = order.indexOf(g);
  return idx < 0 ? 99 : idx;
}

function GradeDelta({ a, b }: { a: string; b: string }) {
  const delta = gradeRank(a) - gradeRank(b);
  if (delta === 0) {
    return <span className="font-mono text-muted text-xs">≈</span>;
  }
  const color = delta < 0 ? "#2E7D32" : "#C62828"; // lower rank number = better grade
  const sign = delta < 0 ? "+" : "−";
  return (
    <span className="font-mono text-xs tabular-nums" style={{ color }}>
      {sign}
      {Math.abs(delta)}
    </span>
  );
}

function StrategyColumn({
  value,
  setValue,
  strategies,
}: {
  value: string;
  setValue: (v: string) => void;
  strategies: StrategyCard[];
}) {
  const s = strategies.find((x) => x.id === value);
  return (
    <div className="border border-hair bg-white">
      <div className="px-4 py-3 border-b border-hair bg-paper">
        <label className="font-mono text-[0.68rem] uppercase tracking-[0.18em] text-muted block mb-1">
          Strategy
        </label>
        <select
          value={value}
          onChange={(e) => setValue(e.target.value)}
          className="w-full bg-transparent border-0 font-display text-2xl text-ink focus:outline-none"
        >
          {strategies.map((x) => (
            <option key={x.id} value={x.id}>
              {x.name} · {x.asset}
            </option>
          ))}
        </select>
      </div>
      {s && (
        <div>
          <dl className="grid grid-cols-2 text-sm">
            <dt className="px-4 py-3 border-b border-r border-hair bg-paper label">Asset</dt>
            <dd className="px-4 py-3 border-b border-hair font-mono tabular-nums">{s.asset}</dd>
            <dt className="px-4 py-3 border-b border-r border-hair bg-paper label">Period</dt>
            <dd className="px-4 py-3 border-b border-hair font-mono tabular-nums">{s.period}</dd>
            <dt className="px-4 py-3 border-b border-r border-hair bg-paper label">Return</dt>
            <dd className="px-4 py-3 border-b border-hair font-mono tabular-nums">
              {s.headline.return ?? <span className="text-muted">—</span>}
            </dd>
            <dt className="px-4 py-3 border-b border-r border-hair bg-paper label">Sharpe</dt>
            <dd className="px-4 py-3 border-b border-hair font-mono tabular-nums">
              {s.headline.sharpe ?? <span className="text-muted">—</span>}
            </dd>
            <dt className="px-4 py-3 border-b border-r border-hair bg-paper label">Grade</dt>
            <dd className="px-4 py-3 border-b border-hair">
              <span className={`grade ${s.grade.startsWith("A") ? "grade--a" : s.grade.startsWith("B") ? "grade--b" : s.grade.startsWith("C") ? "grade--c" : s.grade.startsWith("D") ? "grade--d" : "grade--f"}`}>
                {s.grade}
              </span>
            </dd>
          </dl>
          <div className="p-4 border-t border-hair">
            <img src={s.pages.scorecard} alt="" className="block w-full h-auto border border-hair" />
          </div>
        </div>
      )}
    </div>
  );
}

export default function CompareDiff() {
  const [a, setA] = useState(STRATEGIES[0].id);
  const [b, setB] = useState(STRATEGIES[4].id);
  const A = STRATEGIES.find((x) => x.id === a)!;
  const B = STRATEGIES.find((x) => x.id === b)!;

  return (
    <div>
      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-4 md:gap-6">
        <StrategyColumn value={a} setValue={setA} strategies={STRATEGIES} />
        <div className="hidden md:flex items-center justify-center">
          <div className="border border-hair bg-paper px-3 py-4 text-center min-w-[100px]">
            <div className="font-mono text-[0.68rem] uppercase tracking-[0.18em] text-muted mb-2">Δ Grade</div>
            <GradeDelta a={A.grade} b={B.grade} />
          </div>
        </div>
        <StrategyColumn value={b} setValue={setB} strategies={STRATEGIES} />
      </div>
      <div className="md:hidden mt-4 border border-hair bg-paper px-4 py-3 text-center">
        <div className="font-mono text-[0.68rem] uppercase tracking-[0.18em] text-muted mb-1">Δ Grade</div>
        <GradeDelta a={A.grade} b={B.grade} />
      </div>
    </div>
  );
}
