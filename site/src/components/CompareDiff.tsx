import { useState } from "react";
import { STRATEGIES, type StrategyCard } from "../data/strategies";

const GRADE_ORDER = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"];
function gradeRank(g: string): number {
  const idx = GRADE_ORDER.indexOf(g);
  return idx < 0 ? 99 : idx;
}
function gradeClass(g: string): string {
  const L = g.trim().charAt(0).toUpperCase();
  return L === "A" ? "grade--a" : L === "B" ? "grade--b" : L === "C" ? "grade--c" : L === "D" ? "grade--d" : L === "F" ? "grade--f" : "";
}

function parseNum(v: string | null | undefined): number | null {
  if (!v) return null;
  const m = v.match(/-?\d+(\.\d+)?/);
  return m ? parseFloat(m[0]) : null;
}

type Winner = "a" | "b" | "tie" | "none";

function compare(
  a: string | null | undefined,
  b: string | null | undefined,
  higherIsBetter = true,
): { winner: Winner; gap: string } {
  const na = parseNum(a);
  const nb = parseNum(b);
  if (na === null || nb === null) return { winner: "none", gap: "—" };
  if (na === nb) return { winner: "tie", gap: "≈" };
  const aHigher = na > nb;
  const aWins = higherIsBetter ? aHigher : !aHigher;
  return { winner: aWins ? "a" : "b", gap: Math.abs(na - nb).toFixed(2) };
}

function compareGrade(a: string, b: string): { winner: Winner; gap: string } {
  const ra = gradeRank(a);
  const rb = gradeRank(b);
  if (ra === rb) return { winner: "tie", gap: "≈" };
  return { winner: ra < rb ? "a" : "b", gap: String(Math.abs(ra - rb)) };
}

// Emphasize the winning cell — bold + maroon. The non-winner stays neutral ink.
function cellStyle(winner: Winner, side: "a" | "b"): string {
  return winner === side ? "font-semibold text-primary" : "";
}

function GapCell({ gap }: { gap: string }) {
  return <span className="font-mono tabular-nums text-[0.82rem] text-muted">{gap}</span>;
}

function Picker({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="block border border-hair bg-white hover:border-primary transition-colors">
      <span className="block px-4 pt-3 font-mono text-[0.68rem] uppercase tracking-[0.18em] text-primary">
        {label}
      </span>
      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full appearance-none bg-transparent border-0 px-4 pb-3 pt-1 pr-10 font-body text-[1.02rem] text-ink cursor-pointer focus:outline-none"
          style={{ WebkitAppearance: "none", MozAppearance: "none" }}
        >
          {STRATEGIES.map((x) => (
            <option key={x.id} value={x.id}>
              {x.name} · {x.asset}
            </option>
          ))}
        </select>
        <span
          aria-hidden="true"
          className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 font-mono text-[0.78rem] text-muted"
        >
          ▾
        </span>
      </div>
    </label>
  );
}

function ScorecardPanel({ label, s }: { label: string; s: StrategyCard }) {
  const [open, setOpen] = useState(false);
  return (
    <figure className="border border-hair bg-white">
      <figcaption className="border-b border-hair bg-paper px-5 py-4 flex items-baseline justify-between flex-wrap gap-3">
        <div>
          <div className="font-mono text-[0.68rem] uppercase tracking-[0.18em] text-primary">{label}</div>
          <div className="font-display text-[1.6rem] md:text-[1.85rem] text-ink leading-none mt-1">
            {s.name} <span className="text-muted italic">· {s.asset}</span>
          </div>
        </div>
        <button
          onClick={() => setOpen(true)}
          className="font-mono text-[0.68rem] uppercase tracking-[0.18em] text-primary hover:underline"
        >
          Enlarge ↗
        </button>
      </figcaption>
      <button onClick={() => setOpen(true)} className="block w-full cursor-zoom-in" aria-label={`Open ${s.name} full-size`}>
        <img src={s.pages.scorecard} alt={`${s.name} — scorecard`} className="block w-full h-auto" />
      </button>
      {open && (
        <div
          className="fixed inset-0 z-50 bg-ink/92 p-4 md:p-10 overflow-auto"
          onClick={() => setOpen(false)}
          role="dialog"
          aria-modal="true"
        >
          <button
            onClick={() => setOpen(false)}
            className="fixed top-4 right-4 font-mono text-[0.72rem] uppercase tracking-[0.18em] text-paper bg-transparent border border-paper/40 px-3 py-2 hover:border-paper"
          >
            Close ×
          </button>
          <img
            src={s.pages.scorecard}
            alt=""
            className="block mx-auto max-w-none w-[min(1600px,100%)] h-auto border border-paper/20"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </figure>
  );
}

export default function CompareDiff() {
  const [aId, setA] = useState(STRATEGIES[0].id);
  const [bId, setB] = useState(STRATEGIES[4].id);
  const A = STRATEGIES.find((x) => x.id === aId)!;
  const B = STRATEGIES.find((x) => x.id === bId)!;

  return (
    <div>
      {/* Pickers */}
      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-4 md:gap-6 items-end">
        <Picker label="Strategy A" value={aId} onChange={setA} />
        <div className="hidden md:flex items-center justify-center pb-3">
          <span className="font-display italic text-[1.2rem] text-muted">vs.</span>
        </div>
        <Picker label="Strategy B" value={bId} onChange={setB} />
      </div>

      {/* Metrics table */}
      {(() => {
        const cagr = compare(A.headline.return, B.headline.return, true);
        const sharpe = compare(A.headline.sharpe, B.headline.sharpe, true);
        const mdd = compare(A.headline.maxDrawdown, B.headline.maxDrawdown, true); // -11 > -22 → higher = better
        const grade = compareGrade(A.grade, B.grade);
        return (
          <div className="mt-8 border border-hair bg-white overflow-x-auto">
            <table className="w-full text-sm tabular">
              <thead>
                <tr className="border-b border-ink">
                  <th className="text-left font-mono text-[0.68rem] uppercase tracking-[0.18em] text-muted px-4 py-3 w-[180px]">Metric</th>
                  <th className="text-right font-display text-[1.05rem] text-ink px-4 py-3">
                    A &nbsp; <span className="text-muted italic font-display">{A.name} · {A.asset}</span>
                  </th>
                  <th className="text-center font-mono text-[0.68rem] uppercase tracking-[0.18em] text-muted px-3 py-3 w-[80px]">Gap</th>
                  <th className="text-right font-display text-[1.05rem] text-ink px-4 py-3">
                    B &nbsp; <span className="text-muted italic font-display">{B.name} · {B.asset}</span>
                  </th>
                </tr>
              </thead>
              <tbody className="text-ink">
                <tr className="border-b border-hair">
                  <td className="px-4 py-3 label">Asset class</td>
                  <td className="px-4 py-3 text-right font-mono text-[0.88rem] uppercase tracking-[0.08em]">{A.assetClass}</td>
                  <td className="px-3 py-3 text-center"></td>
                  <td className="px-4 py-3 text-right font-mono text-[0.88rem] uppercase tracking-[0.08em]">{B.assetClass}</td>
                </tr>
                <tr className="border-b border-hair">
                  <td className="px-4 py-3 label">Period</td>
                  <td className="px-4 py-3 text-right font-mono text-[0.88rem]">{A.period}</td>
                  <td className="px-3 py-3 text-center"></td>
                  <td className="px-4 py-3 text-right font-mono text-[0.88rem]">{B.period}</td>
                </tr>
                <tr className="border-b border-hair">
                  <td className="px-4 py-3 label">CAGR</td>
                  <td className={`px-4 py-3 text-right font-mono tabular-nums text-[1.05rem] ${cellStyle(cagr.winner, "a")}`}>{A.headline.return ?? "—"}</td>
                  <td className="px-3 py-3 text-center"><GapCell gap={cagr.gap} /></td>
                  <td className={`px-4 py-3 text-right font-mono tabular-nums text-[1.05rem] ${cellStyle(cagr.winner, "b")}`}>{B.headline.return ?? "—"}</td>
                </tr>
                <tr className="border-b border-hair">
                  <td className="px-4 py-3 label">Sharpe</td>
                  <td className={`px-4 py-3 text-right font-mono tabular-nums text-[1.05rem] ${cellStyle(sharpe.winner, "a")}`}>{A.headline.sharpe ?? "—"}</td>
                  <td className="px-3 py-3 text-center"><GapCell gap={sharpe.gap} /></td>
                  <td className={`px-4 py-3 text-right font-mono tabular-nums text-[1.05rem] ${cellStyle(sharpe.winner, "b")}`}>{B.headline.sharpe ?? "—"}</td>
                </tr>
                <tr className="border-b border-hair">
                  <td className="px-4 py-3 label">
                    Max drawdown
                    <span className="block text-[0.7rem] normal-case tracking-normal text-muted">smaller is better</span>
                  </td>
                  <td className={`px-4 py-3 text-right font-mono tabular-nums text-[1.05rem] ${cellStyle(mdd.winner, "a")}`}>{A.headline.maxDrawdown ?? "—"}</td>
                  <td className="px-3 py-3 text-center"><GapCell gap={mdd.gap} /></td>
                  <td className={`px-4 py-3 text-right font-mono tabular-nums text-[1.05rem] ${cellStyle(mdd.winner, "b")}`}>{B.headline.maxDrawdown ?? "—"}</td>
                </tr>
                <tr className="border-b border-hair">
                  <td className="px-4 py-3 label">Trades</td>
                  <td className="px-4 py-3 text-right font-mono tabular-nums text-[1.05rem]">{A.headline.trades ?? "—"}</td>
                  <td className="px-3 py-3 text-center"></td>
                  <td className="px-4 py-3 text-right font-mono tabular-nums text-[1.05rem]">{B.headline.trades ?? "—"}</td>
                </tr>
                <tr className="border-b border-hair">
                  <td className="px-4 py-3 label">Benchmark (CAGR)</td>
                  <td className="px-4 py-3 text-right font-mono text-[0.88rem]">{A.headline.benchmark ?? "—"}</td>
                  <td className="px-3 py-3 text-center"></td>
                  <td className="px-4 py-3 text-right font-mono text-[0.88rem]">{B.headline.benchmark ?? "—"}</td>
                </tr>
                <tr className="border-b border-hair">
                  <td className="px-4 py-3 label">Engine divergence</td>
                  <td className="px-4 py-3 text-right font-mono text-[0.88rem]">{A.headline.divergence ?? "—"}</td>
                  <td className="px-3 py-3 text-center"></td>
                  <td className="px-4 py-3 text-right font-mono text-[0.88rem]">{B.headline.divergence ?? "—"}</td>
                </tr>
                <tr>
                  <td className="px-4 py-3 label">Overall grade</td>
                  <td className={`px-4 py-3 text-right ${cellStyle(grade.winner, "a")}`}>
                    <span className={`grade ${gradeClass(A.grade)}`}>{A.grade}</span>
                  </td>
                  <td className="px-3 py-3 text-center"><GapCell gap={grade.gap} /></td>
                  <td className={`px-4 py-3 text-right ${cellStyle(grade.winner, "b")}`}>
                    <span className={`grade ${gradeClass(B.grade)}`}>{B.grade}</span>
                  </td>
                </tr>
              </tbody>
            </table>
            <div className="px-4 py-3 border-t border-hair text-[0.78rem] text-muted italic">
              The <span className="font-semibold text-primary not-italic">maroon, bold value</span> wins on that metric.
              The Gap column shows the absolute difference — no sign, no direction math.
            </div>
          </div>
        );
      })()}

      {/* Scorecard PNGs — stacked, full width */}
      <div className="mt-12 space-y-10">
        <ScorecardPanel label="Strategy A" s={A} />
        <ScorecardPanel label="Strategy B" s={B} />
      </div>
    </div>
  );
}
