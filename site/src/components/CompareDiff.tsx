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

function parsePct(v: string | null | undefined): number | null {
  if (!v) return null;
  const m = v.match(/-?\d+(\.\d+)?/);
  return m ? parseFloat(m[0]) : null;
}

function deltaCell(
  a: string | null | undefined,
  b: string | null | undefined,
  opts: { higherIsBetter?: boolean } = {},
) {
  const higherIsBetter = opts.higherIsBetter ?? true;
  const na = parsePct(a);
  const nb = parsePct(b);
  if (na === null || nb === null) return <span className="text-muted">—</span>;
  const diff = na - nb;
  if (diff === 0) return <span className="text-muted font-mono">≈</span>;
  const sign = diff > 0 ? "+" : "−";
  const good = higherIsBetter ? diff > 0 : diff < 0;
  return (
    <span
      className="font-mono tabular-nums text-[0.88rem]"
      style={{ color: good ? "#2E7D32" : "#C62828" }}
    >
      {sign}
      {Math.abs(diff).toFixed(2)}
    </span>
  );
}

function deltaGrade(a: string, b: string) {
  const diff = gradeRank(a) - gradeRank(b);
  if (diff === 0) return <span className="text-muted font-mono">≈</span>;
  // lower rank number = better grade → if A-rank(a) < A-rank(b), a is better
  const good = diff < 0;
  const arrow = diff < 0 ? "↑" : "↓";
  return (
    <span
      className="font-mono text-[0.88rem]"
      style={{ color: good ? "#2E7D32" : "#C62828" }}
    >
      {arrow} {Math.abs(diff)}
    </span>
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

  // Note: Δ for mixed returnLabels (CAGR vs Total return) is mathematically
  // inappropriate. We suppress the delta when labels differ.
  const returnLabelsMatch = A.headline.returnLabel === B.headline.returnLabel;

  return (
    <div>
      {/* Pickers */}
      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-4 md:gap-6 items-end">
        <label className="block border border-hair bg-white px-4 py-3">
          <div className="font-mono text-[0.68rem] uppercase tracking-[0.18em] text-primary mb-1">Strategy A</div>
          <select
            value={aId}
            onChange={(e) => setA(e.target.value)}
            className="w-full bg-transparent border-0 font-display text-[1.5rem] md:text-[1.9rem] text-ink focus:outline-none leading-tight"
          >
            {STRATEGIES.map((x) => (
              <option key={x.id} value={x.id}>
                {x.name} · {x.asset}
              </option>
            ))}
          </select>
        </label>
        <div className="hidden md:flex items-center justify-center pb-3">
          <span className="font-display italic text-[1.2rem] text-muted">vs.</span>
        </div>
        <label className="block border border-hair bg-white px-4 py-3">
          <div className="font-mono text-[0.68rem] uppercase tracking-[0.18em] text-primary mb-1">Strategy B</div>
          <select
            value={bId}
            onChange={(e) => setB(e.target.value)}
            className="w-full bg-transparent border-0 font-display text-[1.5rem] md:text-[1.9rem] text-ink focus:outline-none leading-tight"
          >
            {STRATEGIES.map((x) => (
              <option key={x.id} value={x.id}>
                {x.name} · {x.asset}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* Metrics table */}
      <div className="mt-8 border border-hair bg-white overflow-x-auto">
        <table className="w-full text-sm tabular">
          <thead>
            <tr className="border-b border-ink">
              <th className="text-left font-mono text-[0.68rem] uppercase tracking-[0.18em] text-muted px-4 py-3 w-[180px]">Metric</th>
              <th className="text-right font-display text-[1.05rem] text-ink px-4 py-3">
                A &nbsp; <span className="text-muted italic font-display">{A.name} · {A.asset}</span>
              </th>
              <th className="text-center font-mono text-[0.68rem] uppercase tracking-[0.18em] text-muted px-3 py-3 w-[70px]">Δ</th>
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
              <td className="px-4 py-3 label">
                Return
                <span className="block text-[0.7rem] normal-case tracking-normal text-muted">
                  {A.headline.returnLabel}{returnLabelsMatch ? "" : ` / ${B.headline.returnLabel}`}
                </span>
              </td>
              <td className="px-4 py-3 text-right font-mono tabular-nums text-[1.05rem]">{A.headline.return ?? "—"}</td>
              <td className="px-3 py-3 text-center">
                {returnLabelsMatch ? deltaCell(A.headline.return, B.headline.return) : <span className="text-muted italic text-[0.72rem]">mixed</span>}
              </td>
              <td className="px-4 py-3 text-right font-mono tabular-nums text-[1.05rem]">{B.headline.return ?? "—"}</td>
            </tr>
            <tr className="border-b border-hair">
              <td className="px-4 py-3 label">Sharpe</td>
              <td className="px-4 py-3 text-right font-mono tabular-nums text-[1.05rem]">{A.headline.sharpe ?? "—"}</td>
              <td className="px-3 py-3 text-center">{deltaCell(A.headline.sharpe, B.headline.sharpe)}</td>
              <td className="px-4 py-3 text-right font-mono tabular-nums text-[1.05rem]">{B.headline.sharpe ?? "—"}</td>
            </tr>
            <tr className="border-b border-hair">
              <td className="px-4 py-3 label">Max drawdown</td>
              <td className="px-4 py-3 text-right font-mono tabular-nums text-[1.05rem]">{A.headline.maxDrawdown ?? "—"}</td>
              <td className="px-3 py-3 text-center">{deltaCell(A.headline.maxDrawdown, B.headline.maxDrawdown, { higherIsBetter: true })}</td>
              <td className="px-4 py-3 text-right font-mono tabular-nums text-[1.05rem]">{B.headline.maxDrawdown ?? "—"}</td>
            </tr>
            <tr className="border-b border-hair">
              <td className="px-4 py-3 label">Trades</td>
              <td className="px-4 py-3 text-right font-mono tabular-nums text-[1.05rem]">{A.headline.trades ?? "—"}</td>
              <td className="px-3 py-3 text-center"></td>
              <td className="px-4 py-3 text-right font-mono tabular-nums text-[1.05rem]">{B.headline.trades ?? "—"}</td>
            </tr>
            <tr className="border-b border-hair">
              <td className="px-4 py-3 label">Benchmark</td>
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
              <td className="px-4 py-3 text-right">
                <span className={`grade ${gradeClass(A.grade)}`}>{A.grade}</span>
              </td>
              <td className="px-3 py-3 text-center">{deltaGrade(A.grade, B.grade)}</td>
              <td className="px-4 py-3 text-right">
                <span className={`grade ${gradeClass(B.grade)}`}>{B.grade}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Scorecard PNGs — stacked, full width */}
      <div className="mt-12 space-y-10">
        <ScorecardPanel label="Strategy A" s={A} />
        <ScorecardPanel label="Strategy B" s={B} />
      </div>
    </div>
  );
}
