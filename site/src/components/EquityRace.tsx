import { useMemo, useState, useEffect } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { RACE, MONTH_LABELS } from "../data/equity_race";

type Row = Record<string, number | string>;

export default function EquityRace() {
  // Step the race from 0 → N over ~3.5s on first render.
  const N = RACE[0].values.length;
  const [step, setStep] = useState(0);

  useEffect(() => {
    const total = 3500;
    const frames = 60;
    const interval = total / frames;
    let f = 0;
    const id = setInterval(() => {
      f += 1;
      const progress = Math.min(1, f / frames);
      setStep(Math.floor(progress * (N - 1)));
      if (f >= frames) clearInterval(id);
    }, interval);
    return () => clearInterval(id);
  }, [N]);

  const data: Row[] = useMemo(() => {
    const rows: Row[] = [];
    for (let i = 0; i <= step; i++) {
      const row: Row = { i, label: MONTH_LABELS[i] };
      for (const s of RACE) {
        row[s.id] = Number(s.values[i].toFixed(2));
      }
      rows.push(row);
    }
    return rows;
  }, [step]);

  const hasFinished = step >= N - 1;

  return (
    <div className="w-full">
      <div className="flex items-baseline justify-between gap-4 mb-3 border-b border-hair pb-2">
        <div className="font-mono text-[0.68rem] tracking-[0.18em] uppercase text-muted">
          Equity · Index = 100 · 2022 → 2025
        </div>
        <div className="font-mono text-[0.68rem] tracking-[0.18em] uppercase text-muted">
          Representative · pending JSON export
        </div>
      </div>
      <div style={{ width: "100%", height: 340 }}>
        <ResponsiveContainer>
          <LineChart
            data={data}
            margin={{ top: 12, right: 48, left: 8, bottom: 18 }}
          >
            <CartesianGrid stroke="#0F0F0F" strokeOpacity={0.06} vertical={false} />
            <XAxis
              dataKey="label"
              stroke="#8B8680"
              tick={{ fontSize: 11, fontFamily: "JetBrains Mono" }}
              tickLine={false}
              axisLine={{ stroke: "#E8E4DE" }}
              interval={Math.floor(N / 4)}
              minTickGap={20}
            />
            <YAxis
              stroke="#8B8680"
              tick={{ fontSize: 11, fontFamily: "JetBrains Mono" }}
              tickLine={false}
              axisLine={{ stroke: "#E8E4DE" }}
              width={44}
              domain={[80, "dataMax + 10"]}
              tickFormatter={(v: number) => `${v.toFixed(0)}`}
            />
            <ReferenceLine y={100} stroke="#0F0F0F" strokeOpacity={0.22} strokeDasharray="2 3" />
            <Tooltip
              contentStyle={{
                background: "#FAF7F2",
                border: "1px solid #E8E4DE",
                fontFamily: "JetBrains Mono",
                fontSize: 11,
                letterSpacing: "0.04em",
              }}
              labelStyle={{ color: "#8B8680" }}
              cursor={{ stroke: "#881C1C", strokeOpacity: 0.4, strokeDasharray: "2 3" }}
              formatter={(v: number, n: string) => {
                const s = RACE.find((r) => r.id === n);
                return [v.toFixed(2), s?.name ?? n];
              }}
            />
            {RACE.map((s) => (
              <Line
                key={s.id}
                dataKey={s.id}
                type="monotone"
                stroke={s.color}
                strokeWidth={s.id === "covered-call" ? 1.8 : 1.2}
                dot={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-3 flex flex-wrap gap-x-6 gap-y-2 text-xs">
        {RACE.map((s) => {
          const latest = s.values[Math.min(step, s.values.length - 1)];
          const pct = ((latest - 100) / 100) * 100;
          const sign = pct >= 0 ? "+" : "";
          return (
            <div key={s.id} className="flex items-baseline gap-2">
              <span
                className="inline-block h-2 w-4"
                style={{ background: s.color }}
                aria-hidden="true"
              />
              <span className="font-body text-ink">{s.name}</span>
              <span className="text-muted font-mono tracking-tight">{s.asset}</span>
              <span
                className={`font-mono tabular-nums ml-1 ${
                  hasFinished ? "" : "opacity-80"
                }`}
                style={{ color: pct >= 0 ? "#2E7D32" : "#C62828" }}
              >
                {sign}
                {pct.toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
