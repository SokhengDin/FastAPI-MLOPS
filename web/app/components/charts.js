"use client";

import { useEffect, useRef } from "react";
import uPlot from "uplot";

const LABEL_HEX = { High: "#dc2626", Medium: "#ca8a04", Low: "#16a34a" };

// Horizontal bar chart for class probabilities
export function ProbabilityBars({ probabilities }) {
  if (!probabilities?.length) return null;

  const sorted = [...probabilities].sort((a, b) => b.probability - a.probability);

  return (
    <div className="flex flex-col gap-3 w-full">
      {sorted.map(({ label, probability }) => (
        <div key={label} className="flex flex-col gap-1">
          <div className="flex justify-between text-xs font-medium">
            <span className="text-[var(--text-primary)]">{label}</span>
            <span className="font-mono text-[var(--text-muted)]">{(probability * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2.5 w-full rounded-full bg-slate-100 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width           : `${probability * 100}%`,
                backgroundColor : LABEL_HEX[label] ?? "#94a3b8",
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

// Radial confidence gauge (pure CSS + SVG)
export function ConfidenceGauge({ value = 0 }) {
  const pct     = Math.round(value * 100);
  const radius  = 54;
  const circ    = 2 * Math.PI * radius;
  const dash    = (pct / 100) * circ;
  const color   = pct >= 80 ? "#16a34a" : pct >= 50 ? "#ca8a04" : "#dc2626";

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width="140" height="80" viewBox="0 0 140 80">
        <path
          d={`M 16 70 A ${radius} ${radius} 0 0 1 124 70`}
          fill="none" stroke="#e2e8f0" strokeWidth="10" strokeLinecap="round"
        />
        <path
          d={`M 16 70 A ${radius} ${radius} 0 0 1 124 70`}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${(pct / 100) * 169} 169`}
        />
        <text x="70" y="68" textAnchor="middle" fontSize="22" fontWeight="700" fill="currentColor">
          {pct}%
        </text>
      </svg>
      <span className="text-xs text-[var(--text-muted)]">Confidence</span>
    </div>
  );
}

// Per-class F1 score bars from classification report
export function ClassMetricsBars({ report }) {
  if (!report) return null;

  const classes = ["High", "Medium", "Low"].filter(c => report[c]);

  return (
    <div className="flex flex-col gap-4">
      {classes.map(cls => {
        const { precision, recall, "f1-score": f1 } = report[cls];
        return (
          <div key={cls} className="flex flex-col gap-2">
            <span className="text-xs font-semibold text-[var(--text-primary)]">{cls}</span>
            {[["Precision", precision], ["Recall", recall], ["F1", f1]].map(([name, val]) => (
              <div key={name} className="flex items-center gap-2">
                <span className="w-16 text-xs text-[var(--text-muted)]">{name}</span>
                <div className="flex-1 h-2 rounded-full bg-slate-100 overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${val * 100}%`, backgroundColor: LABEL_HEX[cls] }}
                  />
                </div>
                <span className="w-10 text-right text-xs font-mono text-[var(--text-muted)]">
                  {(val * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}

// Horizontal bar chart for feature importances
export function FeatureImportanceChart({ importance }) {
  if (!importance || !Object.keys(importance).length) return null;

  const entries = Object.entries(importance)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12);

  const max = entries[0][1];

  return (
    <div className="flex flex-col gap-2 w-full">
      {entries.map(([name, val]) => (
        <div key={name} className="flex items-center gap-2">
          <span className="w-40 text-xs text-[var(--text-muted)] truncate shrink-0">{name.replace(/_/g, " ")}</span>
          <div className="flex-1 h-2 rounded-full bg-slate-100 overflow-hidden">
            <div
              className="h-full rounded-full bg-[var(--accent)] transition-all duration-500"
              style={{ width: `${(val / max) * 100}%` }}
            />
          </div>
          <span className="w-10 text-right text-xs font-mono text-[var(--text-muted)] shrink-0">
            {(val * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}


// uPlot line chart — prediction history over session
export function PredictionHistoryChart({ history }) {
  const ref     = useRef(null);
  const plotRef = useRef(null);

  useEffect(() => {
    if (!ref.current || history.length < 2) return;

    const xs      = history.map((_, i) => i);
    const high    = history.map(h => h.probabilities?.find(p => p.label === "High")?.probability ?? 0);
    const medium  = history.map(h => h.probabilities?.find(p => p.label === "Medium")?.probability ?? 0);
    const low     = history.map(h => h.probabilities?.find(p => p.label === "Low")?.probability ?? 0);

    const opts = {
      width   : ref.current.clientWidth,
      height  : 180,
      cursor  : { show: true },
      legend  : { show: true },
      axes    : [
        { stroke: "#94a3b8", ticks: { stroke: "#e2e8f0" } },
        { stroke: "#94a3b8", ticks: { stroke: "#e2e8f0" }, size: 40 },
      ],
      series  : [
        {}
        , { label: "High",   stroke: LABEL_HEX.High,   width: 2, fill: LABEL_HEX.High + "20" }
        , { label: "Medium", stroke: LABEL_HEX.Medium, width: 2, fill: LABEL_HEX.Medium + "20" }
        , { label: "Low",    stroke: LABEL_HEX.Low,    width: 2, fill: LABEL_HEX.Low + "20" }
      ],
    };

    if (plotRef.current) plotRef.current.destroy();
    plotRef.current = new uPlot(opts, [xs, high, medium, low], ref.current);

    return () => { plotRef.current?.destroy(); plotRef.current = null; };
  }, [history]);

  if (history.length < 2) {
    return (
      <div className="flex items-center justify-center h-[180px] text-sm text-[var(--text-muted)]">
        Make 2+ predictions to see history
      </div>
    );
  }

  return <div ref={ref} className="w-full" />;
}
