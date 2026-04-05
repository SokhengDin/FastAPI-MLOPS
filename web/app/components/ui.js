"use client";

import { useEffect, useRef } from "react";

export function Card({ children, className = "" }) {
  return (
    <div className={`bg-[var(--surface)] border border-[var(--border)] rounded-xl p-4 ${className}`}>
      {children}
    </div>
  );
}

export function CardHeader({ title, subtitle }) {
  return (
    <div className="mb-3">
      <h3 className="text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wider">{title}</h3>
      {subtitle && <p className="text-xs text-[var(--text-muted)] mt-0.5">{subtitle}</p>}
    </div>
  );
}

const LABEL_COLORS = {
  Low    : "bg-green-100 text-green-800 border-green-200",
  Medium : "bg-yellow-100 text-yellow-800 border-yellow-200",
  High   : "bg-red-100 text-red-800 border-red-200",
};

export function Badge({ label, large }) {
  const cls = LABEL_COLORS[label] ?? "bg-slate-100 text-slate-800 border-slate-200";
  return (
    <span className={`inline-flex items-center rounded-full font-semibold border ${large ? "px-4 py-1.5 text-base" : "px-2.5 py-0.5 text-xs"} ${cls}`}>
      {label}
    </span>
  );
}

export function StatBlock({ label, value, sub }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-[var(--text-muted)] uppercase tracking-wider">{label}</span>
      <span className="text-2xl font-bold tabular-nums text-[var(--text-primary)]">{value}</span>
      {sub && <span className="text-xs text-[var(--text-muted)]">{sub}</span>}
    </div>
  );
}

export function Slider({ label, name, min, max, step = 0.01, value, onChange }) {
  const decimals = step < 1 ? String(step).split(".")[1]?.length ?? 2 : 0;
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs">
        <span className="text-[var(--text-muted)]">{label}</span>
        <span className="font-mono font-medium text-[var(--text-primary)]">{Number(value).toFixed(decimals)}</span>
      </div>
      <input
        type="range" name={name} min={min} max={max} step={step} value={value} onChange={onChange}
        className="w-full h-1.5 rounded-full accent-[var(--accent)] cursor-pointer"
      />
    </div>
  );
}

export function Select({ label, name, options, value, onChange }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-[var(--text-muted)]">{label}</label>
      <select
        name={name} value={value} onChange={onChange}
        className="w-full border border-[var(--border)] rounded-lg px-2.5 py-1.5 text-sm bg-[var(--surface)] text-[var(--text-primary)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]/30"
      >
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}

export function Button({ children, onClick, type = "button", loading, variant = "primary", className = "" }) {
  const base = "flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all disabled:opacity-50 cursor-pointer";
  const styles = {
    primary   : "bg-[var(--accent)] text-[var(--accent-fg)] hover:opacity-80",
    secondary : "border border-[var(--border)] text-[var(--text-primary)] hover:bg-[var(--background)]",
    ghost     : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--background)]",
  };
  return (
    <button type={type} onClick={onClick} disabled={loading} className={`${base} ${styles[variant]} ${className}`}>
      {loading && <span className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" />}
      {children}
    </button>
  );
}

export function Tabs({ tabs, active, onChange }) {
  return (
    <div className="flex border-b border-[var(--border)] mb-4">
      {tabs.map(t => (
        <button
          key={t}
          onClick={() => onChange(t)}
          className={`px-4 py-2 text-xs font-semibold uppercase tracking-wider border-b-2 -mb-px transition-colors ${
            active === t
              ? "border-[var(--accent)] text-[var(--text-primary)]"
              : "border-transparent text-[var(--text-muted)] hover:text-[var(--text-primary)]"
          }`}
        >
          {t}
        </button>
      ))}
    </div>
  );
}

export function Modal({ open, onClose, title, children }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
      <div ref={ref} className="relative bg-[var(--surface)] border border-[var(--border)] rounded-2xl shadow-xl w-full max-w-md mx-4 p-6 z-10">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-sm font-semibold text-[var(--text-primary)] uppercase tracking-wider">{title}</h2>
          <button onClick={onClose} className="text-[var(--text-muted)] hover:text-[var(--text-primary)] text-lg leading-none">✕</button>
        </div>
        {children}
      </div>
    </div>
  );
}
