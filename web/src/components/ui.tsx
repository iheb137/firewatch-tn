import { LoaderCircle, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";

export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="mb-6 max-w-3xl">
      <h1 className="display text-2xl md:text-3xl">{title}</h1>
      {children && <p className="mt-2 text-sm leading-relaxed text-lichen">{children}</p>}
    </header>
  );
}

export function Panel({ title, children, className = "" }: { title?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-xl border border-rule bg-canopy p-4 ${className}`}>
      {title && <h2 className="mb-3 text-sm font-semibold text-paper">{title}</h2>}
      {children}
    </section>
  );
}

export function Metric({ label, value, hint, tone = "paper" }: {
  label: string;
  value: string;
  hint?: string;
  tone?: "paper" | "ember" | "water" | "sage";
}) {
  const color = { paper: "text-paper", ember: "text-ember", water: "text-water", sage: "text-sage" }[tone];
  return (
    <div>
      <dt className="text-xs text-lichen">{label}</dt>
      <dd className={`tabular text-xl font-semibold ${color}`}>{value}</dd>
      {hint && <dd className="text-xs text-moss">{hint}</dd>}
    </div>
  );
}

export function Loading({ children = "Chargement…" }: { children?: ReactNode }) {
  return (
    <p className="flex items-center gap-2 text-sm text-lichen" role="status">
      <LoaderCircle size={16} className="animate-spin" aria-hidden />
      {children}
    </p>
  );
}

export function ErrorNote({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div role="alert" className="flex gap-3 rounded-xl border border-ember-deep/50 bg-ember-deep/10 p-4 text-sm">
      <TriangleAlert size={18} className="mt-0.5 shrink-0 text-ember" aria-hidden />
      <div>
        <p className="font-semibold">{title}</p>
        {children && <div className="mt-1 text-lichen">{children}</div>}
      </div>
    </div>
  );
}

export function Toggle({ checked, onChange, children }: {
  checked: boolean;
  onChange: (v: boolean) => void;
  children: ReactNode;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 py-1.5 text-sm">
      <span>{children}</span>
      <input
        type="checkbox"
        className="peer sr-only"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span
        aria-hidden
        className="relative h-5 w-9 shrink-0 rounded-full bg-rule transition-colors peer-checked:bg-moss peer-focus-visible:outline-2 peer-focus-visible:outline-sage after:absolute after:left-0.5 after:top-0.5 after:size-4 after:rounded-full after:bg-paper after:transition-transform peer-checked:after:translate-x-4"
      />
    </label>
  );
}
