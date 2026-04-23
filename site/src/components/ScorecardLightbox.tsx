import { useState } from "react";

interface Page {
  label: string;
  src: string;
}

export default function ScorecardLightbox({ pages }: { pages: Page[] }) {
  const [open, setOpen] = useState<string | null>(null);

  return (
    <>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {pages.map((p) => (
          <figure key={p.label} className="scorecard-figure">
            <figcaption className="px-4 py-3 border-b border-hair bg-paper flex items-baseline justify-between">
              <span className="font-display text-lg leading-none text-ink">{p.label}</span>
              <button
                onClick={() => setOpen(p.src)}
                className="font-mono text-[0.68rem] uppercase tracking-[0.18em] text-primary hover:underline"
              >
                Enlarge ↗
              </button>
            </figcaption>
            <button
              onClick={() => setOpen(p.src)}
              className="block w-full cursor-zoom-in"
              aria-label={`Open ${p.label} full-size`}
            >
              <img src={p.src} alt={p.label} className="block w-full h-auto" />
            </button>
          </figure>
        ))}
      </div>

      {open && (
        <div
          className="fixed inset-0 z-50 bg-ink/92 p-4 md:p-10 overflow-auto"
          onClick={() => setOpen(null)}
          role="dialog"
          aria-modal="true"
        >
          <button
            onClick={() => setOpen(null)}
            className="fixed top-4 right-4 font-mono text-[0.72rem] uppercase tracking-[0.18em] text-paper bg-transparent border border-paper/40 px-3 py-2 hover:border-paper"
          >
            Close ×
          </button>
          <img
            src={open}
            alt=""
            className="block mx-auto max-w-none w-[min(1600px,100%)] h-auto border border-paper/20"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </>
  );
}
