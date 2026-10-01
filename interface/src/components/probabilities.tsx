import { Heading } from "./fields";

export function percent(p: number): string {
  if (p > 0 && p < 0.0001) return "<0.01%";
  return `${(p * 100).toFixed(2)}%`;
}

export type ProbabilityRow = { id: string; text: string; p: number };

/**
 * Candidates ranked by probability. While a decision is computed the bars move with
 * each streamed stage; `badge` says which stage and whether it is calibrated.
 */
export function Probabilities({
  rows,
  badge,
  note,
}: {
  rows: ProbabilityRow[];
  badge: string;
  note: string;
}) {
  // Ranked, but keyed by id, so rows animate in place as the ranking changes.
  const ranked = [...rows].sort((a, b) => b.p - a.p);
  const top = ranked[0]?.p ?? 0;
  return (
    <div>
      <div className="mb-4 flex items-baseline justify-between gap-4">
        <Heading>Probabilities</Heading>
        <span className="bg-layer px-2 py-0.5 text-xs text-text-secondary">
          {badge}
        </span>
      </div>
      <ol className="flex flex-col gap-4">
        {ranked.map((row) => (
          <li key={row.id}>
            <div className="mb-1 flex items-baseline justify-between gap-4">
              <span className="text-text-primary">
                <span className="mr-2 font-mono text-text-helper">
                  {row.id}
                </span>
                {row.text}
              </span>
              <span className="font-mono tabular-nums">{percent(row.p)}</span>
            </div>
            {/* Decorative: the percentage above carries the value. */}
            <div className="h-2 bg-layer" aria-hidden="true">
              <div
                className={`bar-animate h-full transition-[width] duration-300 ease-out ${
                  row.p === top ? "bg-interactive" : "bg-bar-muted"
                }`}
                style={{ width: `${row.p * 100}%` }}
              />
            </div>
          </li>
        ))}
      </ol>
      <p className="mt-4 text-xs text-text-helper">{note}</p>
    </div>
  );
}

export const CALIBRATED_NOTE =
  "A temperature fitted on held-out data: on that distribution, stated confidence matched accuracy on average. That is not a guarantee for any single answer.";
export const RAW_NOTE =
  "A plain softmax of the scores: not evidence of reliable confidence.";
export const LIVE_NOTE =
  "Live: the answer so far, as a plain softmax. The final answer (and its calibration, if any) replaces it when the computation ends.";
