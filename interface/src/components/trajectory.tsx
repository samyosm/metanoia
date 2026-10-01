import { Heading } from "./fields";
import { percent } from "./probabilities";

const W = 360;
const H = 160;
const PAD = { left: 36, right: 12, top: 10, bottom: 28 };
const COLORS = 8; // --chart-1 … --chart-8

/**
 * Each candidate's probability as its branch grows: the direct answer (0 statements
 * per candidate), then after reading 1, 2, 4, … statements. Points are added as the
 * stages stream in. Widths are spaced evenly, not to scale.
 */
export function Trajectory({
  ids,
  names,
  snapshots,
}: {
  ids: string[];
  names: Map<string, string>;
  snapshots: { width: number; probabilities: number[] }[];
}) {
  if (snapshots.length === 0) return null;
  const n = snapshots.length;
  const x = (i: number) =>
    PAD.left + (n === 1 ? 0 : (i * (W - PAD.left - PAD.right)) / (n - 1));
  const y = (p: number) => PAD.top + (1 - p) * (H - PAD.top - PAD.bottom);
  const color = (k: number) => `var(--chart-${(k % COLORS) + 1})`;
  return (
    <div>
      <Heading>How the answer was built</Heading>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full max-w-xl"
        role="img"
        aria-label="Each candidate's probability after reading more statements per candidate; the values are in the table below."
      >
        {[0, 0.5, 1].map((p) => (
          <g key={p}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={y(p)}
              y2={y(p)}
              stroke="var(--border-subtle)"
            />
            <text
              x={PAD.left - 6}
              y={y(p) + 3}
              textAnchor="end"
              fontSize="9"
              fill="var(--text-helper)"
            >
              {p * 100}%
            </text>
          </g>
        ))}
        {snapshots.map((s, i) => (
          <text
            key={`w-${s.width}-${i}`}
            x={x(i)}
            y={H - 10}
            textAnchor="middle"
            fontSize="9"
            fill="var(--text-helper)"
          >
            {s.width === 0 ? "direct" : s.width}
          </text>
        ))}
        {ids.map((id, k) => {
          const points = snapshots.map((s, i) => [
            x(i),
            y(s.probabilities[k] ?? 0),
          ]);
          return (
            <g key={id}>
              <polyline
                points={points.map(([px, py]) => `${px},${py}`).join(" ")}
                fill="none"
                stroke={color(k)}
                strokeWidth="2"
              />
              {points.map(([px, py], i) => (
                <circle
                  // biome-ignore lint/suspicious/noArrayIndexKey: one point per stage
                  key={i}
                  cx={px}
                  cy={py}
                  r="2.5"
                  fill={color(k)}
                />
              ))}
            </g>
          );
        })}
      </svg>
      <p className="mt-1 text-xs text-text-helper">
        Statements read per candidate (evenly spaced, not to scale).
        Uncalibrated probabilities.
      </p>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {ids.map((id, k) => (
          <li key={id} className="flex items-center gap-1.5">
            <span
              className="inline-block h-2 w-3"
              style={{ background: color(k) }}
              aria-hidden="true"
            />
            <span className="font-mono text-text-helper">{id}</span>
            <span className="max-w-48 truncate">{names.get(id) ?? id}</span>
          </li>
        ))}
      </ul>
      <table className="sr-only">
        <caption>Probability after reading each number of statements</caption>
        <thead>
          <tr>
            <th scope="col">Candidate</th>
            {snapshots.map((s, i) => (
              <th
                // biome-ignore lint/suspicious/noArrayIndexKey: one column per stage
                key={i}
                scope="col"
              >
                {s.width === 0 ? "direct" : `${s.width} per candidate`}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ids.map((id, k) => (
            <tr key={id}>
              <th scope="row">{names.get(id) ?? id}</th>
              {snapshots.map((s, i) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: one cell per stage
                <td key={i}>{percent(s.probabilities[k] ?? 0)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
