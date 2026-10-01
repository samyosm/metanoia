import type { NodeView } from "@/lib/metanoia/contract";
import { Heading } from "./fields";
import { percent } from "./probabilities";

export type TreeCandidate = {
  id: string;
  text: string;
  /** Current probability (live or final). */
  p: number | null;
  /** The direct answer's probability (no expansion), for the change. */
  direct: number | null;
  nodes: NodeView[];
  /** Leave-one-out effect of each node on this candidate's score, once measured. */
  contributions: number[] | null;
};

/**
 * One level of the tree: a vertical rail with a tick to each child, the rail ending at
 * the last child. Children are positioned relative to their own row.
 */
const BRANCH =
  "relative pl-6 before:absolute before:top-0 before:left-2 before:h-full before:w-px before:bg-border-strong last:before:h-5 after:absolute after:top-5 after:left-2 after:h-px after:w-3 after:bg-border-strong";

/**
 * The expansion tree: the question at the root, each candidate as a branch, and the
 * statements it expanded into as leaves. Statements beyond `readWidth` have been
 * proposed but not read yet; their effect appears once it is measured.
 */
export function ExpansionTree({
  question,
  candidates,
  readWidth,
  leader,
}: {
  question: string;
  candidates: TreeCandidate[];
  readWidth: number;
  leader: string | null;
}) {
  const scale = Math.max(
    1e-9,
    ...candidates.flatMap((c) => (c.contributions ?? []).map(Math.abs)),
  );
  const measured = candidates.some((c) => c.contributions !== null);
  return (
    <div>
      <Heading>Expansion tree</Heading>
      <div className="border border-border-subtle bg-layer px-4 py-3">
        <span className="mr-2 text-xs tracking-[0.32px] text-text-helper">
          Question
        </span>
        {question}
      </div>
      <ol className="ml-2">
        {candidates.map((c) => (
          <li key={c.id} className={`${BRANCH} pt-3`}>
            <CandidateHeader candidate={c} leading={c.id === leader} />
            {c.nodes.length > 0 && (
              <ul className="ml-2">
                {c.nodes.map((node, i) => (
                  <Leaf
                    key={`${c.id}-${node.id}-${i}`}
                    node={node}
                    read={i < readWidth}
                    contribution={c.contributions?.[i] ?? null}
                    scale={scale}
                  />
                ))}
              </ul>
            )}
          </li>
        ))}
      </ol>
      <p className="mt-3 text-xs text-text-helper">
        {measured
          ? "Each statement's number is its effect on its own candidate's score: the score with the whole branch minus the score without that statement. It measures what the model used, not whether the statement is true."
          : "Statements appear as each candidate expands, and turn solid once the model has read them."}
      </p>
    </div>
  );
}

function CandidateHeader({
  candidate,
  leading,
}: {
  candidate: TreeCandidate;
  leading: boolean;
}) {
  const { p, direct } = candidate;
  const change = p !== null && direct !== null ? p - direct : null;
  return (
    <div
      className={`flex flex-wrap items-baseline justify-between gap-x-4 border-l-4 px-3 py-1.5 ${
        leading ? "border-interactive bg-layer" : "border-transparent"
      }`}
    >
      <span>
        <span className="mr-2 font-mono text-text-helper">{candidate.id}</span>
        {candidate.text}
      </span>
      <span className="flex items-baseline gap-3">
        {change !== null && Math.abs(change) >= 0.0005 && (
          <span className="text-xs text-text-helper">
            <span
              className={
                change > 0 ? "text-support-success" : "text-support-error"
              }
            >
              {change > 0 ? "+" : "−"}
              {(Math.abs(change) * 100).toFixed(1)} pts
            </span>{" "}
            vs direct {percent(direct ?? 0)}
          </span>
        )}
        {p !== null && (
          <span className="font-mono tabular-nums">{percent(p)}</span>
        )}
      </span>
    </div>
  );
}

function Leaf({
  node,
  read,
  contribution,
  scale,
}: {
  node: NodeView;
  read: boolean;
  contribution: number | null;
  scale: number;
}) {
  const width = contribution === null ? 0 : Math.abs(contribution) / scale;
  const positive = (contribution ?? 0) >= 0;
  return (
    <li
      className={`node-enter ${BRANCH} pt-1.5 text-sm transition-opacity duration-300 ${
        read ? "opacity-100" : "opacity-40"
      }`}
    >
      <div className="flex items-start justify-between gap-4">
        <span>
          {node.concept && (
            <span className="mr-1.5 bg-layer px-1 font-mono text-xs text-text-secondary">
              {node.concept}
            </span>
          )}
          {node.text}
          {!read && (
            <span className="ml-1 text-xs text-text-helper">
              (not read yet)
            </span>
          )}
        </span>
        {contribution !== null && (
          <span className="flex shrink-0 items-center gap-2 pt-1">
            {/* The number carries the value; the bar is decorative. */}
            <span className="relative h-1.5 w-16 bg-layer" aria-hidden="true">
              <span
                className={`absolute top-0 h-full ${
                  positive
                    ? "left-1/2 bg-support-success"
                    : "right-1/2 bg-support-error"
                }`}
                style={{ width: `${width * 50}%` }}
              />
            </span>
            <span className="w-10 text-right font-mono text-xs tabular-nums text-text-helper">
              {positive ? "+" : "−"}
              {Math.abs(contribution).toFixed(2)}
            </span>
          </span>
        )}
      </div>
    </li>
  );
}
