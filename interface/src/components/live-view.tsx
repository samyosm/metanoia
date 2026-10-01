"use client";

import { useEffect, useState } from "react";
import type { LiveState, Stage } from "@/lib/metanoia/live";
import { branchesFromTrace } from "@/lib/metanoia/live";
import { Path } from "./path";
import {
  CALIBRATED_NOTE,
  LIVE_NOTE,
  Probabilities,
  type ProbabilityRow,
  RAW_NOTE,
} from "./probabilities";
import { Trajectory } from "./trajectory";
import { ExpansionTree, type TreeCandidate } from "./tree";

function describe(stage: Stage | null, names: Map<string, string>): string {
  if (!stage) return "Starting the model (the first request loads it)…";
  switch (stage.kind) {
    case "render":
      return "Reading the input…";
    case "score":
      return stage.width === 0
        ? "Direct answer computed; expanding the candidates…"
        : `Read ${stage.width} statement${stage.width === 1 ? "" : "s"} per candidate…`;
    case "expand":
      return `Expanded “${names.get(stage.candidate) ?? stage.candidate}”…`;
    case "contribution":
      return `Measured each statement's effect on “${names.get(stage.candidate) ?? stage.candidate}”…`;
    case "done":
      return "Done.";
  }
}

/** Seconds since `running` became true, ticking while it stays true. */
function useElapsed(running: boolean): number {
  const [started, setStarted] = useState<number | null>(null);
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (!running) return;
    const t0 = performance.now();
    setStarted(t0);
    setNow(t0);
    const timer = setInterval(() => setNow(performance.now()), 100);
    return () => clearInterval(timer);
  }, [running]);
  return started === null ? 0 : (now - started) / 1000;
}

export function LiveView({
  state,
  running,
}: {
  state: LiveState;
  running: boolean;
}) {
  const elapsed = useElapsed(running);
  const { request, result } = state;
  if (!request) {
    return (
      <p className="text-text-helper">
        Probabilities, the expansion tree and the computation path appear here,
        live, as the model works.
      </p>
    );
  }
  const ids = request.candidates.map((c) => c.id);
  const names = new Map(request.candidates.map((c) => [c.id, c.text]));
  const latest = state.snapshots.at(-1) ?? null;
  const direct =
    state.snapshots.find((s) => s.width === 0)?.probabilities ?? null;

  // Final probabilities replace the live ones when the result arrives.
  const probabilities = result
    ? ids.map(
        (id) => result.probabilities[result.candidate_ids.indexOf(id)] ?? 0,
      )
    : (latest?.probabilities ?? null);
  const calibrated = result?.calibration.status === "fitted";
  const rows: ProbabilityRow[] | null = probabilities
    ? ids.map((id, i) => ({
        id,
        text: names.get(id) ?? id,
        p: probabilities[i] ?? 0,
      }))
    : null;
  const leader =
    rows && rows.length > 0
      ? rows.reduce((best, row) => (row.p > best.p ? row : best)).id
      : null;

  // Streamed branches, or those recorded in the final trace (models that do not stream).
  const fallback = result ? branchesFromTrace(result) : null;
  const streamed = Object.keys(state.branches).length > 0;
  const branches = streamed ? state.branches : (fallback?.branches ?? {});
  const contributions =
    Object.keys(state.contributions).length > 0
      ? state.contributions
      : (fallback?.contributions ?? {});
  const readWidth = result
    ? Number.POSITIVE_INFINITY
    : (state.snapshots.at(-1)?.width ?? 0);
  const tree: TreeCandidate[] = ids.map((id, i) => ({
    id,
    text: names.get(id) ?? id,
    p: probabilities?.[i] ?? null,
    direct: direct?.[i] ?? null,
    nodes: branches[id] ?? [],
    contributions: contributions[id] ?? null,
  }));
  const hasTree = tree.some((c) => c.nodes.length > 0);

  return (
    <div className="flex flex-col gap-10">
      <p className="flex items-baseline justify-between gap-4 border-l-4 border-interactive bg-layer px-4 py-3 text-sm">
        <span>{state.error ? "Stopped." : describe(state.stage, names)}</span>
        <span className="font-mono text-xs tabular-nums text-text-helper">
          {result && state.wallTimeMs !== null
            ? `${(state.wallTimeMs / 1000).toFixed(2)} s`
            : `${elapsed.toFixed(1)} s`}
        </span>
      </p>

      {rows && (
        <Probabilities
          rows={rows}
          badge={
            result
              ? calibrated
                ? `Final · calibrated · T = ${result.calibration.temperature.toFixed(3)}`
                : "Final · uncalibrated"
              : latest && latest.width > 0
                ? `Live · ${latest.width} per candidate`
                : "Live · direct answer"
          }
          note={result ? (calibrated ? CALIBRATED_NOTE : RAW_NOTE) : LIVE_NOTE}
        />
      )}

      {hasTree && (
        <ExpansionTree
          question={request.question}
          candidates={tree}
          readWidth={readWidth}
          leader={leader}
        />
      )}

      {state.snapshots.length >= 2 && (
        <Trajectory ids={ids} names={names} snapshots={state.snapshots} />
      )}

      {result && state.wallTimeMs !== null && (
        <details className="group">
          <summary className="cursor-pointer text-text-secondary hover:text-text-primary">
            Computation details (every step, as recorded in the trace)
          </summary>
          <div className="mt-4">
            <Path
              outcome={{
                ok: true,
                request,
                result,
                wallTimeMs: state.wallTimeMs,
              }}
            />
          </div>
        </details>
      )}
    </div>
  );
}
