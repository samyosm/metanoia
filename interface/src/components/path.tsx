import type { ReactNode } from "react";
import type { DecideResponse, TraceEvent } from "@/lib/metanoia/contract";
import { Heading } from "./fields";
import { percent } from "./probabilities";

type Outcome = Extract<DecideResponse, { ok: true }>;

const TITLES: Record<string, string> = {
  render: "Render the input",
  encode: "Encode",
  direct_score: "Direct answer: score each candidate",
  initial_readout: "Answer before any round",
  workspace: "Open the workspace",
  aggregate: "Bring the evidence back",
  calibrate: "Turn scores into probabilities",
};

function titleFor(event: TraceEvent, names: Map<string, string>): string {
  if (event.op === "expand") {
    const cid = String(event.details.candidate);
    return `Expand “${names.get(cid) ?? cid}”`;
  }
  if (event.op === "aggregate" && typeof event.details.round === "number") {
    return `Round ${event.details.round}`;
  }
  return TITLES[event.op] ?? event.op;
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

/** "cid|source|weight|text" entries of an aggregate event, grouped by candidate. */
function attendedBy(value: unknown): Map<string, [string, number, string][]> {
  const grouped = new Map<string, [string, number, string][]>();
  for (const entry of strings(value)) {
    const [cid = "", source = "", weight = "0", ...text] = entry.split("|");
    const list = grouped.get(cid) ?? [];
    list.push([source, Number(weight), text.join("|")]);
    grouped.set(cid, list);
  }
  return grouped;
}

function Bar({ value, max }: { value: number; max: number }) {
  const width = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  return (
    <span
      className="inline-block h-1.5 w-16 bg-layer align-middle"
      aria-hidden="true"
    >
      <span
        className="block h-full bg-interactive"
        style={{ width: `${width * 100}%` }}
      />
    </span>
  );
}

function Mono({ children }: { children: ReactNode }) {
  return <span className="font-mono">{children}</span>;
}

function ScoreTable({
  ids,
  values,
  names,
  format,
}: {
  ids: string[];
  values: number[];
  names: Map<string, string>;
  format: (v: number) => string;
}) {
  return (
    <table className="mt-2 w-full text-left">
      <tbody>
        {ids.map((id, i) => (
          <tr key={id} className="border-b border-border-subtle">
            <td className="py-1.5 pr-4 font-mono text-text-helper">{id}</td>
            <td className="py-1.5 pr-4">{names.get(id) ?? id}</td>
            <td className="py-1.5 text-right font-mono tabular-nums">
              {format(values[i] ?? 0)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function numbers(value: unknown): number[] {
  return Array.isArray(value) ? value.filter((v) => typeof v === "number") : [];
}

function Step({
  event,
  names,
}: {
  event: TraceEvent;
  names: Map<string, string>;
}) {
  const d = event.details;
  const ids = event.outputs.filter((o) => names.has(o));
  switch (event.op) {
    case "render":
      return (
        <>
          <p className="text-text-secondary">
            The question, context and observations are joined into one text (
            <Mono>{String(d.rendering)}</Mono>). The question comes first, so a
            long context is cut at its end.
          </p>
          <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap bg-layer p-3 font-mono text-xs">
            {String(d.text)}
          </pre>
        </>
      );
    case "encode":
      return (
        <>
          <p className="text-text-secondary">
            <Mono>{String(d.encoder)}</Mono> reads{" "}
            <Mono>
              {String(d.input_tokens)} / {String(d.max_length)}
            </Mono>{" "}
            tokens.{" "}
            {d.candidate_encoding === "joint"
              ? "Each candidate is read together with the input, so its representation depends on the context."
              : "Each candidate is read on its own."}
          </p>
          {d.truncated === true && (
            <p className="mt-2 border-l-4 border-support-error bg-layer p-3">
              The input is longer than {String(d.max_length)} tokens: the end of
              the context was not seen.
            </p>
          )}
        </>
      );
    case "direct_score":
    case "initial_readout":
      return (
        <>
          <p className="text-text-secondary">
            {event.op === "direct_score"
              ? "A shared scorer compares every candidate with the encoded context and with the other candidates, giving one score (logit) each: the answer without any reasoning."
              : "The workspace's reading of the candidates before any round: what the model concludes without deliberating."}
          </p>
          <ScoreTable
            ids={ids}
            values={numbers(d.logits)}
            names={names}
            format={(v) => (v >= 0 ? "+" : "") + v.toFixed(3)}
          />
        </>
      );
    case "calibrate":
      return (
        <>
          <p className="text-text-secondary">
            {d.status === "fitted" ? (
              <>
                Softmax of the scores divided by the fitted temperature{" "}
                <Mono>T = {Number(d.temperature).toFixed(3)}</Mono> (
                <Mono>{String(d.calibrator_id)}</Mono>).
              </>
            ) : (
              "Plain softmax of the scores (no fitted calibration)."
            )}
          </p>
          <ScoreTable
            ids={ids}
            values={numbers(d.probabilities)}
            names={names}
            format={percent}
          />
        </>
      );
    case "expand": {
      const concepts = strings(d.concepts);
      const relevance = numbers(d.relevance);
      const top = Math.max(...relevance.map((r) => Math.abs(r)), 1e-9);
      const branch = d.mode === "branch";
      return (
        <>
          <p className="text-text-secondary">
            {branch
              ? "Statements about related concepts, retrieved for this candidate and read next to it. Each number is how much the statement moves this candidate's score: its score minus its score without that statement."
              : "Concepts retrieved from the concept memory for this candidate, read in the context of the question. Relevance weights how much each can contribute; it is not a probability that the concept is true."}
          </p>
          <ul className="mt-2 flex flex-col gap-1">
            {concepts.map((text, i) => {
              // Statements are shown whole; concepts as "name — gloss".
              const [name, ...gloss] = branch ? [text] : text.split(": ");
              const r = relevance[i] ?? 0;
              return (
                <li key={`${text}-${r}`} className="flex items-baseline gap-3">
                  <span className="w-14 shrink-0 text-right font-mono text-xs tabular-nums text-text-helper">
                    {(r >= 0 ? "+" : "") + r.toFixed(2)}
                  </span>
                  <Bar value={Math.abs(r)} max={top} />
                  <span>
                    <span className="text-text-primary">{name}</span>
                    {gloss.length > 0 && (
                      <span className="text-text-helper">
                        {" "}
                        — {gloss.join(": ")}
                      </span>
                    )}
                  </span>
                </li>
              );
            })}
          </ul>
        </>
      );
    }
    case "aggregate": {
      const attended = attendedBy(d.attended);
      if (typeof d.round === "number") {
        return (
          <>
            <p className="text-text-secondary">
              One round of attention between the sentences and the candidates.
            </p>
            <ScoreTable
              ids={ids}
              values={numbers(d.logits)}
              names={names}
              format={(v) => (v >= 0 ? "+" : "") + v.toFixed(3)}
            />
          </>
        );
      }
      const evidence = numbers(d.evidence);
      const candidateIds = event.outputs;
      const branch = d.mode === "branch";
      return (
        <>
          <p className="text-text-secondary">
            {branch ? (
              <>
                The candidates are compared <em>with</em> their branches, so
                what one branch says also moves the others. The evidence is each
                candidate's score change from its direct score.
              </>
            ) : (
              <>
                Each candidate reads the concepts of <em>every</em> branch, so a
                concept found under one candidate can support or weaken another.
                The evidence is added to its direct score.
              </>
            )}
          </p>
          <ScoreTable
            ids={candidateIds}
            values={evidence}
            names={names}
            format={(v) => (v >= 0 ? "+" : "") + v.toFixed(3)}
          />
          <ul className="mt-3 flex flex-col gap-2">
            {candidateIds.map((cid) => (
              <li key={cid}>
                <span className="text-text-primary">
                  {names.get(cid) ?? cid}
                </span>
                <span className="text-text-helper">
                  {branch ? " moved most by:" : " drew most on:"}
                </span>
                <ul className="mt-1 flex flex-col gap-0.5 pl-4">
                  {(attended.get(cid) ?? []).map(([source, weight, text]) => (
                    <li key={`${source}-${text}`} className="text-xs">
                      <span className="font-mono tabular-nums text-text-helper">
                        {branch
                          ? (weight >= 0 ? "+" : "") + weight.toFixed(2)
                          : `${(weight * 100).toFixed(1)}%`}
                      </span>{" "}
                      {branch ? text : text.split(": ")[0]}
                      {source !== cid && (
                        <span className="text-text-helper">
                          {" "}
                          (from “{names.get(source) ?? source}”)
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        </>
      );
    }
    case "workspace":
      return (
        <p className="text-text-secondary">
          Mode <Mono>{String(d.mode)}</Mono>, {String(d.rounds)} of{" "}
          {String(d.rounds_requested)} rounds ({String(d.termination)}).
        </p>
      );
    default:
      // Operations this view does not know yet (e.g. later reasoning stages).
      return (
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 font-mono text-xs">
          {Object.entries(d).map(([key, value]) => (
            <div key={key} className="contents">
              <dt className="text-text-helper">{key}</dt>
              <dd className="break-all">{JSON.stringify(value)}</dd>
            </div>
          ))}
        </dl>
      );
  }
}

export function Path({ outcome }: { outcome: Outcome }) {
  const { request, result, wallTimeMs } = outcome;
  const names = new Map(request.candidates.map((c) => [c.id, c.text]));
  const events = result.trace ?? [];
  return (
    <div>
      <Heading>Path</Heading>
      {events.length === 0 ? (
        <p className="text-text-helper">This predictor returned no trace.</p>
      ) : (
        <ol>
          {events.map((event, index) => (
            <li
              key={event.seq}
              className="relative border-l border-border-strong pb-6 pl-6 last:border-transparent last:pb-0"
            >
              <span className="absolute -left-2 top-0 flex h-4 w-4 items-center justify-center bg-interactive font-mono text-[10px] text-text-on-color">
                {index + 1}
              </span>
              <h3 className="-mt-0.5 mb-1 text-text-primary">
                {titleFor(event, names)}
              </h3>
              <Step event={event} names={names} />
            </li>
          ))}
        </ol>
      )}
      <p className="mt-6 font-mono text-xs text-text-helper">
        {result.predictor_id} · {result.termination} · {wallTimeMs.toFixed(0)}{" "}
        ms
      </p>
    </div>
  );
}
