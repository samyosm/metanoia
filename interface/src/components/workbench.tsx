"use client";

import { type FormEvent, useId, useReducer, useState } from "react";
import {
  DEFAULT_EFFORT,
  EFFORT_LEVELS,
  effortLabel,
  type Mechanism,
} from "@/lib/metanoia/budgets";
import { EMPTY, readMessages, reduce } from "@/lib/metanoia/live";
import type { ModelCard as Card } from "@/lib/metanoia/models";
import {
  GhostButton,
  Label,
  PrimaryButton,
  Select,
  TextArea,
  TextInput,
} from "./fields";
import { LiveView } from "./live-view";
import { ModelCard } from "./model-card";

export type ModelOption = {
  name: string;
  predictorId: string;
  trainedOn: string | null;
  temperature: number | null;
  mechanism: Mechanism;
  card: Card | null;
};

const EFFORT_HELP: Record<Mechanism, string> = {
  direct: "",
  expansion:
    "Each candidate expands into related concepts from the concept memory; their evidence flows back into every candidate's score. This is the total number of concepts read.",
  branch:
    "Each candidate retrieves statements about related concepts and reads them next to itself; the candidates are then compared with their branches. This is the total number of statements read.",
  workspace:
    "Rounds of attention between the input's sentences and the candidates.",
};

// The analysis searches temperatures up to e^5 ≈ 148.4; a fit near that bound means
// the best calibration is a uniform distribution: the model has no usable signal.
const NEAR_UNIFORM_TEMPERATURE = 100;

function calibrationNote(temperature: number | null): string {
  if (temperature === null)
    return "No fitted calibration: probabilities are an uncalibrated softmax";
  if (temperature >= NEAR_UNIFORM_TEMPERATURE)
    return `Calibrated to near-uniform (T = ${temperature.toFixed(1)}): the model has no usable signal`;
  return `Calibrated (T = ${temperature.toFixed(3)})`;
}

export function Workbench({ models }: { models: ModelOption[] }) {
  const id = useId();
  const [model, setModel] = useState(models[0]?.name ?? "");
  const [effort, setEffort] = useState(
    DEFAULT_EFFORT[models[0]?.mechanism ?? "direct"],
  );
  const [context, setContext] = useState("");
  const [question, setQuestion] = useState("");
  const [candidates, setCandidates] = useState(["", ""]);
  const [running, setRunning] = useState(false);
  const [live, dispatch] = useReducer(reduce, EMPTY);

  const setCandidate = (index: number, text: string) =>
    setCandidates((all) => all.map((c, i) => (i === index ? text : c)));

  async function run(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setRunning(true);
    try {
      const response = await fetch("/api/decide", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, effort, context, question, candidates }),
      });
      // Every stage is shown as it arrives: see LiveView.
      for await (const message of readMessages(response)) dispatch(message);
    } catch (reason) {
      dispatch({
        type: "error",
        error: reason instanceof Error ? reason.message : String(reason),
      });
    } finally {
      setRunning(false);
    }
  }

  if (models.length === 0) {
    return (
      <main className="p-8 text-text-secondary">
        No trained predictor bundles were found under{" "}
        <code className="font-mono">artifacts/predictors</code>.
      </main>
    );
  }

  const selected = models.find((m) => m.name === model);

  return (
    <main className="grid lg:grid-cols-2">
      <form
        onSubmit={run}
        className="flex flex-col gap-6 border-border-subtle p-8 lg:border-r"
      >
        <div>
          <Label htmlFor={`${id}-model`}>Model</Label>
          <Select
            id={`${id}-model`}
            value={model}
            onChange={(e) => {
              setModel(e.target.value);
              const next = models.find((m) => m.name === e.target.value);
              setEffort(DEFAULT_EFFORT[next?.mechanism ?? "direct"]);
            }}
          >
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
                {m.trainedOn ? ` — ${m.trainedOn}` : ""}
              </option>
            ))}
          </Select>
          <p className="mt-1 text-xs text-text-helper">
            {calibrationNote(selected?.temperature ?? null)}
          </p>
          <ModelCard card={selected?.card ?? null} />
        </div>

        {selected && selected.mechanism !== "direct" && (
          <div>
            <Label htmlFor={`${id}-effort`}>
              {selected.mechanism === "expansion"
                ? "Concepts to explore"
                : selected.mechanism === "branch"
                  ? "Statements to read"
                  : "Reasoning rounds"}
            </Label>
            <Select
              id={`${id}-effort`}
              value={effort}
              onChange={(e) => setEffort(Number(e.target.value))}
            >
              {EFFORT_LEVELS[selected.mechanism].map((level) => (
                <option key={level} value={level}>
                  {effortLabel(selected.mechanism, level)}
                </option>
              ))}
            </Select>
            <p className="mt-1 text-xs text-text-helper">
              {EFFORT_HELP[selected.mechanism]}
            </p>
          </div>
        )}

        <div>
          <Label htmlFor={`${id}-context`}>Context</Label>
          <TextArea
            id={`${id}-context`}
            value={context}
            onChange={(e) => setContext(e.target.value)}
            placeholder="Passage the question is about (optional)"
          />
        </div>

        <div>
          <Label htmlFor={`${id}-question`}>Question</Label>
          <TextInput
            id={`${id}-question`}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            required
          />
        </div>

        <fieldset>
          <legend className="mb-2 text-xs tracking-[0.32px] text-text-secondary">
            Candidates
          </legend>
          <ol className="flex flex-col gap-px">
            {candidates.map((text, index) => (
              // Rows are positional: their identity is their index.
              // biome-ignore lint/suspicious/noArrayIndexKey: positional list
              <li key={index} className="flex">
                <span className="flex h-10 w-10 shrink-0 items-center justify-center border-b border-border-strong bg-field font-mono text-text-helper">
                  {String.fromCharCode(97 + index)}
                </span>
                <TextInput
                  aria-label={`Candidate ${index + 1}`}
                  value={text}
                  onChange={(e) => setCandidate(index, e.target.value)}
                  required
                />
                <button
                  type="button"
                  aria-label={`Remove candidate ${index + 1}`}
                  disabled={candidates.length <= 2}
                  onClick={() =>
                    setCandidates((all) => all.filter((_, i) => i !== index))
                  }
                  className="h-10 w-10 shrink-0 border-b border-border-strong bg-field text-text-secondary hover:bg-border-subtle disabled:opacity-30"
                >
                  ×
                </button>
              </li>
            ))}
          </ol>
          <GhostButton
            onClick={() => setCandidates((all) => [...all, ""])}
            disabled={candidates.length >= 26}
          >
            Add candidate +
          </GhostButton>
        </fieldset>

        <div className="flex items-center gap-4">
          <PrimaryButton type="submit" disabled={running}>
            {running ? "Deciding…" : "Decide"}
          </PrimaryButton>
          {running && !live.request && (
            <span className="text-xs text-text-helper">
              The first request loads the model and can take a while.
            </span>
          )}
        </div>

        {live.error && (
          <p
            role="alert"
            className="whitespace-pre-wrap border-l-4 border-support-error bg-layer p-4 text-text-primary"
          >
            {live.error}
          </p>
        )}
      </form>

      <section
        className="flex flex-col gap-10 border-t border-border-subtle p-8 lg:border-t-0"
        aria-live="polite"
      >
        <LiveView state={live} running={running} />
      </section>
    </main>
  );
}
