import { z } from "zod";
import type { Budget } from "./budgets";

/**
 * The public decision contract (metanoia_contracts, contract_version 1), validated at
 * the boundary with the Python worker. Only the fields this interface reads are
 * required; the worker's own validation remains the authority.
 */

export const traceValue = z.union([
  z.string(),
  z.number(),
  z.boolean(),
  z.null(),
  z.array(z.number()),
  z.array(z.string()),
]);

export const traceEvent = z.object({
  seq: z.number().int().nonnegative(),
  op: z.string(),
  inputs: z.array(z.string()),
  outputs: z.array(z.string()),
  details: z.record(z.string(), traceValue),
});
export type TraceEvent = z.infer<typeof traceEvent>;

export const decisionResult = z.object({
  predictor_id: z.string(),
  candidate_ids: z.array(z.string()),
  logits: z.array(z.number()),
  probabilities: z.array(z.number()),
  calibration: z.object({
    status: z.enum(["uncalibrated", "fitted"]),
    calibrator_id: z.string().nullable(),
    temperature: z.number(),
  }),
  termination: z.string(),
  usage: z.record(z.string(), z.number()),
  trace: z.array(traceEvent).nullable(),
});
export type DecisionResult = z.infer<typeof decisionResult>;

/** One memory node of a candidate's branch, as streamed. */
export const nodeView = z.object({
  id: z.string(),
  concept: z.string(),
  text: z.string(),
  similarity: z.number(),
});
export type NodeView = z.infer<typeof nodeView>;

/**
 * A stage of a decision while it is computed (`metanoia serve` with streaming). Its
 * probabilities are an uncalibrated softmax; only the final result is calibrated.
 */
export const decisionUpdate = z.object({
  seq: z.number().int().nonnegative(),
  stage: z.enum(["render", "score", "expand", "contribution"]),
  elapsed_ms: z.number(),
  text: z.string().nullish(),
  width: z.number().int().nullish(),
  candidate_ids: z.array(z.string()),
  logits: z.array(z.number()).nullish(),
  probabilities: z.array(z.number()).nullish(),
  candidate: z.string().nullish(),
  nodes: z.array(nodeView),
  contributions: z.array(z.number()).nullish(),
});
export type DecisionUpdate = z.infer<typeof decisionUpdate>;

export type DecisionRequest = {
  context: string;
  question: string;
  candidates: { id: string; text: string }[];
  budget: Budget;
  trace: "events" | "off";
};

/** What the browser sends to `POST /api/decide`. */
export const decideInput = z.object({
  model: z.string().min(1),
  // Reasoning effort: concepts (expansion) or rounds (workspace); 0 = direct answer.
  effort: z.number().int().min(0).max(128).default(0),
  context: z.string().max(100_000),
  question: z.string().trim().min(1, "Enter a question."),
  candidates: z
    .array(z.string().trim().min(1, "Candidates cannot be empty."))
    .min(2, "Give at least two candidates.")
    .max(26, "At most 26 candidates.")
    .refine(
      (texts) =>
        new Set(texts.map((t) => t.toLowerCase())).size === texts.length,
      "Candidates must be distinct.",
    ),
});
export type DecideInput = z.infer<typeof decideInput>;

/** Lines of the streamed `POST /api/decide` response (newline-delimited JSON). */
export type StreamMessage =
  | { type: "start"; request: DecisionRequest }
  | { type: "update"; update: DecisionUpdate }
  | { type: "result"; result: DecisionResult; wallTimeMs: number }
  | { type: "error"; error: string };

export type DecideResponse =
  | {
      ok: true;
      request: DecisionRequest;
      result: DecisionResult;
      wallTimeMs: number;
    }
  | { ok: false; error: string };
