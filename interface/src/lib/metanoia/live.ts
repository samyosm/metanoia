import type {
  DecisionRequest,
  DecisionResult,
  NodeView,
  StreamMessage,
} from "./contract";

/** The decision as it has been computed so far, built from streamed messages. */
export type LiveState = {
  request: DecisionRequest | null;
  rendered: string | null;
  /** Decisions after reading `width` nodes per candidate (0 = the direct answer). */
  snapshots: { width: number; probabilities: number[]; elapsedMs: number }[];
  /** Each candidate's proposed nodes, in reading order. */
  branches: Record<string, NodeView[]>;
  /** Each node's leave-one-out effect on its own candidate's score. */
  contributions: Record<string, number[]>;
  /** The most recent stage, for the status line. */
  stage: Stage | null;
  result: DecisionResult | null;
  wallTimeMs: number | null;
  error: string | null;
};

export type Stage =
  | { kind: "render" }
  | { kind: "score"; width: number }
  | { kind: "expand"; candidate: string }
  | { kind: "contribution"; candidate: string }
  | { kind: "done" };

export const EMPTY: LiveState = {
  request: null,
  rendered: null,
  snapshots: [],
  branches: {},
  contributions: {},
  stage: null,
  result: null,
  wallTimeMs: null,
  error: null,
};

export function reduce(state: LiveState, message: StreamMessage): LiveState {
  switch (message.type) {
    case "start":
      return { ...EMPTY, request: message.request };
    case "error":
      return { ...state, error: message.error };
    case "result":
      return {
        ...state,
        result: message.result,
        wallTimeMs: message.wallTimeMs,
        stage: { kind: "done" },
      };
    case "update": {
      const u = message.update;
      switch (u.stage) {
        case "render":
          return {
            ...state,
            rendered: u.text ?? "",
            stage: { kind: "render" },
          };
        case "score":
          return {
            ...state,
            snapshots: [
              ...state.snapshots,
              {
                width: u.width ?? 0,
                probabilities: u.probabilities ?? [],
                elapsedMs: u.elapsed_ms,
              },
            ],
            stage: { kind: "score", width: u.width ?? 0 },
          };
        case "expand": {
          const cid = u.candidate ?? "";
          return {
            ...state,
            branches: { ...state.branches, [cid]: u.nodes },
            stage: { kind: "expand", candidate: cid },
          };
        }
        case "contribution": {
          const cid = u.candidate ?? "";
          return {
            ...state,
            contributions: {
              ...state.contributions,
              [cid]: u.contributions ?? [],
            },
            stage: { kind: "contribution", candidate: cid },
          };
        }
      }
    }
  }
}

/** Streamed messages of a newline-delimited JSON response, as they arrive. */
export async function* readMessages(
  response: Response,
): AsyncGenerator<StreamMessage> {
  if (!response.body) throw new Error("the response has no body");
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let newline = buffer.indexOf("\n");
    while (newline >= 0) {
      const text = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (text) yield JSON.parse(text) as StreamMessage;
      newline = buffer.indexOf("\n");
    }
  }
  if (buffer.trim()) yield JSON.parse(buffer) as StreamMessage;
}

/**
 * Branches and contributions from a finished result's trace, for models that do not
 * stream their expansion (EXPAND v1) or when only the result is available.
 */
export function branchesFromTrace(result: DecisionResult): {
  branches: Record<string, NodeView[]>;
  contributions: Record<string, number[]>;
} {
  const branches: Record<string, NodeView[]> = {};
  const contributions: Record<string, number[]> = {};
  for (const event of result.trace ?? []) {
    if (event.op !== "expand") continue;
    const cid = String(event.details.candidate);
    const texts = event.details.concepts;
    const relevance = event.details.relevance;
    if (!Array.isArray(texts)) continue;
    branches[cid] = texts.map((text, i) => ({
      id: event.outputs[i] ?? `${cid}-${i}`,
      concept: "",
      text: String(text),
      similarity: 0,
    }));
    contributions[cid] = Array.isArray(relevance)
      ? relevance.map((r) => Number(r))
      : [];
  }
  return { branches, contributions };
}
