/**
 * Budgets the interface sends, mirroring the evaluation suites so a model is used at the
 * operating points it was measured at:
 * - expansion (knowledge_v0 r2): the total number of concepts read, at depth 1;
 * - branch (knowledge_v0 r3): the total number of statements in the candidates' branches;
 * - workspace (reasoning_v1 r2): the number of rounds.
 */

export type Mechanism = "direct" | "expansion" | "branch" | "workspace";

export type Budget = {
  max_nodes: number;
  max_queries: number;
  max_proposals: number;
  max_pairs: number;
  max_edges: number;
  max_composites: number;
  max_rounds: number;
  max_depth: number;
};

const ZERO: Budget = {
  max_nodes: 0,
  max_queries: 0,
  max_proposals: 0,
  max_pairs: 0,
  max_edges: 0,
  max_composites: 0,
  max_rounds: 0,
  max_depth: 0,
};

export const EFFORT_LEVELS: Record<Mechanism, number[]> = {
  direct: [0],
  expansion: [0, 8, 16, 32, 64, 128],
  branch: [0, 4, 8, 16, 32],
  workspace: [0, 1, 2, 4, 8, 16],
};

export const DEFAULT_EFFORT: Record<Mechanism, number> = {
  direct: 0,
  expansion: 64,
  branch: 16,
  workspace: 4,
};

export function effortLabel(mechanism: Mechanism, effort: number): string {
  if (effort === 0) return "0 — direct answer, no reasoning";
  if (mechanism === "expansion") return `${effort} concepts in total`;
  if (mechanism === "branch") return `${effort} statements in total`;
  return `${effort} round${effort === 1 ? "" : "s"}`;
}

export function budgetFor(mechanism: Mechanism, effort: number): Budget {
  if (effort <= 0 || mechanism === "direct") return ZERO;
  if (mechanism === "expansion" || mechanism === "branch") {
    return {
      ...ZERO,
      max_depth: 1,
      max_queries: 8,
      max_proposals: effort,
      max_nodes: effort,
    };
  }
  return {
    ...ZERO,
    max_rounds: effort,
    max_nodes: 64,
    max_pairs: effort * 4096,
  };
}

export function sameBudget(a: Partial<Budget>, b: Partial<Budget>): boolean {
  return (Object.keys(ZERO) as (keyof Budget)[]).every(
    (key) => (a[key] ?? 0) === (b[key] ?? 0),
  );
}
