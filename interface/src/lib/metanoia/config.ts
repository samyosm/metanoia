import "server-only";
import path from "node:path";

/**
 * Where the Python project lives and how its worker runs. Override with env vars.
 * The project is the parent of this app, so it is excluded from Turbopack's file tracing.
 */
export const metanoiaRoot = path.resolve(
  process.env.METANOIA_ROOT ??
    path.join(/* turbopackIgnore: true */ process.cwd(), ".."),
);

/** Model cards (capabilities, grounded in RESULT_LOG.md), at the project root. */
export const modelCardsPath =
  process.env.METANOIA_MODEL_CARDS ??
  path.join(metanoiaRoot, "model_cards.toml");

/** The `metanoia` command: the project's virtual environment unless overridden. */
export const metanoiaBin =
  process.env.METANOIA_BIN ??
  path.join(metanoiaRoot, ".venv", "bin", "metanoia");

/** `cpu` by default so the interface never competes with a training run for the GPU. */
export const device = process.env.METANOIA_DEVICE ?? "cpu";

export const precision = process.env.METANOIA_PRECISION ?? "fp32";

/** Preferred model (a directory name under artifacts/predictors). */
export const defaultModel =
  process.env.METANOIA_DEFAULT_MODEL ?? "mbft_joint_mcqa_v0";

export const requestTimeoutMs = Number(
  process.env.METANOIA_TIMEOUT_MS ?? 120_000,
);
