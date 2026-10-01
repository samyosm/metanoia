import "server-only";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { parse as parseToml } from "smol-toml";
import { z } from "zod";
import type { Mechanism } from "./budgets";
import { defaultModel, metanoiaRoot, modelCardsPath } from "./config";

const bundleManifest = z.object({
  predictor_id: z.string(),
  created_at: z.string(),
  training: z.object({ dataset_id: z.string().optional() }).partial(),
  resolved_config: z
    .object({
      model: z
        .object({
          expansion: z.unknown().optional(),
          branch: z.unknown().optional(),
          workspace: z.unknown().optional(),
        })
        .partial(),
    })
    .partial(),
});

const calibrationArtifact = z.object({
  calibrator_id: z.string(),
  compatible_predictor_id: z.string(),
  temperature: z.number(),
  budget: z.record(z.string(), z.number()),
});

export type Calibration = {
  path: string;
  id: string;
  temperature: number;
  budget: Record<string, number>;
};

const modelCard = z.object({
  predictor_id: z.string(),
  // The dev-selected checkpoint to serve (under <bundle>/checkpoints/); else the final weights.
  checkpoint: z
    .string()
    .regex(/^step_\d+$/)
    .optional(),
  status: z.enum(["recommended", "baseline", "not useful"]),
  summary: z.string(),
  headline: z.string(),
  good_at: z.array(z.string()),
  weak_at: z.array(z.string()),
  different: z.string(),
  evidence: z.array(z.string()),
});
export type ModelCard = z.infer<typeof modelCard>;

const cardFile = z.object({ models: z.record(z.string(), modelCard) });

export type Model = {
  /** Directory name under artifacts/predictors; the only handle the browser sends. */
  name: string;
  predictorId: string;
  trainedOn: string | null;
  createdAt: string;
  bundlePath: string;
  /** The served checkpoint when a card pins one (dev-selected); null = final weights. */
  checkpoint: string | null;
  mechanism: Mechanism;
  /** Fitted calibrations for this predictor; each applies only at its own budget. */
  calibrations: Calibration[];
  /** Only present when the card describes exactly this bundle's weights. */
  card: ModelCard | null;
};

const artifacts = path.join(metanoiaRoot, "artifacts");

async function readJson(file: string): Promise<unknown> {
  return JSON.parse(await readFile(file, "utf8"));
}

async function listDir(dir: string): Promise<string[]> {
  try {
    return await readdir(dir);
  } catch {
    return [];
  }
}

async function calibrations(): Promise<Map<string, Calibration[]>> {
  const dir = path.join(artifacts, "calibrations");
  const byPredictor = new Map<string, Calibration[]>();
  for (const file of (await listDir(dir)).sort()) {
    if (!file.endsWith(".json")) continue;
    const parsed = calibrationArtifact.safeParse(
      await readJson(path.join(dir, file)),
    );
    if (!parsed.success) continue;
    const list = byPredictor.get(parsed.data.compatible_predictor_id) ?? [];
    list.push({
      path: path.join(dir, file),
      id: parsed.data.calibrator_id,
      temperature: parsed.data.temperature,
      budget: parsed.data.budget,
    });
    byPredictor.set(parsed.data.compatible_predictor_id, list);
  }
  return byPredictor;
}

/**
 * Cards keyed by bundle name. A missing or invalid file yields no cards rather than
 * breaking the page; a malformed file is reported in the server log.
 */
async function cards(): Promise<Record<string, ModelCard>> {
  let text: string;
  try {
    text = await readFile(modelCardsPath, "utf8");
  } catch {
    return {};
  }
  let data: unknown;
  try {
    data = parseToml(text);
  } catch (error) {
    console.error(`cannot parse ${modelCardsPath}:`, error);
    return {};
  }
  const parsed = cardFile.safeParse(data);
  if (!parsed.success) {
    console.error(`invalid ${modelCardsPath}:`, parsed.error.issues);
    return {};
  }
  return parsed.data.models;
}

/**
 * Finished predictor bundles, each with the calibrations fitted for exactly that
 * predictor. The default model comes first.
 */
export async function listModels(): Promise<Model[]> {
  const dir = path.join(artifacts, "predictors");
  const [fitted, described] = await Promise.all([calibrations(), cards()]);
  const models: Model[] = [];
  for (const name of await listDir(dir)) {
    if (name.endsWith(".incomplete")) continue;
    // A card may pin the dev-selected checkpoint; the bundle's final weights otherwise.
    const checkpoint = described[name]?.checkpoint ?? null;
    const bundlePath = checkpoint
      ? path.join(dir, name, "checkpoints", checkpoint)
      : path.join(dir, name);
    let manifest: z.infer<typeof bundleManifest>;
    try {
      manifest = bundleManifest.parse(
        await readJson(path.join(bundlePath, "bundle.json")),
      );
    } catch {
      continue;
    }
    models.push({
      name,
      predictorId: manifest.predictor_id,
      trainedOn: manifest.training.dataset_id ?? null,
      createdAt: manifest.created_at,
      bundlePath,
      checkpoint,
      mechanism: manifest.resolved_config.model?.branch
        ? "branch"
        : manifest.resolved_config.model?.expansion
          ? "expansion"
          : manifest.resolved_config.model?.workspace
            ? "workspace"
            : "direct",
      calibrations: fitted.get(manifest.predictor_id) ?? [],
      card:
        described[name]?.predictor_id === manifest.predictor_id
          ? (described[name] ?? null)
          : null,
    });
  }
  const rank = (m: Model) => (m.name === defaultModel ? 0 : 1);
  return models.sort(
    (a, b) => rank(a) - rank(b) || b.createdAt.localeCompare(a.createdAt),
  );
}

export async function findModel(name: string): Promise<Model | undefined> {
  return (await listModels()).find((m) => m.name === name);
}
