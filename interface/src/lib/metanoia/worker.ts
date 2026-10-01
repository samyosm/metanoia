import "server-only";
import { type ChildProcess, spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { z } from "zod";
import {
  device,
  metanoiaBin,
  metanoiaRoot,
  precision,
  requestTimeoutMs,
} from "./config";
import {
  type DecisionRequest,
  type DecisionResult,
  type DecisionUpdate,
  decisionResult,
  decisionUpdate,
} from "./contract";
import type { Calibration, Model } from "./models";

/**
 * One long-lived `metanoia serve` process per loaded model. It speaks JSON lines:
 * `{"id", "request", "stream"}` in, `{"id", "result" | "error"}` out, after a
 * `{"ready"}` line. A streamed request is answered by `{"id", "update"}` lines first,
 * one per computed stage. Loading a model takes seconds, so the process stays warm.
 */

const reply = z.union([
  z.object({ ready: z.literal(true) }),
  z.object({ id: z.number(), update: decisionUpdate }),
  z.object({
    id: z.number(),
    result: decisionResult,
    wall_time_ms: z.number(),
  }),
  z.object({ id: z.number().nullable(), error: z.string() }),
]);

type Pending = {
  onUpdate?: (update: DecisionUpdate) => void;
  resolve: (value: { result: DecisionResult; wallTimeMs: number }) => void;
  reject: (error: Error) => void;
  timer: NodeJS.Timeout;
};

const STDERR_LINES = 20;

class ModelWorker {
  readonly ready: Promise<void>;
  private readonly child: ChildProcess;
  private readonly pending = new Map<number, Pending>();
  private readonly stderr: string[] = [];
  private nextId = 1;
  private exitError: Error | null = null;

  constructor(
    readonly model: Model,
    readonly calibration: Calibration | null,
  ) {
    const args = [
      "serve",
      "--bundle",
      model.bundlePath,
      "--device",
      device,
      "--precision",
      precision,
    ];
    if (calibration) args.push("--calibration", calibration.path);
    this.child = spawn(metanoiaBin, args, {
      cwd: metanoiaRoot,
      stdio: ["pipe", "pipe", "pipe"],
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
        // On CPU, hide the GPU entirely so no CUDA context is created.
        ...(device === "cpu" ? { CUDA_VISIBLE_DEVICES: "" } : {}),
      },
    });

    let markReady: () => void = () => {};
    let failReady: (error: Error) => void = () => {};
    this.ready = new Promise((resolve, reject) => {
      markReady = resolve;
      failReady = reject;
    });
    // Avoid an unhandled rejection when nobody awaits `ready` yet.
    this.ready.catch(() => {});

    if (!this.child.stdout || !this.child.stderr) {
      throw new Error("worker streams are unavailable");
    }
    createInterface({ input: this.child.stdout }).on("line", (line) => {
      const parsed = reply.safeParse(safeJson(line));
      if (!parsed.success) {
        this.remember(`unexpected worker output: ${line.slice(0, 200)}`);
        return;
      }
      const message = parsed.data;
      if ("ready" in message) {
        markReady();
        return;
      }
      if (message.id === null) return;
      const waiter = this.pending.get(message.id);
      if (!waiter) return;
      if ("update" in message) {
        waiter.onUpdate?.(message.update);
        return;
      }
      this.pending.delete(message.id);
      clearTimeout(waiter.timer);
      if ("error" in message) waiter.reject(new Error(message.error));
      else
        waiter.resolve({
          result: message.result,
          wallTimeMs: message.wall_time_ms,
        });
    });
    createInterface({ input: this.child.stderr }).on("line", (line) =>
      this.remember(line),
    );
    this.child.on("error", (error) => this.fail(error, failReady));
    this.child.on("exit", (code, signal) =>
      this.fail(
        new Error(
          `model worker exited (${signal ?? `code ${code}`})\n${this.stderr.join("\n")}`,
        ),
        failReady,
      ),
    );
  }

  get alive(): boolean {
    return this.exitError === null;
  }

  async decide(
    request: DecisionRequest,
    onUpdate?: (update: DecisionUpdate) => void,
  ): Promise<{ result: DecisionResult; wallTimeMs: number }> {
    await this.ready;
    if (this.exitError) throw this.exitError;
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`no answer within ${requestTimeoutMs} ms`));
      }, requestTimeoutMs);
      this.pending.set(id, { onUpdate, resolve, reject, timer });
      const stream = onUpdate !== undefined;
      this.child.stdin?.write(`${JSON.stringify({ id, request, stream })}\n`);
    });
  }

  dispose(): void {
    if (this.alive) this.child.kill();
  }

  private remember(line: string): void {
    this.stderr.push(line);
    if (this.stderr.length > STDERR_LINES) this.stderr.shift();
  }

  private fail(error: Error, failReady: (error: Error) => void): void {
    if (this.exitError) return;
    this.exitError = error;
    failReady(error);
    for (const waiter of this.pending.values()) {
      clearTimeout(waiter.timer);
      waiter.reject(error);
    }
    this.pending.clear();
  }
}

function safeJson(line: string): unknown {
  try {
    return JSON.parse(line);
  } catch {
    return null;
  }
}

// Survives dev-server module reloads, so a reload never leaks a second model process.
const registry = globalThis as typeof globalThis & {
  metanoiaWorker?: ModelWorker;
  metanoiaExitHook?: boolean;
};

function sameWorker(
  worker: ModelWorker,
  model: Model,
  calibration: Calibration | null,
): boolean {
  return (
    worker.model.bundlePath === model.bundlePath &&
    worker.model.predictorId === model.predictorId &&
    worker.calibration?.path === calibration?.path
  );
}

/**
 * Answer one request with `model` and an optional calibration fitted for the request's
 * budget, starting or replacing the worker as needed. With `onUpdate`, each computed
 * stage is reported as it happens.
 */
export function decide(
  model: Model,
  calibration: Calibration | null,
  request: DecisionRequest,
  onUpdate?: (update: DecisionUpdate) => void,
): Promise<{ result: DecisionResult; wallTimeMs: number }> {
  let worker = registry.metanoiaWorker;
  if (!worker?.alive || !sameWorker(worker, model, calibration)) {
    worker?.dispose();
    worker = new ModelWorker(model, calibration);
    registry.metanoiaWorker = worker;
  }
  if (!registry.metanoiaExitHook) {
    registry.metanoiaExitHook = true;
    process.once("exit", () => registry.metanoiaWorker?.dispose());
  }
  return worker.decide(request, onUpdate);
}
