import { budgetFor, sameBudget } from "@/lib/metanoia/budgets";
import {
  type DecisionRequest,
  decideInput,
  type StreamMessage,
} from "@/lib/metanoia/contract";
import { findModel } from "@/lib/metanoia/models";
import { decide } from "@/lib/metanoia/worker";

const NDJSON = { "content-type": "application/x-ndjson; charset=utf-8" };

function line(message: StreamMessage): string {
  return `${JSON.stringify(message)}\n`;
}

/** A one-line stream, for requests refused before any work starts. */
function refuse(error: string, status: number): Response {
  return new Response(line({ type: "error", error }), {
    status,
    headers: NDJSON,
  });
}

/**
 * Decide, streaming every computed stage as newline-delimited JSON: a `start` line with
 * the request, `update` lines as the model works, then `result` (or `error`).
 */
export async function POST(request: Request): Promise<Response> {
  const parsed = decideInput.safeParse(await request.json().catch(() => null));
  if (!parsed.success) {
    return refuse(parsed.error.issues[0]?.message ?? "Invalid input.", 400);
  }
  const input = parsed.data;
  // Only models listed from artifacts/ are accepted; the browser never sends a path.
  const model = await findModel(input.model);
  if (!model) return refuse(`Unknown model ${input.model}.`, 404);
  const budget = budgetFor(model.mechanism, input.effort);
  // A calibration applies only at the budget it was fitted for; otherwise none.
  const calibration =
    model.calibrations.find((c) => sameBudget(c.budget, budget)) ?? null;
  const decisionRequest: DecisionRequest = {
    context: input.context,
    question: input.question,
    budget,
    // Stable positional ids: a, b, c, ...
    candidates: input.candidates.map((text, i) => ({
      id: String.fromCharCode(97 + i),
      text,
    })),
    trace: "events",
  };

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let open = true;
      const send = (message: StreamMessage) => {
        if (!open) return;
        try {
          controller.enqueue(encoder.encode(line(message)));
        } catch {
          open = false; // the browser went away; the model finishes on its own
        }
      };
      send({ type: "start", request: decisionRequest });
      try {
        const { result, wallTimeMs } = await decide(
          model,
          calibration,
          decisionRequest,
          (update) => send({ type: "update", update }),
        );
        send({ type: "result", result, wallTimeMs });
      } catch (error) {
        send({
          type: "error",
          error: error instanceof Error ? error.message : String(error),
        });
      }
      if (open) controller.close();
    },
  });
  return new Response(stream, {
    headers: { ...NDJSON, "cache-control": "no-store" },
  });
}
