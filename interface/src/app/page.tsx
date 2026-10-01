import { connection } from "next/server";
import { Workbench } from "@/components/workbench";
import { listModels } from "@/lib/metanoia/models";

export default async function Page() {
  // Read artifacts at request time, so newly trained models appear on reload.
  await connection();
  const models = (await listModels()).map((m) => ({
    name: m.name,
    predictorId: m.predictorId,
    trainedOn: m.trainedOn,
    mechanism: m.mechanism,
    // The direct-answer calibration, shown next to the model name.
    temperature:
      m.calibrations.find((c) => Object.values(c.budget).every((v) => v === 0))
        ?.temperature ?? null,
    card: m.card,
  }));
  return <Workbench models={models} />;
}
