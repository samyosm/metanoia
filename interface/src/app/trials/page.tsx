const TRIALS: {
  name: string;
  what: string;
  result: string;
  verdict: string;
}[] = [
  {
    name: "Direct model",
    what: "ModernBERT-base fine-tuned on 99k reading-comprehension questions, one pass, probability per option. Frozen encoders were tried first.",
    result:
      "76.8% on RACE + SciQ test, well calibrated. Frozen encoders stayed at chance.",
    verdict: "Works",
  },
  {
    name: "Rule puzzles",
    what: "Synthetic chains of rules, to test multi-step reasoning in a workspace.",
    result:
      "The direct model solved the trained depths but not longer ones, and stayed overconfident. A workspace with rounds went unused; small modules trained from scratch failed.",
    verdict: "Negative, parked",
  },
  {
    name: "Expand into concepts (v1)",
    what: "Each option expands into WordNet concepts, read as pooled vectors.",
    result:
      "Random concepts gave the same answers as retrieved ones. No effect.",
    verdict: "Failed",
  },
  {
    name: "Expand into statements (v2)",
    what: "Each option retrieves statements from a 1M-statement fact base and reads them next to itself.",
    result:
      "Accuracy rose from 53.7% to 61.1% as more statements were read: +2.9 points over a matched model without them. The gain was all on ARC-Easy.",
    verdict: "Works, small",
  },
  {
    name: "Statements focused on the option",
    what: "Prefer statements that mention the option.",
    result:
      "65.2% against 58.1% for the matched direct model (+7.0 points). The best result. No gain on OpenBookQA.",
    verdict: "Works",
  },
  {
    name: "Better selection",
    what: "Reranker fusion, concept paths, follow-up search.",
    result:
      "No further gain. OpenBookQA is limited by what the fact base contains and by questions that need two facts.",
    verdict: "Plateau",
  },
  {
    name: "All options in one sequence",
    what: "Scores read at a marker per option, after Laya. Quick runs on 2,400 questions.",
    result:
      "+2.6 points over per-option encoding (+5.2 averaging four option orders), trained 3× faster. But answers change with option order: 27–67% of questions.",
    verdict: "Promising, not shown at full scale",
  },
];

export default function Trials() {
  return (
    <main className="mx-auto max-w-4xl p-8">
      <h1 className="text-2xl font-light">What we tried</h1>
      <p className="mt-3 text-text-secondary">
        Metanoia is an attempt at a decision model that does not generate text:
        given a context, a question and options, it returns a probability for
        each option. The idea is to let each option expand into related
        knowledge, read it, and gather the evidence back. These are the
        experiments, in order. Numbers are from a single training run unless
        stated; each is recorded in{" "}
        <code className="font-mono">RESULT_LOG.md</code> in the repository.
      </p>
      <ol className="mt-8 flex flex-col gap-px bg-border-subtle">
        {TRIALS.map((trial, index) => (
          <li key={trial.name} className="bg-background py-4">
            <div className="flex items-baseline justify-between gap-4">
              <h2 className="font-semibold">
                {index + 1}. {trial.name}
              </h2>
              <span className="shrink-0 rounded-full bg-layer px-2 py-0.5 text-xs text-text-secondary">
                {trial.verdict}
              </span>
            </div>
            <p className="mt-1 text-text-secondary">{trial.what}</p>
            <p className="mt-1">{trial.result}</p>
          </li>
        ))}
      </ol>
      <h2 className="mt-10 font-semibold">Where it stands</h2>
      <p className="mt-1 text-text-secondary">
        Paused. The gains came on science questions with a small fact base, on
        top of a weak direct model. Whether they survive a stronger direct
        model, or hold on other domains, is untested.
      </p>
    </main>
  );
}
