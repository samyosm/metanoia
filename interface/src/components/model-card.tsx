import type { ModelCard as Card } from "@/lib/metanoia/models";

const STATUS: Record<Card["status"], string> = {
  recommended:
    "bg-[#defbe6] text-[#0e6027] dark:bg-[#044317] dark:text-[#a7f0ba]",
  baseline: "bg-[#e0e0e0] text-[#161616] dark:bg-[#393939] dark:text-[#f4f4f4]",
  "not useful":
    "bg-[#fff1f1] text-[#a2191f] dark:bg-[#750e13] dark:text-[#ffd7d9]",
};

function List({ title, items }: { title: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <div>
      <h4 className="mb-1 text-xs tracking-[0.32px] text-text-secondary">
        {title}
      </h4>
      <ul className="list-disc space-y-1 pl-4">
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

/** What a model can do, from model_cards.toml (grounded in RESULT_LOG.md). */
export function ModelCard({ card }: { card: Card | null }) {
  if (!card) {
    return (
      <p className="mt-3 text-xs text-text-helper">
        No model card yet: this model has not been evaluated and described in
        model_cards.toml.
      </p>
    );
  }
  return (
    <div className="mt-3 bg-layer p-4">
      <div className="flex items-start justify-between gap-4">
        <p className="text-text-primary">{card.summary}</p>
        <span
          className={`shrink-0 rounded-full px-2 py-0.5 text-xs ${STATUS[card.status]}`}
        >
          {card.status}
        </span>
      </div>
      <p className="mt-1 text-xs text-text-secondary">{card.headline}</p>
      <details className="group mt-3">
        <summary className="cursor-pointer list-none text-interactive dark:text-[#78a9ff]">
          <svg
            aria-hidden="true"
            viewBox="0 0 16 16"
            className="mr-1 inline h-4 w-4 align-[-3px] transition-transform group-open:rotate-90"
          >
            <path d="M6 3.5 10.5 8 6 12.5" fill="none" stroke="currentColor" />
          </svg>
          Capabilities
        </summary>
        <div className="mt-3 space-y-3">
          <List title="Good at" items={card.good_at} />
          <List title="Weak at" items={card.weak_at} />
          <div>
            <h4 className="mb-1 text-xs tracking-[0.32px] text-text-secondary">
              What makes it different
            </h4>
            <p>{card.different}</p>
          </div>
          <p className="font-mono text-xs text-text-helper">
            Evidence: {card.evidence.join(" · ")} in RESULT_LOG.md
            {card.checkpoint
              ? ` · serves the dev-selected checkpoint ${card.checkpoint}`
              : ""}
          </p>
        </div>
      </details>
    </div>
  );
}
