"""Plan verification (CPU only): does an off-the-shelf reranker, reading (question +
candidate, statement) together, rank the gold fact above retrieved statements better
than the retrieval similarity does?

For N OBQA dev questions: pool = the top 31 generics_v0 statements for
"question gold-candidate" plus the gold core fact (32). Rank of the gold fact by
(a) bge-small similarity to the query, (b) bge-reranker-base on (question + candidate,
statement). Also: for the same pool read against each *distractor* query, is the gold
fact's reranker score lower (i.e. the reranker is candidate-sensitive)?
Run from metanoia/: .venv/bin/python <this> <out_dir> [n]
"""

import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from metanoia.memory.encoder import RetrievalEncoder
from metanoia.memory.store import ConceptMemory

OUT = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {
    "base": ("BAAI/bge-reranker-base", "2cfc18c9415c912f9d8155881c133215df768a70"),
    "m3": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
}
RERANKER, REVISION = MODELS[sys.argv[3] if len(sys.argv) > 3 else "base"]
# "concat": "question candidate"; "qa": "Question: … Answer: …" (reranker query only).
FORMAT = sys.argv[4] if len(sys.argv) > 4 else "concat"


def rquery(question: str, candidate: str) -> str:
    if FORMAT == "qa":
        return f"Question: {question} Answer: {candidate}"
    return f"{question} {candidate}"

torch.set_num_threads(6)

D = Path("artifacts/datasets/knowledge_v0")
labels = {r["example_id"]: r for r in pq.read_table(D / "labels.parquet").to_pylist()}
facts = {json.loads(l)["example_id"]: json.loads(l)["core_fact"] for l in (D / "proofs.jsonl").open()}
cases = [json.loads(l) for l in (D / "cases.jsonl").open()]
cases = [c for c in cases if c["split"] == "dev" and c["example_id"] in facts][:N]

memory = ConceptMemory(Path("artifacts/memories/generics_v0"), device="cpu")
spec = memory.manifest.encoder
retriever = RetrievalEncoder(spec["model"], spec["requested_revision"], spec["pooling"], spec["max_length"], spec["query_prefix"])
tok = AutoTokenizer.from_pretrained(RERANKER, revision=REVISION)
rer = AutoModelForSequenceClassification.from_pretrained(RERANKER, revision=REVISION).eval()


def rerank(query: str, statements: list[str]) -> list[float]:
    with torch.inference_mode():
        batch = tok([query] * len(statements), statements, padding=True, truncation=True, max_length=192, return_tensors="pt")
        return rer(**batch).logits.view(-1).float().tolist()


def rank_of(scores: list[float], index: int) -> int:
    return 1 + sum(s > scores[index] for s in scores)


rows, t0, pairs = [], time.time(), 0
for c in cases:
    r = c["request"]
    gold_id = labels[c["example_id"]]["label_id"]
    gold = next(x["text"] for x in r["candidates"] if x["id"] == gold_id)
    fact = facts[c["example_id"]]
    query = f"{r['question']} {gold}"
    with torch.inference_mode():
        qv = retriever.embed([query], query=True)
        top = memory.search(qv, 31)[0][0].tolist()
        pool = [memory.concepts[i].text for i in top] + [fact]
        sims = (retriever.embed(pool) @ qv[0]).tolist()
    rr = rerank(rquery(r["question"], gold), pool)
    pairs += len(pool)
    # Candidate sensitivity: the gold fact's reranker score under each distractor query.
    distractor_scores = [rerank(rquery(r["question"], x["text"]), [fact])[0] for x in r["candidates"] if x["id"] != gold_id]
    pairs += len(distractor_scores)
    rows.append({
        "rank_similarity": rank_of(sims, len(pool) - 1),
        "rank_reranker": rank_of(rr, len(pool) - 1),
        "fact_score_gold_query": rr[-1],
        "fact_score_best_distractor_query": max(distractor_scores),
    })
seconds = time.time() - t0
n = len(rows)
summary = {
    "questions": n,
    "pool": 32,
    "mean_rank_similarity": sum(x["rank_similarity"] for x in rows) / n,
    "mean_rank_reranker": sum(x["rank_reranker"] for x in rows) / n,
    "top2_similarity": sum(x["rank_similarity"] <= 2 for x in rows) / n,
    "top2_reranker": sum(x["rank_reranker"] <= 2 for x in rows) / n,
    "top8_similarity": sum(x["rank_similarity"] <= 8 for x in rows) / n,
    "top8_reranker": sum(x["rank_reranker"] <= 8 for x in rows) / n,
    "fact_scores_higher_for_gold_than_every_distractor": sum(x["fact_score_gold_query"] > x["fact_score_best_distractor_query"] for x in rows) / n,
    "cpu_ms_per_pair": 1000 * seconds / pairs,
}
print(json.dumps(summary, indent=1))
(OUT / f"reranker_feasibility_{sys.argv[3] if len(sys.argv) > 3 else 'base'}_{FORMAT}.json").write_text(json.dumps({"reranker": RERANKER, "revision": REVISION, "summary": summary, "rows": rows}, indent=1))
