"""R027: are OBQA's gold core facts in generics_v0, and do focused branches contain them?

Run inline from metanoia/ on 2026-09-30 (output in output.txt). For each OBQA dev question
(500) with a gold fact: the memory statement closest to the fact (bge-small, passage
embeddings, exact search), and the closest statement among the focused branches (8 per
candidate, G1/J1 from R025's branches.json.gz).
"""

import gzip
import json
import statistics
from pathlib import Path

import torch

from metanoia.memory.encoder import RetrievalEncoder
from metanoia.memory.store import ConceptMemory

mem = ConceptMemory(Path("artifacts/memories/generics_v0"), device="cuda")
spec = mem.manifest.encoder
enc = RetrievalEncoder(
    spec["model"], spec["requested_revision"], spec["pooling"], spec["max_length"], spec["query_prefix"]
).to("cuda")
facts = {
    json.loads(l)["example_id"]: json.loads(l)["core_fact"]
    for l in open("artifacts/datasets/knowledge_v0/proofs.jsonl")
}
screen = json.loads(
    gzip.open("../result_logs/R025_2026-09-29_proposal-screening-round1/branches.json.gz").read()
)
ids = screen["example_ids"]
focused = screen["configs"]["G1/J1"]
rows = [(j, e) for j, e in enumerate(ids) if e in facts]
texts = [facts[e] for _, e in rows]
with torch.inference_mode():
    fv = enc.embed(texts, query=False).to(mem.keys.dtype)
    best_idx, best_sim = mem.search(fv, 1)
best_sim = best_sim[:, 0].tolist()
best_idx = best_idx[:, 0].tolist()
in_branch = []
for (j, _), v in zip(rows, fv):
    nodes = sorted({n for b in focused[j] for n in b[:8]})
    sims = (mem.keys[torch.tensor(nodes, device="cuda")] @ v).float()
    in_branch.append(float(sims.max()))
n = len(rows)
print(f"OBQA dev questions with a gold fact: {n}")
for t in (0.95, 0.9, 0.85, 0.8):
    in_mem = sum(s >= t for s in best_sim) / n
    found = sum(s >= t for s in in_branch) / n
    print(
        f"  similarity >= {t}: in the memory {in_mem:.3f} | in the focused branches (8/cand) "
        f"{found:.3f} | found when present {found / in_mem if in_mem else 0:.3f}"
    )
print(
    "median best similarity in memory",
    round(statistics.median(best_sim), 3),
    "| in branches",
    round(statistics.median(in_branch), 3),
)
for k in range(0, 60, 12):
    print(f"\nGOLD: {texts[k]}\n  memory best ({best_sim[k]:.2f}): {mem.concepts[best_idx[k]].text}")
