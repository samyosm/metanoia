"""Gate (b) of decision 0012: does the frozen R005 reader use retrieved statements?

Evaluation only, on knowledge_v0 dev + calibration (1,368; test stays untouched). Memory given as argv[2] (e.g. generics_v0).
Formats:
  shared   one context for all candidates: the union of each candidate's top-k nodes
           (R017's protocol)
  branch   EXPAND v2's format: candidate i reads only its own top-k nodes, in its own
           joint input "answer: c / question: q / context: nodes"; the question-only
           encoding keeps an empty context (as in the direct model)
Node sources: retrieved ("question candidate" query, as EXPAND v1), random (seeded per
example and candidate). Widths k per candidate from argv[3] (default 1,2,4,8).
Run from metanoia/: .venv/bin/python <this> <out_dir> <memory> [ks]
"""

import hashlib
import json
import math
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch

from metanoia.api import load_predictor
from metanoia.encoding.inputs import render_candidate_joint, render_input
from metanoia.memory.encoder import RetrievalEncoder
from metanoia.memory.store import ConceptMemory
from metanoia_contracts import DecisionRequest

OUT = Path(sys.argv[1])
MEMORY = sys.argv[2]
KS = [int(k) for k in (sys.argv[3] if len(sys.argv) > 3 else "1,2,4,8").split(",")]
OUT.mkdir(parents=True, exist_ok=True)
D = Path("artifacts/datasets/knowledge_v0")
cases = [json.loads(line) for line in (D / "cases.jsonl").open()]
cases = [c for c in cases if c["split"] in ("dev", "calibration")]  # test stays untouched
labels = {r["example_id"]: r for r in pq.read_table(D / "labels.parquet").to_pylist()}
requests = [DecisionRequest.model_validate_json(json.dumps(c["request"])) for c in cases]

memory = ConceptMemory(Path("artifacts/memories") / MEMORY, device="cuda")
spec = memory.manifest.encoder
retriever = RetrievalEncoder(
    spec["model"], spec["requested_revision"], spec["pooling"], spec["max_length"], spec["query_prefix"]
).to("cuda")
KMAX = max(KS)
queries = [f"{r.question} {c.text}" for r in requests for c in r.candidates]
with torch.inference_mode():
    vecs = torch.cat([retriever.embed(queries[i : i + 256], query=True) for i in range(0, len(queries), 256)])
    top = memory.search(vecs, KMAX)[0].tolist()
del retriever
torch.cuda.empty_cache()
branches: dict[str, list[list[list[int]]]] = {"retrieved": [], "random": []}
pos = 0
for case, r in zip(cases, requests, strict=True):
    ret, rnd = [], []
    for c in r.candidates:
        ret.append(top[pos])
        pos += 1
        seed = int.from_bytes(hashlib.sha256(f"{case['example_id']}\x1f{c.id}".encode()).digest()[:8], "big")
        rnd.append(random.Random(seed).sample(range(len(memory)), KMAX))
    branches["retrieved"].append(ret)
    branches["random"].append(rnd)


def text(nodes: list[int]) -> str:
    return " ".join(f"[{i + 1}] {memory.concepts[j].text}" for i, j in enumerate(nodes))


pred = load_predictor(Path("artifacts/predictors/mbft_joint_mcqa_v0"), device="cuda", precision="fp16")
model = pred.model


def score(batch: list[DecisionRequest], contexts: list[list[str]] | None) -> list[list[float]]:
    """Logits. contexts[b][i] is candidate i's own context (branch format); None = as is."""
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        if contexts is None:
            return [row[: len(r.candidates)] for r, row in zip(batch, model(batch).logits.float().tolist(), strict=True)]
        rendered = [render_input(r) for r in batch]  # the question-only encoding: empty context
        ctx = model.encoder.encode(rendered)
        texts = [
            render_candidate_joint(c.text, render_input(r.model_copy(update={"context": contexts[b][i]})))
            for b, r in enumerate(batch)
            for i, c in enumerate(r.candidates)
        ]
        flat = model.encoder.encode(texts).pooled
        n = max(len(r.candidates) for r in batch)
        cands = flat.new_zeros(len(batch), n, flat.shape[-1])
        mask = torch.zeros(len(batch), n, dtype=torch.bool, device=flat.device)
        s = 0
        for b, r in enumerate(batch):
            k = len(r.candidates)
            cands[b, :k] = flat[s : s + k]
            mask[b, :k] = True
            s += k
        logits = model.scorer(ctx, cands, mask).float().tolist()
        return [row[: len(r.candidates)] for r, row in zip(batch, logits, strict=True)]


def evaluate(name: str, build) -> None:
    per = {}
    for i in range(0, len(requests), 16):
        idx = range(i, min(i + 16, len(requests)))
        batch, contexts = build(list(idx))
        for j, row in zip(idx, score(batch, contexts), strict=True):
            lab = labels[cases[j]["example_id"]]
            probs = torch.softmax(torch.tensor(row), 0).tolist()
            gold = [c.id for c in requests[j].candidates].index(lab["label_id"])
            ok = max(range(len(probs)), key=probs.__getitem__) == gold
            for s in (lab["source"], "all"):
                per.setdefault(s, []).append((ok, -math.log(max(probs[gold], 1e-15))))
    results[name] = {s: {"n": len(v), "accuracy": sum(o for o, _ in v) / len(v), "nll": sum(x for _, x in v) / len(v)} for s, v in per.items()}
    print(name, {s: (round(m["accuracy"], 3), round(m["nll"], 3)) for s, m in results[name].items()}, flush=True)


results: dict = {}
evaluate("none", lambda idx: ([requests[j] for j in idx], None))
# Sanity: the branch path with empty branches must reproduce "none".
evaluate("branch/empty", lambda idx: ([requests[j] for j in idx], [[""] * len(requests[j].candidates) for j in idx]))
for source in ("retrieved", "random"):
    for k in KS:
        def shared(idx, source=source, k=k):
            out = []
            for j in idx:
                union: list[int] = []
                for nodes in branches[source][j]:
                    union += [x for x in nodes[:k] if x not in union]
                out.append(requests[j].model_copy(update={"context": text(union)}))
            return out, None

        def branch(idx, source=source, k=k):
            return [requests[j] for j in idx], [[text(nodes[:k]) for nodes in branches[source][j]] for j in idx]

        evaluate(f"branch/{source}/k{k}", branch)
        if k in (1, 2):
            evaluate(f"shared/{source}/k{k}", shared)

examples = []
for j in range(0, 60, 6):
    r = requests[j]
    examples.append({
        "example_id": cases[j]["example_id"], "question": r.question,
        "label": labels[cases[j]["example_id"]]["label_id"],
        "branches": {c.id: [memory.concepts[x].text for x in branches["retrieved"][j][i][:4]] for i, c in enumerate(r.candidates)},
    })
(OUT / f"branch_probe_{MEMORY}.json").write_text(json.dumps({"memory": MEMORY, "memory_manifest_id": memory.manifest.memory_id, "results": results, "examples": examples}, indent=1))
