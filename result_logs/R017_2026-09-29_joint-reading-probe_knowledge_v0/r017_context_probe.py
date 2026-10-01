"""R017 diagnostic: does concept/fact knowledge help when the reader reads it jointly?

Evaluation only (no training). Two readers:
  r005    frozen R005 reader (mbft_joint_mcqa_v0) = the expansion models' direct path
  kv0     mbft_joint_knowledge_v0 selected checkpoint (step 521)
Context conditions on knowledge_v0 test:
  none       empty context (as trained / as collected)
  retrieved  union of the top-4 wordnet_v0 concepts per candidate ("question candidate")
  random     the same number of random concepts (seeded per example)
  qret       top-16 concepts for the question alone
  gold       OpenBookQA core fact (OBQA only)
Also a concept-sensitivity probe of the frozen reader, as EXPAND v1 uses it.
Run from metanoia/: .venv/bin/python <this> <out_dir>
"""

import hashlib
import json
import math
import random
import sys
from pathlib import Path

import torch

from metanoia.api import load_predictor
from metanoia.memory.encoder import RetrievalEncoder
from metanoia.memory.store import ConceptMemory
from metanoia_contracts import DecisionRequest

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
D = Path("artifacts/datasets/knowledge_v0")
cases = [json.loads(line) for line in (D / "cases.jsonl").open()]
cases = [c for c in cases if c["split"] == "test"]
proofs = {json.loads(line)["example_id"]: json.loads(line)["core_fact"] for line in (D / "proofs.jsonl").open()}
import pyarrow.parquet as pq

labels = {r["example_id"]: r for r in pq.read_table(D / "labels.parquet").to_pylist()}

memory = ConceptMemory(Path("artifacts/memories/wordnet_v0"), device="cuda")
spec = memory.manifest.encoder
retriever = RetrievalEncoder(
    spec["model"], spec["requested_revision"], spec["pooling"], spec["max_length"], spec["query_prefix"]
).to("cuda")


def embed(texts):
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), 256):
            out.append(retriever.embed(texts[i : i + 256], query=True))
    return torch.cat(out)


# Retrieval, as EXPAND v1 does it: query = "question candidate", top k.
K = 4
per_cand_q = [f"{c['request']['question']} {x['text']}" for c in cases for x in c["request"]["candidates"]]
idx = memory.search(embed(per_cand_q), K)[0].tolist()
qidx = memory.search(embed([c["request"]["question"] for c in cases]), 16)[0].tolist()
contexts = {"none": [], "retrieved": [], "random": [], "qret": [], "gold": []}
pos = 0
for n, c in enumerate(cases):
    chosen = []
    for _ in c["request"]["candidates"]:
        for j in idx[pos]:
            if j not in chosen:
                chosen.append(j)
        pos += 1
    seed = int.from_bytes(hashlib.sha256(c["example_id"].encode()).digest()[:8], "big")
    rnd = random.Random(seed).sample(range(len(memory)), len(chosen))
    contexts["none"].append("")
    contexts["retrieved"].append(" ".join(memory.concepts[j].text + "." for j in chosen))
    contexts["random"].append(" ".join(memory.concepts[j].text + "." for j in rnd))
    contexts["qret"].append(" ".join(memory.concepts[j].text + "." for j in qidx[n]))
    contexts["gold"].append(proofs.get(c["example_id"]))

readers = {
    "r005": Path("artifacts/predictors/mbft_joint_mcqa_v0"),
    "kv0": Path("artifacts/predictors/mbft_joint_knowledge_v0/checkpoints/step_0000521"),
}
summary = {}
examples = []
for rname, path in readers.items():
    pred = load_predictor(path, device="cuda", precision="fp16")
    for cond, ctx in contexts.items():
        rows = [(c, x) for c, x in zip(cases, ctx) if x is not None]
        reqs = [DecisionRequest.model_validate_json(json.dumps({**c["request"], "context": x})) for c, x in rows]
        results = []
        for i in range(0, len(reqs), 32):
            results.extend(pred.decide_batch(reqs[i : i + 32]))
        by_src = {}
        for (c, _), res in zip(rows, results):
            lab = labels[c["example_id"]]
            p = res.probabilities[res.candidate_ids.index(lab["label_id"])]
            ok = res.candidate_ids[max(range(len(res.probabilities)), key=res.probabilities.__getitem__)] == lab["label_id"]
            for s in (lab["source"], "all"):
                by_src.setdefault(s, []).append((ok, -math.log(max(p, 1e-15))))
            if rname == "r005" and cond in ("retrieved", "gold") and len(examples) < 12 and lab["source"] == "obqa":
                examples.append({"cond": cond, "id": c["example_id"], "context": _[:600]})
        summary[f"{rname}/{cond}"] = {
            s: {"n": len(v), "accuracy": sum(o for o, _ in v) / len(v), "nll": sum(l for _, l in v) / len(v)}
            for s, v in by_src.items()
        }
        print(rname, cond, {s: round(m["accuracy"], 3) for s, m in summary[f"{rname}/{cond}"].items()}, flush=True)
    del pred
    torch.cuda.empty_cache()

# Concept sensitivity of the frozen R005 reader as EXPAND v1 feeds it:
# "concept: {text}\n{rendered input}" -> pooled. Within-example spread vs across examples.
pred = load_predictor(readers["r005"], device="cuda", precision="fp16")
enc = pred.model.encoder
from metanoia.encoding.inputs import render_input

probe = cases[:64]
within, across, zs = [], [], []
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    for n, c in enumerate(probe):
        req = DecisionRequest.model_validate_json(json.dumps(c["request"]))
        rendered = render_input(req)
        first = idx[sum(len(p["request"]["candidates"]) for p in cases[:n])]
        extra = memory.search(embed([f"{req.question} {req.candidates[0].text}"]), 16)[0].tolist()[0]
        texts = [f"concept: {memory.concepts[j].text}\n{rendered}" for j in extra]
        z = torch.nn.functional.normalize(enc.encode(texts).pooled.float(), dim=-1)
        sim = z @ z.T
        within.append(((sim.sum() - sim.diag().sum()) / (16 * 15)).item())
        zs.append(z.mean(0))
Z = torch.nn.functional.normalize(torch.stack(zs), dim=-1)
S = Z @ Z.T
across = ((S.sum() - S.diag().sum()) / (len(zs) * (len(zs) - 1))).item()
summary["probe"] = {
    "mean_cos_between_concepts_same_input": sum(within) / len(within),
    "mean_cos_between_inputs": across,
    "n_inputs": len(probe),
    "concepts_per_input": 16,
}
print(summary["probe"])
(OUT / "context_probe.json").write_text(json.dumps({"summary": summary, "examples": examples}, indent=1))
