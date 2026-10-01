"""Decision 0014, screening round 1, part B: a trained reader reads each configuration.

Inputs: branches.json from screen_build.py; a trained branch bundle (the reader); the
configurations to evaluate (comma-separated, or "all"). Widths 2 and 8 per candidate.
Metrics per configuration and width: accuracy / NLL overall and per source; branch
quality (on-candidate rate, mean pairwise Jaccard, OBQA novel gold-fact recall).
Run from metanoia/: .venv/bin/python <this> <dir> <bundle> <reader_name> <configs|all>
"""

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import pyarrow.parquet as pq
import torch

from metanoia.api import load_predictor
from metanoia.encoding.inputs import render_input
from metanoia.reasoning.branch import _stems
from metanoia_contracts import DecisionRequest

DIR, BUNDLE, READER = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
data = json.loads((DIR / "branches.json").read_text())
names = list(data["configs"]) if sys.argv[4] == "all" else sys.argv[4].split(",")
WIDTHS = (2, 8)
D = Path("artifacts/datasets/knowledge_v0")
by_id = {json.loads(l)["example_id"]: json.loads(l) for l in (D / "cases.jsonl").open()}
cases = [by_id[e] for e in data["example_ids"]]
labels = {r["example_id"]: r for r in pq.read_table(D / "labels.parquet").to_pylist()}
facts = {json.loads(l)["example_id"]: json.loads(l)["core_fact"] for l in (D / "proofs.jsonl").open()}
requests = [DecisionRequest.model_validate_json(json.dumps(c["request"])) for c in cases]

pred = load_predictor(BUNDLE, device="cuda", precision="fp16")
model = pred.model.eval()
memory = model.memory()


def logits_for(idx, statements):
    batch = [requests[j] for j in idx]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        ctx = model.encoder.encode([render_input(r) for r in batch])
        texts = [model._candidate_text(r, i, statements[b][i]) for b, r in enumerate(batch) for i in range(len(r.candidates))]
        cands, mask = model._pack(model.encoder.encode(texts).pooled, [len(r.candidates) for r in batch])
        out = model.scorer(ctx, cands, mask).float()
    return [out[b, : len(r.candidates)].tolist() for b, r in enumerate(batch)]


out_path = DIR / f"eval_{READER}.json"
results = json.loads(out_path.read_text()) if out_path.exists() else {}
for name in names:
    for k in WIDTHS:
        key = f"{name}/k{k}"
        if key in results:
            continue
        branches = data["configs"][name]
        text = [[[memory.concepts[n].text for n in b[:k]] for b in per] for per in branches]
        per_src = {}
        for s in range(0, len(requests), 16):
            idx = list(range(s, min(s + 16, len(requests))))
            for j, lg in zip(idx, logits_for(idx, [text[j] for j in idx]), strict=True):
                lab = labels[cases[j]["example_id"]]
                probs = torch.softmax(torch.tensor(lg), 0).tolist()
                gold = [c.id for c in requests[j].candidates].index(lab["label_id"])
                best = max(range(len(probs)), key=probs.__getitem__)
                for src in (lab["source"], "all"):
                    per_src.setdefault(src, []).append((best == gold, -math.log(max(probs[gold], 1e-15))))
        jac, on, recall = [], [], []
        for j, r in enumerate(requests):
            sets = [set(b[:k]) for b in branches[j]]
            jac += [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if a | b]
            q = _stems(r.question)
            for c, t in zip(r.candidates, text[j], strict=True):
                own = _stems(c.text) - q
                if own:
                    on.append(bool(own & _stems(" ".join(t))))
            fact = facts.get(cases[j]["example_id"])
            if fact:
                novel = _stems(fact) - q - set().union(*(_stems(c.text) for c in r.candidates))
                if novel:
                    found = set().union(*(_stems(" ".join(t)) for t in text[j]))
                    recall.append(len(novel & found) / len(novel))
        results[key] = {
            **{src: {"n": len(v), "accuracy": sum(o for o, _ in v) / len(v), "nll": sum(x for _, x in v) / len(v)} for src, v in per_src.items()},
            "quality": {"on_candidate": sum(on) / len(on), "jaccard": sum(jac) / len(jac), "obqa_gold_recall": sum(recall) / len(recall)},
        }
        m = results[key]
        print(f"{READER:9} {key:28} all {m['all']['accuracy']:.3f}/{m['all']['nll']:.3f}  obqa {m['obqa']['accuracy']:.3f}  easy {m['arc_easy']['accuracy']:.3f}  chal {m['arc_challenge']['accuracy']:.3f}  on {m['quality']['on_candidate']:.2f} jac {m['quality']['jaccard']:.2f} rec {m['quality']['obqa_gold_recall']:.3f}", flush=True)
        out_path.write_text(json.dumps(results, indent=1))  # saved after each, so reruns resume
(DIR / f"eval_{READER}_reader.json").write_text(json.dumps({"bundle": str(BUNDLE), "predictor_id": pred.predictor_id}))
