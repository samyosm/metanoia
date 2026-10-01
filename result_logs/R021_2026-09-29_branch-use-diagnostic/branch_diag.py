"""R021 diagnostic: what does the trained EXPAND v2 model take from its branches?

Evaluation only, knowledge_v0 dev + calibration (1,368; test untouched). The model is
argv[2] (a branch bundle). Conditions change only the branches' content:
  empty      no nodes (budget 0)
  own        each candidate reads its own top-k nodes (normal)
  swapped    candidate i reads candidate i+1's branch (does the answer follow content?)
  shared     every candidate reads the same k nodes: the round-robin union of all
             branches (topic knowledge without candidate-specific evidence)
  question   every candidate reads the top-k nodes for the question alone
  gold       OBQA only: every candidate reads the gold core fact (an oracle)
  gold+own   OBQA only: the gold fact first, then k-1 own nodes
Also: overlap between the branches of a question, and per-node leave-one-out
contributions on a subset.
Run from metanoia/: .venv/bin/python <this> <out_dir> <bundle> [ks]
"""

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import pyarrow.parquet as pq
import torch

from metanoia.api import load_predictor
from metanoia_contracts import DecisionRequest

OUT = Path(sys.argv[1])
BUNDLE = Path(sys.argv[2])
KS = [int(k) for k in (sys.argv[3] if len(sys.argv) > 3 else "2,8").split(",")]
OUT.mkdir(parents=True, exist_ok=True)
D = Path("artifacts/datasets/knowledge_v0")
cases = [json.loads(line) for line in (D / "cases.jsonl").open()]
cases = [c for c in cases if c["split"] in ("dev", "calibration")]
labels = {r["example_id"]: r for r in pq.read_table(D / "labels.parquet").to_pylist()}
facts = {json.loads(l)["example_id"]: json.loads(l)["core_fact"] for l in (D / "proofs.jsonl").open()}
requests = [DecisionRequest.model_validate_json(json.dumps(c["request"])) for c in cases]

pred = load_predictor(BUNDLE, device="cuda", precision="fp16")
model = pred.model
model.eval()
memory = model.memory()
KMAX = max(KS)
with torch.inference_mode():
    pairs = [(r.question, c.text) for r in requests for c in r.candidates]
    flat, _ = model.propose(pairs, KMAX, "retrieved")
    qflat, _ = model.propose([(r.question, "") for r in requests], KMAX, "retrieved")
own_all, row = [], 0
for r in requests:
    own_all.append([flat[row + i] for i in range(len(r.candidates))])
    row += len(r.candidates)


def texts(ids):
    return [memory.concepts[j].text for j in ids]


def logits_for(batch_idx, statements):
    """statements[b][i]: list of texts for candidate i of example b."""
    batch = [requests[j] for j in batch_idx]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        from metanoia.encoding.inputs import render_input

        ctx = model.encoder.encode([render_input(r) for r in batch])
        cand_texts = [
            model._candidate_text(r, i, statements[b][i])
            for b, r in enumerate(batch)
            for i in range(len(r.candidates))
        ]
        pooled = model.encoder.encode(cand_texts).pooled
        cands, mask = model._pack(pooled, [len(r.candidates) for r in batch])
        out = model.scorer(ctx, cands, mask).float()
    return [out[b, : len(r.candidates)].tolist() for b, r in enumerate(batch)]


def condition(j, name, k):
    r = requests[j]
    n = len(r.candidates)
    own = [texts(b[:k]) for b in own_all[j]]
    if name == "empty":
        return [[] for _ in range(n)]
    if name == "own":
        return own
    if name == "swapped":
        return [own[(i + 1) % n] for i in range(n)]
    if name == "shared":
        union = []
        for m in range(KMAX):
            for b in own_all[j]:
                t = memory.concepts[b[m]].text
                if t not in union:
                    union.append(t)
        return [union[:k] for _ in range(n)]
    if name == "question":
        return [texts(qflat[j][:k]) for _ in range(n)]
    fact = facts.get(cases[j]["example_id"])
    if name == "gold":
        return [[fact] for _ in range(n)]
    if name == "gold+own":
        return [[fact] + o[: k - 1] for o in own]
    raise ValueError(name)


results, decisions = {}, {}
obqa = [j for j, c in enumerate(cases) if labels[c["example_id"]]["source"] == "obqa" and facts.get(c["example_id"])]
plan = [("empty", 0, range(len(cases)))]
for k in KS:
    plan += [(n, k, range(len(cases))) for n in ("own", "swapped", "shared", "question")]
    plan += [("gold+own", k, obqa)]
plan += [("gold", 1, obqa)]
for name, k, subset in plan:
    subset = list(subset)
    per, chosen = {}, []
    for s in range(0, len(subset), 16):
        idx = subset[s : s + 16]
        rows = logits_for(idx, [condition(j, name, k) for j in idx])
        for j, row in zip(idx, rows, strict=True):
            lab = labels[cases[j]["example_id"]]
            probs = torch.softmax(torch.tensor(row), 0).tolist()
            gold = [c.id for c in requests[j].candidates].index(lab["label_id"])
            best = max(range(len(probs)), key=probs.__getitem__)
            chosen.append(best)
            for src in (lab["source"], "all"):
                per.setdefault(src, []).append((best == gold, -math.log(max(probs[gold], 1e-15))))
    key = f"{name}/k{k}"
    decisions[key] = dict(zip(subset, chosen, strict=True))
    results[key] = {s: {"n": len(v), "accuracy": sum(o for o, _ in v) / len(v), "nll": sum(x for _, x in v) / len(v)} for s, v in per.items()}
    print(key, {s: (round(m["accuracy"], 3), round(m["nll"], 3), m["n"]) for s, m in results[key].items()}, flush=True)

# How often does the answer follow the branch? With swapped branches, the choice moves
# to the candidate whose (original) branch was the chosen one's.
follow = {}
for k in KS:
    moved = followed = 0
    for j in range(len(cases)):
        n = len(requests[j].candidates)
        a = decisions[f"own/k{k}"][j]
        b = decisions[f"swapped/k{k}"][j]
        if a != b:
            moved += 1
        # candidate (a-1) mod n now reads the branch that led to choosing a
        followed += b == (a - 1) % n
    follow[f"k{k}"] = {"changed": moved / len(cases), "followed_branch": followed / len(cases), "chance": 0.25}
print("follow", follow)

# Branch overlap within a question: mean pairwise Jaccard of node sets, and the share of
# a candidate's nodes that also appear in another candidate's branch.
overlap = {}
for k in KS:
    jac, shared_share = [], []
    for branches in own_all:
        sets = [set(b[:k]) for b in branches]
        jac += [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
        for i, s in enumerate(sets):
            others = set().union(*(t for m, t in enumerate(sets) if m != i))
            shared_share.append(len(s & others) / len(s))
    overlap[f"k{k}"] = {"mean_pairwise_jaccard": sum(jac) / len(jac), "nodes_also_in_another_branch": sum(shared_share) / len(shared_share)}
print("overlap", overlap)

# Candidate-word presence: does a candidate's branch mention its own candidate's words?
import re

WORD = re.compile(r"[a-z]{3,}")
mention = {}
for k in KS:
    hits = []
    for r, branches in zip(requests, own_all, strict=True):
        for c, b in zip(r.candidates, branches, strict=True):
            words = set(WORD.findall(c.text.lower())) - set(WORD.findall(r.question.lower()))
            if not words:
                continue
            text = " ".join(texts(b[:k])).lower()
            hits.append(any(w in text for w in words))
    mention[f"k{k}"] = sum(hits) / len(hits)
print("branch mentions its candidate's own (non-question) words", mention)

# Leave-one-out contributions on 160 examples at width 2 and 8.
loo = {}
for k in KS:
    mags, moves, spreads = [], [], []
    for s in range(0, 160, 8):
        batch = [requests[j] for j in range(s, s + 8)]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            out = model(batch, width=k, explain=True)
        for b, r in enumerate(batch):
            n = len(r.candidates)
            mags += [abs(x) for c in out.contributions[b] for x in c]
            final = out.logits[b, :n].float()
            init = out.initial_logits[b, :n].float()
            moves.append(float((final - final.mean() - (init - init.mean())).abs().mean()))
            spreads.append(float(init.std()))
    loo[f"k{k}"] = {"mean_abs_node_contribution": sum(mags) / len(mags), "mean_abs_centered_logit_change": sum(moves) / len(moves), "mean_direct_logit_std": sum(spreads) / len(spreads)}
print("loo", loo)

examples = []
for j in list(range(0, 200, 20)):
    r = requests[j]
    lab = labels[cases[j]["example_id"]]["label_id"]
    examples.append({
        "question": r.question, "label": lab,
        "choice_empty": r.candidates[decisions["empty/k0"][j]].id,
        f"choice_own_k{KMAX}": r.candidates[decisions[f"own/k{KMAX}"][j]].id,
        "branches": {c.text: texts(own_all[j][i][:4]) for i, c in enumerate(r.candidates)},
    })
(OUT / "branch_diagnostics.json").write_text(json.dumps({"bundle": str(BUNDLE), "predictor_id": pred.predictor_id, "results": results, "follow": follow, "overlap": overlap, "mentions_own_words": mention, "loo": loo, "examples": examples}, indent=1))
