"""R022 probe (part 2: focused and distinct reorderings of the retrieved pool): concept-anchored and chained proposals, read by the trained v2 model.

Evaluation only, knowledge_v0 dev + calibration (test untouched). No retraining: the
trained branch model (argv[2]) reads branches built by different proposal rules.
  retrieved  as trained: top-k by similarity to "question candidate"
  anchored   statements whose concept term is named in the candidate (1-3-word spans,
             lemmatized), ranked by similarity to the question; fallback: retrieved
  chain      anchored first hop; second hop from concept terms mentioned in the chosen
             first-hop statements (not the candidate's own, not the question's), ranked
             by similarity to the question; branch = hop1, hop2 interleaved
Reports accuracy/NLL, branch overlap, own-word mention and OBQA gold-fact word recall.
Run from metanoia/: .venv/bin/python <this> <out_dir> <bundle> [ks]
"""

import json
import math
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import nltk
import pyarrow.parquet as pq
import torch

from metanoia.api import load_predictor
from metanoia.encoding.inputs import render_input
from metanoia_contracts import DecisionRequest

nltk.data.path.insert(0, str(Path("artifacts/nltk_data").resolve()))
from nltk.corpus import wordnet  # noqa: E402

OUT = Path(sys.argv[1])
BUNDLE = Path(sys.argv[2])
KS = [int(k) for k in (sys.argv[3] if len(sys.argv) > 3 else "2,4,8").split(",")]
OUT.mkdir(parents=True, exist_ok=True)
D = Path("artifacts/datasets/knowledge_v0")
cases = [json.loads(line) for line in (D / "cases.jsonl").open()]
cases = [c for c in cases if c["split"] in ("dev", "calibration")]
labels = {r["example_id"]: r for r in pq.read_table(D / "labels.parquet").to_pylist()}
facts = {json.loads(l)["example_id"]: json.loads(l)["core_fact"] for l in (D / "proofs.jsonl").open()}
requests = [DecisionRequest.model_validate_json(json.dumps(c["request"])) for c in cases]

pred = load_predictor(BUNDLE, device="cuda", precision="fp16")
model = pred.model.eval()
memory = model.memory()
by_term: dict[str, list[int]] = defaultdict(list)
for idx, concept in enumerate(memory.concepts):
    by_term[concept.name].append(idx)
WORD = re.compile(r"[a-z]+(?:-[a-z]+)?")
STOP = set("a an the of to in on at for from by with and or but not no is are was were be been it its this that these those as than then so if when which what who how why do does did can will would should could may might have has had they them their there here some any all most more less very much many each other such only also into out up down over under about after before during between through one two three".split())


def lemma(word: str) -> str:
    return wordnet.morphy(word, "n") or wordnet.morphy(word, "v") or word


def terms_in(text: str) -> list[str]:
    """Concept terms named in a text: 1-3-word spans (lemmatized last word), longest first."""
    words = WORD.findall(text.lower())
    found = []
    for size in (3, 2, 1):
        for i in range(len(words) - size + 1):
            span = words[i : i + size]
            if size == 1 and span[0] in STOP:
                continue
            for cand in (" ".join(span), " ".join(span[:-1] + [lemma(span[-1])])):
                if cand in by_term and cand not in found:
                    found.append(cand)
    return found


with torch.inference_mode():
    qvec = model.retriever().embed([r.question for r in requests], query=True).to(memory.keys.dtype)


def rank(nodes: list[int], j: int, k: int, exclude: set[int]) -> list[int]:
    nodes = [n for n in dict.fromkeys(nodes) if n not in exclude]
    if not nodes:
        return []
    idx = torch.tensor(nodes, device=memory.keys.device)
    scores = memory.keys[idx] @ qvec[j]
    top = torch.topk(scores.float(), k=min(k, len(nodes))).indices.tolist()
    return [nodes[t] for t in top]


KMAX = max(KS)
POOL = 64
pairs = [(r.question, c.text) for r in requests for c in r.candidates]
with torch.inference_mode():
    flat, _ = model.propose(pairs, POOL, "retrieved")
branches = {"retrieved": [], "focused": [], "distinct": []}
anchored_rate = []
row = 0
for j, r in enumerate(requests):
    qwords = {lemma(w) for w in WORD.findall(r.question.lower())}
    pools = [flat[row + i] for i in range(len(r.candidates))]
    row += len(r.candidates)
    ret, foc, dis = [], [], []
    for i, c in enumerate(r.candidates):
        own = {lemma(w) for w in WORD.findall(c.text.lower()) if w not in STOP and len(w) > 2} - qwords
        def mentions(n):
            return bool(own & {lemma(w) for w in WORD.findall(memory.concepts[n].text.lower())})
        pool = pools[i]
        others = set().union(*(set(p) for m, p in enumerate(pools) if m != i))
        focused = [n for n in pool if mentions(n)] + [n for n in pool if not mentions(n)]
        distinct = [n for n in focused if n not in others] + [n for n in focused if n in others]
        anchored_rate.append(any(mentions(n) for n in pool[:KMAX]))
        ret.append(pool[:KMAX]); foc.append(focused[:KMAX]); dis.append(distinct[:KMAX])
    branches["retrieved"].append(ret)
    branches["focused"].append(foc)
    branches["distinct"].append(dis)
print("candidates whose top-k retrieval mentions them:", sum(anchored_rate) / len(anchored_rate), flush=True)


def texts(ids):
    return [memory.concepts[x].text for x in ids]


def logits_for(idx, statements):
    batch = [requests[j] for j in idx]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        ctx = model.encoder.encode([render_input(r) for r in batch])
        cand_texts = [model._candidate_text(r, i, statements[b][i]) for b, r in enumerate(batch) for i in range(len(r.candidates))]
        cands, mask = model._pack(model.encoder.encode(cand_texts).pooled, [len(r.candidates) for r in batch])
        out = model.scorer(ctx, cands, mask).float()
    return [out[b, : len(r.candidates)].tolist() for b, r in enumerate(batch)]


def content_words(text):
    return {lemma(w) for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 2}


results, quality = {}, {}
for source in branches:
    for k in KS:
        per = {}
        for s in range(0, len(requests), 16):
            idx = list(range(s, min(s + 16, len(requests))))
            rows = logits_for(idx, [[texts(b[:k]) for b in branches[source][j]] for j in idx])
            for j, lg in zip(idx, rows, strict=True):
                lab = labels[cases[j]["example_id"]]
                probs = torch.softmax(torch.tensor(lg), 0).tolist()
                gold = [c.id for c in requests[j].candidates].index(lab["label_id"])
                best = max(range(len(probs)), key=probs.__getitem__)
                for src in (lab["source"], "all"):
                    per.setdefault(src, []).append((best == gold, -math.log(max(probs[gold], 1e-15))))
        key = f"{source}/k{k}"
        results[key] = {s: {"n": len(v), "accuracy": sum(o for o, _ in v) / len(v), "nll": sum(x for _, x in v) / len(v)} for s, v in per.items()}
        # Branch quality: overlap, own-word mention, OBQA gold-fact novel-word recall.
        jac, mention, recall = [], [], []
        for j, r in enumerate(requests):
            sets = [set(b[:k]) for b in branches[source][j]]
            jac += [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if a | b]
            qwords = content_words(r.question)
            for c, b in zip(r.candidates, branches[source][j], strict=True):
                own = content_words(c.text) - qwords
                if own:
                    mention.append(bool(own & content_words(" ".join(texts(b[:k])))))
            fact = facts.get(cases[j]["example_id"])
            if fact:
                novel = content_words(fact) - qwords - set().union(*(content_words(c.text) for c in r.candidates))
                if novel:
                    found = set().union(*(content_words(" ".join(texts(b[:k]))) for b in branches[source][j]))
                    recall.append(len(novel & found) / len(novel))
        quality[key] = {"pairwise_jaccard": sum(jac) / len(jac), "mentions_own_words": sum(mention) / len(mention), "obqa_novel_fact_recall": sum(recall) / len(recall)}
        print(key, {s: (round(m["accuracy"], 3), round(m["nll"], 3)) for s, m in results[key].items()}, {q: round(v, 3) for q, v in quality[key].items()}, flush=True)

examples = []
for j in range(0, 300, 30):
    r = requests[j]
    examples.append({"question": r.question, "label": labels[cases[j]["example_id"]]["label_id"], **{src: {c.text: texts(branches[src][j][i][:4]) for i, c in enumerate(r.candidates)} for src in branches}})
(OUT / "focused_probe.json").write_text(json.dumps({"bundle": str(BUNDLE), "predictor_id": pred.predictor_id, "anchored_rate": sum(anchored_rate) / len(anchored_rate), "results": results, "quality": quality, "examples": examples}, indent=1))
