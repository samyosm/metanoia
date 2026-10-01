"""Decision 0014, screening round 1, part A: build proposal pools and judge them.

knowledge_v0 dev + calibration (1,368 questions; test untouched), memory generics_v0.
Generators (per question, candidate):
  G1 similarity   top 64 for "question candidate" (bge-small, exact search)
  G3 paths        concept paths from the candidate's concepts to the question's:
                  one hop: statements mentioning a candidate concept and a question
                  concept, ranked by idf of the question concepts matched (top 32);
                  two hops: s1 (candidate concept + bridge b) and s2 (b + question
                  concept), b not in either side and in at most 5,000 statements
                  (top 8 pairs, both statements added)
  G4 follow-up    top 16 for "question candidate s1", s1 = the focused top statement
Judges over the pool:
  J0 similarity to "question candidate" (memory keys)
  J1 focused: statements mentioning the candidate's own words first, then J0
  J2 bge-reranker-v2-m3, query "Question: q Answer: c" (GPU, fp16)
Configurations (pool, ordering) are written as branches of up to 8 nodes per candidate.
Run from metanoia/: .venv/bin/python <this> <out_dir>
"""

import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from metanoia.memory.encoder import RetrievalEncoder
from metanoia.memory.store import ConceptMemory
from metanoia.reasoning.branch import _singular, _stems

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
KMAX = 8
RERANKER, REVISION = "BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
WORD = re.compile(r"[a-z]+")
STOP = set(
    "a an the of to in on at for from by with and or but not no is are was were be been it its "
    "this that these those as than then so if when which what who how why do does did can will "
    "would should could may might have has had they them their there here some any all most more "
    "less very much many each other such only also into out up down over under about after before "
    "during between through one two three use used make made get thing way part kind type".split()
)
t_start = time.time()

D = Path("artifacts/datasets/knowledge_v0")
cases = [json.loads(line) for line in (D / "cases.jsonl").open()]
cases = [c for c in cases if c["split"] in ("dev", "calibration")]
memory = ConceptMemory(Path("artifacts/memories/generics_v0"), device="cuda")
spec = memory.manifest.encoder
retriever = RetrievalEncoder(
    spec["model"], spec["requested_revision"], spec["pooling"], spec["max_length"], spec["query_prefix"]
).to("cuda")
terms = {c.name for c in memory.concepts}


def mentioned(text: str) -> set[str]:
    words = WORD.findall(text.lower())
    found = set()
    for size in (3, 2, 1):
        for i in range(len(words) - size + 1):
            span = words[i : i + size]
            if size == 1 and (span[0] in STOP or len(span[0]) < 3):
                continue
            for cand in (" ".join(span), " ".join(span[:-1] + [_singular(span[-1])])):
                if cand in terms:
                    found.add(cand)
    return found


mentions = [mentioned(c.text) | {c.name} for c in memory.concepts]
by_concept: dict[str, list[int]] = defaultdict(list)
for i, ms in enumerate(mentions):
    for m in ms:
        by_concept[m].append(i)
N = len(memory.concepts)
idf = {m: math.log(N / len(v)) for m, v in by_concept.items()}
print("index built", round(time.time() - t_start), "s", flush=True)

questions = [c["request"]["question"] for c in cases]
cands = [[x["text"] for x in c["request"]["candidates"]] for c in cases]
flat_pairs = [(q, a) for q, cs in zip(questions, cands) for a in cs]


def embed(texts):
    with torch.inference_mode():
        return torch.cat([retriever.embed(texts[i : i + 256], query=True) for i in range(0, len(texts), 256)])


qc_vec = embed([f"{q} {a}" for q, a in flat_pairs]).to(memory.keys.dtype)
q_vec = embed(questions).to(memory.keys.dtype)
g1_idx, _ = memory.search(qc_vec, 64)
g1_idx = g1_idx.tolist()


def sim(vec, nodes):
    if not nodes:
        return []
    return (memory.keys[torch.tensor(nodes, device=memory.keys.device)] @ vec).float().tolist()


pools, row = [], 0
t0 = time.time()
for j, (q, cs) in enumerate(zip(questions, cands)):
    qcon = mentioned(q)
    per = []
    for a in cs:
        vec = qc_vec[row]
        g1 = g1_idx[row]
        row += 1
        own = _stems(a) - _stems(q)
        ccon = mentioned(a) - qcon
        # G3 one hop
        one = {}
        for c in ccon:
            for s in by_concept.get(c, ())[:20000]:
                hit = (mentions[s] & qcon) - {c}
                if hit:
                    one[s] = max(one.get(s, 0.0), idf.get(c, 0) + sum(idf[h] for h in hit))
        one_top = sorted(one, key=lambda s: -one[s])[:200]
        if one_top:
            sims = sim(q_vec[j], one_top)
            one_top = [s for _, s in sorted(zip([one[s] + 5 * x for s, x in zip(one_top, sims)], one_top), reverse=True)][:32]
        # G3 two hops
        first = []
        for c in ccon:
            first += by_concept.get(c, ())[:5000]
        first = list(dict.fromkeys(first))
        if len(first) > 300:
            sims = sim(vec, first)
            first = [s for _, s in sorted(zip(sims, first), reverse=True)[:300]]
        pairs = {}
        if qcon:
            q_statements = set()
            for h in qcon:
                q_statements.update(by_concept.get(h, ())[:20000])
            for s1 in first:
                for b in mentions[s1] - ccon - qcon:
                    df = len(by_concept[b])
                    if df > 5000 or b in pairs:
                        continue
                    linked = [s2 for s2 in by_concept[b] if s2 in q_statements and s2 != s1]
                    if linked:
                        best = max(zip(sim(q_vec[j], linked), linked))[1]
                        pairs[b] = (s1, best, idf[b])
        two = sorted(pairs.values(), key=lambda t: -t[2])[:40]
        if two:
            s_first = sim(vec, [t[0] for t in two])
            s_second = sim(q_vec[j], [t[1] for t in two])
            two = [t for _, t in sorted(zip([a_ + b_ for a_, b_ in zip(s_first, s_second)], two), key=lambda z: -z[0])][:8]
        g3 = list(dict.fromkeys(one_top + [s for t in two for s in t[:2]]))
        per.append({"own": sorted(own), "g1": g1, "g3": g3, "g3_one": one_top, "g3_two": [list(t[:2]) for t in two]})
    pools.append(per)
print("G3 built", round(time.time() - t0), "s", flush=True)


def mentions_own(node, own):
    return bool(own & _stems(memory.concepts[node].text))


def focused(nodes, own):
    return [n for n in nodes if mentions_own(n, own)] + [n for n in nodes if not mentions_own(n, own)]


# G4 follow-up: second search from the focused top statement.
follow_queries, row = [], 0
for j, per in enumerate(pools):
    for i, entry in enumerate(per):
        s1 = focused(entry["g1"], set(entry["own"]))[0]
        follow_queries.append(f"{questions[j]} {cands[j][i]} {memory.concepts[s1].text}")
fv = embed(follow_queries).to(memory.keys.dtype)
g4_idx = memory.search(fv, 24)[0].tolist()
row = 0
for per in pools:
    for entry in per:
        seen = set(entry["g1"])
        entry["g4"] = [n for n in g4_idx[row] if n not in seen][:16]
        row += 1

# Judges. J0 similarity to "question candidate"; J2 reranker.
del retriever
torch.cuda.empty_cache()
tok = AutoTokenizer.from_pretrained(RERANKER, revision=REVISION)
rer = AutoModelForSequenceClassification.from_pretrained(RERANKER, revision=REVISION, torch_dtype=torch.float16).to("cuda").eval()
t0 = time.time()
row, n_pairs = 0, 0
for j, per in enumerate(pools):
    for i, entry in enumerate(per):
        pool = list(dict.fromkeys(entry["g1"][:32] + entry["g3"] + entry["g4"]))
        entry["pool"] = pool
        entry["j0"] = dict(zip(pool, sim(qc_vec[row], pool)))
        row += 1
        query = f"Question: {questions[j]} Answer: {cands[j][i]}"
        scores = []
        with torch.inference_mode():
            for s in range(0, len(pool), 128):
                chunk = pool[s : s + 128]
                batch = tok([query] * len(chunk), [memory.concepts[n].text for n in chunk], padding=True, truncation=True, max_length=192, return_tensors="pt").to("cuda")
                scores += rer(**batch).logits.view(-1).float().tolist()
        entry["j2"] = dict(zip(pool, scores))
        n_pairs += len(pool)
    if j % 200 == 0:
        print("reranked", j, "questions", round(time.time() - t0), "s", flush=True)
rerank_ms = 1000 * (time.time() - t0) / n_pairs


def rrf(orderings):
    score = defaultdict(float)
    for order in orderings:
        for r, n in enumerate(order):
            score[n] += 1 / (60 + r)
    return sorted(score, key=lambda n: -score[n])


CONFIGS = {}


def order_by(entry, nodes, judge):
    own = set(entry["own"])
    if judge == "J0":
        return sorted(nodes, key=lambda n: -entry["j0"][n])
    if judge == "J1":
        return focused(sorted(nodes, key=lambda n: -entry["j0"][n]), own)
    if judge == "J2":
        return sorted(nodes, key=lambda n: -entry["j2"][n])
    if judge == "J1+J2":  # mention first, reranker within each group
        return focused(sorted(nodes, key=lambda n: -entry["j2"][n]), own)
    if judge == "RRF(J0,J2)":
        return rrf([order_by(entry, nodes, "J0"), order_by(entry, nodes, "J2")])
    if judge == "J1+RRF(J0,J2)":
        return focused(rrf([order_by(entry, nodes, "J0"), order_by(entry, nodes, "J2")]), own)
    raise ValueError(judge)


GENERATORS = {
    "G1": lambda e: e["g1"][:32],
    "G3": lambda e: e["g3"],
    "G1+G3": lambda e: list(dict.fromkeys(e["g1"][:32] + e["g3"])),
    "G1+G3+G4": lambda e: e["pool"],
}
PLAN = [
    ("G1", "J0"), ("G1", "J1"), ("G1", "J2"), ("G1", "J1+J2"), ("G1", "RRF(J0,J2)"), ("G1", "J1+RRF(J0,J2)"),
    ("G3", "J1"), ("G3", "J2"),
    ("G1+G3", "J1"), ("G1+G3", "J1+J2"), ("G1+G3", "J1+RRF(J0,J2)"),
    ("G1+G3+G4", "J1"), ("G1+G3+G4", "J1+J2"), ("G1+G3+G4", "J1+RRF(J0,J2)"),
]
branches = {}
for gen, judge in PLAN:
    name = f"{gen}/{judge}"
    out = []
    for per in pools:
        rows = []
        for entry in per:
            nodes = GENERATORS[gen](entry)
            ordered = order_by(entry, nodes, judge)
            if gen == "G3" and len(ordered) < KMAX:  # paths may be few: fill from G1
                ordered += [n for n in order_by(entry, entry["g1"][:32], "J1") if n not in ordered]
            rows.append(ordered[:KMAX])
        out.append(rows)
    branches[name] = out

stats = {
    "questions": len(cases),
    "g3_one_hop_nonempty": sum(bool(e["g3_one"]) for per in pools for e in per) / sum(len(p) for p in pools),
    "g3_two_hop_nonempty": sum(bool(e["g3_two"]) for per in pools for e in per) / sum(len(p) for p in pools),
    "mean_pool": sum(len(e["pool"]) for per in pools for e in per) / sum(len(p) for p in pools),
    "rerank_ms_per_pair_gpu": rerank_ms,
    "seconds": round(time.time() - t_start),
}
print(json.dumps(stats), flush=True)
examples = []
for j in range(0, 200, 25):
    examples.append({
        "question": questions[j],
        "paths": {cands[j][i]: {"one_hop": [memory.concepts[n].text for n in e["g3_one"][:3]], "two_hop": [[memory.concepts[a].text, memory.concepts[b].text] for a, b in e["g3_two"][:2]]} for i, e in enumerate(pools[j])},
    })
(OUT / "branches.json").write_text(json.dumps({"example_ids": [c["example_id"] for c in cases], "configs": branches}))
(OUT / "build_stats.json").write_text(json.dumps({"stats": stats, "reranker": RERANKER, "revision": REVISION, "examples": examples}, indent=1))
