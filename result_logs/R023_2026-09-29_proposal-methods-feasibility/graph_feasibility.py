"""Plan verification (CPU only): can generics_v0 become a concept graph, and do paths
from a candidate's concepts to the question's concepts separate right from wrong answers?

1. Mention index: for each statement, the memory concept terms it mentions (1-3-word
   spans, last word made singular). Timed; degree statistics of concepts.
2. Path existence on knowledge_v0 dev + calibration: for each candidate, is there a
   statement mentioning both one of its concepts and one of the question's concepts
   (1 hop), or a bridge concept linking them through two statements (2 hops)?
   Compared between gold and distractor candidates, with hub concepts capped.
Run from metanoia/: .venv/bin/python <this> <out_dir>
"""

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from metanoia.reasoning.branch import _singular

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
WORD = re.compile(r"[a-z]+")
STOP = set(
    "a an the of to in on at for from by with and or but not no is are was were be been it its "
    "this that these those as than then so if when which what who how why do does did can will "
    "would should could may might have has had they them their there here some any all most more "
    "less very much many each other such only also into out up down over under about after before "
    "during between through one two three use used make made get thing way part kind type".split()
)

t0 = time.time()
rows = [json.loads(line) for line in open("artifacts/memories/generics_v0/concepts.jsonl")]
terms = {r["name"] for r in rows}
print("loaded", len(rows), "statements,", len(terms), "terms", round(time.time() - t0, 1), "s", flush=True)


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


t0 = time.time()
mentions = [mentioned(r["text"]) | {r["name"]} for r in rows]
index_seconds = time.time() - t0
by_concept: dict[str, list[int]] = defaultdict(list)
for i, ms in enumerate(mentions):
    for m in ms:
        by_concept[m].append(i)
degrees = sorted((len(v) for v in by_concept.values()), reverse=True)
per_statement = sum(len(m) for m in mentions) / len(mentions)
stats = {
    "statements": len(rows),
    "terms": len(terms),
    "index_seconds": round(index_seconds, 1),
    "concepts_per_statement": round(per_statement, 2),
    "top_concepts": Counter({k: len(v) for k, v in by_concept.items()}).most_common(15),
    "degree_percentiles": {p: degrees[int(len(degrees) * p / 100)] for p in (1, 10, 50, 90)},
}
print(json.dumps(stats), flush=True)

# Path existence. Hubs (concepts in more than HUB statements) do not count as links.
HUB = 2000
D = Path("artifacts/datasets/knowledge_v0")
import pyarrow.parquet as pq  # noqa: E402

labels = {r["example_id"]: r for r in pq.read_table(D / "labels.parquet").to_pylist()}
cases = [json.loads(line) for line in (D / "cases.jsonl").open()]
cases = [c for c in cases if c["split"] in ("dev", "calibration")]


def concepts_of(text: str) -> set[str]:
    return {m for m in mentioned(text) if len(by_concept.get(m, ())) <= HUB}


def one_hop(cand: set[str], quest: set[str]) -> int:
    """Statements mentioning a candidate concept and a question concept."""
    if not cand or not quest:
        return 0
    qs = set()
    for q in quest:
        qs.update(by_concept.get(q, ()))
    return sum(1 for c in cand for s in by_concept.get(c, ()) if s in qs)


def two_hop(cand: set[str], quest: set[str]) -> int:
    """Bridge concepts b with a statement (cand, b) and a statement (b, quest)."""
    if not cand or not quest:
        return 0
    near_c, near_q = set(), set()
    for c in cand:
        for s in by_concept.get(c, ())[:HUB]:
            near_c.update(m for m in mentions[s] if len(by_concept[m]) <= HUB)
    for q in quest:
        for s in by_concept.get(q, ())[:HUB]:
            near_q.update(m for m in mentions[s] if len(by_concept[m]) <= HUB)
    return len((near_c & near_q) - cand - quest)


t0 = time.time()
table = defaultdict(lambda: defaultdict(list))
for case in cases:
    request = case["request"]
    lab = labels[case["example_id"]]
    quest = concepts_of(request["question"])
    for cand in request["candidates"]:
        own = concepts_of(cand["text"]) - quest
        role = "gold" if cand["id"] == lab["label_id"] else "distractor"
        for src in (lab["source"], "all"):
            row = table[src][role]
            row.append((bool(own), one_hop(own, quest), two_hop(own, quest)))
path_seconds = time.time() - t0
paths = {}
for src, roles in table.items():
    paths[src] = {}
    for role, values in roles.items():
        n = len(values)
        paths[src][role] = {
            "n": n,
            "has_concept": sum(v[0] for v in values) / n,
            "one_hop_any": sum(v[1] > 0 for v in values) / n,
            "one_hop_mean": sum(v[1] for v in values) / n,
            "two_hop_any": sum(v[2] > 0 for v in values) / n,
        }
    # Within-question ranking: does the gold candidate have the most one-hop links?
print(json.dumps(paths, indent=1), flush=True)
ranked = Counter()
for case in cases:
    request = case["request"]
    lab = labels[case["example_id"]]
    quest = concepts_of(request["question"])
    counts = [one_hop(concepts_of(c["text"]) - quest, quest) for c in request["candidates"]]
    gold = [c["id"] for c in request["candidates"]].index(lab["label_id"])
    if max(counts) == 0:
        ranked["no_links"] += 1
    elif counts[gold] == max(counts) and counts.count(max(counts)) == 1:
        ranked["gold_most_links_alone"] += 1
    elif counts[gold] == max(counts):
        ranked["gold_tied_most"] += 1
    else:
        ranked["distractor_more"] += 1
print(dict(ranked), "path seconds", round(path_seconds, 1))
(OUT / "graph_feasibility.json").write_text(
    json.dumps({"stats": stats, "hub_cap": HUB, "paths": paths, "one_hop_ranking": dict(ranked)}, indent=1)
)
