"""Step 0 of PLAN_V3: shortcut and length statistics on validation splits of candidates.

Downloads only small validation files. For each source it reports sizes, passage and
option lengths (words), answer-position balance, and the accuracy of three trivial
rules that need no learning: the longest option, the option with the most word overlap
with the passage, and the option with the most overlap with the question. These are the
lexical fixtures of PLAN §8.3. Prints one JSON line per source.
"""

import json
import re
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

_W = re.compile(r"\w+")


def words(s: str) -> list[str]:
    return [w.lower() for w in _W.findall(s)]


def _read(path: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(**row) for row in pq.read_table(path).to_pylist()]


def load(name: str) -> list[dict]:
    """Return rows as {passage, question, options, gold}."""
    rows: list[dict] = []
    if name == "race":
        df = _read(
            hf_hub_download(
                "ehovy/race",
                "all/validation-00000-of-00001.parquet",
                repo_type="dataset",
                revision="2fec9fd81f1dc971569a9b729c43f2f0e6436637",
            )
        )
        for r in df:
            rows.append(
                dict(
                    passage=r.article,
                    question=r.question,
                    options=list(r.options),
                    gold="ABCD".index(r.answer),
                )
            )
    elif name == "casehold":
        df = _read(
            hf_hub_download(
                "coastalcph/lex_glue",
                "case_hold/validation-00000-of-00001.parquet",
                repo_type="dataset",
                revision="c23fdff1a6bf74e0e1a71cb86f1e781d37da888c",
            )
        )
        for r in df:
            rows.append(
                dict(passage=r.context, question="", options=list(r.endings), gold=int(r.label))
            )
    elif name == "quality":
        df = _read(
            hf_hub_download(
                "emozilla/quality",
                "data/validation-00000-of-00001-77baeb9538209706.parquet",
                repo_type="dataset",
                revision="7142df0407e16a99e5cda9188edc7db01e8c151a",
            )
        )
        for r in df:
            rows.append(
                dict(
                    passage=r.article,
                    question=r.question,
                    options=list(r.options),
                    gold=int(r.answer),
                )
            )
    return rows


def stats(name: str) -> dict:
    rows = load(name)
    pw = [len(words(r["passage"])) for r in rows]
    ow = [np.mean([len(words(o)) for o in r["options"]]) for r in rows]
    nopt = Counter(len(r["options"]) for r in rows)
    pos = Counter(r["gold"] for r in rows)
    longest = overlap_p = overlap_q = 0
    dup = 0
    for r in rows:
        opts = r["options"]
        if len({o.strip().lower() for o in opts}) < len(opts):
            dup += 1
        lens = [len(words(o)) for o in opts]
        longest += int(int(np.argmax(lens)) == r["gold"])
        pset = set(words(r["passage"]))
        qset = set(words(r["question"]))
        sp = [len(set(words(o)) & pset) / max(1, len(set(words(o)))) for o in opts]
        sq = [len(set(words(o)) & qset) / max(1, len(set(words(o)))) for o in opts]
        overlap_p += int(int(np.argmax(sp)) == r["gold"])
        overlap_q += int(int(np.argmax(sq)) == r["gold"])
    n = len(rows)
    return {
        "source": name,
        "n_val": n,
        "n_options": dict(nopt),
        "chance": round(1 / max(nopt), 3),
        "passage_words_p10_p50_p90_max": [int(x) for x in np.percentile(pw, [10, 50, 90, 100])],
        "option_words_mean": round(float(np.mean(ow)), 1),
        "gold_position": {int(k): round(v / n, 3) for k, v in sorted(pos.items())},
        "dup_option_rows": dup,
        "acc_longest_option": round(longest / n, 3),
        "acc_overlap_with_passage": round(overlap_p / n, 3),
        "acc_overlap_with_question": round(overlap_q / n, 3),
    }


if __name__ == "__main__":
    for s in ["race", "casehold", "quality"]:
        print(json.dumps(stats(s)), flush=True)
