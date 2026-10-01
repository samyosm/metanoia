"""Audit a dataset bundle: counts per source and split, option counts, split leakage, and
trivial-rule accuracy (longest option, overlap with the passage) per source on test.

Usage: audit_dataset.py DATASET_DIR. Prints one JSON object.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from metanoia_research.data.bundle import read_cases, read_labels

_W = re.compile(r"\w+")


def words(s: str) -> set[str]:
    return {w.lower() for w in _W.findall(s)}


def norm(case) -> str:
    r = case.request
    return " ".join([r.question.lower(), *sorted(c.text.lower() for c in r.candidates)])


def main(path: str) -> None:
    manifest, cases = read_cases(Path(path))
    labels = {label.example_id: label for label in read_labels(Path(path))}
    counts: dict[str, Counter] = defaultdict(Counter)
    groups: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    texts: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    longest, overlap, total = Counter(), Counter(), Counter()
    options: dict[str, Counter] = defaultdict(Counter)
    passage_words: dict[str, list[int]] = defaultdict(list)
    for case in cases:
        label = labels[case.example_id]
        source = label.annotations["source"]
        counts[source][case.split] += 1
        groups[source][case.split].add(case.group_id)
        texts[source][case.split].add(norm(case))
        options[source][len(case.request.candidates)] += 1
        passage_words[source].append(len(case.request.context.split()))
        if case.split == "test":
            idx = case.request.candidate_ids.index(label.label_id)
            lens = [len(c.text.split()) for c in case.request.candidates]
            pw = words(case.request.context)
            ov = [len(words(c.text) & pw) / max(1, len(words(c.text))) for c in case.request.candidates]
            total[source] += 1
            longest[source] += int(max(range(len(lens)), key=lens.__getitem__) == idx)
            overlap[source] += int(max(range(len(ov)), key=ov.__getitem__) == idx)
    out = {"dataset": manifest.dataset_id, "splits": manifest.splits, "sources": {}}
    for source in sorted(counts):
        train_texts = texts[source]["train"]
        test_texts = texts[source]["test"]
        dup = len(test_texts & train_texts)
        out["sources"][source] = {
            "counts": dict(counts[source]),
            "options": dict(options[source]),
            "median_passage_words": sorted(passage_words[source])[len(passage_words[source]) // 2],
            "group_overlap_train_test": len(groups[source]["train"] & groups[source]["test"]),
            "group_overlap_train_dev": len(groups[source]["train"] & groups[source]["dev"]),
            "group_overlap_dev_calibration": len(groups[source]["dev"] & groups[source]["calibration"]),
            "test_question_and_options_also_in_train": dup,
            "test_dup_fraction": round(dup / max(1, len(test_texts)), 4),
            "chance": round(1 / max(options[source]), 3),
            "test_acc_longest_option": round(longest[source] / max(1, total[source]), 3),
            "test_acc_overlap_with_passage": round(overlap[source] / max(1, total[source]), 3),
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
