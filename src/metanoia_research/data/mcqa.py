"""Realistic exclusive-choice data: human-written multiple-choice reading comprehension.

Sources are pinned Hugging Face dataset revisions, read as Parquet files (no ``datasets``
dependency). Each row becomes a :class:`DecisionRequest` with the passage as context,
the question, and the options as candidates; the gold option is stored only in labels.

Split policy: each source's train split -> ``train``; its validation split is divided by
group (passage) into ``dev`` and ``calibration``; its test split -> ``test``. Groups are
passages, so questions about one passage never straddle two partitions.

Candidate policy: candidate IDs are ``a``, ``b``, ... in the presented order. SciQ lists
its correct answer separately, so its options are shuffled with a per-example seeded
RNG; otherwise the position would reveal the answer. Rows with empty or duplicate
(case-insensitive) option texts are skipped and counted, not repaired.
"""

import ast
import hashlib
import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from metanoia_contracts import Candidate, DecisionRequest
from metanoia_contracts.artifacts import CaseRecord, LabelRecord
from metanoia_contracts.base import StrictModel

GENERATOR_REVISION = "mcqa/1"

SourceName = Literal["race", "sciq", "obqa", "arc_easy", "arc_challenge"]

# Pinned dataset revisions and Parquet files per split.
_SOURCES: dict[str, dict[str, object]] = {
    "race": {
        "repo": "ehovy/race",
        "revision": "2fec9fd81f1dc971569a9b729c43f2f0e6436637",
        "files": {
            "train": "all/train-00000-of-00001.parquet",
            "validation": "all/validation-00000-of-00001.parquet",
            "test": "all/test-00000-of-00001.parquet",
        },
    },
    "obqa": {
        # The "additional" config adds fact1, the gold open-book core fact per question.
        "repo": "allenai/openbookqa",
        "revision": "388097ea7776314e93a529163e0fea805b8a6454",
        "files": {
            "train": "additional/train-00000-of-00001.parquet",
            "validation": "additional/validation-00000-of-00001.parquet",
            "test": "additional/test-00000-of-00001.parquet",
        },
    },
    "arc_easy": {
        "repo": "allenai/ai2_arc",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "files": {
            "train": "ARC-Easy/train-00000-of-00001.parquet",
            "validation": "ARC-Easy/validation-00000-of-00001.parquet",
            "test": "ARC-Easy/test-00000-of-00001.parquet",
        },
    },
    "arc_challenge": {
        "repo": "allenai/ai2_arc",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "files": {
            "train": "ARC-Challenge/train-00000-of-00001.parquet",
            "validation": "ARC-Challenge/validation-00000-of-00001.parquet",
            "test": "ARC-Challenge/test-00000-of-00001.parquet",
        },
    },
    "sciq": {
        "repo": "allenai/sciq",
        "revision": "2c94ad3e1aafab77146f384e23536f97a4849815",
        "files": {
            "train": "data/train-00000-of-00001.parquet",
            "validation": "data/validation-00000-of-00001.parquet",
            "test": "data/test-00000-of-00001.parquet",
        },
    },
}


class McqaConfig(StrictModel):
    generator: Literal["mcqa"] = "mcqa"
    dataset_id: str
    seed: int
    sources: tuple[SourceName, ...] = Field(min_length=1)
    # Fraction of validation groups assigned to ``dev``; the rest go to ``calibration``.
    dev_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)


@dataclass(frozen=True)
class RawItem:
    source: str
    source_split: str
    row: int
    group: str
    context: str
    question: str
    options: list[str]
    answer: int
    # Benchmark-only gold evidence (OpenBookQA fact1). Written to proofs.jsonl; never
    # part of the request.
    core_fact: str | None = None


def _race_items(split: str, rows: list[dict]) -> Iterator[RawItem]:
    for i, r in enumerate(rows):
        options = r["options"]
        if isinstance(options, str):
            options = ast.literal_eval(options)
        yield RawItem(
            "race",
            split,
            i,
            f"race:{r['example_id']}",
            r["article"],
            r["question"],
            list(options),
            "ABCDEFGH".index(r["answer"]),
        )


def _sciq_items(split: str, rows: list[dict], seed: int) -> Iterator[RawItem]:
    for i, r in enumerate(rows):
        options = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        order = list(range(len(options)))
        random.Random(f"{seed}:sciq:{split}:{i}").shuffle(order)
        support = r["support"] or ""
        group = hashlib.sha256(support.encode()).hexdigest()[:16] if support else f"q{split}{i}"
        yield RawItem(
            "sciq",
            split,
            i,
            f"sciq:{group}",
            support,
            r["question"],
            [options[j] for j in order],
            order.index(0),
        )


def _choice_items(source: str, split: str, rows: list[dict]) -> Iterator[RawItem]:
    """OpenBookQA and ARC: a question, labelled choices and an answer key; no passage.
    Each question is its own group (they share no passage)."""
    for i, r in enumerate(rows):
        choices = r["choices"]
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        keys = [str(k) for k in choices["label"]]
        answer = str(r["answerKey"]).strip()
        if answer not in keys:
            raise ValueError(f"{source} {split} row {i}: answer {answer!r} not in {keys}")
        yield RawItem(
            source,
            split,
            i,
            f"{source}:{r['id']}",
            "",
            r["question_stem"] if source == "obqa" else r["question"],
            list(choices["text"]),
            keys.index(answer),
            core_fact=r.get("fact1") if source == "obqa" else None,
        )


def _load_rows(source: str, split: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    spec = _SOURCES[source]
    files = spec["files"]
    assert isinstance(files, dict)
    path = hf_hub_download(
        str(spec["repo"]), files[split], repo_type="dataset", revision=str(spec["revision"])
    )
    return pq.read_table(path).to_pylist()


def _assign_split(item: RawItem, config: McqaConfig) -> str:
    if item.source_split == "train":
        return "train"
    if item.source_split == "test":
        return "test"
    digest = hashlib.sha256(f"{config.seed}:{item.group}".encode()).digest()
    return (
        "dev" if int.from_bytes(digest[:8], "big") / 2**64 < config.dev_fraction else "calibration"
    )


def build_cases(
    config: McqaConfig, items: list[RawItem]
) -> tuple[list[CaseRecord], list[LabelRecord], dict[str, int]]:
    cases, labels = [], []
    skipped: dict[str, int] = {"empty_option": 0, "duplicate_option": 0}
    for item in items:
        texts = [str(o).strip() for o in item.options]
        if any(not t for t in texts):
            skipped["empty_option"] += 1
            continue
        if len({t.lower() for t in texts}) != len(texts):
            skipped["duplicate_option"] += 1
            continue
        ids = [chr(ord("a") + k) for k in range(len(texts))]
        split = _assign_split(item, config)
        example_id = _example_id(config, item)
        request = DecisionRequest(
            context=item.context.strip(),
            question=item.question.strip(),
            candidates=tuple(Candidate(id=i, text=t) for i, t in zip(ids, texts, strict=True)),
        )
        cases.append(
            CaseRecord(example_id=example_id, group_id=item.group, split=split, request=request)
        )
        labels.append(
            LabelRecord(
                example_id=example_id,
                group_id=item.group,
                split=split,
                label_id=ids[item.answer],
                annotations={"source": item.source, "n_candidates": len(ids)},
            )
        )
    return cases, labels, skipped


def generate(
    config: McqaConfig,
) -> tuple[list[CaseRecord], list[LabelRecord], dict[str, int], list[dict]]:
    """Download (cached) pinned sources and convert them. Deterministic given the config.
    Returns cases, labels, skip counts and proofs (gold core facts, where the source has them)."""
    items: list[RawItem] = []
    for source in config.sources:
        for split in ("train", "validation", "test"):
            rows = _load_rows(source, split)
            if source == "race":
                items.extend(_race_items(split, rows))
            elif source == "sciq":
                items.extend(_sciq_items(split, rows, config.seed))
            else:
                items.extend(_choice_items(source, split, rows))
    cases, labels, skipped = build_cases(config, items)
    facts = {_example_id(config, i): i.core_fact for i in items if i.core_fact}
    proofs = [
        {"example_id": c.example_id, "core_fact": facts[c.example_id]}
        for c in cases
        if c.example_id in facts
    ]
    return cases, labels, skipped, proofs


def _example_id(config: McqaConfig, item: RawItem) -> str:
    return f"{config.dataset_id}-{item.source}-{item.source_split}-{item.row:06d}"


def source_revisions(config: McqaConfig) -> dict[str, object]:
    return {s: {k: _SOURCES[s][k] for k in ("repo", "revision", "files")} for s in config.sources}
