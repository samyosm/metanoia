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

SourceName = Literal[
    "race",
    "sciq",
    "obqa",
    "arc_easy",
    "arc_challenge",
    "casehold",
    "medmcqa",
    "qasc",
    "commonsenseqa",
    "boolq",
    "quality",
]

# Sources added for mix_v1 (PLAN_V3). Their test labels are public only for CaseHOLD; the
# others' validation split is the test split, and dev and calibration are carved from
# train by passage/question group, each with this fraction of the groups.
_CARVE: dict[str, float] = {
    "medmcqa": 0.01,
    "qasc": 0.06,
    "commonsenseqa": 0.06,
    "boolq": 0.06,
    "quality": 0.06,
}

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
    "casehold": {
        # Law: a citing passage with the holding masked, and five candidate holdings.
        "repo": "coastalcph/lex_glue",
        "revision": "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c",
        "files": {
            "train": "case_hold/train-00000-of-00001.parquet",
            "validation": "case_hold/validation-00000-of-00001.parquet",
            "test": "case_hold/test-00000-of-00001.parquet",
        },
    },
    "medmcqa": {
        # Medicine, no passage. Test labels are hidden, so validation is the test split.
        "repo": "openlifescienceai/medmcqa",
        "revision": "91c6572c454088bf71b679ad90aa8dffcd0d5868",
        "files": {
            "train": "data/train-00000-of-00001.parquet",
            "test": "data/validation-00000-of-00001.parquet",
        },
    },
    "qasc": {
        "repo": "allenai/qasc",
        "revision": "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070",
        "files": {
            "train": "data/train-00000-of-00001.parquet",
            "test": "data/validation-00000-of-00001.parquet",
        },
    },
    "commonsenseqa": {
        "repo": "tau/commonsense_qa",
        "revision": "94630fe30dad47192a8546eb75f094926d47e155",
        "files": {
            "train": "data/train-00000-of-00001.parquet",
            "test": "data/validation-00000-of-00001.parquet",
        },
    },
    "boolq": {
        "repo": "google/boolq",
        "revision": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
        "files": {
            "train": "data/train-00000-of-00001.parquet",
            "test": "data/validation-00000-of-00001.parquet",
        },
    },
    "quality": {
        # Long articles (median about 4,800 words): needs a long window to be answerable.
        "repo": "emozilla/quality",
        "revision": "7142df0407e16a99e5cda9188edc7db01e8c151a",
        "files": {
            "train": "data/train-00000-of-00001-0bffe0bf790d89eb.parquet",
            "test": "data/validation-00000-of-00001-77baeb9538209706.parquet",
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


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _casehold_items(split: str, rows: list[dict]) -> Iterator[RawItem]:
    """Law: the citing passage (holding masked) is the context; the five holdings are options."""
    for i, r in enumerate(rows):
        yield RawItem(
            "casehold",
            split,
            i,
            f"casehold:{_digest(r['context'])}",
            r["context"],
            "Which holding completes the citing text?",
            list(r["endings"]),
            int(r["label"]),
        )


def _medmcqa_items(split: str, rows: list[dict]) -> Iterator[RawItem]:
    """Medicine, no passage. The expert explanation, where present, is kept as a proof
    annotation only (it enables a with-passage ablation later); it is never in the request."""
    for i, r in enumerate(rows):
        answer = int(r["cop"])
        if answer not in range(4):
            continue
        exp = (r.get("exp") or "").strip()
        yield RawItem(
            "medmcqa",
            split,
            i,
            f"medmcqa:{r['id']}",
            "",
            r["question"],
            [r["opa"], r["opb"], r["opc"], r["opd"]],
            answer,
            core_fact=exp if exp and exp != "None" else None,
        )


def _qasc_items(split: str, rows: list[dict]) -> Iterator[RawItem]:
    """QASC: eight options, no passage. The two composed gold facts are proof annotations."""
    for i, r in enumerate(rows):
        choices = r["choices"]
        keys = [str(k) for k in choices["label"]]
        yield RawItem(
            "qasc",
            split,
            i,
            f"qasc:{r['id']}",
            "",
            r["question"],
            list(choices["text"]),
            keys.index(str(r["answerKey"]).strip()),
            core_fact=f"{r['fact1']} {r['fact2']}".strip() or None,
        )


def _commonsense_items(split: str, rows: list[dict]) -> Iterator[RawItem]:
    for i, r in enumerate(rows):
        choices = r["choices"]
        keys = [str(k) for k in choices["label"]]
        yield RawItem(
            "commonsenseqa",
            split,
            i,
            f"commonsenseqa:{r['id']}",
            "",
            r["question"],
            list(choices["text"]),
            keys.index(str(r["answerKey"]).strip()),
        )


def _boolq_items(split: str, rows: list[dict], seed: int) -> Iterator[RawItem]:
    """BoolQ as a two-option choice. The order of "yes" and "no" is seeded per example, so
    position does not reveal the answer. The option words are a known label-word shortcut
    risk (LAYA_STUDY.md section 2.3); report BoolQ per source."""
    for i, r in enumerate(rows):
        options = ["yes", "no"]
        order = [0, 1]
        random.Random(f"{seed}:boolq:{split}:{i}").shuffle(order)
        question = r["question"].strip()
        question = question[:1].upper() + question[1:]
        if not question.endswith("?"):
            question += "?"
        gold = 0 if r["answer"] in (True, "True", "true") else 1
        yield RawItem(
            "boolq",
            split,
            i,
            f"boolq:{_digest(r['passage'])}",
            r["passage"],
            question,
            [options[j] for j in order],
            order.index(gold),
        )


def _quality_items(split: str, rows: list[dict]) -> Iterator[RawItem]:
    for i, r in enumerate(rows):
        options = r["options"]
        if isinstance(options, str):
            options = ast.literal_eval(options)
        yield RawItem(
            "quality",
            split,
            i,
            f"quality:{_digest(r['article'])}",
            r["article"],
            r["question"].strip(),
            list(options),
            int(r["answer"]),
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
        fraction = _CARVE.get(item.source)
        if fraction is None:
            return "train"
        u = (
            int.from_bytes(
                hashlib.sha256(f"{config.seed}:carve:{item.group}".encode()).digest()[:8], "big"
            )
            / 2**64
        )
        return "dev" if u < fraction else "calibration" if u < 2 * fraction else "train"
    if item.source_split == "test":
        return "test"
    return _assign_validation(item, config)


def _assign_validation(item: RawItem, config: McqaConfig) -> str:
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
        files = _SOURCES[source]["files"]
        assert isinstance(files, dict)
        for split in ("train", "validation", "test"):
            if split not in files:
                continue
            rows = _load_rows(source, split)
            if source == "race":
                items.extend(_race_items(split, rows))
            elif source == "sciq":
                items.extend(_sciq_items(split, rows, config.seed))
            elif source == "casehold":
                items.extend(_casehold_items(split, rows))
            elif source == "medmcqa":
                items.extend(_medmcqa_items(split, rows))
            elif source == "qasc":
                items.extend(_qasc_items(split, rows))
            elif source == "commonsenseqa":
                items.extend(_commonsense_items(split, rows))
            elif source == "boolq":
                items.extend(_boolq_items(split, rows, config.seed))
            elif source == "quality":
                items.extend(_quality_items(split, rows))
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
