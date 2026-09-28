"""Access recall of a concept memory (PLAN §10.1 gate; foundations §5.2).

For each question with a gold core fact (OpenBookQA ``fact1``, never a model input), ask
whether concepts retrieved from ``question + candidate`` queries cover the fact's
content words. It reports "all" content words and "novel" ones (absent from the question
and candidates: knowledge the model does not already see). A random-concept draw of
equal size is the chance reference. Retrieval quality is not usefulness: EXPAND still has
to select concepts in context.
"""

import json
import random
import re
from pathlib import Path
from typing import Any

from metanoia.memory.encoder import RetrievalEncoder
from metanoia.memory.store import ConceptMemory
from metanoia_research.data.bundle import read_cases, read_manifest

_WORD = re.compile(r"[a-z]+")
# A small closed-class list; content words are what remains.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "from",
        "by",
        "with",
        "and",
        "or",
        "but",
        "not",
        "no",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "as",
        "than",
        "then",
        "so",
        "if",
        "when",
        "which",
        "who",
        "whom",
        "what",
        "how",
        "why",
        "where",
        "can",
        "could",
        "will",
        "would",
        "may",
        "might",
        "must",
        "should",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "more",
        "most",
        "less",
        "least",
        "very",
        "also",
        "such",
        "into",
        "over",
        "under",
        "about",
        "between",
        "through",
        "each",
        "other",
        "some",
        "any",
        "all",
        "many",
        "much",
        "one",
        "two",
        "there",
        "their",
        "they",
        "them",
        "he",
        "she",
        "his",
        "her",
        "we",
        "our",
        "you",
        "your",
        "i",
        "my",
        "me",
        "us",
    ]
)


def _lemmas(text: str, morphy: Any) -> set[str]:
    out = set()
    for word in _WORD.findall(text.lower()):
        if len(word) < 3 or word in STOPWORDS:
            continue
        out.add(morphy(word) or word)
    return out


def access_recall(
    dataset_dir: Path,
    memory: ConceptMemory,
    encoder: RetrievalEncoder,
    ks: tuple[int, ...],
    splits: tuple[str, ...],
    seed: int = 0,
) -> dict[str, Any]:
    import nltk

    nltk.data.path.insert(0, str(Path(memory.manifest.source["config"]["nltk_data"]).resolve()))
    from nltk.corpus import wordnet

    def morphy(word: str) -> str | None:
        for pos in ("n", "v", "a"):
            if lemma := wordnet.morphy(word, pos):
                return str(lemma)
        return None

    manifest = read_manifest(dataset_dir)
    if manifest.proofs is None:
        raise ValueError(f"{dataset_dir} has no proofs (gold core facts)")
    facts = {
        row["example_id"]: row["core_fact"]
        for row in map(json.loads, (dataset_dir / manifest.proofs.path).read_text().splitlines())
    }
    _, cases = read_cases(dataset_dir)
    cases = [c for c in cases if c.example_id in facts and c.split in splits]
    concept_words = [
        {morphy(w) or w for name in c.names for w in _WORD.findall(name) if w not in STOPWORDS}
        for c in memory.concepts
    ]
    queries = [f"{c.request.question} {cand.text}" for c in cases for cand in c.request.candidates]
    vectors = encoder.embed(queries, query=True)
    kmax = max(ks)
    indices, _ = memory.search(vectors, kmax)
    rng = random.Random(seed)
    totals: dict[str, dict[str, float]] = {}
    counted = {"all": 0, "novel": 0}
    row = 0
    for case in cases:
        n = len(case.request.candidates)
        per_candidate = indices[row : row + n].tolist()
        row += n
        fact = _lemmas(facts[case.example_id], morphy)
        seen = _lemmas(
            case.request.question + " " + " ".join(c.text for c in case.request.candidates), morphy
        )
        targets = {"all": fact, "novel": fact - seen}
        for kind, target in targets.items():
            if not target:
                continue
            counted[kind] += 1
            for k in ks:
                retrieved = set().union(
                    *(concept_words[i] for ids in per_candidate for i in ids[:k])
                )
                random_ids = rng.sample(range(len(memory)), k * n)
                chance = set().union(*(concept_words[i] for i in random_ids))
                bucket = totals.setdefault(f"{kind}@{k}", {"retrieved": 0.0, "random": 0.0})
                bucket["retrieved"] += len(target & retrieved) / len(target)
                bucket["random"] += len(target & chance) / len(target)
    recall = {
        key: {name: value / counted[key.split("@")[0]] for name, value in bucket.items()}
        for key, bucket in totals.items()
    }
    return {
        "memory_id": memory.memory_id,
        "encoder": encoder.identity(),
        "dataset_id": manifest.dataset_id,
        "splits": list(splits),
        "questions": len(cases),
        "questions_with_novel_fact_words": counted["novel"],
        "ks_per_candidate": list(ks),
        "recall": recall,
    }
