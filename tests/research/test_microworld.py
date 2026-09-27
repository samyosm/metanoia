import re
from collections import Counter

from metanoia_research.data.microworld import MicroworldConfig, generate, parse_context
from metanoia_research.oracles import MicroworldOracle

CONFIG = MicroworldConfig(
    dataset_id="t",
    seed=3,
    n_worlds=40,
    examples_per_world=4,
    depths=(1, 2, 3),
    heldout_depths=(4,),
    conjunction_probability=0.5,
    splits={"train": 0.5, "dev": 0.1, "calibration": 0.2, "test": 0.2},
)


def test_generation_is_deterministic() -> None:
    assert generate(CONFIG) == generate(CONFIG)


def test_splits_are_by_world_and_heldout_depth_is_test_only() -> None:
    cases, labels, _ = generate(CONFIG)
    world_splits: dict[str, set[str]] = {}
    for case in cases:
        world_splits.setdefault(case.group_id, set()).add(case.split)
    assert all(len(s) == 1 for s in world_splits.values())
    assert {label.split for label in labels if label.annotations["depth"] == 4} <= {"test"}
    assert Counter(s for (s,) in world_splits.values()) == {
        "train": 20,
        "dev": 4,
        "calibration": 8,
        "test": 8,
    }


def test_oracle_recovers_every_label_from_text_alone() -> None:
    cases, labels, _ = generate(CONFIG)
    oracle = MicroworldOracle()
    for case, label in zip(cases, labels, strict=True):
        result = oracle.decide(case.request)
        best = max(range(len(result.probabilities)), key=result.probabilities.__getitem__)
        assert result.candidate_ids[best] == label.label_id


def test_candidate_mentions_carry_no_label_cue() -> None:
    cases, _, _ = generate(CONFIG)
    for case in cases:
        counts = {
            len(re.findall(rf"\b{c.text}\b", case.request.context)) for c in case.request.candidates
        }
        assert len(counts) == 1, case.request.context


def test_every_fact_predicate_appears_in_a_rule() -> None:
    cases, _, _ = generate(CONFIG)
    for case in cases:
        world = parse_context(case.request.context)
        in_rules = {p for r in world.rules for p in (*r.premises, r.conclusion)}
        assert {p for _, p in world.facts} <= in_rules


def test_proofs_have_declared_depth() -> None:
    _, labels, proofs = generate(CONFIG)
    for label, proof in zip(labels, proofs, strict=True):
        assert proof["example_id"] == label.example_id
        assert len(proof["steps"]) >= label.annotations["depth"]
