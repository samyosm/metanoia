"""mix_v1 builders and mixed-source epoch sampling (PLAN_V3 A2)."""

import random
from collections import Counter

from metanoia_research.data import mcqa
from metanoia_research.training.config import TrainingConfig
from metanoia_research.training.loop import epoch_order


def _config(**kw):
    return TrainingConfig(name="t", model_config_path="m.toml", dataset="d", epochs=1, **kw)


def test_plain_epoch_is_a_shuffled_pass_and_unchanged_by_the_new_code():
    order = epoch_order(10, None, _config(), random.Random(0))
    reference = list(range(10))
    random.Random(0).shuffle(reference)
    assert order == reference


def test_mix_alpha_zero_gives_each_source_the_same_share():
    sources = ["a"] * 90 + ["b"] * 9 + ["c"] * 1
    order = epoch_order(100, sources, _config(mix_alpha=0.0, epoch_examples=300), random.Random(0))
    counts = Counter(sources[i] for i in order)
    assert counts == {"a": 100, "b": 100, "c": 100}


def test_mix_alpha_one_is_proportional_and_small_sources_repeat():
    sources = ["a"] * 80 + ["b"] * 20
    order = epoch_order(100, sources, _config(mix_alpha=1.0, epoch_examples=200), random.Random(0))
    counts = Counter(sources[i] for i in order)
    assert counts == {"a": 160, "b": 40}
    assert len({i for i in order if sources[i] == "b"}) == 20  # all 20 reused, none invented


def test_casehold_and_boolq_items():
    rows = [{"context": "Text <HOLDING> more.", "endings": ["x", "y", "z", "w", "v"], "label": 3}]
    (item,) = mcqa._casehold_items("train", rows)
    assert item.options[item.answer] == "w" and item.context.startswith("Text")
    rows = [{"question": "is it so", "answer": True, "passage": "P"}]
    (b,) = mcqa._boolq_items("test", rows, seed=1)
    assert b.question == "Is it so?" and sorted(b.options) == ["no", "yes"]
    assert b.options[b.answer] == "yes"


def test_medmcqa_skips_hidden_labels_and_keeps_explanation_as_proof_only():
    base = {"id": "1", "question": "q", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "exp": "E"}
    items = list(
        mcqa._medmcqa_items(
            "train",
            [{**base, "cop": "-1"}, {**base, "cop": "2"}, {**base, "cop": "1", "exp": "None"}],
        )
    )
    assert [i.answer for i in items] == [2, 1]
    assert items[0].core_fact == "E" and items[1].core_fact is None
    assert items[0].context == ""


def test_new_sources_without_labelled_test_carve_dev_and_calibration_from_train():
    config = mcqa.McqaConfig(dataset_id="t", seed=1, sources=("qasc",))
    items = [
        mcqa.RawItem("qasc", "train", i, f"qasc:{i}", "", "q", ["a", "b"], 0) for i in range(5000)
    ]
    splits = Counter(mcqa._assign_split(i, config) for i in items)
    assert 200 < splits["dev"] < 400 and 200 < splits["calibration"] < 400
    assert splits["train"] > 4000
