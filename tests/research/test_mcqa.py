"""Conversion of multiple-choice rows: split policy, skipping, and label separation."""

from metanoia_research.data.mcqa import McqaConfig, RawItem, _sciq_items, build_cases

CONFIG = McqaConfig(dataset_id="t", seed=1, sources=("race",))


def item(split: str, row: int, options: list[str], answer: int = 1) -> RawItem:
    return RawItem("race", split, row, f"race:g{row % 3}", "ctx", "q?", options, answer)


def test_splits_labels_and_skips() -> None:
    items = [
        item("train", 0, ["a", "b", "c", "d"]),
        item("validation", 1, ["a", "b", "c", "d"]),
        item("test", 2, ["a", "b", "c", "d"]),
        item("train", 3, ["a", "A", "c", "d"]),
        item("train", 4, ["a", " ", "c", "d"]),
    ]
    cases, labels, skipped = build_cases(CONFIG, items)
    assert [c.split for c in cases][0::2] == ["train", "test"]
    assert cases[1].split in {"dev", "calibration"}
    assert skipped == {"empty_option": 0 + 1, "duplicate_option": 1}
    assert [label.label_id for label in labels] == ["b", "b", "b"]
    assert "label" not in cases[0].request.model_dump_json()


def test_validation_split_is_grouped() -> None:
    items = [item("validation", r, ["a", "b"]) for r in range(30)]
    cases, _, _ = build_cases(CONFIG, items)
    by_group: dict[str, set[str]] = {}
    for case in cases:
        by_group.setdefault(case.group_id, set()).add(case.split)
    assert all(len(splits) == 1 for splits in by_group.values())


def test_sciq_answer_position_is_shuffled_and_deterministic() -> None:
    rows = [
        {
            "question": f"q{i}",
            "correct_answer": "right",
            "distractor1": "x",
            "distractor2": "y",
            "distractor3": "z",
            "support": "",
        }
        for i in range(40)
    ]
    first = list(_sciq_items("train", rows, seed=3))
    again = list(_sciq_items("train", rows, seed=3))
    assert [i.options for i in first] == [i.options for i in again]
    assert all(i.options[i.answer] == "right" for i in first)
    assert len({i.answer for i in first}) == 4
