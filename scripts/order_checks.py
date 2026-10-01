"""Option-order robustness checks for a marker-encoding bundle (decision 0015, Gate A1).

1. Identical-option check (after Laya's presentation checks): the same text in every slot;
   a position-free model gives equal scores, so any spread is pure position bias.
2. Permutation check: each question is scored with its options in every cyclic rotation
   (and several random orders). Reports how often the predicted option changes with the
   order, accuracy with the original order, with rotation averaging, and the worst order.
Usage: order_checks.py BUNDLE DATASET SPLIT N [rotations]
"""

import json
import random
import sys
from pathlib import Path

import torch

from metanoia.api import load_predictor
from metanoia_contracts import Candidate, DecisionRequest
from metanoia_research.training.loop import load_examples


def scores(model, request: DecisionRequest, order: list[int]) -> list[float]:
    """Raw logits of the original options when presented in ``order``."""
    permuted = request.model_copy(
        update={"candidates": tuple(request.candidates[i] for i in order)}
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        logits = model([permuted]).logits[0].float().cpu()
    out = [0.0] * len(order)
    for slot, original in enumerate(order):
        out[original] = float(logits[slot])
    return out


def main(bundle: str, dataset: str, split: str, n: int, rotations: int) -> None:
    predictor = load_predictor(Path(bundle), device="cuda", precision="fp16")
    model = predictor.model
    model.config = model.config.model_copy(update={"rotations": 1, "shuffle_options": False})
    examples = load_examples(Path(dataset), split, n, "random", 0)
    rng = random.Random(0)

    # 1. identical options
    spread = []
    by_slot = torch.zeros(4)
    for request, _ in examples[:200]:
        same = request.model_copy(
            update={"candidates": tuple(Candidate(id=f"c{i}", text="the answer") for i in range(4))}
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits = model([same]).logits[0].float().cpu()
        centred = logits - logits.mean()
        by_slot += centred
        spread.append(float(logits.max() - logits.min()))
    identical = {
        "mean_logit_by_slot_centred": [
            round(float(x), 3) for x in by_slot / min(200, len(examples))
        ],
        "mean_max_minus_min_logit": round(sum(spread) / len(spread), 3),
        "identical_option_questions": min(200, len(examples)),
    }

    # 2. permutations
    changed = total = 0
    correct_plain = correct_rot = correct_worst = 0
    prob_gap = []
    evaluated = 0
    for request, gold in examples:
        k = len(request.candidates)
        orders = [[(i + s) % k for i in range(k)] for s in range(k)]
        orders += [rng.sample(range(k), k) for _ in range(4)]
        per_order = [scores(model, request, o) for o in orders]
        arg = [max(range(k), key=lambda i: s[i]) for s in per_order]
        changed += int(len(set(arg)) > 1)
        total += 1
        correct_plain += int(arg[0] == gold)
        correct_worst += int(all(a == gold for a in arg))
        logp = [torch.log_softmax(torch.tensor(s), 0) for s in per_order[:k]]
        mean = torch.stack(logp).mean(0)
        correct_rot += int(int(mean.argmax()) == gold)
        prob_gap.append(float(torch.softmax(torch.tensor(per_order[0]), 0).max()))
        evaluated += 1
    print(
        json.dumps(
            {
                "bundle": bundle,
                "dataset": dataset,
                "split": split,
                "n": total,
                "identical_option_check": identical,
                "answer_changes_with_order": round(changed / total, 4),
                "accuracy_original_order": round(correct_plain / total, 4),
                "accuracy_rotation_average": round(correct_rot / total, 4),
                "accuracy_correct_in_every_order": round(correct_worst / total, 4),
                "mean_top_probability": round(sum(prob_gap) / len(prob_gap), 3),
            }
        )
    )


if __name__ == "__main__":
    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        int(sys.argv[4]),
        int(sys.argv[5]) if len(sys.argv) > 5 else 4,
    )
