"""Decision 0015 smoke probe on real ModernBERT-base: positions, memory, throughput.

1. Does ModernBERT honour ``position_ids`` on this attention path? (Compares encoder states
   with and without position resets on identical token ids.)
2. Training throughput and peak GPU memory of the marker model against the joint model
   on real RACE questions (fp16 autocast, gradient checkpointing).
Prints JSON lines.
"""

import json
import sys
import time
import tomllib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from survey_samples import load  # noqa: E402

from metanoia.build import build_model  # noqa: E402
from metanoia.config import ModelConfig  # noqa: E402
from metanoia_contracts import Candidate, DecisionRequest  # noqa: E402


def requests(n: int) -> tuple[list[DecisionRequest], list[int]]:
    rows = load("race")[:n]
    out = [
        DecisionRequest(
            context=r["passage"],
            question=r["question"],
            candidates=tuple(Candidate(id=f"c{i}", text=t) for i, t in enumerate(r["options"])),
        )
        for r in rows
    ]
    return out, [r["gold"] for r in rows]


def model_from(path: str) -> torch.nn.Module:
    config = ModelConfig.model_validate(tomllib.loads(Path(path).read_text()))
    return build_model(config, seed=0).cuda()


def positions_probe() -> None:
    reqs, _ = requests(2)
    plain = model_from("configs/models/seq_modernbert_base.toml").eval()
    reset = model_from("configs/models/seq_modernbert_base_reset.toml").eval()
    reset.load_state_dict(plain.state_dict())
    with torch.no_grad():
        a = plain.encoder.encode_ids(
            *[t.cuda() for t in (plain.build(reqs).input_ids, plain.build(reqs).attention_mask)]
        )
        built = reset.build(reqs)
        b = reset.encoder.encode_ids(
            built.input_ids.cuda(), built.attention_mask.cuda(), built.position_ids.cuda()
        )
        c = reset.encoder.encode_ids(built.input_ids.cuda(), built.attention_mask.cuda(), None)
    m = built.attention_mask.cuda().bool()
    print(
        json.dumps(
            {
                "probe": "position_ids honoured",
                "attn_impl": plain.encoder.model.config._attn_implementation,
                "max_abs_diff_reset_vs_default": float((b - c).abs()[m].max()),
                "max_abs_diff_plain_vs_default_ids": float((a - c).abs()[m].max()),
            }
        )
    )
    del plain, reset
    torch.cuda.empty_cache()


def throughput(path: str, batch: int, steps: int) -> None:
    reqs, gold = requests(batch * (steps + 1))
    model = model_from(path)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler()
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for s in range(steps + 1):
        chunk = reqs[s * batch : (s + 1) * batch]
        target = torch.tensor(gold[s * batch : (s + 1) * batch]).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(chunk).logits
        loss = torch.nn.functional.cross_entropy(logits.float(), target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if s == 0:
            torch.cuda.synchronize()
            t0 = time.time()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(
        json.dumps(
            {
                "probe": "throughput",
                "config": path,
                "batch": batch,
                "examples_per_s": round(batch * steps / dt, 2),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "last_loss": round(float(loss), 3),
            }
        ),
        flush=True,
    )
    del model, opt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    positions_probe()
    throughput("configs/models/seq_modernbert_base.toml", 8, 10)
    throughput("configs/models/direct_modernbert_ft_joint.toml", 8, 10)
