"""How different are the candidates' pooled vectors, per dataset and model?"""
import torch
from pathlib import Path
from transformers.utils import logging; logging.set_verbosity_error(); logging.disable_progress_bar()
from metanoia.api import load_predictor
from metanoia.encoding.inputs import render_input, render_candidate_joint
from metanoia_research.training.loop import load_examples

def stats(model, examples):
    rel, cos, spread = [], [], []
    for req, _ in examples:
        rendered = render_input(req)
        texts = [render_candidate_joint(c.text, rendered) for c in req.candidates]
        with torch.inference_mode():
            pooled = model.encoder.encode(texts).pooled.float()
            logits = model([req]).logits[0].float()
        centre = pooled.mean(0, keepdim=True)
        rel.append(((pooled - centre).norm(dim=1).mean() / pooled.norm(dim=1).mean()).item())
        n = torch.nn.functional.normalize(pooled, dim=1); sim = n @ n.T
        cos.append(sim[~torch.eye(len(pooled), dtype=bool)].mean().item())
        spread.append(logits.std().item())
    k = len(rel)
    return f"between-candidate spread / norm {sum(rel)/k:.4f} | mean cosine {sum(cos)/k:.5f} | logit std {sum(spread)/k:.4f}"

micro = load_examples(Path("artifacts/datasets/microworld_v0"), "dev")[:40]
mcqa = load_examples(Path("artifacts/datasets/mcqa_v0"), "dev")[:40]
for name in ["mbft_joint_mcqa_v0", "mbft_joint_reasoning_train20k"]:
    m = load_predictor(Path(f"artifacts/predictors/{name}"), device="cuda").model.eval()
    print(name); print("  microworld:", stats(m, micro)); print("  mcqa      :", stats(m, mcqa))
    del m; torch.cuda.empty_cache()
