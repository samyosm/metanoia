import sys, tomllib
from pathlib import Path
sys.path.insert(0, "scripts")
import torch
from probe_sequence import requests, model_from
reqs, _ = requests(6)
for cfg in ("seq_modernbert_base", "seq_modernbert_base_reset"):
    m = model_from(f"configs/models/{cfg}.toml").eval()
    b = m.build(reqs)
    with torch.no_grad():
        h = m.encoder.encode_ids(b.input_ids.cuda(), b.attention_mask.cuda(), None if b.position_ids is None else b.position_ids.cuda())
    sims = []
    stds = []
    for i in range(len(reqs)):
        v = h[i, b.markers[i].cuda()]  # [4, D]
        v = torch.nn.functional.normalize(v.float(), dim=-1)
        sims.append((v @ v.T)[0, 1:].mean().item())
        stds.append(h[i].float().std().item())
    print(cfg, "mean cos(marker0, others)", round(sum(sims)/len(sims), 4), "state std", round(sum(stds)/len(stds), 3), "attn", m.encoder.model.config._attn_implementation)
    if b.position_ids is not None:
        print(b.position_ids[0, :40].tolist())
    del m; torch.cuda.empty_cache()
