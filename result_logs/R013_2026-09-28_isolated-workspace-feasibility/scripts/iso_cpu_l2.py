"""CPU feasibility: can isolated-encoding workspace rounds learn rule chaining?"""
import random, time, torch, torch.nn.functional as F
from metanoia.build import build_model
from metanoia.config import DirectScorerConfig, HashingEncoderConfig, ModelConfig, WorkspaceConfig
from metanoia_contracts import Budget
from metanoia_research.data.microworld import MicroworldConfig, generate

torch.manual_seed(0)
cfg = MicroworldConfig(dataset_id="t", seed=7, n_worlds=900, examples_per_world=4, depths=(1, 2), heldout_depths=(3,),
                       heldout_depth_splits=("test",), conjunction_probability=0.0, splits={"train": 0.8, "test": 0.2})
cases, labels, _ = generate(cfg)
data = [(c.request, c.request.candidate_ids.index(l.label_id), l.annotations["depth"], c.split) for c, l in zip(cases, labels)]
train = [d for d in data if d[3] == "train"]; test = [d for d in data if d[3] == "test"]
mc = ModelConfig(name="iso", encoder=HashingEncoderConfig(buckets=4096, dim=64, layers=2, heads=4),
                 scorer=DirectScorerConfig(latent_dim=64, hidden_dim=64),
                 workspace=WorkspaceConfig(dim=64, heads=4, hidden_dim=128, encoding="isolated", readout="workspace"))
model = build_model(mc, seed=0)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
rng = random.Random(0); start = time.time()
for epoch in range(24):
    model.train(); rng.shuffle(train); tot = 0
    for i in range(0, len(train), 32):
        b = train[i:i+32]; R = rng.randint(1, 6)
        out = model([r for r, *_ in b], rounds=R)
        y = torch.tensor([t for _, t, *_ in b])
        loss = torch.stack([F.cross_entropy(l, y) for l in out.round_logits]).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item()
    model.eval(); res = {}
    with torch.no_grad():
        for R in (0, 1, 2, 4, 8):
            acc = {}
            for d in (1, 2, 3):
                ex = [x for x in test if x[2] == d]
                reqs = [r.model_copy(update={"budget": Budget(max_rounds=R, max_nodes=64, max_pairs=R*4096)}) for r, *_ in ex]
                logits = torch.cat([model(reqs[j:j+64]).logits for j in range(0, len(reqs), 64)])
                acc[d] = (logits.argmax(1) == torch.tensor([t for _, t, *_ in ex])).float().mean().item()
            res[R] = acc
    print(f"epoch {epoch} loss {tot/(len(train)/32):.3f} {time.time()-start:.0f}s | " + " ".join(f"r{R}:" + "/".join(f"{a:.2f}" for a in v.values()) for R, v in res.items()), flush=True)
