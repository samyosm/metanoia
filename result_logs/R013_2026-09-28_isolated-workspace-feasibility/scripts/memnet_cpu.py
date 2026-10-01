"""CPU probe: can a memory-network-style recurrent read (one hop per rule) learn rule
chaining that transfers to unseen worlds and to more hops than trained?"""
import random, re, time, torch, torch.nn as nn, torch.nn.functional as F
from metanoia.encoding.inputs import render_input
from metanoia.reasoning.workspace import sentence_spans
from metanoia_research.data.microworld import MicroworldConfig, generate

torch.manual_seed(0); rng = random.Random(0)
cfg = MicroworldConfig(dataset_id="t", seed=7, n_worlds=1500, examples_per_world=4, depths=(1, 2), heldout_depths=(3, 4),
                       heldout_depth_splits=("test",), conjunction_probability=0.0, splits={"train": 0.8, "test": 0.2})
cases, labels, _ = generate(cfg)
TOK = re.compile(r"\w+|[^\w\s]")
vocab = {}
def ids(text): return [vocab.setdefault(t.lower(), len(vocab) + 1) for t in TOK.findall(text)]
def encode(case, label):
    r = case.request; text = render_input(r)
    sents = [ids(text[a:b]) for a, b in sentence_spans(text)]
    return sents[0], sents[1:], [ids(c.text)[0] for c in r.candidates], r.candidate_ids.index(label.label_id), label.annotations["depth"], case.split
data = [encode(c, l) for c, l in zip(cases, labels)]
train = [d for d in data if d[5] == "train"]; test = [d for d in data if d[5] == "test"]
V, D, L = len(vocab) + 1, 64, 12

class MemNet(nn.Module):
    """Question -> query u. Each hop: attend over position-encoded sentences, read, u <- H u + o.
    Same weights every hop (recurrent), so the hop count is the budget."""
    def __init__(self):
        super().__init__()
        self.A, self.C, self.B = (nn.Embedding(V, D, padding_idx=0) for _ in range(3))
        self.pos = nn.Parameter(torch.ones(L, D))  # learned position weighting (MemN2N PE)
        self.H = nn.Linear(D, D)
    def sent(self, emb, x):  # x [B, S, L]
        return (emb(x) * self.pos[: x.shape[-1]]).sum(-2)
    def forward(self, q, mem, mmask, cands, hops):
        u = self.sent(self.B, q.unsqueeze(1)).squeeze(1)
        m, c = self.sent(self.A, mem), self.sent(self.C, mem)
        outs = []
        for _ in range(hops):
            p = (m @ u.unsqueeze(-1)).squeeze(-1).masked_fill(~mmask, -1e9).softmax(-1)
            u = self.H(u) + (p.unsqueeze(-1) * c).sum(1)
            outs.append((self.C(cands) @ u.unsqueeze(-1)).squeeze(-1))  # score candidate names
        return outs

def batch(ex):
    S = max(len(e[1]) for e in ex)
    q = torch.zeros(len(ex), L, dtype=torch.long); mem = torch.zeros(len(ex), S, L, dtype=torch.long)
    for i, e in enumerate(ex):
        q[i, : len(e[0][:L])] = torch.tensor(e[0][:L])
        for j, s in enumerate(e[1]): mem[i, j, : len(s[:L])] = torch.tensor(s[:L])
    return q, mem, mem.sum(-1) > 0, torch.tensor([e[2] for e in ex]), torch.tensor([e[3] for e in ex])

model = MemNet(); opt = torch.optim.Adam(model.parameters(), lr=3e-3); start = time.time()
for epoch in range(40):
    rng.shuffle(train); tot = 0
    for i in range(0, len(train), 32):
        q, mem, mm, cands, y = batch(train[i:i+32]); hops = rng.randint(1, 4)
        loss = torch.stack([F.cross_entropy(o, y) for o in model(q, mem, mm, cands, hops)]).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); tot += loss.item()
    if epoch % 5 == 4:
        with torch.no_grad():
            row = []
            for hops in (1, 2, 3, 4, 6, 8):
                accs = []
                for d in (1, 2, 3, 4):
                    ex = [e for e in test if e[4] == d]; q, mem, mm, cands, y = batch(ex)
                    accs.append((model(q, mem, mm, cands, hops)[-1].argmax(1) == y).float().mean().item())
                row.append(f"h{hops}:" + "/".join(f"{a:.2f}" for a in accs))
        print(f"epoch {epoch} loss {tot/(len(train)/32):.3f} {time.time()-start:.0f}s | " + " ".join(row), flush=True)
