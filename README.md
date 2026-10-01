# Metanoia

A decision model that does not generate text. Give it a context, a question and a list of
options; it returns a probability for each option. The expansion models also show which
retrieved statements they read to get there.

**Status: paused.** The prototype works and is documented; development is on hold.

## The idea

Each option expands into related knowledge from a memory. The model reads that knowledge
next to the option, and the evidence flows back into every option's probability. A larger
budget means more statements read.

## Results

| | Test accuracy |
|---|---|
| Reading comprehension (RACE + SciQ), direct model | 76.8% |
| Science questions without a passage, direct model | 58.1% |
| The same, with expansion over a 1M-statement memory | 65.2% |

Single training runs, one seed. Every experiment, including the failures, is in
[RESULT_LOG.md](RESULT_LOG.md), and summarized on the website's "What we tried" tab.

## Run the website and model server

Needs Docker. Trained models are not in the repository (about 0.6 GB per model, 0.9 GB
for the statement memory).

```sh
python scripts/export_serving.py     # copies the served models into serving/artifacts
docker compose up -d --build         # http://127.0.0.1:3000
```

The first request downloads the pretrained encoders (about 0.7 GB, cached in a volume).
It runs on CPU by default; for a GPU, build with
`--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128`, add `gpus: all` to the
service and set `METANOIA_DEVICE=cuda`, `METANOIA_PRECISION=fp16`.

**Cloudflare tunnel** (served at https://metanoia.samy.dev). Put the tunnel token in `.env`
(see `.env.example`) and run `docker compose --profile tunnel up -d`. In the Cloudflare
dashboard (Zero Trust, Networks, Tunnels), give the tunnel one public hostname,
`metanoia.samy.dev`, with service `http://metanoia:3000`. The site has no login; use
Cloudflare Access if it should not be open to everyone.

Models are chosen from `serving/artifacts`. Pass other bundle names to
`export_serving.py` to serve them. To train your own, see [docs/WORKFLOW.md](docs/WORKFLOW.md).

## Develop

```sh
uv sync --extra research --extra analysis --extra hf
uv run pytest && uv run ruff check src tests && uv run mypy
cd interface && pnpm install && pnpm dev       # needs a trained model in artifacts/
```

## Layout

- `src/`: `metanoia_contracts` (schemas), `metanoia` (model and server),
  `metanoia_research` (data, training), `metanoia_analysis` (metrics from saved predictions).
- `interface/`: the website (Next.js). It talks to the model only through `metanoia serve`.
- `configs/`, `suites/`, `analysis/`: experiment settings and fixed evaluation protocols.
- `docs/`: [workflow](docs/WORKFLOW.md), [plan](docs/PLAN.md), [decisions](docs/decisions/).
- `RESULT_LOG.md`, `result_logs/`, `model_cards.toml`: what was tried and what each model can do.

## Licence

MIT, see [LICENSE](LICENSE). Datasets and pretrained models keep their own licences.

## Data and credits

Training data comes from public sources pinned by revision (RACE, SciQ, ARC, OpenBookQA,
GenericsKB, and others). It is not redistributed; check each licence before use, since some
are research-only. The marker-sequence work follows [Laya](https://github.com/NandhaKishorM/laya)
(Apache-2.0). Pretrained encoders: ModernBERT and BGE.
