"""Step 0 of PLAN_V3: metadata survey of candidate public QA sources (no data downloads).

Reads Hugging Face dataset metadata only: licence, gating, files and their sizes, last
modified. Prints JSON lines. Row counts come from the card's ``dataset_info`` when present.
"""

import json
import sys

from huggingface_hub import HfApi

CANDIDATES = {
    "race": ["ehovy/race"],
    "dream": ["dataset-org/dream", "community-datasets/dream"],
    "reclor": ["metaeval/reclor", "voidful/ReClor"],
    "multirc": ["aps/super_glue"],
    "boolq": ["google/boolq"],
    "cosmosqa": ["allenai/cosmos_qa"],
    "quality": ["emozilla/quality", "tasksource/quality"],
    "hotpotqa": ["hotpotqa/hotpot_qa"],
    "musique": ["dgslibisey/MuSiQue", "bdsaglam/musique"],
    "2wiki": ["framolfese/2WikiMultihopQA", "scholarly-shadows-syndicate/2wikimultihopqa"],
    "casehold": ["coastalcph/lex_glue", "casehold/casehold"],
    "contractnli": ["kiddothe2b/contract-nli", "tasksource/contract-nli"],
    "cuad": [
        "theatticusproject/cuad-qa",
        "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification",
    ],
    "legalbench": ["nguha/legalbench"],
    "pubmedqa": ["qiaojin/PubMedQA"],
    "medmcqa": ["openlifescienceai/medmcqa"],
    "squad2": ["rajpurkar/squad_v2"],
    "drop": ["ucinlp/drop"],
    "narrativeqa": ["deepmind/narrativeqa"],
    "finqa": ["ibm-research/finqa", "dreamerdeo/finqa"],
    "qasc": ["allenai/qasc"],
    "commonsenseqa": ["tau/commonsense_qa"],
}


def main() -> None:
    api = HfApi()
    for name, repos in CANDIDATES.items():
        for repo in repos:
            try:
                info = api.dataset_info(repo, files_metadata=True)
            except Exception as exc:  # noqa: BLE001 - survey tool: report and continue
                print(json.dumps({"source": name, "repo": repo, "error": type(exc).__name__}))
                continue
            card = info.card_data.to_dict() if info.card_data else {}
            sizes = {
                s.rfilename: s.size
                for s in (info.siblings or [])
                if s.rfilename.endswith((".parquet", ".json", ".jsonl", ".csv", ".zip", ".gz"))
            }
            infos = card.get("dataset_info")
            splits = {}
            if isinstance(infos, dict):
                infos = [infos]
            for cfg in infos or []:
                for sp in cfg.get("splits", []):
                    splits[f"{cfg.get('config_name', 'default')}/{sp['name']}"] = sp.get(
                        "num_examples"
                    )
            print(
                json.dumps(
                    {
                        "source": name,
                        "repo": repo,
                        "license": card.get("license"),
                        "gated": info.gated,
                        "sha": info.sha,
                        "downloads": info.downloads,
                        "n_files": len(sizes),
                        "total_mb": round(sum(v or 0 for v in sizes.values()) / 1e6, 1),
                        "splits": splits,
                        "files": dict(list(sizes.items())[:8]),
                    }
                )
            )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
