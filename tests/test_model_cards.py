"""model_cards.toml stays consistent with RESULT_LOG.md and, when present, the bundles."""

import json
import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CARDS = REPO / "model_cards.toml"
FIELDS = {"predictor_id", "status", "summary", "headline", "good_at", "weak_at"}
FIELDS |= {"different", "evidence"}
OPTIONAL = {"checkpoint"}  # the dev-selected checkpoint the interface serves


def cards() -> dict:
    return tomllib.loads(CARDS.read_text())["models"]


def test_cards_are_complete_and_cite_logged_results() -> None:
    logged = set(re.findall(r"^## (R\d{3})", (REPO / "RESULT_LOG.md").read_text(), re.M))
    for name, card in cards().items():
        assert FIELDS <= set(card) <= FIELDS | OPTIONAL, name
        assert re.fullmatch(r"step_\d+", card.get("checkpoint", "step_0")), name
        assert card["status"] in {"recommended", "baseline", "not useful"}, name
        assert card["evidence"] and set(card["evidence"]) <= logged, name


def test_cards_match_local_bundles() -> None:
    predictors = REPO / "artifacts" / "predictors"
    if not predictors.exists():
        pytest.skip("no local artifacts")
    for name, card in cards().items():
        bundle = predictors / name
        if "checkpoint" in card:
            bundle = bundle / "checkpoints" / card["checkpoint"]
        manifest = bundle / "bundle.json"
        if manifest.exists():
            assert json.loads(manifest.read_text())["predictor_id"] == card["predictor_id"]
