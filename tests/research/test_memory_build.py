"""Memory sources (decision 0012): GenericsKB statements become concept-keyed nodes."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from metanoia_contracts.provenance import sha256_file
from metanoia_research.memory_build import MemoryConfig, generics_concepts

ENCODER = {"model": "m", "revision": "r", "pooling": "cls"}


def test_generics_rows_become_statement_nodes(tmp_path) -> None:
    path = tmp_path / "g.parquet"
    table = pa.table(
        {
            "term": ["Metal", "metal", "sun"],
            "generic_sentence": ["Metals  conduct heat.", "Metals conduct heat.", "Suns shine."],
        }
    )
    pq.write_table(table, path)
    nodes = generics_concepts(str(path), sha256_file(path))
    # Whitespace is normalized, and exact duplicate statements are kept once.
    assert [n["text"] for n in nodes] == ["Metals conduct heat.", "Suns shine."]
    assert nodes[0] == {
        "id": "gkb:0",
        "name": "metal",
        "names": ["metal"],
        "pos": "statement",
        "text": "Metals conduct heat.",
    }
    assert nodes[1]["id"] == "gkb:2"  # ids keep the source row


def test_generics_hash_is_checked(tmp_path) -> None:
    path = tmp_path / "g.parquet"
    pq.write_table(pa.table({"term": ["a"], "generic_sentence": ["A."]}), path)
    with pytest.raises(ValueError, match="sha256"):
        generics_concepts(str(path), "0" * 64)


def test_config_requires_the_source_fields() -> None:
    with pytest.raises(ValueError, match="needs"):
        MemoryConfig.model_validate({"memory_id": "x", "source": "generics_kb", "encoder": ENCODER})
    with pytest.raises(ValueError, match="needs"):
        MemoryConfig.model_validate({"memory_id": "x", "encoder": ENCODER})
