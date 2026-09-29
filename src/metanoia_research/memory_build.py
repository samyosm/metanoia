"""Build a versioned concept memory bundle (PLAN §6.4, decision 0010).

Sources:
- ``wordnet`` (``wordnet_v0``): WordNet 3.0 synsets through nltk. Each synset is one
  concept: id = the synset name (e.g. ``photosynthesis.n.01``), name = its first lemma,
  names = all its lemmas, text = ``"{name}: {gloss}"``.
- ``generics_kb`` (``generics_v0``, decision 0012): GenericsKB-Best generic statements.
  Each node is a statement about a concept: id = ``gkb:{row}``, name = the concept term,
  text = the statement. Exact duplicate statements are kept once.

The frozen embedding of ``text`` is the key.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from metanoia.memory.encoder import RetrievalEncoder
from metanoia.memory.store import MEMORY_MANIFEST
from metanoia_contracts.artifacts import FileRef, MemoryManifest
from metanoia_contracts.base import StrictModel
from metanoia_contracts.provenance import (
    canonical_json,
    content_sha256,
    environment_snapshot,
    sha256_file,
    source_revision,
    utc_now,
)
from metanoia_research.artifacts import (
    accelerator_snapshot,
    discard_incomplete,
    finalize_directory,
    new_incomplete_directory,
)

TEXT_TEMPLATE = "{name}: {gloss}"
GENERICS_TEMPLATE = "{statement}"


class RetrievalEncoderConfig(StrictModel):
    model: str
    revision: str
    pooling: Literal["cls", "mean"]
    max_length: int = Field(default=128, ge=8)
    query_prefix: str = ""


class MemoryConfig(StrictModel):
    memory_id: str
    source: Literal["wordnet", "generics_kb"] = "wordnet"
    # wordnet: the directory holding corpora/wordnet.zip, and that zip's hash
    nltk_data: str | None = None
    wordnet_zip_sha256: str | None = None
    # generics_kb: the downloaded parquet, its hash and the Hugging Face revision
    parquet: str | None = None
    parquet_sha256: str | None = None
    hf_revision: str | None = None
    encoder: RetrievalEncoderConfig
    device: str = "cuda"
    batch_size: int = Field(default=512, ge=1)

    @model_validator(mode="after")
    def _source_fields(self) -> "MemoryConfig":
        needed = {
            "wordnet": ("nltk_data", "wordnet_zip_sha256"),
            "generics_kb": ("parquet", "parquet_sha256", "hf_revision"),
        }[self.source]
        missing = [f for f in needed if getattr(self, f) is None]
        if missing:
            raise ValueError(f"source {self.source} needs {missing}")
        return self


def wordnet_concepts(nltk_data: str, expected_sha256: str) -> list[dict]:
    zip_path = Path(nltk_data) / "corpora" / "wordnet.zip"
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"{zip_path} has sha256 {digest}, expected {expected_sha256}")
    import nltk

    nltk.data.path.insert(0, str(Path(nltk_data).resolve()))
    from nltk.corpus import wordnet

    concepts = []
    for synset in sorted(wordnet.all_synsets(), key=lambda s: s.name()):
        names = [lemma.name().replace("_", " ").lower() for lemma in synset.lemmas()]
        name = names[0]
        concepts.append(
            {
                "id": synset.name(),
                "name": name,
                "names": names,
                "pos": synset.pos(),
                "text": TEXT_TEMPLATE.format(name=name, gloss=synset.definition()),
            }
        )
    return concepts


def generics_concepts(parquet: str, expected_sha256: str) -> list[dict]:
    digest = sha256_file(Path(parquet))
    if digest != expected_sha256:
        raise ValueError(f"{parquet} has sha256 {digest}, expected {expected_sha256}")
    import pyarrow.parquet as pq

    table = pq.read_table(parquet, columns=["term", "generic_sentence"])
    seen: set[str] = set()
    concepts = []
    terms = table.column("term").to_pylist()
    statements = table.column("generic_sentence").to_pylist()
    for row, (term, statement) in enumerate(zip(terms, statements, strict=True)):
        statement = " ".join(statement.split())
        if not statement or statement in seen:
            continue
        seen.add(statement)
        term = term.strip().lower()
        concepts.append(
            {
                "id": f"gkb:{row}",
                "name": term,
                "names": [term],
                "pos": "statement",
                "text": GENERICS_TEMPLATE.format(statement=statement),
            }
        )
    return concepts


def _source(config: MemoryConfig) -> tuple[list[dict], dict, str]:
    """(concepts, source description, text template) for the configured source."""
    if config.source == "wordnet":
        assert config.nltk_data and config.wordnet_zip_sha256
        concepts = wordnet_concepts(config.nltk_data, config.wordnet_zip_sha256)
        described = {"kind": "wordnet", "version": "3.0"}
        described["wordnet_zip_sha256"] = config.wordnet_zip_sha256
        return concepts, described, TEXT_TEMPLATE
    assert config.parquet and config.parquet_sha256 and config.hf_revision
    concepts = generics_concepts(config.parquet, config.parquet_sha256)
    described = {"kind": "generics_kb", "config": "generics_kb_best"}
    described |= {"hf_revision": config.hf_revision, "parquet_sha256": config.parquet_sha256}
    return concepts, described, GENERICS_TEMPLATE


def build_memory(config: MemoryConfig, out_dir: Path) -> MemoryManifest:
    concepts, described, template = _source(config)
    encoder = RetrievalEncoder(
        config.encoder.model,
        config.encoder.revision,
        config.encoder.pooling,
        config.encoder.max_length,
        config.encoder.query_prefix,
    ).to(config.device)
    work = new_incomplete_directory(out_dir)
    try:
        vectors = encoder.embed([c["text"] for c in concepts], batch_size=config.batch_size)
        array = vectors.cpu().numpy().astype(np.float16)
        np.save(work / "vectors.npy", array)
        (work / "concepts.jsonl").write_text("".join(canonical_json(c) + "\n" for c in concepts))
        manifest = MemoryManifest(
            memory_id=config.memory_id,
            source={**described, "config": config.model_dump(mode="json")},
            text_template=template,
            encoder=encoder.identity(),
            concepts=FileRef(
                path="concepts.jsonl",
                sha256=sha256_file(work / "concepts.jsonl"),
                rows=len(concepts),
            ),
            vectors=FileRef(
                path="vectors.npy", sha256=sha256_file(work / "vectors.npy"), rows=len(concepts)
            ),
            dim=int(array.shape[1]),
            environment=environment_snapshot(accelerator_snapshot(config.device)),
            code_source=source_revision(),
            created_at=utc_now(),
        )
        (work / MEMORY_MANIFEST).write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n"
        )
        (work / "content_sha256.txt").write_text(content_sha256(concepts) + "\n")
    except BaseException:
        discard_incomplete(work)
        raise
    finalize_directory(work, out_dir)
    return manifest
