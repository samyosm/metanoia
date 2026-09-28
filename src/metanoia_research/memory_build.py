"""Build a versioned concept memory bundle (PLAN §6.4, decision 0010).

v0 source: WordNet 3.0 synsets through nltk. Each synset is one concept:
- id: the synset name (e.g. ``photosynthesis.n.01``);
- name: its first lemma;
- names: all its lemmas;
- text: ``"{name}: {gloss}"``, whose frozen embedding is the key.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

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


class RetrievalEncoderConfig(StrictModel):
    model: str
    revision: str
    pooling: Literal["cls", "mean"]
    max_length: int = Field(default=128, ge=8)
    query_prefix: str = ""


class MemoryConfig(StrictModel):
    memory_id: str
    source: Literal["wordnet"] = "wordnet"
    nltk_data: str  # directory holding corpora/wordnet.zip
    wordnet_zip_sha256: str
    encoder: RetrievalEncoderConfig
    device: str = "cuda"
    batch_size: int = Field(default=512, ge=1)


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


def build_memory(config: MemoryConfig, out_dir: Path) -> MemoryManifest:
    concepts = wordnet_concepts(config.nltk_data, config.wordnet_zip_sha256)
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
            source={
                "kind": "wordnet",
                "version": "3.0",
                "wordnet_zip_sha256": config.wordnet_zip_sha256,
                "config": config.model_dump(mode="json"),
            },
            text_template=TEXT_TEMPLATE,
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
