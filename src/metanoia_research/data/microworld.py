"""Controlled micro-worlds: named entities, nonce predicates, Horn rules, exact answers.

Each example asks which candidate entity has a target property. Exactly one candidate
does under forward chaining; distractors hold dead-end chains, consequences of the
target (the converse is not valid), or one half of a conjunctive premise. Depth is the
number of rule applications on the longest path of the answer's derivation
(SCIENTIFIC_FOUNDATIONS §11.1 and §13.2).

v0 limitations: unary predicates only (no binary relations or role binding), no
negation, closed candidate set.

Revision 2 (``revision = 2``; revision-1 configs generate byte-identical output) adds:
- a 400-word nonce vocabulary (:mod:`.microworld_vocab`), so deeper chains fit;
- optional length control (``pad_to_sentences``): every example is padded to the same
  sentence count with facts and dead-end chains about *non-candidate* entities, so
  input length does not track depth and candidate mention counts stay equal;
- an ``n_words`` annotation (whitespace words in the context). The rendered text is parsed back by
:func:`parse_context`, which the symbolic oracle uses; the parser is exact only for
the templates below.
"""

import random
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field, model_validator

from metanoia_contracts import Candidate, DecisionRequest
from metanoia_contracts.artifacts import CaseRecord, LabelRecord
from metanoia_contracts.base import StrictModel
from metanoia_research.data.microworld_vocab import PREDICATES_V2

GENERATOR_REVISION = "microworld/1"  # revision 1; see generator_revision()

_SYLLABLES = [
    "ka",
    "lo",
    "mi",
    "ra",
    "ven",
    "dor",
    "sil",
    "tu",
    "bex",
    "qua",
    "zor",
    "nim",
    "fal",
    "pry",
    "gub",
    "wex",
    "tam",
    "rho",
    "ush",
    "eld",
    "vri",
    "po",
    "sen",
    "dak",
    "lir",
    "mox",
    "tel",
    "fen",
]

PREDICATES = [
    "blicket",
    "dax",
    "wuggy",
    "feppish",
    "tomal",
    "zibbic",
    "kikish",
    "mernal",
    "ploxy",
    "quonic",
    "raskish",
    "sprillous",
    "tulvic",
    "vornal",
    "yeldish",
    "glimmic",
    "bropish",
    "crintal",
    "drellous",
    "fraskic",
    "gorpish",
    "hundal",
    "jixous",
    "kempic",
    "lusty",
    "morvic",
    "nubbish",
    "obbal",
    "pintic",
    "quillous",
    "restic",
    "sorvish",
    "tabbic",
    "umbral",
    "vexic",
    "worlish",
    "yappic",
    "zentic",
    "brisk",
    "cloven",
    "dunnic",
    "elvish",
    "fornic",
    "grelic",
]

TEMPLATE_FAMILIES: tuple[dict[str, str], ...] = (
    {
        "fact": "{e} is {p}.",
        "rule": "Everything that is {a} is also {b}.",
        "conj": "Everything that is {a} and {b} is also {c}.",
        "question": "Which one is {t}?",
    },
    {
        "fact": "{e} is known to be {p}.",
        "rule": "If something is {a}, then it is {b}.",
        "conj": "If something is {a} and {b}, then it is {c}.",
        "question": "Who is {t}?",
    },
    {
        "fact": "It is stated that {e} is {p}.",
        "rule": "All {a} things are {b}.",
        "conj": "All things that are both {a} and {b} are {c}.",
        "question": "Which candidate is {t}?",
    },
)


class MicroworldConfig(StrictModel):
    generator: Literal["microworld"] = "microworld"
    dataset_id: str
    seed: int
    n_worlds: int = Field(ge=1)
    examples_per_world: int = Field(ge=1)
    entities_per_world: int = Field(default=8, ge=2)
    predicates_per_world: int = Field(default=44, ge=8)
    n_candidates: int = Field(default=4, ge=2)
    depths: tuple[int, ...] = (1, 2, 3)
    heldout_depths: tuple[int, ...] = ()
    heldout_depth_splits: tuple[str, ...] = ("test",)
    conjunction_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    splits: dict[str, float]
    revision: Literal[1, 2] = 1
    pad_to_sentences: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check(self) -> "MicroworldConfig":
        if abs(sum(self.splits.values()) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        if any(d < 1 for d in (*self.depths, *self.heldout_depths)):
            raise ValueError("depths must be >= 1")
        if set(self.depths) & set(self.heldout_depths):
            raise ValueError("held-out depths must not appear in training depths")
        if self.n_candidates > self.entities_per_world:
            raise ValueError("n_candidates exceeds entities_per_world")
        if self.revision == 1 and self.pad_to_sentences is not None:
            raise ValueError("pad_to_sentences requires revision 2")
        if self.pad_to_sentences is not None and self.n_candidates >= self.entities_per_world:
            raise ValueError("padding needs at least one non-candidate entity")
        vocabulary = vocabulary_for(self.revision)
        if self.predicates_per_world > len(vocabulary):
            raise ValueError(f"at most {len(vocabulary)} predicates are available")
        max_depth = max((*self.depths, *self.heldout_depths))
        needed = 3 + max_depth + 2 * (self.n_candidates - 1) * (max_depth + 1)
        if self.pad_to_sentences is not None:
            # Each padded sentence consumes at most one fresh predicate.
            needed = max(needed, self.pad_to_sentences + 3)
        if needed > self.predicates_per_world:
            raise ValueError(f"predicates_per_world must be >= {needed} for these depths")
        return self


def vocabulary_for(revision: int) -> tuple[str, ...] | list[str]:
    return PREDICATES if revision == 1 else PREDICATES_V2


def generator_revision(config: "MicroworldConfig") -> str:
    return f"microworld/{config.revision}"


@dataclass(frozen=True)
class Rule:
    premises: tuple[str, ...]
    conclusion: str


Fact = tuple[str, str]  # (entity, predicate)


@dataclass
class World:
    facts: list[Fact] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)


def forward_chain(
    facts: list[Fact], rules: list[Rule]
) -> dict[Fact, tuple[int, list[Fact]] | None]:
    """All derivable facts, each with its (rule index, premise facts), or None if given."""
    known: dict[Fact, tuple[int, list[Fact]] | None] = dict.fromkeys(facts)
    changed = True
    while changed:
        changed = False
        entities = sorted({e for e, _ in known})
        for index, rule in enumerate(rules):
            for entity in entities:
                fact = (entity, rule.conclusion)
                premises = [(entity, p) for p in rule.premises]
                if fact not in known and all(p in known for p in premises):
                    known[fact] = (index, premises)
                    changed = True
    return known


def derivation_depth(fact: Fact, known: dict[Fact, tuple[int, list[Fact]] | None]) -> int:
    step = known[fact]
    if step is None:
        return 0
    return 1 + max(derivation_depth(p, known) for p in step[1])


def proof_steps(fact: Fact, known: dict[Fact, tuple[int, list[Fact]] | None]) -> list[dict]:
    step = known[fact]
    if step is None:
        return []
    steps = [s for p in step[1] for s in proof_steps(p, known)]
    steps.append(
        {"rule": step[0], "premises": [list(p) for p in step[1]], "conclusion": list(fact)}
    )
    return steps


def _world_names(rng: random.Random, count: int) -> list[str]:
    names: set[str] = set()
    while len(names) < count:
        names.add("".join(rng.sample(_SYLLABLES, 2)).capitalize())
    return sorted(names)


def _render(world: World, family: dict[str, str]) -> list[str]:
    sentences = [family["fact"].format(e=e, p=p) for e, p in world.facts]
    for rule in world.rules:
        if len(rule.premises) == 1:
            sentences.append(family["rule"].format(a=rule.premises[0], b=rule.conclusion))
        else:
            a, b = rule.premises
            sentences.append(family["conj"].format(a=a, b=b, c=rule.conclusion))
    return sentences


def _build_example(
    rng: random.Random,
    names: list[str],
    predicates: list[str],
    n_candidates: int,
    depth: int,
    conjunctive: bool,
    pad_to: int | None = None,
) -> tuple[World, list[str], str, str]:
    """Returns (world, candidate names, answer, target predicate)."""
    candidates = rng.sample(names, n_candidates)
    answer = rng.choice(candidates)
    fresh: Iterator[str] = iter(rng.sample(predicates, len(predicates)))
    target = next(fresh)
    world = World()

    if conjunctive:
        chain = [next(fresh) for _ in range(depth)]  # depth-1 unary steps, then one conjunction
        joint = next(fresh)
        world.rules += [Rule((a,), b) for a, b in zip(chain, chain[1:], strict=False)]
        world.rules.append(Rule((chain[-1], joint), target))
        world.facts += [(answer, chain[0]), (answer, joint)]
        halves = [chain[0], joint]
    else:
        chain = [next(fresh) for _ in range(depth)] + [target]
        world.rules += [Rule((a,), b) for a, b in zip(chain, chain[1:], strict=False)]
        world.facts.append((answer, chain[0]))
        halves = []

    # Target implies a consequence; holding the consequence does not imply the target.
    consequence = next(fresh)
    world.rules.append(Rule((target,), consequence))

    distractors = [c for c in candidates if c != answer]
    rng.shuffle(distractors)
    for index, entity in enumerate(distractors):
        if index < len(halves):
            world.facts.append((entity, halves[index]))  # near miss: one conjunct only
        elif rng.random() < 0.5:
            world.facts.append((entity, consequence))
        else:
            world.facts.append((entity, _dead_end(world, fresh, depth)))

    # Every candidate is mentioned equally often, so mention counts carry no label cue
    # (SCIENTIFIC_FOUNDATIONS §9.3). Padding starts dead-end chains rather than using
    # predicates no rule mentions, which would itself mark a distractor.
    answer_facts = sum(e == answer for e, _ in world.facts)
    for entity in distractors:
        while sum(e == entity for e, _ in world.facts) < answer_facts:
            world.facts.append((entity, _dead_end(world, fresh, 1 + rng.randrange(depth))))
    if pad_to is not None:
        fillers = [n for n in names if n not in candidates]
        _pad(world, rng, fresh, fillers, pad_to)
    return world, candidates, answer, target


def _pad(
    world: World, rng: random.Random, fresh: Iterator[str], fillers: list[str], total: int
) -> None:
    """Add facts and dead-end chains about non-candidate entities up to ``total`` sentences.

    Only non-candidates are touched, so candidate mention counts stay equal and no
    padding sentence can make a candidate hold the target.
    """
    remaining = total - len(world.facts) - len(world.rules)
    if remaining < 0:
        raise ValueError(f"example has {total - remaining} sentences, more than {total}")
    try:
        while remaining > 0:
            entity = rng.choice(fillers)
            if remaining == 1:
                world.facts.append((entity, next(fresh)))
                remaining -= 1
            else:
                length = rng.randint(1, min(3, remaining - 1))
                world.facts.append((entity, _dead_end(world, fresh, length)))
                remaining -= 1 + length
    except StopIteration:
        raise ValueError("predicates_per_world is too small for pad_to_sentences") from None


def _dead_end(world: World, fresh: Iterator[str], length: int) -> str:
    """Add a chain of ``length`` rules that never reaches the target; return its start."""
    chain = [next(fresh) for _ in range(length + 1)]
    world.rules += [Rule((a,), b) for a, b in zip(chain, chain[1:], strict=False)]
    return chain[0]


def generate(config: MicroworldConfig) -> tuple[list[CaseRecord], list[LabelRecord], list[dict]]:
    rng = random.Random(config.seed)
    split_names = sorted(config.splits)
    world_order = list(range(config.n_worlds))
    rng.shuffle(world_order)
    world_split: dict[int, str] = {}
    start = 0.0
    for name in split_names:
        stop = start + config.splits[name] * config.n_worlds
        for position in range(round(start), round(stop)):
            world_split[world_order[position]] = name
        start = stop

    cases, labels, proofs = [], [], []
    for world_index in range(config.n_worlds):
        world_rng = random.Random(f"{config.seed}:{world_index}")
        names = _world_names(world_rng, config.entities_per_world)
        predicates = world_rng.sample(vocabulary_for(config.revision), config.predicates_per_world)
        split = world_split[world_index]
        depths = config.depths
        if split in config.heldout_depth_splits:
            depths = (*depths, *config.heldout_depths)
        group_id = f"w{world_index:05d}"
        for example_index in range(config.examples_per_world):
            depth = world_rng.choice(depths)
            conjunctive = world_rng.random() < config.conjunction_probability
            family_index = world_rng.randrange(len(TEMPLATE_FAMILIES))
            family = TEMPLATE_FAMILIES[family_index]
            world, candidates, answer, target = _build_example(
                world_rng,
                names,
                predicates,
                config.n_candidates,
                depth,
                conjunctive,
                config.pad_to_sentences,
            )
            known = forward_chain(world.facts, world.rules)
            holders = [c for c in candidates if (c, target) in known]
            if holders != [answer] or derivation_depth((answer, target), known) != depth:
                raise AssertionError(f"generator invariant violated in world {world_index}")

            sentences = _render(world, family)
            world_rng.shuffle(sentences)
            annotations: dict[str, str | int | float | bool] = {
                "depth": depth,
                "conjunctive": conjunctive,
                "template_family": family_index,
                "n_candidates": len(candidates),
                "n_sentences": len(sentences),
            }
            if config.revision >= 2:
                annotations["n_words"] = sum(len(s.split()) for s in sentences)
            example_id = f"{config.dataset_id}-{group_id}-e{example_index:03d}"
            request = DecisionRequest(
                context=" ".join(sentences),
                question=family["question"].format(t=target),
                candidates=tuple(Candidate(id=c.lower(), text=c) for c in candidates),
            )
            cases.append(
                CaseRecord(example_id=example_id, group_id=group_id, split=split, request=request)
            )
            labels.append(
                LabelRecord(
                    example_id=example_id,
                    group_id=group_id,
                    split=split,
                    label_id=answer.lower(),
                    annotations=annotations,
                )
            )
            proofs.append({"example_id": example_id, "steps": proof_steps((answer, target), known)})
    return cases, labels, proofs


# --- Parsing rendered text back to symbols (used by the oracle) ---------------------


def _pattern(template: str) -> re.Pattern[str]:
    escaped = re.escape(template)
    for slot in ("e", "p", "a", "b", "c", "t"):
        escaped = escaped.replace(re.escape("{" + slot + "}"), rf"(?P<{slot}>\w+)")
    return re.compile(escaped)


_PATTERNS = [
    {kind: _pattern(template) for kind, template in family.items()} for family in TEMPLATE_FAMILIES
]
_SENTENCE = re.compile(r"[^.?!]+[.?!]")


def parse_context(context: str) -> World:
    """Invert the templates exactly; raises ValueError on any unrecognized sentence."""
    world = World()
    for raw in _SENTENCE.findall(context):
        sentence = raw.strip()
        matches = [
            (kind, match)
            for family in _PATTERNS
            for kind, pattern in family.items()
            if kind != "question" and (match := pattern.fullmatch(sentence))
        ]
        if len({kind for kind, _ in matches}) != 1:
            raise ValueError(f"cannot parse sentence: {sentence!r}")
        kind, match = matches[0]
        slots = match.groupdict()
        if kind == "fact":
            world.facts.append((slots["e"], slots["p"]))
        elif kind == "rule":
            world.rules.append(Rule((slots["a"],), slots["b"]))
        else:
            world.rules.append(Rule((slots["a"], slots["b"]), slots["c"]))
    return world


def parse_question(question: str) -> str:
    for family in _PATTERNS:
        if match := family["question"].fullmatch(question.strip()):
            return match.group("t")
    raise ValueError(f"cannot parse question: {question!r}")
