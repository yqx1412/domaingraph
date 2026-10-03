"""D2: score extracted knowledge against hand-labeled gold sets.

A gold file (``benchmarks/extraction/gold/*.yaml``) lists one lecture's concepts, with
aliases, and the relations between them. Scoring is per source, on the merged concepts:

* **Concept match (strict).** An extracted concept matches a gold concept when any of its
  surface forms (name or alias) normalizes to any of the gold concept's forms. Each gold
  concept can be claimed once; a second extracted concept that maps to an already matched
  gold concept is a *duplicate*: counted against precision, because merging should have
  caught it.
* **Concept match (lenient).** Strict, plus embedding similarity >= ``lenient_threshold``
  between names. Reported alongside, to show how much of the gap is wording.
* **Relations.** An extracted relation is correct when both ends match gold concepts and
  that (subject, predicate, object) is in the gold set. Also reported: *any predicate*
  (the pair is related in gold, in either direction).

Gold sets are never complete, so precision is a lower bound: a real concept the annotator
left out counts as a false positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from domaingraph.extraction import CONCEPT_TYPES, PREDICATES
from domaingraph.llm import Embedder
from domaingraph.merge import Knowledge, _cos, normalize


class GoldConcept(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    type: str
    core: bool = False
    chunks: list[int] = Field(default_factory=list)

    def forms(self) -> set[str]:
        return {normalize(x) for x in [self.name, *self.aliases]}


class GoldRelation(BaseModel):
    subject: str
    predicate: str
    object: str
    chunks: list[int] = Field(default_factory=list)


class Gold(BaseModel):
    source_id: str
    title: str
    concepts: list[GoldConcept]
    relations: list[GoldRelation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Gold:
        names = [c.name for c in self.concepts]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.source_id}: duplicate gold concept names")
        owner: dict[str, str] = {}
        for c in self.concepts:
            if c.type not in CONCEPT_TYPES:
                raise ValueError(f"{c.name}: unknown type {c.type!r}")
            for f in c.forms():
                if f in owner and owner[f] != c.name:
                    raise ValueError(f"form {f!r} belongs to both {owner[f]!r} and {c.name!r}")
                owner[f] = c.name
        for r in self.relations:
            if r.predicate not in PREDICATES:
                raise ValueError(f"unknown predicate {r.predicate!r}")
            for end in (r.subject, r.object):
                if end not in names:
                    raise ValueError(f"relation end {end!r} is not a gold concept")
        return self


def load_gold(path: Path) -> Gold:
    return Gold.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_gold_dir(d: Path) -> list[Gold]:
    return [load_gold(p) for p in sorted(d.glob("*.yaml"))]


@dataclass
class PRF:
    tp: int = 0
    n_pred: int = 0
    n_gold: int = 0

    @property
    def precision(self) -> float:
        return self.tp / self.n_pred if self.n_pred else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_gold if self.n_gold else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def __add__(self, o: PRF) -> PRF:
        return PRF(self.tp + o.tp, self.n_pred + o.n_pred, self.n_gold + o.n_gold)


@dataclass
class SourceScore:
    source_id: str
    title: str
    concepts: PRF
    concepts_lenient: PRF
    core_recall: PRF  # tp/n_gold over core gold concepts only
    relations: PRF
    relations_any_predicate: PRF
    relations_grounded: PRF = field(default_factory=PRF)  # only rels with both ends in gold
    duplicates: int = 0
    matched: dict[str, str] = field(default_factory=dict)  # extracted name -> gold name
    false_positives: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    missed_core: list[str] = field(default_factory=list)


def score_source(
    kn: Knowledge,
    gold: Gold,
    *,
    embedder: Embedder | None = None,
    lenient_threshold: float = 0.85,
    min_chunks: int = 1,
) -> SourceScore:
    """``min_chunks``: count an extracted concept only if at least that many of this
    source's chunks mention it (a support filter; 1 = keep everything)."""
    concepts = [
        c
        for c in kn.concepts
        if len({m.chunk_id for m in c.mentions if m.source_id == gold.source_id}) >= min_chunks
    ]
    form_owner = {f: g.name for g in gold.concepts for f in g.forms()}

    def strict(c) -> str | None:
        # The canonical name decides first, so a generic alias can't steal the match.
        for f in [c.name, *c.aliases]:
            g = form_owner.get(normalize(f))
            if g:
                return g
        return None

    strict_map = {c.id: strict(c) for c in concepts}
    lenient_map = dict(strict_map)
    if embedder is not None:
        todo = [c for c in concepts if strict_map[c.id] is None]
        gold_forms = sorted(form_owner)
        if todo and gold_forms:
            vec = embedder.embed([normalize(c.name) for c in todo] + gold_forms)
            cv, gv = vec[: len(todo)], vec[len(todo) :]
            for c, v in zip(todo, cv, strict=True):
                sims = [(_cos(v, w), f) for w, f in zip(gv, gold_forms, strict=True)]
                best, f = max(sims)
                if best >= lenient_threshold:
                    lenient_map[c.id] = form_owner[f]

    def prf(mapping: dict[str, str | None]) -> tuple[PRF, int, set[str]]:
        claimed: set[str] = set()
        dup = 0
        # Most-mentioned extracted concept claims a gold concept first.
        for c in sorted(concepts, key=lambda c: -len(c.mentions)):
            g = mapping[c.id]
            if g is None:
                continue
            if g in claimed:
                dup += 1
            claimed.add(g)
        return PRF(len(claimed), len(concepts), len(gold.concepts)), dup, claimed

    cp, dup, claimed = prf(strict_map)
    lp, _, _ = prf(lenient_map)
    core = [g.name for g in gold.concepts if g.core]

    gold_rel = {(r.subject, r.predicate, r.object) for r in gold.relations}
    gold_pairs = {frozenset((r.subject, r.object)) for r in gold.relations}
    kept = {c.id for c in concepts}
    rels = [
        r
        for r in kn.relations
        if r.subject in kept
        and r.object in kept
        and any(m.source_id == gold.source_id for m in r.mentions)
    ]
    # Several extracted relations can map to one gold relation (duplicate concepts); count
    # each gold relation once as a hit, but every extracted relation in the denominator.
    hit, hit_any = set(), set()
    grounded = 0  # relations whose both ends are (different) gold concepts
    for r in rels:
        s, o = strict_map.get(r.subject), strict_map.get(r.object)
        if s is None or o is None or s == o:
            continue
        grounded += 1
        if (s, r.predicate, o) in gold_rel:
            hit.add((s, r.predicate, o))
        if frozenset((s, o)) in gold_pairs:
            hit_any.add(frozenset((s, o)))

    by_id = {c.id: c for c in concepts}
    return SourceScore(
        source_id=gold.source_id,
        title=gold.title,
        concepts=cp,
        concepts_lenient=lp,
        core_recall=PRF(len(claimed & set(core)), 0, len(core)),
        relations=PRF(len(hit), len(rels), len(gold_rel)),
        relations_any_predicate=PRF(len(hit_any), len(rels), len(gold_pairs)),
        relations_grounded=PRF(len(hit), grounded, len(gold_rel)),
        duplicates=dup,
        matched={by_id[i].name: g for i, g in strict_map.items() if g},
        false_positives=sorted(by_id[i].name for i, g in strict_map.items() if g is None),
        missed=sorted(g.name for g in gold.concepts if g.name not in claimed),
        missed_core=sorted(n for n in core if n not in claimed),
    )


def merge_quality(kn: Knowledge, golds: list[Gold]) -> PRF:
    """Pairwise merge precision/recall, over mentions whose own surface name matches a gold
    concept. A pair of mentions from the same source *should* be merged when both name the
    same gold concept, and should *not* be when they name different ones. Pairs across
    sources are skipped: each gold file names things its own way."""
    gold_by_source = {
        g.source_id: {f: c.name for c in g.concepts for f in c.forms()} for g in golds
    }
    labelled: dict[str, list[tuple[str, str]]] = {}  # source -> [(gold name, concept id)]
    for c in kn.concepts:
        for m in c.mentions:
            owner = gold_by_source.get(m.source_id)
            if owner is None or m.surface is None:
                continue
            g = owner.get(normalize(m.surface))
            if g:
                labelled.setdefault(m.source_id, []).append((g, c.id))
    tp = merged = should = 0
    for items in labelled.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                same_gold = items[i][0] == items[j][0]
                same_concept = items[i][1] == items[j][1]
                should += same_gold
                merged += same_concept
                tp += same_gold and same_concept
    return PRF(tp, merged, should)


def _pct(x: float) -> str:
    return f"{x:.0%}"


def summary_rows(scores: dict[str, list[SourceScore]]) -> list[list[str]]:
    """One row per model, totals over its sources (micro-averaged)."""
    rows = []
    for model, ss in scores.items():
        c = sum((s.concepts for s in ss), PRF())
        lc = sum((s.concepts_lenient for s in ss), PRF())
        core = sum((s.core_recall for s in ss), PRF())
        r = sum((s.relations for s in ss), PRF())
        ra = sum((s.relations_any_predicate for s in ss), PRF())
        rg = sum((s.relations_grounded for s in ss), PRF())
        rows.append(
            [
                model,
                f"{_pct(c.precision)} / {_pct(c.recall)} / {_pct(c.f1)}",
                f"{_pct(lc.precision)} / {_pct(lc.recall)}",
                _pct(core.recall),
                str(sum(s.duplicates for s in ss)),
                f"{_pct(r.precision)} / {_pct(r.recall)}",
                f"{_pct(ra.precision)} / {_pct(ra.recall)}",
                _pct(rg.precision),
                f"{c.n_pred} / {r.n_pred}",
            ]
        )
    return rows


SUMMARY_HEADER = [
    "Model",
    "Concepts P / R / F1",
    "Lenient P / R",
    "Core R",
    "Dups",
    "Relations P / R",
    "Any-pred P / R",
    "Rel P, ends in gold",
    "#concepts / #rels",
]


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)
