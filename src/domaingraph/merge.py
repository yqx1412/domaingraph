"""D2: merge duplicate concepts across chunks (and sources) into one knowledge set.

Two passes:

1. **Names.** Every surface form is normalized (case, hyphens, articles, plurals). Concepts
   whose names normalize the same are merged, and so is a concept whose name equals another's
   alias (``"BST"`` -> ``binary search tree``). Two concepts that merely share an alias are
   not merged: models hand out generic aliases ("tree", "search") too freely.
2. **Embeddings.** The remaining groups' names are embedded (bge-m3) and groups whose
   cosine similarity is at least ``threshold`` are merged, if their types agree. Names only,
   not definitions: two definitions of different ideas from the same lecture read alike.

The result keeps every mention (chunk, time range, confidence), so each merged concept,
relation and fact still points back to where it was said.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from pydantic import BaseModel, Field

from domaingraph.extraction import ChunkResult
from domaingraph.llm import Embedder

_ARTICLES = re.compile(r"^(the|a|an)\s+")
_NONWORD = re.compile(r"[^a-z0-9+#'\s]")
_IRREGULAR = {"vertices": "vertex", "indices": "index", "matrices": "matrix", "children": "child"}
_KEEP_S = ("ss", "us", "is", "sis")  # class, radius, analysis


def _singular(word: str) -> str:
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if len(word) <= 3 or word.endswith(_KEEP_S):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def normalize(name: str) -> str:
    """``"The AVL-Trees"`` -> ``"avl tree"``; ``"Stirling's formula"`` -> ``"stirling formula"``."""
    s = name.lower().replace("-", " ").replace("_", " ").replace("\u2019", "'")
    s = s.replace("'s ", " ").replace("'", "")
    s = _NONWORD.sub(" ", s)
    s = _ARTICLES.sub("", " ".join(s.split()))
    words = s.split()
    if len(words) > 1 and words[-1] in ("operation", "operations"):
        words.pop()  # "insert operation" -> "insert"
    if words:
        words[-1] = _singular(words[-1])  # "binary search trees" -> "... tree"
    return " ".join(words)


class Mention(BaseModel):
    chunk_id: str
    source_id: str
    locator: str
    start: float | None = None
    end: float | None = None
    page: int | None = None
    confidence: float
    surface: str | None = None  # the name the chunk used, before merging


class Concept(BaseModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    type: str
    definition: str
    confidence: float  # highest single-mention confidence
    mentions: list[Mention] = Field(default_factory=list)

    def surface_forms(self) -> list[str]:
        return [self.name, *self.aliases]


class Relation(BaseModel):
    subject: str  # concept id
    predicate: str
    object: str
    confidence: float
    mentions: list[Mention] = Field(default_factory=list)


class Fact(BaseModel):
    statement: str
    concepts: list[str]  # concept ids
    confidence: float
    mention: Mention


class Knowledge(BaseModel):
    model: str
    embed_model: str | None = None
    threshold: float | None = None
    sources: list[str] = Field(default_factory=list)
    concepts: list[Concept] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    stats: dict[str, int] = Field(default_factory=dict)


class _UF:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def contrast_block(a: str, b: str) -> bool:
    """True when two normalized names look alike but likely name different ideas, which
    embeddings can't tell apart:

    * one adds words to the other: ``binary tree`` / ``binary search tree``,
      ``log n`` / ``n log n``, ``rotation`` / ``double rotation`` (a kind of, not the same);
    * they swap exactly one word: ``left rotate`` / ``right rotate``, ``find min`` /
      ``find max``, ``sorted array`` / ``sorted list``.

    Names with no word in common (``insert`` / ``insertion``) or that differ by more than
    one word each way (``AVL property`` / ``AVL balance condition``) are left to the
    similarity threshold."""
    ta, tb = Counter(a.split()), Counter(b.split())  # counts: "log n" vs "n log n"
    if ta == tb or not ta & tb:
        return False
    extra_a, extra_b = ta - tb, tb - ta
    if not extra_a or not extra_b:  # one name is the other plus words
        return True
    return extra_a.total() == 1 and extra_b.total() == 1


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", normalize(name)).strip("-") or "concept"


def _mention(r: ChunkResult, confidence: float, surface: str | None = None) -> Mention:
    return Mention(
        chunk_id=r.chunk_id,
        source_id=r.source_id,
        locator=r.locator,
        start=r.start,
        end=r.end,
        page=r.page_start,
        confidence=confidence,
        surface=surface,
    )


def name_groups(names: list[str], aliases: list[list[str]]) -> list[int]:
    """Pass 1: group index per item by normalized name (spaces ignored, so "inorder" and
    "in order" agree), and name == another's alias."""
    uf = _UF(len(names))
    by_name: dict[str, int] = {}

    def key(n: str) -> str:
        return normalize(n).replace(" ", "")

    for i, n in enumerate(names):
        k = key(n)
        if k in by_name:
            uf.union(by_name[k], i)
        else:
            by_name[k] = i
    for i, al in enumerate(aliases):
        for a in al:
            j = by_name.get(key(a))
            if j is not None:
                uf.union(i, j)
    return [uf.find(i) for i in range(len(names))]


def merge(
    results: Iterable[ChunkResult],
    *,
    embedder: Embedder | None = None,
    threshold: float = 0.9,
    model: str = "",
) -> Knowledge:
    results = list(results)
    # Every concept mention across all chunks.
    items: list[tuple[ChunkResult, object]] = [
        (r, c) for r in results for c in r.extraction.concepts
    ]
    names = [c.name for _, c in items]  # type: ignore[attr-defined]
    aliases = [list(c.aliases) for _, c in items]  # type: ignore[attr-defined]
    types = [c.type for _, c in items]  # type: ignore[attr-defined]
    group = name_groups(names, aliases)

    # Pass 2: embeddings over one representative name per group.
    stats = {"mentions": len(items), "after_names": len(set(group)), "embedding_merges": 0}
    if embedder is not None and items:
        roots = sorted(set(group))
        rep = {
            g: Counter(normalize(names[i]) for i in range(len(items)) if group[i] == g)
            for g in roots
        }
        rep_name = {g: rep[g].most_common(1)[0][0] for g in roots}
        gtype = {
            g: Counter(types[i] for i in range(len(items)) if group[i] == g).most_common(1)[0][0]
            for g in roots
        }
        vecs = dict(zip(roots, embedder.embed([rep_name[g] for g in roots]), strict=True))
        uf = _UF(max(roots) + 1)
        for a_i, a in enumerate(roots):
            for b in roots[a_i + 1 :]:
                if (
                    gtype[a] == gtype[b]
                    and not contrast_block(rep_name[a], rep_name[b])
                    and _cos(vecs[a], vecs[b]) >= threshold
                ):
                    if uf.find(a) != uf.find(b):
                        stats["embedding_merges"] += 1
                    uf.union(a, b)
        group = [uf.find(g) for g in group]

    members: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(group):
        members[g].append(i)

    concepts: list[Concept] = []
    id_of_item: dict[int, str] = {}
    used_ids: set[str] = set()
    for idx in sorted(members.values(), key=lambda m: m[0]):
        surface = Counter(names[i] for i in idx)
        # Most frequent wording; ties go to the shorter, then alphabetical.
        name = sorted(surface, key=lambda n: (-surface[n], len(n), n))[0]
        best = max(idx, key=lambda i: items[i][1].confidence)  # type: ignore[attr-defined]
        all_forms = list(dict.fromkeys([*surface, *(a for i in idx for a in aliases[i])]))
        seen_norm = {normalize(name)}
        alias_list = []
        for f in all_forms:
            k = normalize(f)
            if k not in seen_norm:
                seen_norm.add(k)
                alias_list.append(f)
        cid = base = _slug(name)
        n = 2
        while cid in used_ids:
            cid, n = f"{base}-{n}", n + 1
        used_ids.add(cid)
        mentions = [
            _mention(items[i][0], items[i][1].confidence, names[i])  # type: ignore[attr-defined]
            for i in idx
        ]
        concepts.append(
            Concept(
                id=cid,
                name=name,
                aliases=alias_list,
                type=Counter(types[i] for i in idx).most_common(1)[0][0],
                definition=items[best][1].definition,  # type: ignore[attr-defined]
                confidence=max(m.confidence for m in mentions),
                mentions=mentions,
            )
        )
        for i in idx:
            id_of_item[i] = cid

    # Map each chunk's (result, name) to the merged id, for relations and facts.
    local: dict[tuple[str, str], str] = {}
    for i, (r, c) in enumerate(items):
        local[(r.chunk_id, c.name)] = id_of_item[i]  # type: ignore[attr-defined]

    rels: dict[tuple[str, str, str], Relation] = {}
    for r in results:
        for rel in r.extraction.relations:
            s, o = local.get((r.chunk_id, rel.subject)), local.get((r.chunk_id, rel.object))
            if s is None or o is None or s == o:  # merging can make a relation reflexive
                continue
            key = (s, rel.predicate, o)
            m = _mention(r, rel.confidence)
            if key in rels:
                rels[key].mentions.append(m)
                rels[key].confidence = max(rels[key].confidence, rel.confidence)
            else:
                rels[key] = Relation(
                    subject=s,
                    predicate=rel.predicate,
                    object=o,
                    confidence=rel.confidence,
                    mentions=[m],
                )

    facts = [
        Fact(
            statement=f.statement,
            concepts=list(
                dict.fromkeys(
                    local[(r.chunk_id, n)] for n in f.concepts if (r.chunk_id, n) in local
                )
            ),
            confidence=f.confidence,
            mention=_mention(r, f.confidence),
        )
        for r in results
        for f in r.extraction.facts
    ]
    stats |= {"concepts": len(concepts), "relations": len(rels), "facts": len(facts)}
    return Knowledge(
        model=model,
        embed_model=getattr(embedder, "model", None),
        threshold=threshold if embedder else None,
        sources=list(dict.fromkeys(r.source_id for r in results)),
        concepts=concepts,
        relations=list(rels.values()),
        facts=facts,
        stats=stats,
    )
