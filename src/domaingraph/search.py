"""D4: three ways to search the graph store, plus a keyword baseline.

Every mode returns passages (``Chunk`` ids), best first, so they can be scored on the same
labels:

* **vector**: the query's bge-m3 embedding against Neo4j's ``chunk_embedding`` index.
* **graph**: no passage embeddings at all. The query is matched to concepts, by name or
  alias in its text and by the ``concept_embedding`` index, and passages are ranked by the
  concepts they mention (see :class:`GraphSearch`).
* **hybrid**: vector's top 50 passages, re-scored with the graph: ``cosine + 0.05 x graph
  score / best graph score among the 50``. The graph reorders, it never adds passages.
* **hybrid-rrf**: reciprocal rank fusion of vector and graph, then a small re-rank bonus for
  the query's concepts. This was the hybrid fixed before any scoring; it lost to vector.
* **bm25**: plain keyword search over the passage text, as a reference point. It is not one
  of the roadmap's three modes; it shows what the embeddings and the graph add over words.

The first version of every weight was committed before the query set was scored. The query
set is split by a hash of the query id into ``dev`` and ``test`` halves
(:func:`~domaingraph.evaluate_search.split_of`). The re-rank hybrid and its 0.05 bonus were
chosen on ``dev`` only, so ``test`` is the honest number for it.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from domaingraph.graph import GraphStore, _forms, _tokens, link_fact

Ranking = list[tuple[str, float]]  # (chunk id, score), best first


class Searcher(Protocol):
    name: str

    def search(self, query: str, vec: list[float] | None, k: int) -> Ranking: ...


# --- vector ----------------------------------------------------------------------------


class VectorSearch:
    name = "vector"

    def __init__(self, store: GraphStore) -> None:
        self.store = store

    def search(self, query: str, vec: list[float] | None, k: int) -> Ranking:
        assert vec is not None
        rows = self.store.run(
            """CALL db.index.vector.queryNodes('chunk_embedding', $k, $vec)
            YIELD node, score RETURN node.id AS id, score""",
            k=k,
            vec=vec,
        )
        return [(r["id"], r["score"]) for r in rows]


# --- graph -----------------------------------------------------------------------------


@dataclass
class GraphParams:
    seeds_by_vector: int = 3  # concept_embedding neighbours of the query
    seed_min_score: float = 0.55  # ... with at least this cosine similarity
    hop: float = 0.25  # weight of a 1-hop neighbour's mentions, relative to the seed's
    fact_bonus: float = 0.5  # per fact ABOUT the concept stated in the passage (max 3)


@dataclass
class _Concept:
    id: str
    name: str
    n_chunks: int
    forms: list[tuple[str, ...]]
    chunks: dict[str, int] = field(default_factory=dict)  # chunk -> facts about it there
    neighbours: set[str] = field(default_factory=set)


class GraphSearch:
    """Rank passages by the concepts they mention.

    1. **Seeds.** Concepts named in the query (name or alias as whole words, longest match
       first, as in fact linking) get weight 1. The ``concept_embedding`` index adds up to
       ``seeds_by_vector`` more, weighted by similarity, if above ``seed_min_score``.
    2. **Passages.** Each seed adds ``weight x idf`` to every passage that mentions it, where
       ``idf = log(1 + N / chunks mentioning the concept)``: a passage naming "radix sort"
       says more than one naming "node". Each fact about the seed stated in that passage
       adds ``fact_bonus`` more, up to three, since facts mark where a concept is explained
       rather than just named.
    3. **One hop.** Concepts linked to a seed by ``RELATED_TO`` or ``PART_OF`` add their
       passages too, at ``hop`` times the seed's weight.

    The mention/edge/fact structure is read from Neo4j once, at construction."""

    name = "graph"

    def __init__(self, store: GraphStore | None, params: GraphParams | None = None) -> None:
        self.store = store
        self.p = params or GraphParams()
        self.concepts: dict[str, _Concept] = {}
        self.n_chunks = 0
        self.order: dict[str, int] = {}  # deterministic tie-break
        if store is not None:
            self._load(store)

    def _load(self, store: GraphStore) -> None:
        concepts = store.run(
            "MATCH (c:Concept) RETURN c.id AS id, c.name AS name, c.aliases AS aliases"
        )
        mentions = store.run(
            """MATCH (c:Concept)-[:MENTIONED_IN]->(ch:Chunk)
            OPTIONAL MATCH (f:Fact)-[:ABOUT]->(c), (f)-[:MENTIONED_IN]->(ch)
            RETURN c.id AS c, ch.id AS ch, count(f) AS facts"""
        )
        edges = store.run(
            "MATCH (a:Concept)-[:RELATED_TO|PART_OF]->(b:Concept) RETURN a.id AS a, b.id AS b"
        )
        chunks = store.run(
            "MATCH (ch:Chunk)-[:PART_OF]->(s:Source) RETURN ch.id AS id ORDER BY s.id, ch.index"
        )
        self.build(concepts, mentions, edges, [r["id"] for r in chunks])

    def build(
        self,
        concepts: Sequence[dict[str, Any]],
        mentions: Sequence[dict[str, Any]],
        edges: Sequence[dict[str, Any]],
        chunk_ids: Sequence[str],
    ) -> None:
        self.n_chunks = len(chunk_ids)
        self.order = {c: i for i, c in enumerate(chunk_ids)}
        for c in concepts:
            self.concepts[c["id"]] = _Concept(
                id=c["id"],
                name=c["name"],
                n_chunks=0,
                forms=_forms([c["name"], *(c.get("aliases") or [])]),
            )
        for m in mentions:
            self.concepts[m["c"]].chunks[m["ch"]] = m.get("facts", 0)
        for c in self.concepts.values():
            c.n_chunks = len(c.chunks)
        for e in edges:
            if e["a"] in self.concepts and e["b"] in self.concepts and e["a"] != e["b"]:
                self.concepts[e["a"]].neighbours.add(e["b"])
                self.concepts[e["b"]].neighbours.add(e["a"])

    def idf(self, c: _Concept) -> float:
        return math.log(1 + self.n_chunks / c.n_chunks) if c.n_chunks else 0.0

    def seeds(self, query: str, vec: list[float] | None) -> dict[str, float]:
        forms = {cid: c.forms for cid, c in self.concepts.items()}
        seeds = dict.fromkeys(link_fact(query, self.concepts, forms), 1.0)
        if vec is not None and self.store is not None and self.p.seeds_by_vector:
            for r in self.store.run(
                """CALL db.index.vector.queryNodes('concept_embedding', $k, $vec)
                YIELD node, score RETURN node.id AS id, score""",
                k=self.p.seeds_by_vector,
                vec=vec,
            ):
                if r["score"] >= self.p.seed_min_score and r["id"] in self.concepts:
                    seeds[r["id"]] = max(seeds.get(r["id"], 0.0), r["score"])
        return seeds

    def score(self, seeds: dict[str, float]) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)

        def add(c: _Concept, w: float) -> None:
            base = w * self.idf(c)
            for ch, facts in c.chunks.items():
                scores[ch] += base * (1 + self.p.fact_bonus * min(facts, 3))

        for cid, w in seeds.items():
            c = self.concepts[cid]
            add(c, w)
            if self.p.hop:
                for nid in c.neighbours:
                    add(self.concepts[nid], w * self.p.hop)
        return scores

    def search(self, query: str, vec: list[float] | None, k: int) -> Ranking:
        scores = self.score(self.seeds(query, vec))
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.order.get(kv[0], 0)))
        return ranked[:k]


# --- hybrid ------------------------------------------------------------------------------


def rrf(rankings: Sequence[Ranking], k: int = 60, weights: Sequence[float] | None = None):
    """Reciprocal rank fusion: ``sum(w / (k + rank))`` over the rankings a passage is in."""
    weights = weights or [1.0] * len(rankings)
    out: dict[str, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, (cid, _) in enumerate(ranking, 1):
            out[cid] += w / (k + rank)
    return sorted(out.items(), key=lambda kv: -kv[1])


class HybridSearch:
    """Combine vector and graph, in one of two ways:

    * ``fusion="rrf"``: reciprocal rank fusion of both top ``pool`` lists (``weights`` per
      list), then a re-rank of the fused top ``rerank`` that adds ``bonus`` times the share
      of the query's seed concepts (by seed weight) the passage mentions.
    * ``fusion="rerank"``: keep vector's top ``pool`` candidates and re-score each as
      ``cosine + bonus * graph score / best graph score among them``. The graph can only
      reorder what vector found, never add passages."""

    name = "hybrid"

    def __init__(
        self,
        vector: VectorSearch,
        graph: GraphSearch,
        *,
        pool: int = 50,
        rrf_k: int = 60,
        rerank: int = 20,
        bonus: float = 0.05,
        weights: tuple[float, float] = (1.0, 1.0),
        fusion: str = "rerank",
    ) -> None:
        self.vector, self.graph = vector, graph
        self.pool, self.rrf_k, self.rerank, self.bonus = pool, rrf_k, rerank, bonus
        self.weights, self.fusion = weights, fusion

    def search(self, query: str, vec: list[float] | None, k: int) -> Ranking:
        seeds = self.graph.seeds(query, vec)
        v = self.vector.search(query, vec, self.pool)
        gscores = self.graph.score(seeds)
        if self.fusion == "rerank":
            best = max((gscores.get(cid, 0.0) for cid, _ in v), default=0.0) or 1.0
            out = [(cid, s + self.bonus * gscores.get(cid, 0.0) / best) for cid, s in v]
            out.sort(key=lambda kv: -kv[1])
            return out[:k]
        ranked_graph = sorted(
            gscores.items(),
            key=lambda kv: (-kv[1], self.graph.order.get(kv[0], 0)),
        )[: self.pool]
        fused = rrf([v, ranked_graph], self.rrf_k, self.weights)
        total = sum(seeds.values())
        if total and self.bonus:
            head = []
            for cid, s in fused[: self.rerank]:
                covered = sum(
                    w for sid, w in seeds.items() if cid in self.graph.concepts[sid].chunks
                )
                head.append((cid, s + self.bonus * covered / total))
            head.sort(key=lambda kv: -kv[1])
            fused = head + fused[self.rerank :]
        return fused[:k]


# --- keyword baseline ------------------------------------------------------------------------


class BM25Search:
    name = "bm25"

    def __init__(self, chunks: Sequence[tuple[str, str]], k1: float = 1.2, b: float = 0.75):
        self.ids = [cid for cid, _ in chunks]
        self.docs = [Counter(_tokens(text)) for _, text in chunks]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, vec: list[float] | None, k: int) -> Ranking:
        terms = [t for t in _tokens(query) if t in self.idf]
        scores = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in terms:
                f = d.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                    s += self.idf[t] * f * (self.k1 + 1) / norm
            if s:
                scores.append((self.ids[i], s))
        scores.sort(key=lambda kv: -kv[1])
        return scores[:k]
