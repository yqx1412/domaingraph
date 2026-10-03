"""D4 tests: search modes on a hand-built graph (no database) and the metrics."""

from __future__ import annotations

import pytest

from domaingraph.evaluate_search import (
    Query,
    QueryResult,
    load_queries,
    paired_bootstrap,
    split_of,
    summarize,
)
from domaingraph.search import (
    BM25Search,
    GraphParams,
    GraphSearch,
    HybridSearch,
    gold_graph,
    rrf,
)

CHUNKS = ["a:0", "a:1", "a:2", "a:3"]


def _graph(params: GraphParams | None = None) -> GraphSearch:
    g = GraphSearch(None, params or GraphParams(seeds_by_vector=0))
    g.build(
        concepts=[
            {"id": "radix", "name": "radix sort", "aliases": []},
            {"id": "counting", "name": "counting sort", "aliases": []},
            {"id": "node", "name": "node", "aliases": ["nodes"]},
            {"id": "stable", "name": "stable sort", "aliases": ["stability"]},
        ],
        mentions=[
            {"c": "radix", "ch": "a:2", "facts": 0},
            {"c": "counting", "ch": "a:1", "facts": 2},
            {"c": "counting", "ch": "a:2", "facts": 0},
            {"c": "node", "ch": "a:0", "facts": 0},
            {"c": "node", "ch": "a:1", "facts": 0},
            {"c": "node", "ch": "a:2", "facts": 0},
            {"c": "node", "ch": "a:3", "facts": 0},
            {"c": "stable", "ch": "a:3", "facts": 0},
        ],
        edges=[{"a": "radix", "b": "counting"}, {"a": "radix", "b": "stable"}],
        chunk_ids=CHUNKS,
    )
    return g


def test_graph_seeds_from_names_and_aliases():
    g = _graph()
    assert g.seeds("How does radix sort use stability?", None) == {"radix": 1.0, "stable": 1.0}
    assert g.seeds("nothing relevant here", None) == {}


def test_graph_ranks_by_idf_facts_and_hops():
    g = _graph()
    # A rare concept outweighs a common one: "node" is in every chunk, so its idf is lowest.
    assert g.idf(g.concepts["radix"]) > g.idf(g.concepts["node"])
    top = g.search("radix sort", None, 4)
    assert top[0][0] == "a:2"  # the seed's own passage
    # The 1-hop neighbours' passages follow; counting sort's chunk 1 carries 2 facts.
    assert [cid for cid, _ in top[1:3]] == ["a:1", "a:3"]
    no_hop = _graph(GraphParams(seeds_by_vector=0, hop=0)).search("radix sort", None, 4)
    assert [cid for cid, _ in no_hop] == ["a:2"]


def test_rrf_rewards_agreement():
    fused = rrf([[("x", 1), ("y", 1)], [("y", 1), ("z", 1)]], k=60)
    assert fused[0][0] == "y"
    assert {c for c, _ in fused} == {"x", "y", "z"}


class _FixedVector:
    name = "vector"

    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query, vec, k):
        return self.ranking[:k]


def test_hybrid_rerank_prefers_passages_with_query_concepts():
    g = _graph()
    # Vector puts a:0 first; it shares no concept with the query, a:2 has radix sort.
    vec = _FixedVector([("a:0", 0.9), ("a:2", 0.8), ("a:3", 0.7)])
    h = HybridSearch(vec, g, fusion="rrf", bonus=0.0)
    plain = [c for c, _ in h.search("radix sort", None, 3)]
    assert plain[0] == "a:2"  # first in graph, second in vector: RRF already agrees
    h_vec_only = HybridSearch(
        _FixedVector([("a:0", 0.9), ("a:3", 0.8)]), _graph(), fusion="rrf", bonus=0.05
    )
    ranked = [c for c, _ in h_vec_only.search("radix sort", None, 4)]
    assert ranked[0] == "a:2"


def test_hybrid_rerank_only_reorders_vector_candidates():
    g = _graph()
    vec = _FixedVector([("a:0", 0.80), ("a:3", 0.79), ("a:1", 0.70)])
    ranked = HybridSearch(vec, g, bonus=0.05).search("radix sort", None, 5)
    # a:2 has the best graph score but vector never returned it, so it stays out.
    assert [c for c, _ in ranked] == ["a:3", "a:0", "a:1"]
    # a:3 (stable sort, 1 hop) overtakes a:0 (no concept). a:1 has the best graph score of
    # the three (counting sort, 1 hop, 2 facts) but is too far behind on cosine.
    gs = g.score(g.seeds("radix sort", None))
    best = max(gs["a:1"], gs["a:3"])
    assert best == gs["a:1"]
    assert ranked[0][1] == pytest.approx(0.79 + 0.05 * gs["a:3"] / best)
    assert [c for c, _ in HybridSearch(vec, g, bonus=0.0).search("x", None, 5)] == [
        "a:0",
        "a:3",
        "a:1",
    ]


def test_bm25_prefers_rare_terms():
    bm = BM25Search(
        [("a:0", "the tree has nodes"), ("a:1", "radix sort sorts digits"), ("a:2", "the tree")]
    )
    assert bm.search("how does radix sort work", None, 3)[0][0] == "a:1"
    assert bm.search("unrelated words", None, 3) == []


def test_gold_graph_uses_gold_concepts_and_chunks():
    from types import SimpleNamespace as NS

    gold = NS(
        source_id="a",
        concepts=[
            NS(name="radix sort", aliases=[], chunks=[2]),
            NS(name="stable sort", aliases=["stability"], chunks=[3, 9]),  # 9 doesn't exist
        ],
        relations=[NS(subject="radix sort", object="stable sort")],
    )
    g = gold_graph([gold], CHUNKS, lambda s, i: f"{s}:{i}" if i < 4 else None)
    assert g.concepts["a/stable sort"].chunks == {"a:3": 0}
    ranked = [c for c, _ in g.search("why is stability needed", None, 4)]
    assert ranked == ["a:3", "a:2"]  # the seed, then its gold neighbour at 0.25x


def test_metrics():
    q = Query("q1", "x", "fact", frozenset({"a:1", "a:3"}))
    r = QueryResult(q, ["a:0", "a:3", "a:2", "a:1"])
    assert r.first_hit() == 2
    assert r.recall(1) == 0 and r.recall(2) == 0.5 and r.recall(4) == 1.0
    assert r.hit(1) == 0 and r.hit(5) == 1 and r.rr() == 0.5
    miss = QueryResult(q, ["a:0"])
    s = summarize([r, miss])
    assert s["MRR@10"] == pytest.approx(0.25) and s["Hit@5"] == 0.5


def test_load_queries_both_formats_and_bad_labels(tmp_path):
    (tmp_path / "lecture1.yaml").write_text(
        "source_id: s1\nqueries:\n"
        "  - {id: q1, query: 'What?', type: fact, relevant: [0, 2], answer: a}\n",
        encoding="utf-8",
    )
    (tmp_path / "cross.yaml").write_text(
        "sources: {l1: s1, l2: s2}\nqueries:\n"
        "  - {id: x1, query: 'Both?', type: cross, relevant: {l1: [1], l2: [0]}}\n",
        encoding="utf-8",
    )
    known = {("s1", 0), ("s1", 1), ("s1", 2), ("s2", 0)}

    def cid(s, i):
        return f"{s}:{i}" if (s, i) in known else None

    qs = {q.id: q for q in load_queries(tmp_path, cid)}
    assert qs["q1"].relevant == {"s1:0", "s1:2"}
    assert qs["x1"].relevant == {"s1:1", "s2:0"} and qs["x1"].type == "cross"

    (tmp_path / "lecture2.yaml").write_text(
        "source_id: s2\nqueries:\n  - {id: q2, query: 'Q', type: fact, relevant: [7]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no chunk 7"):
        load_queries(tmp_path, cid)


def test_split_is_stable_and_roughly_even():
    ids = [f"q-{i}" for i in range(200)]
    assert [split_of(i) for i in ids] == [split_of(i) for i in ids]
    assert 70 < sum(split_of(i) == "dev" for i in ids) < 130


def test_paired_bootstrap():
    d, lo, hi = paired_bootstrap([0.0] * 50, [1.0] * 50)
    assert (d, lo, hi) == (1.0, 1.0, 1.0)
    d, lo, hi = paired_bootstrap([0, 1] * 25, [1, 0] * 25)
    assert d == 0 and lo < 0 < hi
