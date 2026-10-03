"""D7: training pairs come only from the allowed sources (live test Neo4j, see test_graph)."""

from __future__ import annotations

import os

import pytest
from test_graph import _chunk, _knowledge, _m, _source  # pytest puts tests/ on sys.path

from domaingraph.finetune import Pair, build_pairs, load_pairs, save_pairs
from domaingraph.graph import GraphConfig, GraphStore, build_rows
from domaingraph.merge import Fact, Relation

TEST_URI = os.environ.get("NEO4J_TEST_URI")
live = pytest.mark.skipif(not TEST_URI, reason="set NEO4J_TEST_URI to a throwaway Neo4j")


@pytest.fixture
def store():
    s = GraphStore(GraphConfig(uri=TEST_URI, password=os.environ.get("NEO4J_TEST_PASSWORD")))
    s.reset()
    kn = _knowledge()
    kn.sources = ["s1", "s2"]
    # A second "test" lecture: one fact, and a relation stated in both lectures.
    kn.facts.append(
        Fact(statement="AVL trees rotate.", concepts=["avl-tree"], confidence=0.9,
             mention=_m(0, sid="s2"))
    )  # fmt: skip
    kn.concepts[0].mentions.append(_m(0, sid="s2", surface="AVL tree"))
    kn.relations.append(
        Relation(subject="bst", predicate="contrasts_with", object="rotation", confidence=0.9,
                 mentions=[_m(1), _m(0, sid="s2")])
    )  # fmt: skip
    sources = [
        (_source("s1"), [_chunk(i) for i in range(3)]),
        (_source("s2", "Lecture 2"), [_chunk(0, "s2")]),
    ]
    s.load(build_rows(kn, sources, "d"), model="m1")
    yield s
    s.reset()
    s.close()


@live
def test_pairs_never_use_excluded_sources(store):
    pairs = build_pairs(store, {"s1"})
    assert pairs and {p.source for p in pairs} == {"s1"}
    assert {p.kind for p in pairs} == {"fact", "concept", "relation"}
    texts = {p.positive for p in pairs}
    assert "passage 0 of s2" not in texts
    assert "AVL trees rotate." not in {p.anchor for p in pairs}
    # The relation also stated in s2 is left out; the s1-only one stays.
    rels = {(p.anchor, p.positive) for p in pairs if p.kind == "relation"}
    assert ("AVL tree", "binary search tree") in rels
    assert ("binary search tree", "rotation") not in rels
    # Concept anchors are names only (a merged definition may come from another lecture).
    assert {p.anchor for p in pairs if p.kind == "concept"} <= {
        "AVL tree", "binary search tree", "rotation",
    }  # fmt: skip
    assert build_pairs(store, {"nothing"}) == []


def test_pairs_roundtrip(tmp_path):
    pairs = [Pair("a", "b", "fact", "s1"), Pair("c", "d", "relation", "s2")]
    save_pairs(pairs, tmp_path / "p.jsonl")
    assert load_pairs(tmp_path / "p.jsonl") == pairs
