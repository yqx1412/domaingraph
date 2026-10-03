"""D3 tests.

Unit tests build rows without a database. Integration tests need a throwaway Neo4j, named by
``NEO4J_TEST_URI`` (+ ``NEO4J_TEST_PASSWORD``); they EMPTY that database, so they never fall
back to the normal ``NEO4J_URI``. CI starts one as a service container. Locally::

    docker run -d --rm --name dg-neo4j-test -p 127.0.0.1:7688:7687 \
        -e NEO4J_AUTH=neo4j/testpassword neo4j:5.26.31-community
    NEO4J_TEST_URI=bolt://127.0.0.1:7688 NEO4J_TEST_PASSWORD=testpassword uv run pytest
"""

from __future__ import annotations

import hashlib
import os

import pytest

from domaingraph.graph import (
    GraphConfig,
    GraphError,
    GraphStore,
    _forms,
    build_rows,
    concept_text,
    embed_hash,
    link_fact,
    needs_embedding,
)
from domaingraph.merge import Concept, Fact, Knowledge, Mention, Relation
from domaingraph.models import Chunk, Source


def _source(sid="s1", title="Lecture 1") -> Source:
    return Source(
        id=sid, path=f"{sid}.mp3", kind="audio", title=title, sha256=sid, bytes=1, duration=600
    )


def _chunk(i, sid="s1") -> Chunk:
    return Chunk(
        id=f"{sid}:{i}",
        source_id=sid,
        index=i,
        text=f"passage {i} of {sid}",
        n_words=4,
        start=60.0 * i,
        end=60.0 * i + 75,
    )


def _m(i, sid="s1", surface=None, conf=0.9) -> Mention:
    ch = _chunk(i, sid)
    return Mention(
        chunk_id=ch.id,
        source_id=sid,
        locator=ch.locator(),
        start=ch.start,
        end=ch.end,
        confidence=conf,
        surface=surface,
    )


def _knowledge(model="m1") -> Knowledge:
    return Knowledge(
        model=model,
        sources=["s1"],
        concepts=[
            Concept(
                id="avl-tree",
                name="AVL tree",
                aliases=["AVL"],
                type="data_structure",
                definition="A balanced BST.",
                confidence=0.9,
                mentions=[
                    _m(2, surface="AVL tree"),
                    _m(0, surface="AVL"),
                    _m(0, surface="AVL tree"),
                ],
            ),
            Concept(
                id="bst",
                name="binary search tree",
                aliases=["BST"],
                type="data_structure",
                definition="",
                confidence=0.8,
                mentions=[_m(1)],
            ),
            Concept(
                id="rotation",
                name="rotation",
                type="operation",
                definition="Restructures a subtree.",
                confidence=0.9,
                mentions=[_m(2)],
            ),
        ],
        relations=[
            Relation(
                subject="avl-tree",
                predicate="is_a",
                object="bst",
                confidence=0.9,
                mentions=[_m(0), _m(2)],
            ),
            Relation(
                subject="rotation",
                predicate="part_of",
                object="avl-tree",
                confidence=0.8,
                mentions=[_m(2)],
            ),
            Relation(
                subject="avl-tree",
                predicate="uses",
                object="ghost",
                confidence=0.5,
                mentions=[_m(2)],
            ),
        ],
        facts=[
            Fact(
                statement="AVL trees have height O(log n).",
                concepts=["avl-tree", "ghost"],
                confidence=0.9,
                mention=_m(0),
            ),
            Fact(
                statement="AVL trees have height O(log n).",
                concepts=["avl-tree"],
                confidence=0.9,
                mention=_m(0),
            ),
            Fact(
                statement="From a source that isn't loaded.",
                concepts=[],
                confidence=0.9,
                mention=_m(0, sid="s9"),
            ),
        ],
    )


def _sources():
    return [(_source(), [_chunk(i) for i in range(3)])]


# --- unit ------------------------------------------------------------------------------------


def test_build_rows_shapes_the_graph():
    rows = build_rows(_knowledge(), _sources(), "algorithms")
    assert rows.domain == "algorithms"
    assert [c["id"] for c in rows.chunks] == ["s1:0", "s1:1", "s1:2"]
    assert rows.chunks[1]["locator"] == "01:00-02:15"

    # Two mentions in chunk 0 collapse into one edge listing both surface forms.
    avl = [m for m in rows.mentions if m["concept"] == "avl-tree"]
    assert sorted(m["chunk"] for m in avl) == ["s1:0", "s1:2"]
    assert next(m for m in avl if m["chunk"] == "s1:0")["surfaces"] == ["AVL", "AVL tree"]
    concept = next(c for c in rows.concepts if c["id"] == "avl-tree")
    assert (concept["n_mentions"], concept["n_chunks"]) == (3, 2)

    # part_of goes to PART_OF; the relation to an unknown concept is dropped.
    assert [(r["subject"], r["predicate"], r["object"]) for r in rows.relations] == [
        ("avl-tree", "is_a", "bst")
    ]
    assert rows.relations[0]["chunks"] == ["s1:0", "s1:2"]
    assert [(r["subject"], r["object"]) for r in rows.part_of] == [("rotation", "avl-tree")]

    # Duplicate facts collapse, unknown concepts and unloaded chunks are dropped.
    assert len(rows.facts) == 1
    assert rows.facts[0]["concepts"] == ["avl-tree"]


def test_mentions_of_unloaded_chunks_are_dropped():
    kn = _knowledge()
    kn.concepts[1].mentions = [_m(0, sid="s9")]
    rows = build_rows(kn, _sources(), "d")
    assert not [m for m in rows.mentions if m["concept"] == "bst"]
    assert next(c for c in rows.concepts if c["id"] == "bst")["n_chunks"] == 0


def test_link_fact_matches_whole_names_longest_first():
    forms = {
        "bst": _forms(["binary search tree", "BST"]),
        "bs": _forms(["binary search"]),
        "tree": _forms(["tree"]),
        "heap": _forms(["heap"]),
        "rot": _forms(["rotation"]),
    }
    cands = {"bst", "bs", "tree", "heap", "rot"}
    # "binary search trees" covers "binary search" and "tree"; plural folded; order kept.
    assert link_fact("Rotations keep binary search trees balanced.", cands, forms) == [
        "rot",
        "bst",
    ]
    assert link_fact("A BST is not a heap; heaps aren't sorted.", cands, forms) == [
        "bst",
        "heap",
    ]
    assert link_fact("Binary search halves the range.", cands, forms) == ["bs"]
    assert link_fact("A heapify step.", cands, forms) == []  # whole words only
    assert link_fact("A heap.", {"bst"}, forms) == []  # only concepts in the chunk


def test_facts_without_model_links_are_linked_by_text():
    kn = _knowledge()
    kn.facts = [
        Fact(
            statement="A rotation keeps an AVL tree balanced.",
            concepts=[],
            confidence=0.9,
            mention=_m(2),
        ),
        Fact(statement="Unrelated.", concepts=[], confidence=0.9, mention=_m(2)),
    ]
    rows = build_rows(kn, _sources(), "d")
    assert [(f["concepts"], f["linked_by"]) for f in rows.facts] == [
        (["rotation", "avl-tree"], "text"),
        ([], "text"),
    ]


def test_concept_text_and_needs_embedding():
    assert concept_text({"name": "heap", "definition": "A tree."}) == "heap: A tree."
    assert concept_text({"name": "heap", "definition": ""}) == "heap"
    items = [("a", "x"), ("b", "y"), ("c", "z")]
    existing = {"a": embed_hash("m", "x"), "b": embed_hash("m", "old"), "c": embed_hash("m2", "z")}
    assert [i for i, _, _ in needs_embedding(items, existing, "m")] == ["b", "c"]


def test_config_from_env_file(tmp_path, monkeypatch):
    for k in list(os.environ):
        if k.startswith("NEO4J_"):
            monkeypatch.delenv(k)
    env = tmp_path / ".env"
    env.write_text("# comment\nNEO4J_PASSWORD='secret'\nNEO4J_URI=bolt://h:1\n", encoding="utf-8")
    cfg = GraphConfig.from_env(env)
    assert (cfg.uri, cfg.user, cfg.password) == ("bolt://h:1", "neo4j", "secret")
    monkeypatch.setenv("NEO4J_PASSWORD", "from-env")
    assert GraphConfig.from_env(env).password == "from-env"
    assert GraphConfig.from_env(tmp_path / "missing").uri == "bolt://127.0.0.1:7687"


def test_store_without_password_explains(monkeypatch):
    with pytest.raises(GraphError, match="NEO4J_PASSWORD"):
        GraphStore(GraphConfig(password=None))


# --- integration -----------------------------------------------------------------------------

TEST_URI = os.environ.get("NEO4J_TEST_URI")
live = pytest.mark.skipif(not TEST_URI, reason="set NEO4J_TEST_URI to a throwaway Neo4j")


class FakeEmbedder:
    """Deterministic vectors from a hash, so different texts point different ways."""

    def __init__(self, model="fake", dims=4):
        self.model, self.dims, self.calls = model, dims, 0

    def embed(self, texts):
        self.calls += len(texts)
        return [
            [b / 255 + 0.01 for b in hashlib.sha256(t.encode()).digest()[: self.dims]]
            for t in texts
        ]


@pytest.fixture
def store():
    cfg = GraphConfig(uri=TEST_URI, password=os.environ.get("NEO4J_TEST_PASSWORD"))
    s = GraphStore(cfg)
    s.reset()
    yield s
    s.reset()
    s.close()


@live
def test_load_is_idempotent(store):
    rows = build_rows(_knowledge(), _sources(), "algorithms")
    emb = FakeEmbedder()
    first = store.load(rows, model="m1", embedder=emb)
    before = store.counts()
    calls = emb.calls
    second = store.load(rows, model="m1", embedder=emb)
    assert store.counts() == before
    # Unchanged text isn't re-embedded, and the stored dims make the probe unnecessary.
    assert second.embedded == 0 and emb.calls == calls
    assert first.embedded == 3 + 3  # 3 concepts + 3 chunks
    assert before["Concept"] == 3 and before["Chunk"] == 3 and before["Fact"] == 1
    assert before["RELATED_TO"] == 1
    # PART_OF: 1 concept->concept, 3 chunk->source, 1 source->domain, 3 concept->domain
    # (bst's only mention is chunk 1, which is loaded).
    assert before["PART_OF"] == 1 + 3 + 1 + 3
    assert before["MENTIONED_IN"] == 4 + 1  # concept edges (avl 2, bst 1, rotation 1) + fact
    assert before["ABOUT"] == 1


@live
def test_trace_walks_concept_to_timestamps(store):
    store.load(build_rows(_knowledge(), _sources(), "algorithms"), model="m1")
    c = store.find_concept("avl")  # by alias, case-insensitive
    assert c["id"] == "avl-tree"
    trace = store.trace("avl-tree")
    assert [(r["source"], r["locator"]) for r in trace] == [
        ("Lecture 1", "00:00-01:15"),
        ("Lecture 1", "02:00-03:15"),
    ]
    assert trace[0]["said_as"] == ["AVL", "AVL tree"]
    rel = {(r["dir"], r["predicate"], r["other"]) for r in store.neighbours("avl-tree")}
    assert rel == {("out", "is_a", "binary search tree"), ("in", "part_of", "rotation")}


@live
def test_reload_prunes_what_disappeared(store):
    store.load(build_rows(_knowledge(), _sources(), "d"), model="m1")
    kn = _knowledge()
    kn.concepts = [c for c in kn.concepts if c.id != "rotation"]
    kn.facts = []
    st = store.load(build_rows(kn, _sources(), "d"), model="m1")
    assert st.removed["concepts"] == 1 and st.removed["facts"] == 1
    counts = store.counts()
    assert counts["Concept"] == 2 and counts["Fact"] == 0 and "ABOUT" not in counts


@live
def test_refuses_to_mix_models_unless_replace(store):
    rows = build_rows(_knowledge(), _sources(), "d")
    store.load(rows, model="m1")
    with pytest.raises(GraphError, match="--replace"):
        store.load(rows, model="m2")
    store.load(rows, model="m2", replace=True)
    assert store.meta()["model"] == "m2"
    assert store.counts()["Concept"] == 3


@live
def test_vector_index_search(store):
    emb = FakeEmbedder()
    store.load(build_rows(_knowledge(), _sources(), "d"), model="m1", embedder=emb)
    assert store.meta()["dims"] == 4
    vec = emb.embed([concept_text({"name": "rotation", "definition": "Restructures a subtree."})])
    hits = store.similar(vec[0], "Concept", 3)
    assert hits[0]["name"] == "rotation" and hits[0]["score"] == pytest.approx(
        1.0, abs=1e-4
    )  # stored as float32
    chunk_hits = store.similar(emb.embed(["passage 1 of s1"])[0], "Chunk", 1)
    assert chunk_hits[0]["locator"] == "01:00-02:15"
