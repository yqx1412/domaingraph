"""D5 tests: graph memory and the service, against a throwaway Neo4j (see test_graph.py),
plus the MCP server's tool list. A fake embedder gives related texts related vectors."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
from test_graph import _knowledge, _sources  # pytest puts tests/ on sys.path

from domaingraph.graph import GraphConfig, GraphStore, build_rows
from domaingraph.memory import GraphMemory
from domaingraph.service import DomainGraphService, ServiceError

TEST_URI = os.environ.get("NEO4J_TEST_URI")
live = pytest.mark.skipif(not TEST_URI, reason="set NEO4J_TEST_URI to a throwaway Neo4j")

# Topic words -> axis. Texts sharing topics point the same way; "exam" and "test" share one,
# so recall by meaning can be checked without a real model.
_AXES = {
    "exam": 0, "test": 0, "quiz": 0,
    "wifi": 1, "network": 1,
    "avl": 2, "tree": 2, "rotation": 2,
    "passage": 3, "task": 4, "deploy": 5,
}  # fmt: skip


class TopicEmbedder:
    model = "topic"

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.05] * 6
            for w in t.lower().replace(".", " ").replace(",", " ").split():
                if w in _AXES:
                    v[_AXES[w]] += 1.0
            n = math.sqrt(sum(x * x for x in v))
            out.append([x / n for x in v])
        return out


@pytest.fixture
def store():
    s = GraphStore(GraphConfig(uri=TEST_URI, password=os.environ.get("NEO4J_TEST_PASSWORD")))
    s.reset()
    yield s
    s.reset()
    s.close()


@pytest.fixture
def service(store, tmp_path):
    emb = TopicEmbedder()
    store.load(build_rows(_knowledge(), _sources(), "algorithms"), model="m1", embedder=emb)
    return DomainGraphService(store, emb, out=tmp_path / "data", ingest_roots=[tmp_path / "in"])


@live
def test_memory_add_dedup_recall_forget(store):
    mem = GraphMemory(store, TopicEmbedder())
    a, new_a = mem.add("My exam is on Friday.", scope="u1")
    again, new_again = mem.add("my  exam is on friday", scope="u1")
    assert new_a and not new_again and again.id == a.id
    b, _ = mem.add("The wifi network is Coffeehouse.", scope="u1")
    other, _ = mem.add("My exam is on Friday.", scope="u2")  # same text, other scope
    assert len({a.id, b.id, other.id}) == 3

    hits = mem.search("when is the test", scope="u1")
    assert [h.text for h in hits] == ["My exam is on Friday."]
    assert mem.search("when is the test", scope="nobody") == []
    assert mem.forget(a.id, scope="u2") is False  # scopes are separate
    assert mem.forget(a.id, scope="u1") is True
    assert [m.id for m in mem.list(scope="u1")] == [b.id]
    assert mem.clear("u2") == 1


@live
def test_newest_first_and_episode_details(store):
    mem = GraphMemory(store, TopicEmbedder())
    mem.add("The wifi network is Teahouse.", scope="s")
    mem.add("Update: the wifi network was renamed to Coffeehouse.", scope="s")
    hits = mem.search("wifi network name", scope="s")
    assert hits[0].text.startswith("Update")
    ep, _ = mem.add(
        "Task: deploy. Outcome: completed.",
        scope="s",
        kind="episode",
        data={"task": "deploy", "outcome": "completed", "answer": "ok"},
    )
    assert ep.data == {"task": "deploy", "outcome": "completed", "answer": "ok"}
    assert [m.kind for m in mem.search("deploy task", scope="s", kind="episode")] == ["episode"]


@live
def test_memories_survive_graph_reloads_and_link_concepts(store):
    emb = TopicEmbedder()
    rows = build_rows(_knowledge(), _sources(), "d")
    store.load(rows, model="m1", embedder=emb)
    mem = GraphMemory(store, emb)
    m, _ = mem.add("An AVL tree needs a rotation after some inserts.", scope="s")
    assert set(m.concepts) == {"AVL tree", "rotation"}
    store.load(rows, model="m1", embedder=emb)  # reload: prunes facts not in the file
    store.load(rows, model="m2", embedder=emb, replace=True)
    assert [x.text for x in mem.list(scope="s")] == [m.text]


@live
def test_service_search_concept_and_errors(service, tmp_path):
    hits = service.search("passage 1 of s1", k=2)
    # The topic embedder gives every chunk the same vector, so only the shape is checked.
    assert len(hits) == 2 and hits[0]["source"] == "Lecture 1"
    assert hits[0]["at"] in {"00:00-01:15", "01:00-02:15", "02:00-03:15"}
    assert set(hits[0]) == {"rank", "source", "at", "start_seconds", "text", "score", "chunk"}
    c = service.get_concept("avl")
    assert c["name"] == "AVL tree"
    assert {"relation": "AVL tree is_a binary search tree", "concept": "binary search tree",
            "times_said": 2} in c["related"]  # fmt: skip
    assert c["mentioned_in"][0] == {"source": "Lecture 1", "at": "00:00-01:15"}
    assert [r["concept"] for r in service.related_concepts("AVL tree", "part_of")] == ["rotation"]
    with pytest.raises(ServiceError, match="closest"):
        service.get_concept("splay tree")
    with pytest.raises(ServiceError, match="mode"):
        service.search("x", mode="magic")
    with pytest.raises(ServiceError, match="default scope"):
        service.clear_scope("default")
    with pytest.raises(ServiceError, match="identifiers"):
        service.add_fact("x", "s", "fact", **{"bad key": "v"})


@live
def test_ingest_is_confined_and_adds_passages(service, tmp_path):
    inbox = tmp_path / "in"
    inbox.mkdir()
    (inbox / "notes.md").write_text(
        "# Splay trees\n\nA splay tree moves each accessed node to the root.\n", encoding="utf-8"
    )
    outside = tmp_path / "secret.txt"
    outside.write_text("password", encoding="utf-8")
    with pytest.raises(ServiceError, match="outside the allowed"):
        service.ingest_source(str(outside))
    with pytest.raises(ServiceError, match="outside the allowed"):
        service.ingest_source(str(inbox / ".." / "secret.txt"))
    r = service.ingest_source(str(inbox / "notes.md"), title="My notes")
    assert r["title"] == "My notes" and r["chunks"] == 1 and r["embedded"] == 1
    again = service.ingest_source(str(inbox / "notes.md"))
    assert again["already_ingested"] is True and again["embedded"] == 0
    assert service.store.counts()["Source"] == 2


def test_ingest_disabled_without_roots():
    svc = DomainGraphService.__new__(DomainGraphService)
    svc.ingest_roots = ()
    with pytest.raises(ServiceError, match="--allow-ingest"):
        svc._allowed(Path("x.txt"))


def test_server_lists_its_tools():
    import asyncio

    from domaingraph.mcp_server import build_server

    server = build_server(DomainGraphService.__new__(DomainGraphService))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {
        "search", "get_concept", "related_concepts", "add_fact", "recall_facts",
        "forget_fact", "list_facts", "clear_scope", "ingest_source",
    }  # fmt: skip
