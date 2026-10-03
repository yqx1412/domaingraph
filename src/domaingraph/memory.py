"""D5: agent memory stored in the graph, the DomainGraph replacement for AgentOS's SQLite store.

A memory is a ``(:Fact:Memory)`` node: the ``Fact`` label puts it next to the extracted facts,
``Memory`` marks it as written by an agent, so graph reloads (which mirror one knowledge
file) never delete it. Each memory has

* a small integer ``seq`` (the id an agent sees, like SQLite's rowid),
* a ``scope``, so separate users or benchmark runs don't see each other's memories,
* a ``kind``: ``fact`` (something to remember) or ``episode`` (a finished task),
* a bge-m3 ``embedding`` in the ``memory_embedding`` vector index,
* ``ABOUT`` edges to the concepts it names, found the same way facts are linked in D3.

Recall is by meaning, not keywords: the query is embedded and compared with the memories in
its scope. A hit needs cosine similarity of at least ``min_score``, and among the top ``k``
the newest come first, so of two conflicting facts the newer is read first.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any

from domaingraph.graph import GraphError, GraphStore, _forms, link_fact
from domaingraph.llm import Embedder

MAX_CHARS = 2000
KINDS = ("fact", "episode")
_WORD = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


@dataclass
class Memory:
    id: int
    text: str
    kind: str
    scope: str
    created: float  # unix seconds
    score: float | None = None
    concepts: list[str] | None = None
    data: dict[str, str] | None = None  # episode fields: task, outcome, answer, model, agent

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class GraphMemory:
    def __init__(self, store: GraphStore, embedder: Embedder, *, min_score: float = 0.5) -> None:
        self.store = store
        self.embedder = embedder
        self.min_score = min_score
        self._forms: dict[str, list[tuple[str, ...]]] | None = None
        self._names: dict[str, str] = {}

    # -- setup ------------------------------------------------------------------------

    def _ensure_index(self) -> None:
        meta = self.store.meta()
        model = meta.get("embed_model")
        if model not in (None, self.embedder.model):
            raise GraphError(
                f"the graph's embeddings come from {model!r}, not {self.embedder.model!r}"
            )
        dims = meta.get("dims")
        if not dims:
            dims = len(self.embedder.embed(["dimension probe"])[0])
            self.store.run(
                "MERGE (m:Meta {key: 'graph'}) SET m.embed_model = $e, m.dims = $d",
                e=self.embedder.model,
                d=dims,
            )
        self.store.init_schema(dims)

    def _concept_forms(self) -> dict[str, list[tuple[str, ...]]]:
        if self._forms is None:
            rows = self.store.run(
                "MATCH (c:Concept) RETURN c.id AS id, c.name AS name, c.aliases AS aliases"
            )
            self._forms = {r["id"]: _forms([r["name"], *(r["aliases"] or [])]) for r in rows}
            self._names = {r["id"]: r["name"] for r in rows}
        return self._forms

    def refresh(self) -> None:
        """Forget the cached concept names, e.g. after a graph load."""
        self._forms = None

    # -- writing ----------------------------------------------------------------------

    def add(
        self,
        text: str,
        *,
        scope: str = "default",
        kind: str = "fact",
        data: dict[str, str] | None = None,
    ) -> tuple[Memory, bool]:
        """Store a memory. Returns ``(memory, created)``; the same text in the same scope and
        kind returns the existing one."""
        text = " ".join(text.split())
        if not text:
            raise ValueError("memory is empty")
        if len(text) > MAX_CHARS:
            raise ValueError(f"memory is too long ({len(text)} chars, limit {MAX_CHARS})")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        norm = _norm(text)
        found = self.store.run(
            """MATCH (m:Memory {scope: $scope, kind: $kind, norm: $norm})
            RETURN m.seq AS seq LIMIT 1""",
            scope=scope,
            kind=kind,
            norm=norm,
        )
        if found:
            return self.get(found[0]["seq"], scope=scope), False  # type: ignore[return-value]

        self._ensure_index()
        forms = self._concept_forms()
        about = link_fact(text, forms, forms)
        vec = self.embedder.embed([text])[0]
        seq = self.store.run(
            """MERGE (c:Meta {key: 'memory'})
            SET c.seq = coalesce(c.seq, 0) + 1
            RETURN c.seq AS seq"""
        )[0]["seq"]
        self.store.run(
            """CREATE (m:Fact:Memory {id: 'mem-' + toString($seq), seq: $seq, statement: $text,
                norm: $norm, scope: $scope, kind: $kind, origin: 'agent', created: $created})
            SET m += $data
            WITH m
            CALL db.create.setNodeVectorProperty(m, 'embedding', $vec)
            WITH m
            UNWIND $about AS cid
            MATCH (c:Concept {id: cid})
            MERGE (m)-[:ABOUT]->(c)""",
            seq=seq,
            text=text,
            norm=norm,
            scope=scope,
            kind=kind,
            created=time.time(),
            data={f"ep_{k}": v for k, v in (data or {}).items()},
            vec=vec,
            about=about,
        )
        return self.get(seq, scope=scope), True  # type: ignore[return-value]

    def forget(self, seq: int, *, scope: str = "default") -> bool:
        rows = self.store.run(
            """MATCH (m:Memory {seq: $seq, scope: $scope})
            DETACH DELETE m RETURN count(*) AS n""",
            seq=seq,
            scope=scope,
        )
        return rows[0]["n"] > 0

    def clear(self, scope: str) -> int:
        return self.store.run(
            "MATCH (m:Memory {scope: $scope}) DETACH DELETE m RETURN count(*) AS n",
            scope=scope,
        )[0]["n"]

    # -- reading ----------------------------------------------------------------------

    _RETURN = """RETURN m.seq AS id, m.statement AS text, m.kind AS kind, m.scope AS scope,
        m.created AS created, properties(m) AS props,
        [(m)-[:ABOUT]->(c:Concept) | c.name] AS concepts"""

    @staticmethod
    def _memory(r: dict[str, Any], score: float | None = None) -> Memory:
        data = {k[3:]: v for k, v in (r.get("props") or {}).items() if k.startswith("ep_")}
        return Memory(
            id=r["id"],
            text=r["text"],
            kind=r["kind"],
            scope=r["scope"],
            created=r["created"],
            score=score,
            concepts=r.get("concepts") or [],
            data=data or None,
        )

    def get(self, seq: int, *, scope: str = "default") -> Memory | None:
        rows = self.store.run(
            "MATCH (m:Memory {seq: $seq, scope: $scope}) " + self._RETURN,
            seq=seq,
            scope=scope,
        )
        return self._memory(rows[0]) if rows else None

    def list(self, *, scope: str = "default", kind: str | None = None, limit: int = 100):
        rows = self.store.run(
            "MATCH (m:Memory {scope: $scope}) WHERE $kind IS NULL OR m.kind = $kind "
            + self._RETURN
            + " ORDER BY m.created DESC LIMIT $limit",
            scope=scope,
            kind=kind,
            limit=limit,
        )
        return [self._memory(r) for r in rows]

    def search(
        self,
        query: str,
        *,
        scope: str = "default",
        kind: str | None = None,
        k: int = 5,
        min_score: float | None = None,
    ) -> list[Memory]:
        """The ``k`` most similar memories with cosine similarity of at least ``min_score``,
        newest first. (Neo4j reports cosine as ``(1 + cos) / 2``; this converts.)"""
        if not query.strip():
            return []
        floor = self.min_score if min_score is None else min_score
        if not self.store.run("MATCH (m:Memory {scope: $scope}) RETURN m LIMIT 1", scope=scope):
            return []
        vec = self.embedder.embed([query])[0]
        # The index ranks every memory, so ask for enough to survive the scope filter.
        total = self.store.run("MATCH (m:Memory) RETURN count(m) AS n")[0]["n"]
        rows = self.store.run(
            """CALL db.index.vector.queryNodes('memory_embedding', $n, $vec)
            YIELD node AS m, score
            WHERE m.scope = $scope AND ($kind IS NULL OR m.kind = $kind) AND score >= $floor
            WITH m, score ORDER BY score DESC LIMIT $k
            """
            + self._RETURN
            + ", score",
            n=max(total, 1),
            vec=vec,
            scope=scope,
            kind=kind,
            floor=(1 + floor) / 2,
            k=k,
        )
        hits = [self._memory(r, round(2 * r["score"] - 1, 4)) for r in rows]
        return sorted(hits, key=lambda m: m.created, reverse=True)
