"""D5: the operations the MCP server exposes, independent of MCP (so they are unit-testable).

Every passage comes back with its source title and timestamp (or page range), so an agent can
cite where something was said: "Lecture 7, 41:12-42:30".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from domaingraph.graph import GraphError, GraphStore
from domaingraph.llm import Embedder
from domaingraph.memory import GraphMemory
from domaingraph.models import timestamp
from domaingraph.pipeline import ingest, list_sources, load_source

SNIPPET = 700  # characters of passage text returned per hit
SEARCH_MODES = ("vector", "hybrid", "graph")


class ServiceError(RuntimeError):
    """A request the caller can fix: unknown concept, bad argument, disallowed path."""


@dataclass
class DomainGraphService:
    store: GraphStore
    embedder: Embedder
    out: Path = Path("data")
    ingest_roots: Sequence[Path] = ()  # ingest_source only reads files under these
    domain: str = "algorithms"
    memory: GraphMemory = field(init=False)
    _searchers: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.memory = GraphMemory(self.store, self.embedder)

    # -- search ---------------------------------------------------------------------

    def _searcher(self, mode: str):
        if mode not in SEARCH_MODES:
            raise ServiceError(f"mode must be one of {SEARCH_MODES}")
        if mode not in self._searchers:
            from domaingraph.search import GraphSearch, HybridSearch, VectorSearch

            vector = self._searchers.get("vector") or VectorSearch(self.store)
            self._searchers["vector"] = vector
            if mode in ("graph", "hybrid"):
                graph = self._searchers.get("graph") or GraphSearch(self.store)
                self._searchers["graph"] = graph
                self._searchers["hybrid"] = HybridSearch(vector, graph)
        return self._searchers[mode]

    def search(self, query: str, k: int = 5, mode: str = "vector") -> list[dict[str, Any]]:
        if not query.strip():
            raise ServiceError("query is empty")
        k = max(1, min(int(k), 20))
        searcher = self._searcher(mode)
        vec = self.embedder.embed([query])[0]
        ranked = searcher.search(query, vec, k)
        if not ranked:
            return []
        rows = self.store.run(
            """UNWIND $ids AS id
            MATCH (c:Chunk {id: id})-[:PART_OF]->(s:Source)
            RETURN c.id AS id, s.title AS source, c.locator AS at, c.start AS start,
                   c.page_start AS page, c.text AS text""",
            ids=[cid for cid, _ in ranked],
        )
        by_id = {r["id"]: r for r in rows}
        out = []
        for rank, (cid, score) in enumerate(ranked, 1):
            r = by_id.get(cid)
            if r is None:
                continue
            text = r["text"]
            out.append(
                {
                    "rank": rank,
                    "source": r["source"],
                    "at": r["at"],
                    "start_seconds": r["start"],
                    "text": text if len(text) <= SNIPPET else text[: SNIPPET - 3] + "...",
                    "score": round(score, 4),
                    "chunk": cid,
                }
            )
        return out

    # -- concepts -------------------------------------------------------------------

    def _concept(self, name: str) -> dict[str, Any]:
        c = self.store.find_concept(name)
        if c is not None:
            return c
        rows = self.store.run(
            """CALL db.index.vector.queryNodes('concept_embedding', 5, $vec)
            YIELD node RETURN node.name AS name""",
            vec=self.embedder.embed([name])[0],
        )
        hint = ", ".join(r["name"] for r in rows)
        raise ServiceError(f"no concept named {name!r}" + (f"; closest: {hint}" if hint else ""))

    def get_concept(self, name: str, max_mentions: int = 8) -> dict[str, Any]:
        c = self._concept(name)
        mentions = self.store.trace(c["id"])
        facts = self.store.run(
            """MATCH (f:Fact)-[:ABOUT]->(:Concept {id: $id})
            WHERE NOT f:Memory
            MATCH (f)-[:MENTIONED_IN]->(ch:Chunk)-[:PART_OF]->(s:Source)
            RETURN f.statement AS statement, s.title AS source, ch.locator AS at
            ORDER BY s.title, ch.start LIMIT 8""",
            id=c["id"],
        )
        return {
            "name": c["name"],
            "type": c["type"],
            "aliases": c["aliases"],
            "definition": c["definition"],
            "related": self.related_concepts(c["name"], k=8),
            "mentioned_in": [
                {"source": m["source"], "at": m["locator"]} for m in mentions[:max_mentions]
            ],
            "n_mentions": len(mentions),
            "facts": facts,
        }

    def related_concepts(
        self, name: str, predicate: str | None = None, k: int = 10
    ) -> list[dict[str, Any]]:
        c = self._concept(name)
        rows = self.store.neighbours(c["id"])
        if predicate:
            rows = [r for r in rows if r["predicate"] == predicate]
        return [
            {
                "relation": (
                    f"{c['name']} {r['predicate']} {r['other']}"
                    if r["dir"] == "out"
                    else f"{r['other']} {r['predicate']} {c['name']}"
                ),
                "concept": r["other"],
                "times_said": r["n"],
            }
            for r in rows[: max(1, k)]
        ]

    # -- memory ---------------------------------------------------------------------

    def add_fact(self, statement: str, scope: str = "default", kind: str = "fact", **data):
        data = {k: str(v) for k, v in data.items() if v is not None}
        bad = [k for k in data if not k.isidentifier()]
        if bad:
            raise ServiceError(f"detail keys must be identifiers: {bad}")
        try:
            mem, created = self.memory.add(statement, scope=scope, kind=kind, data=data or None)
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc
        return {**mem.as_dict(), "new": created}

    def recall_facts(self, query: str, scope: str = "default", k: int = 5, kind: str | None = None):
        return [m.as_dict() for m in self.memory.search(query, scope=scope, kind=kind, k=k)]

    def forget_fact(self, id: int, scope: str = "default") -> dict[str, Any]:
        return {"deleted": self.memory.forget(int(id), scope=scope), "id": int(id)}

    def list_facts(self, scope: str = "default", kind: str | None = None, limit: int = 100):
        return [m.as_dict() for m in self.memory.list(scope=scope, kind=kind, limit=limit)]

    def clear_scope(self, scope: str) -> int:
        if scope == "default":
            raise ServiceError("refusing to clear the default scope")
        return self.memory.clear(scope)

    # -- ingest ---------------------------------------------------------------------

    def _allowed(self, path: Path) -> Path:
        p = path.expanduser().resolve()
        for root in self.ingest_roots:
            if p.is_relative_to(Path(root).expanduser().resolve()):
                return p
        if not self.ingest_roots:
            raise ServiceError(
                "ingest_source is disabled: start the server with --allow-ingest <dir>"
            )
        allowed = ", ".join(str(r) for r in self.ingest_roots)
        raise ServiceError(f"{p} is outside the allowed ingest folders ({allowed})")

    def ingest_source(
        self,
        path: str,
        title: str | None = None,
        extract: bool = False,
        model: str = "qwen3:8b",
    ) -> dict[str, Any]:
        p = self._allowed(Path(path))
        try:
            r = ingest(p, out=self.out, transcriber=self._transcriber)
        except (FileNotFoundError, ValueError) as exc:
            raise ServiceError(str(exc)) from exc
        source, chunks = r.source, r.chunks
        if title:
            from domaingraph.pipeline import source_dir

            source.title = " ".join(title.split())
            (source_dir(self.out, source.id) / "source.json").write_text(
                source.model_dump_json(indent=2), encoding="utf-8"
            )
        embedded = self.store.load_passages(source, chunks, self.domain, self.embedder)
        result: dict[str, Any] = {
            "source_id": source.id,
            "title": source.title,
            "kind": source.kind,
            "chunks": len(chunks),
            "already_ingested": r.skipped,
            "embedded": embedded,
            "duration": timestamp(source.duration) if source.duration else None,
        }
        if extract:
            result["extraction"] = self._extract_and_load(source, chunks, model)
        self._searchers.clear()
        self.memory.refresh()
        return result

    @staticmethod
    def _transcriber():
        from domaingraph.asr import DEFAULT_MODEL, WhisperTranscriber

        return WhisperTranscriber(DEFAULT_MODEL)

    def _extract_and_load(self, source, chunks, model: str) -> dict[str, Any]:
        from domaingraph.extraction import extract_source, load_results, model_slug
        from domaingraph.graph import build_rows
        from domaingraph.llm import OllamaStructured
        from domaingraph.merge import merge

        meta = self.store.meta()
        if meta.get("model") not in (None, model):
            raise ServiceError(
                f"the graph holds knowledge from {meta['model']!r}; extract with that model"
            )
        llm = OllamaStructured(model)
        try:
            done = extract_source(llm, source, list(chunks), out=self.out)
        finally:
            llm.close()
        failed = sum(r.error is not None for r in done)
        results = []
        for s in list_sources(self.out):
            results += [r for r in load_results(self.out, s.id, model) if r.error is None]
        kn = merge(results, embedder=self.embedder, threshold=0.85, model=model)
        path = self.out / "knowledge" / f"{model_slug(model)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(kn.model_dump_json(indent=1), encoding="utf-8")
        sources = [load_source(self.out, s) for s in kn.sources]
        st = self.store.load(
            build_rows(kn, sources, self.domain), model=model, embedder=self.embedder
        )
        return {"failed_chunks": failed, "concepts": st.concepts, "relations": st.relations}


__all__ = ["DomainGraphService", "GraphError", "ServiceError"]
