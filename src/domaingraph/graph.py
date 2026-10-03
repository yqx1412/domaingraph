"""D3: the Neo4j graph store.

Schema::

    (:Domain {name})
    (:Source {id, title, kind, path, duration})       -[:PART_OF]->     (:Domain)
    (:Chunk  {id, index, text, start, end, locator})  -[:PART_OF]->     (:Source)
    (:Concept {id, name, aliases, type, definition})  -[:PART_OF]->     (:Domain)
    (:Concept) -[:RELATED_TO {predicate, ...}]-> (:Concept)   is_a, uses, solves, ...
    (:Concept) -[:PART_OF {predicate: "part_of", ...}]-> (:Concept)
    (:Concept) -[:MENTIONED_IN {surfaces, confidence, start, end, locator}]-> (:Chunk)
    (:Fact {id, statement}) -[:MENTIONED_IN]-> (:Chunk),  (:Fact) -[:ABOUT]-> (:Concept)

``ABOUT`` comes from the fact's own concept list when the extractor filled it in, and
otherwise from the concepts of the same chunk whose names occur in the statement
(:func:`link_fact`); ``Fact.linked_by`` says which.

``Chunk`` is the one addition to the roadmap's node list: a timestamp belongs to a passage,
not to a whole lecture, and D4's vector-only search ranks passages.

Concepts and chunks carry an ``embedding`` in Neo4j's own vector indexes
(``concept_embedding``, ``chunk_embedding``). A concept is embedded as "name: definition".

**Loading is idempotent.** Every node and edge is written with ``MERGE`` on a stable key,
so loading the same knowledge twice leaves the graph unchanged. Concepts, facts and their
edges from an earlier load that the new one no longer contains are removed, so the graph
always mirrors one knowledge file. Embeddings are recomputed only for text that changed
(each node stores a hash of the embedded text and the model).
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from domaingraph.llm import Embedder
from domaingraph.merge import Knowledge, _singular
from domaingraph.models import Chunk, Source

DEFAULT_URI = "bolt://127.0.0.1:7687"
BATCH = 500


class GraphError(RuntimeError):
    """The graph store is unreachable, misconfigured, or holds incompatible data."""


# --- connection settings -------------------------------------------------------------------


@dataclass
class GraphConfig:
    uri: str = DEFAULT_URI
    user: str = "neo4j"
    password: str | None = None
    database: str | None = None

    @classmethod
    def from_env(cls, env_file: Path | None = Path(".env")) -> GraphConfig:
        """``NEO4J_URI`` / ``NEO4J_USER`` / ``NEO4J_PASSWORD``, with ``.env`` as a fallback.
        Real environment variables win over the file."""
        values = _read_env_file(env_file) if env_file else {}
        values |= {k: v for k, v in os.environ.items() if k.startswith("NEO4J_")}
        return cls(
            uri=values.get("NEO4J_URI", DEFAULT_URI),
            user=values.get("NEO4J_USER", "neo4j"),
            password=values.get("NEO4J_PASSWORD"),
            database=values.get("NEO4J_DATABASE"),
        )


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


# --- rows: what gets written, built without a database (unit-testable) ----------------------


@dataclass
class GraphRows:
    domain: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    concepts: list[dict[str, Any]] = field(default_factory=list)
    mentions: list[dict[str, Any]] = field(default_factory=list)  # concept -> chunk
    relations: list[dict[str, Any]] = field(default_factory=list)  # RELATED_TO
    part_of: list[dict[str, Any]] = field(default_factory=list)  # concept PART_OF concept
    facts: list[dict[str, Any]] = field(default_factory=list)


def fact_id(statement: str, chunk_id: str) -> str:
    return hashlib.sha1(f"{chunk_id}\n{statement}".encode()).hexdigest()[:16]


def concept_text(c: dict[str, Any]) -> str:
    return f"{c['name']}: {c['definition']}" if c.get("definition") else c["name"]


def build_rows(
    kn: Knowledge, sources: Sequence[tuple[Source, Sequence[Chunk]]], domain: str
) -> GraphRows:
    """Turn a knowledge file plus its sources' chunks into rows for the writer.

    Several mentions of one concept in one chunk become a single ``MENTIONED_IN`` edge that
    lists every surface form. A mention whose chunk isn't among ``sources`` is dropped, since
    the edge would have nowhere to point."""
    rows = GraphRows(domain=domain)
    chunk_ids: set[str] = set()
    for src, chunks in sources:
        rows.sources.append(
            {
                "id": src.id,
                "title": src.title,
                "kind": src.kind,
                "path": src.path,
                "duration": src.duration,
                "pages": src.pages,
                "url": src.url,
            }
        )
        for ch in chunks:
            chunk_ids.add(ch.id)
            rows.chunks.append(
                {
                    "id": ch.id,
                    "source_id": ch.source_id,
                    "index": ch.index,
                    "text": ch.text,
                    "n_words": ch.n_words,
                    "start": ch.start,
                    "end": ch.end,
                    "page_start": ch.page_start,
                    "page_end": ch.page_end,
                    "heading": ch.heading,
                    "locator": ch.locator(),
                }
            )

    concept_ids = set()
    for c in kn.concepts:
        concept_ids.add(c.id)
        per_chunk: dict[str, dict[str, Any]] = {}
        for m in c.mentions:
            if m.chunk_id not in chunk_ids:
                continue
            row = per_chunk.setdefault(
                m.chunk_id,
                {
                    "concept": c.id,
                    "chunk": m.chunk_id,
                    "confidence": m.confidence,
                    "surfaces": [],
                    "start": m.start,
                    "end": m.end,
                    "locator": m.locator,
                },
            )
            row["confidence"] = max(row["confidence"], m.confidence)
            if m.surface and m.surface not in row["surfaces"]:
                row["surfaces"].append(m.surface)
        rows.mentions += per_chunk.values()
        rows.concepts.append(
            {
                "id": c.id,
                "name": c.name,
                "aliases": c.aliases,
                "type": c.type,
                "definition": c.definition,
                "confidence": c.confidence,
                "n_mentions": len(c.mentions),
                "n_chunks": len(per_chunk),
            }
        )

    for r in kn.relations:
        if r.subject not in concept_ids or r.object not in concept_ids:
            continue
        chunks = sorted({m.chunk_id for m in r.mentions})
        row = {
            "subject": r.subject,
            "object": r.object,
            "predicate": r.predicate,
            "confidence": r.confidence,
            "n_mentions": len(r.mentions),
            "chunks": chunks,
        }
        (rows.part_of if r.predicate == "part_of" else rows.relations).append(row)

    seen_facts: set[str] = set()
    forms = {c.id: _forms(c.surface_forms()) for c in kn.concepts}
    in_chunk: dict[str, set[str]] = {}
    for m in rows.mentions:
        in_chunk.setdefault(m["chunk"], set()).add(m["concept"])
    for f in kn.facts:
        if f.mention.chunk_id not in chunk_ids:
            continue
        fid = fact_id(f.statement, f.mention.chunk_id)
        if fid in seen_facts:
            continue
        seen_facts.add(fid)
        about = [x for x in f.concepts if x in concept_ids]
        linked_by = "model"
        if not about:  # the extractor almost never fills this in; match the text instead
            linked_by = "text"
            about = link_fact(f.statement, in_chunk.get(f.mention.chunk_id, set()), forms)
        rows.facts.append(
            {
                "id": fid,
                "statement": f.statement,
                "confidence": f.confidence,
                "chunk": f.mention.chunk_id,
                "concepts": about,
                "linked_by": linked_by,
                "start": f.mention.start,
                "end": f.mention.end,
                "locator": f.mention.locator,
            }
        )
    return rows


def _tokens(text: str) -> list[str]:
    return [_singular(w) for w in normalize_words(text)]


def normalize_words(text: str) -> list[str]:
    s = text.lower().replace("-", " ").replace("_", " ").replace("\u2019", "'")
    s = s.replace("'s ", " ").replace("'", "")
    return re.sub(r"[^a-z0-9+#\s]", " ", s).split()


def _forms(names: Iterable[str]) -> list[tuple[str, ...]]:
    out = {tuple(_tokens(n)) for n in names}
    return sorted((f for f in out if f), key=len, reverse=True)


def link_fact(
    statement: str, candidates: Iterable[str], forms: dict[str, list[tuple[str, ...]]]
) -> list[str]:
    """Concepts (among ``candidates``, those mentioned in the fact's chunk) whose name or an
    alias occurs in ``statement`` as a whole-word sequence, plurals folded. A form that
    lies inside a longer matched form is not counted on its own: "binary search tree"
    links that concept, not also "binary search" or "tree"."""
    toks = _tokens(statement)
    spans: list[tuple[int, int, str]] = []
    for cid in candidates:
        for form in forms.get(cid, []):
            n = len(form)
            for i in range(len(toks) - n + 1):
                if tuple(toks[i : i + n]) == form:
                    spans.append((i, i + n, cid))
    spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
    taken: list[tuple[int, int]] = []
    found: list[tuple[int, str]] = []
    for a, b, cid in spans:
        if any(a >= x and b <= y for x, y in taken):
            continue
        taken.append((a, b))
        if cid not in (c for _, c in found):
            found.append((a, cid))
    return [cid for _, cid in sorted(found)]


def embed_hash(model: str, text: str) -> str:
    return hashlib.sha1(f"{model}\n{text}".encode()).hexdigest()


def needs_embedding(
    items: Iterable[tuple[str, str]], existing: dict[str, str | None], model: str
) -> list[tuple[str, str, str]]:
    """``(id, text)`` pairs whose stored hash is missing or stale -> ``(id, text, hash)``."""
    out = []
    for id_, text in items:
        h = embed_hash(model, text)
        if existing.get(id_) != h:
            out.append((id_, text, h))
    return out


def _batches(rows: list[dict[str, Any]], n: int = BATCH) -> Iterator[list[dict[str, Any]]]:
    for i in range(0, len(rows), n):
        yield rows[i : i + n]


# --- the store -----------------------------------------------------------------------------


@dataclass
class LoadStats:
    sources: int = 0
    chunks: int = 0
    concepts: int = 0
    mentions: int = 0
    relations: int = 0
    part_of: int = 0
    facts: int = 0
    embedded: int = 0
    removed: dict[str, int] = field(default_factory=dict)


_CONSTRAINTS = [
    "CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT meta_key IF NOT EXISTS FOR (m:Meta) REQUIRE m.key IS UNIQUE",
    "CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name_lc)",
]

_VECTOR_INDEX = (
    "CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:{label}) ON n.embedding "
    "OPTIONS {{indexConfig: {{`vector.dimensions`: {dims}, "
    "`vector.similarity_function`: 'cosine'}}}}"
)


class GraphStore:
    def __init__(self, config: GraphConfig, *, driver: Any = None) -> None:
        self.config = config
        if driver is None:
            if not config.password:
                raise GraphError(
                    "no Neo4j password: set NEO4J_PASSWORD in the environment or in .env "
                    "(see .env.example)"
                )
            from neo4j import GraphDatabase

            # Notifications are hints such as "label Meta does not exist" on a fresh graph.
            driver = GraphDatabase.driver(
                config.uri,
                auth=(config.user, config.password),
                notifications_min_severity="OFF",
            )
        self._driver = driver

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> GraphStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        from neo4j.exceptions import AuthError, ServiceUnavailable

        try:
            records, _, _ = self._driver.execute_query(
                query, params, database_=self.config.database
            )
        except ServiceUnavailable as exc:
            raise GraphError(
                f"Neo4j is not reachable at {self.config.uri} ({exc}); "
                "start it with `docker compose up -d`"
            ) from exc
        except AuthError as exc:
            raise GraphError(f"Neo4j rejected the credentials: {exc}") from exc
        return [r.data() for r in records]

    # schema

    def meta(self) -> dict[str, Any]:
        rows = self.run("MATCH (m:Meta {key: 'graph'}) RETURN properties(m) AS p")
        return rows[0]["p"] if rows else {}

    def init_schema(self, dims: int | None = None) -> None:
        for q in _CONSTRAINTS:
            self.run(q)
        if dims:
            for name, label in (
                ("concept_embedding", "Concept"),
                ("chunk_embedding", "Chunk"),
                ("memory_embedding", "Memory"),
            ):
                self.run(_VECTOR_INDEX.format(name=name, label=label, dims=int(dims)))
            self.run("CALL db.awaitIndexes(300)")

    # loading

    def load(
        self,
        rows: GraphRows,
        *,
        model: str,
        embedder: Embedder | None = None,
        replace: bool = False,
    ) -> LoadStats:
        """Write ``rows``. Refuses to mix two extraction models (or two embedding models)
        in one graph unless ``replace``, which deletes concepts and facts first."""
        meta = self.meta()
        emb_model = getattr(embedder, "model", None)
        if meta and not replace:
            if meta.get("model") not in (None, model):
                raise GraphError(
                    f"the graph holds knowledge from {meta['model']!r}, not {model!r}; "
                    "use --replace to swap it"
                )
            if emb_model and meta.get("embed_model") not in (None, emb_model):
                raise GraphError(
                    f"the graph's embeddings come from {meta['embed_model']!r}, not "
                    f"{emb_model!r}; use --replace"
                )

        dims = None
        if embedder is not None:
            dims = meta.get("dims") if meta.get("embed_model") == emb_model else None
            if dims is None:
                dims = len(embedder.embed(["dimension probe"])[0])
            if replace and meta.get("dims") not in (None, dims):
                self.run("DROP INDEX concept_embedding IF EXISTS")
                self.run("DROP INDEX chunk_embedding IF EXISTS")
        if replace:
            # Agent memories (D5) are not part of a knowledge file: they stay.
            self.run("MATCH (n) WHERE (n:Concept OR n:Fact) AND NOT n:Memory DETACH DELETE n")
            if embedder is not None and meta.get("embed_model") != emb_model:
                self.run("MATCH (c:Chunk) REMOVE c.embedding, c.embed_hash")
        self.init_schema(dims)

        load_id = uuid.uuid4().hex
        st = LoadStats()
        self.run("MERGE (d:Domain {name: $name})", name=rows.domain)
        for b in _batches(rows.sources):
            self.run(
                """UNWIND $rows AS r
                MERGE (s:Source {id: r.id})
                SET s.title = r.title, s.kind = r.kind, s.path = r.path,
                    s.duration = r.duration, s.pages = r.pages, s.url = r.url
                WITH s
                MATCH (d:Domain {name: $domain})
                MERGE (s)-[:PART_OF]->(d)""",
                rows=b,
                domain=rows.domain,
            )
            st.sources += len(b)
        for b in _batches(rows.chunks):
            self.run(
                """UNWIND $rows AS r
                MATCH (s:Source {id: r.source_id})
                MERGE (c:Chunk {id: r.id})
                SET c.index = r.index, c.text = r.text, c.n_words = r.n_words,
                    c.start = r.start, c.end = r.end, c.page_start = r.page_start,
                    c.page_end = r.page_end, c.heading = r.heading, c.locator = r.locator
                MERGE (c)-[:PART_OF]->(s)""",
                rows=b,
            )
            st.chunks += len(b)
        for b in _batches(rows.concepts):
            self.run(
                """UNWIND $rows AS r
                MERGE (c:Concept {id: r.id})
                SET c.name = r.name, c.name_lc = toLower(r.name),
                    c.aliases = r.aliases, c.aliases_lc = [a IN r.aliases | toLower(a)],
                    c.type = r.type, c.definition = r.definition,
                    c.confidence = r.confidence, c.n_mentions = r.n_mentions,
                    c.n_chunks = r.n_chunks, c.load_id = $load_id""",
                rows=b,
                load_id=load_id,
            )
            st.concepts += len(b)
        for b in _batches(rows.mentions):
            self.run(
                """UNWIND $rows AS r
                MATCH (c:Concept {id: r.concept}), (ch:Chunk {id: r.chunk})
                MERGE (c)-[m:MENTIONED_IN]->(ch)
                SET m.confidence = r.confidence, m.surfaces = r.surfaces,
                    m.start = r.start, m.end = r.end, m.locator = r.locator,
                    m.load_id = $load_id""",
                rows=b,
                load_id=load_id,
            )
            st.mentions += len(b)
        for b in _batches(rows.relations):
            self.run(
                """UNWIND $rows AS r
                MATCH (a:Concept {id: r.subject}), (b:Concept {id: r.object})
                MERGE (a)-[e:RELATED_TO {predicate: r.predicate}]->(b)
                SET e.confidence = r.confidence, e.n_mentions = r.n_mentions,
                    e.chunks = r.chunks, e.load_id = $load_id""",
                rows=b,
                load_id=load_id,
            )
            st.relations += len(b)
        for b in _batches(rows.part_of):
            self.run(
                """UNWIND $rows AS r
                MATCH (a:Concept {id: r.subject}), (b:Concept {id: r.object})
                MERGE (a)-[e:PART_OF]->(b)
                SET e.predicate = 'part_of', e.confidence = r.confidence,
                    e.n_mentions = r.n_mentions, e.chunks = r.chunks, e.load_id = $load_id""",
                rows=b,
                load_id=load_id,
            )
            st.part_of += len(b)
        for b in _batches(rows.facts):
            self.run(
                """UNWIND $rows AS r
                MATCH (ch:Chunk {id: r.chunk})
                MERGE (f:Fact {id: r.id})
                SET f.statement = r.statement, f.confidence = r.confidence,
                    f.linked_by = r.linked_by, f.load_id = $load_id
                MERGE (f)-[m:MENTIONED_IN]->(ch)
                SET m.start = r.start, m.end = r.end, m.locator = r.locator,
                    m.confidence = r.confidence, m.load_id = $load_id
                WITH f, r
                UNWIND r.concepts AS cid
                MATCH (c:Concept {id: cid})
                MERGE (f)-[a:ABOUT]->(c)
                SET a.load_id = $load_id""",
                rows=b,
                load_id=load_id,
            )
            st.facts += len(b)

        # Concepts belong to the domains of the sources that mention them.
        self.run(
            """MATCH (c:Concept {load_id: $load_id})-[:MENTIONED_IN]->(:Chunk)
                  -[:PART_OF]->(:Source)-[:PART_OF]->(d:Domain)
            MERGE (c)-[:PART_OF]->(d)""",
            load_id=load_id,
        )
        st.removed = self._prune(load_id)

        if embedder is not None:
            st.embedded = self._embed(embedder, rows)
        self.run(
            """MERGE (m:Meta {key: 'graph'})
            SET m.model = $model, m.embed_model = coalesce($embed_model, m.embed_model),
                m.dims = coalesce($dims, m.dims), m.loaded_at = datetime(),
                m.load_id = $load_id""",
            model=model,
            embed_model=emb_model,
            dims=dims,
            load_id=load_id,
        )
        return st

    def _prune(self, load_id: str) -> dict[str, int]:
        """Delete what an earlier load wrote and this one didn't."""
        removed = {}
        for name, q in {
            "concepts": "MATCH (n:Concept) WHERE n.load_id <> $id DETACH DELETE n",
            "facts": "MATCH (n:Fact) WHERE NOT n:Memory AND n.load_id <> $id DETACH DELETE n",
            "edges": (
                "MATCH (f:Concept|Fact)-[e:RELATED_TO|PART_OF|MENTIONED_IN|ABOUT]->(t) "
                "WHERE NOT t:Domain AND NOT f:Memory AND e.load_id <> $id DELETE e"
            ),
        }.items():
            removed[name] = self.run(q + " RETURN count(*) AS n", id=load_id)[0]["n"]
        return removed

    def _embed(self, embedder: Embedder, rows: GraphRows) -> int:
        n = 0
        for label, items in (
            ("Concept", [(c["id"], concept_text(c)) for c in rows.concepts]),
            ("Chunk", [(c["id"], c["text"]) for c in rows.chunks]),
        ):
            existing = {
                r["id"]: r["h"]
                for r in self.run(
                    f"UNWIND $ids AS id MATCH (n:{label} {{id: id}}) "
                    "RETURN n.id AS id, n.embed_hash AS h",
                    ids=[i for i, _ in items],
                )
            }
            todo = needs_embedding(items, existing, embedder.model)
            for i in range(0, len(todo), 64):
                part = todo[i : i + 64]
                vecs = embedder.embed([t for _, t, _ in part])
                self.run(
                    f"""UNWIND $rows AS r
                    MATCH (n:{label} {{id: r.id}})
                    CALL db.create.setNodeVectorProperty(n, 'embedding', r.vec)
                    SET n.embed_hash = r.h""",
                    rows=[
                        {"id": id_, "h": h, "vec": v}
                        for (id_, _, h), v in zip(part, vecs, strict=True)
                    ],
                )
            n += len(todo)
        return n

    # reading

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for label in ("Domain", "Source", "Chunk", "Concept", "Fact"):
            out[label] = self.run(f"MATCH (n:{label}) RETURN count(n) AS n")[0]["n"]
        for r in self.run("MATCH ()-[e]->() RETURN type(e) AS t, count(e) AS n ORDER BY t"):
            out[r["t"]] = r["n"]
        out["embedded Concept"] = self.run(
            "MATCH (n:Concept) WHERE n.embedding IS NOT NULL RETURN count(n) AS n"
        )[0]["n"]
        out["embedded Chunk"] = self.run(
            "MATCH (n:Chunk) WHERE n.embedding IS NOT NULL RETURN count(n) AS n"
        )[0]["n"]
        return out

    def find_concept(self, name: str) -> dict[str, Any] | None:
        rows = self.run(
            """MATCH (c:Concept)
            WHERE c.name_lc = toLower($name) OR toLower($name) IN c.aliases_lc
            RETURN c {.id, .name, .aliases, .type, .definition, .n_mentions} AS c
            ORDER BY c.n_mentions DESC LIMIT 1""",
            name=name,
        )
        return rows[0]["c"] if rows else None

    def trace(self, concept_id: str) -> list[dict[str, Any]]:
        """The demo walk: concept -> chunks that mention it -> their sources, in time order."""
        return self.run(TRACE_QUERY, id=concept_id)

    def neighbours(self, concept_id: str) -> list[dict[str, Any]]:
        return self.run(
            """MATCH (c:Concept {id: $id})-[e:RELATED_TO|PART_OF]-(o:Concept)
            RETURN CASE WHEN startNode(e) = c THEN 'out' ELSE 'in' END AS dir,
                   coalesce(e.predicate, 'part_of') AS predicate, o.name AS other,
                   e.n_mentions AS n
            ORDER BY n DESC, other""",
            id=concept_id,
        )

    def similar(self, vector: list[float], label: str = "Concept", k: int = 5):
        index = {"Concept": "concept_embedding", "Chunk": "chunk_embedding"}[label]
        return self.run(
            f"""CALL db.index.vector.queryNodes('{index}', $k, $vec) YIELD node, score
            OPTIONAL MATCH (node)-[:PART_OF]->(s:Source)
            RETURN coalesce(node.name, s.title) AS name, node.locator AS locator,
                   node.definition AS definition, score
            ORDER BY score DESC""",
            k=k,
            vec=vector,
        )

    def reset(self) -> None:
        self.run("MATCH (n) DETACH DELETE n")
        for name in ("concept_embedding", "chunk_embedding", "memory_embedding"):
            self.run(f"DROP INDEX {name} IF EXISTS")

    def load_passages(
        self,
        source: Source,
        chunks: Sequence[Chunk],
        domain: str,
        embedder: Embedder | None = None,
    ) -> int:
        """Add one source and its chunks without touching concepts or facts (D5
        ``ingest_source`` before any extraction). Returns the number of chunks embedded."""
        rows = build_rows(Knowledge(model=""), [(source, chunks)], domain)
        meta = self.meta()
        if embedder is not None and meta.get("embed_model") not in (None, embedder.model):
            raise GraphError(
                f"the graph's embeddings come from {meta['embed_model']!r}, not {embedder.model!r}"
            )
        self.run("MERGE (d:Domain {name: $name})", name=domain)
        self.run(
            """UNWIND $rows AS r
            MERGE (s:Source {id: r.id})
            SET s.title = r.title, s.kind = r.kind, s.path = r.path,
                s.duration = r.duration, s.pages = r.pages, s.url = r.url
            WITH s MATCH (d:Domain {name: $domain}) MERGE (s)-[:PART_OF]->(d)""",
            rows=rows.sources,
            domain=domain,
        )
        for b in _batches(rows.chunks):
            self.run(
                """UNWIND $rows AS r
                MATCH (s:Source {id: r.source_id})
                MERGE (c:Chunk {id: r.id})
                SET c.index = r.index, c.text = r.text, c.n_words = r.n_words,
                    c.start = r.start, c.end = r.end, c.page_start = r.page_start,
                    c.page_end = r.page_end, c.heading = r.heading, c.locator = r.locator
                MERGE (c)-[:PART_OF]->(s)""",
                rows=b,
            )
        if embedder is None:
            return 0
        if not meta.get("dims"):
            dims = len(embedder.embed(["dimension probe"])[0])
            self.init_schema(dims)
            self.run(
                "MERGE (m:Meta {key: 'graph'}) SET m.embed_model = $e, m.dims = $d",
                e=embedder.model,
                d=dims,
            )
        rows.concepts = []
        return self._embed(embedder, rows)


TRACE_QUERY = """
MATCH (c:Concept {id: $id})-[m:MENTIONED_IN]->(ch:Chunk)-[:PART_OF]->(s:Source)
RETURN s.title AS source, ch.index AS chunk, ch.locator AS locator,
       ch.start AS start, m.surfaces AS said_as, m.confidence AS confidence
ORDER BY s.title, ch.start, ch.index
"""
