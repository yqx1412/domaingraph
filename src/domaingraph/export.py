"""D6: one-way export from the graph to an Obsidian vault.

Layout::

    <vault>/Concepts/<concept name>.md    one note per concept
    <vault>/Sources/<source title>.md     one note per lecture or document
    <vault>/.obsidian/graph.json          graph-view colour groups by concept type (only if absent)

A concept note has YAML front matter (``domain``, ``type``, ``confidence``, ``aliases`` so
Obsidian resolves links by alias, ``tags``, ``domaingraph_id``), then the definition, its
relations as ``[[links]]`` grouped by predicate, where it is taught (source and time range,
each time a link to that moment in the video when the source has a URL) and facts about it.

**Re-running updates notes in place.** Everything the exporter writes sits between
``<!-- domaingraph:begin -->`` and ``<!-- domaingraph:end -->``; text outside the markers is
the user's and is kept. Front matter keys the exporter doesn't own are kept too. A note is
found again by its ``domaingraph_id``, not its file name, so a renamed note is updated where
it is. A concept that disappeared from the graph is never deleted: its note gets
``domaingraph_status: removed`` (``prune=True`` deletes such notes only when they hold no user
text). A note whose content didn't change is not rewritten, so Obsidian sees no churn.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from domaingraph.graph import GraphStore
from domaingraph.models import timestamp

BEGIN = "<!-- domaingraph:begin -->"
END = "<!-- domaingraph:end -->"
OWNED_KEYS = (
    "domaingraph_id",
    "domaingraph_kind",
    "domaingraph_status",
    "domain",
    "type",
    "confidence",
    "aliases",
    "mentions",
    "sources",
    "tags",
)
PREDICATE_HEADINGS = {
    "is_a": "Is a",
    "part_of": "Part of",
    "uses": "Uses",
    "solves": "Solves",
    "has_property": "Has property",
    "contrasts_with": "Contrasts with",
}
INVERSE_HEADINGS = {
    "is_a": "Kinds",
    "part_of": "Parts",
    "uses": "Used by",
    "solves": "Solved by",
    "has_property": "Property of",
    "contrasts_with": "Contrasts with",
}
TYPE_COLOURS = {  # Obsidian graph colour groups (RGB as an int)
    "algorithm": 0x4E79A7,
    "data_structure": 0x59A14F,
    "operation": 0xF28E2B,
    "property": 0xB07AA1,
    "complexity": 0xE15759,
    "problem": 0xEDC948,
    "technique": 0x76B7B2,
    "other": 0x9C9C9C,
}
_BAD = re.compile(r'[\\/:*?"<>|#^\[\]]')


def safe_name(name: str, limit: int = 120) -> str:
    """A file name Obsidian and Windows accept, close to the original."""
    s = _BAD.sub(" ", name).strip().strip(".")
    s = re.sub(r"\s+", " ", s)
    return s[:limit].rstrip() or "untitled"


# --- reading the graph ---------------------------------------------------------------------


@dataclass
class ExportData:
    concepts: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    mentions: list[dict[str, Any]]
    facts: list[dict[str, Any]]


def read_graph(store: GraphStore) -> ExportData:
    concepts = store.run(
        """MATCH (c:Concept)
        OPTIONAL MATCH (c)-[:PART_OF]->(d:Domain)
        RETURN c.id AS id, c.name AS name, c.aliases AS aliases, c.type AS type,
               c.definition AS definition, c.confidence AS confidence,
               c.n_mentions AS n_mentions, c.n_chunks AS n_chunks,
               collect(DISTINCT d.name) AS domains"""
    )
    sources = store.run(
        """MATCH (s:Source) OPTIONAL MATCH (s)-[:PART_OF]->(d:Domain)
        RETURN s.id AS id, s.title AS title, s.kind AS kind, s.url AS url,
               s.duration AS duration, s.pages AS pages, collect(d.name) AS domains"""
    )
    relations = store.run(
        """MATCH (a:Concept)-[e:RELATED_TO|PART_OF]->(b:Concept)
        RETURN a.id AS subject, coalesce(e.predicate, 'part_of') AS predicate,
               b.id AS object, e.n_mentions AS n"""
    )
    mentions = store.run(
        """MATCH (c:Concept)-[m:MENTIONED_IN]->(ch:Chunk)-[:PART_OF]->(s:Source)
        RETURN c.id AS concept, s.id AS source, ch.index AS chunk, ch.start AS start,
               ch.locator AS at, m.surfaces AS surfaces"""
    )
    facts = store.run(
        """MATCH (f:Fact)-[:ABOUT]->(c:Concept) WHERE NOT f:Memory
        MATCH (f)-[:MENTIONED_IN]->(ch:Chunk)-[:PART_OF]->(s:Source)
        RETURN c.id AS concept, f.statement AS statement, s.id AS source,
               ch.start AS start, ch.locator AS at"""
    )
    return ExportData(concepts, sources, relations, mentions, facts)


# --- notes ---------------------------------------------------------------------------------


@dataclass
class ExportStats:
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0  # marked as removed
    deleted: int = 0  # removed and pruned
    skipped: int = 0  # below min_chunks
    paths: dict[str, Path] = field(default_factory=dict)


def _time_link(src: dict[str, Any], start: float | None, at: str) -> str:
    """``at`` as a link to that moment of the video when the source is on YouTube."""
    url = src.get("url") or ""
    if start is not None and ("youtube.com/" in url or "youtu.be/" in url):
        sep = "&" if "?" in url else "?"
        return f"[{at}]({url}{sep}t={int(start)}s)"
    return at


def _split(text: str) -> tuple[dict[str, Any], str, str]:
    """Front matter, user text before the managed block, user text after it."""
    meta: dict[str, Any] = {}
    body = text
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            try:
                meta = yaml.safe_load(text[4:end]) or {}
            except yaml.YAMLError:
                meta = {}
            body = text[end + 5 :]
    if BEGIN in body and END in body:
        before, rest = body.split(BEGIN, 1)
        _, after = rest.split(END, 1)
        return meta, before, after
    return meta, "", "\n" + body.strip() + "\n" if body.strip() else ""


def _user_text(before: str, after: str) -> bool:
    return bool(before.strip() or after.strip())


def render_note(meta: dict[str, Any], block: str, before: str = "", after: str = "") -> str:
    fm = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=1000).strip()
    head = before if before.strip() else ""
    tail = after if after.strip() else "\n"
    return f"---\n{fm}\n---\n{head}{BEGIN}\n{block.strip()}\n{END}{tail}"


class VaultExporter:
    def __init__(self, vault: Path, *, min_chunks: int = 1, prune: bool = False) -> None:
        self.vault = Path(vault)
        self.min_chunks = min_chunks
        self.prune = prune

    def _index(self) -> dict[str, Path]:
        """domaingraph_id -> existing note, wherever the user moved it."""
        found: dict[str, Path] = {}
        for p in self.vault.rglob("*.md"):
            if ".obsidian" in p.parts:
                continue
            try:
                head = p.read_text(encoding="utf-8")[:2000]
            except OSError:
                continue
            m = re.search(r"^domaingraph_id:\s*['\"]?([^'\"\n]+)", head, re.M)
            if m and head.startswith("---"):
                found[m.group(1).strip()] = p
        return found

    def _write(self, path: Path, meta: dict[str, Any], block: str, st: ExportStats) -> None:
        if path.is_file():
            old = path.read_text(encoding="utf-8")
            old_meta, before, after = _split(old)
            # Ours first, then the keys the user added (kept as they are).
            extra = {k: v for k, v in old_meta.items() if k not in OWNED_KEYS}
            new = render_note({**meta, **extra}, block, before, after)
            if new == old:
                st.unchanged += 1
                return
            path.write_text(new, encoding="utf-8", newline="\n")
            st.updated += 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_note(meta, block), encoding="utf-8", newline="\n")
            st.created += 1

    def _place(self, kind: str, title: str, nid: str, index: dict[str, Path], used: set[str]):
        if nid in index:
            p = index[nid]
        else:
            base = safe_name(title)
            name, n = base, 2
            while name.lower() in used or (self.vault / kind / f"{name}.md").exists():
                name, n = f"{base} ({n})", n + 1
            p = self.vault / kind / f"{name}.md"
        used.add(p.stem.lower())
        return p

    def export(self, data: ExportData) -> ExportStats:
        st = ExportStats()
        index = self._index()
        used: set[str] = set()
        keep = {c["id"]: c for c in data.concepts if (c.get("n_chunks") or 0) >= self.min_chunks}
        st.skipped = len(data.concepts) - len(keep)
        sources = {s["id"]: s for s in data.sources}

        # Note names first, so every [[link]] points at the file name it will have.
        src_path = {
            sid: self._place("Sources", s["title"], f"source:{sid}", index, used)
            for sid, s in sorted(sources.items(), key=lambda kv: kv[1]["title"])
        }
        con_path = {
            cid: self._place("Concepts", c["name"], cid, index, used)
            for cid, c in sorted(keep.items(), key=lambda kv: (-(kv[1]["n_mentions"] or 0), kv[0]))
        }

        def link(cid: str) -> str:
            p, name = con_path[cid], keep[cid]["name"]
            return f"[[{p.stem}]]" if p.stem == name else f"[[{p.stem}|{name}]]"

        def slink(sid: str) -> str:
            return f"[[{src_path[sid].stem}]]"

        out_rel: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        in_rel: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
        for r in data.relations:
            if r["subject"] in keep and r["object"] in keep and r["subject"] != r["object"]:
                out_rel[r["subject"]][r["predicate"]].append((r["n"] or 1, r["object"]))
                in_rel[r["object"]][r["predicate"]].append((r["n"] or 1, r["subject"]))
        by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for m in data.mentions:
            if m["concept"] in keep:
                by_concept[m["concept"]].append(m)
                by_source[m["source"]].append(m)
        facts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for f in data.facts:
            if f["concept"] in keep:
                facts[f["concept"]].append(f)

        def order(m: dict[str, Any]) -> tuple:
            return (sources[m["source"]]["title"], m["start"] or 0, m["chunk"])

        for cid, c in keep.items():
            lines = [f"# {c['name']}", ""]
            if c.get("definition"):
                lines += [c["definition"], ""]
            rel_lines = []
            for pred, heading in PREDICATE_HEADINGS.items():
                items = sorted(
                    out_rel[cid].get(pred, []), key=lambda t: (-t[0], keep[t[1]]["name"])
                )
                if items:
                    rel_lines.append(f"- **{heading}:** " + ", ".join(link(o) for _, o in items))
            for pred, heading in INVERSE_HEADINGS.items():
                items = sorted(in_rel[cid].get(pred, []), key=lambda t: (-t[0], keep[t[1]]["name"]))
                if items and not (pred == "contrasts_with" and out_rel[cid].get(pred)):
                    rel_lines.append(f"- **{heading}:** " + ", ".join(link(o) for _, o in items))
            if rel_lines:
                lines += ["## Related", "", *rel_lines, ""]
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for m in sorted(by_concept[cid], key=order):
                grouped[m["source"]].append(m)
            if grouped:
                lines += ["## Taught in", ""]
                for sid, ms in grouped.items():
                    times = ", ".join(_time_link(sources[sid], m["start"], m["at"]) for m in ms)
                    lines.append(f"- {slink(sid)}: {times}")
                lines.append("")
            fs = sorted(facts[cid], key=lambda f: (sources[f["source"]]["title"], f["start"] or 0))
            if fs:
                lines += ["## Facts", ""]
                seen: set[str] = set()
                for f in fs[:12]:
                    if f["statement"] in seen:
                        continue
                    seen.add(f["statement"])
                    at = _time_link(sources[f["source"]], f["start"], f["at"])
                    lines.append(f"- {f['statement']} ({slink(f['source'])}, {at})")
                lines.append("")
            domains = sorted({d for d in (c.get("domains") or []) if d})
            meta = {
                "domaingraph_id": cid,
                "domaingraph_kind": "concept",
                "domain": domains[0] if len(domains) == 1 else domains,
                "type": c["type"],
                "confidence": round(float(c.get("confidence") or 0), 2),
                "aliases": list(c.get("aliases") or []),
                "mentions": int(c.get("n_mentions") or 0),
                "sources": sorted({sources[m["source"]]["title"] for m in by_concept[cid]}),
                "tags": [f"type/{c['type']}"] + [f"domain/{d}" for d in domains],
            }
            self._write(con_path[cid], meta, "\n".join(lines), st)
            st.paths[cid] = con_path[cid]

        for sid, s in sources.items():
            lines = [f"# {s['title']}", ""]
            facts_line = []
            if s.get("url"):
                facts_line.append(f"[Watch]({s['url']})")
            if s.get("duration"):
                facts_line.append(timestamp(s["duration"]))
            if facts_line:
                lines += [" · ".join(facts_line), ""]
            first: dict[str, dict[str, Any]] = {}
            for m in sorted(by_source[sid], key=lambda m: (m["start"] or 0, m["chunk"])):
                first.setdefault(m["concept"], m)
            if first:
                lines += ["## Concepts, in the order they come up", ""]
                for cid, m in first.items():
                    lines.append(f"- {_time_link(s, m['start'], m['at'])} {link(cid)}")
                lines.append("")
            meta = {
                "domaingraph_id": f"source:{sid}",
                "domaingraph_kind": "source",
                "domain": (s.get("domains") or [None])[0],
                "type": s["kind"],
                "tags": ["source"],
            }
            self._write(src_path[sid], meta, "\n".join(lines), st)

        live = set(keep) | {f"source:{sid}" for sid in sources}
        for nid, p in index.items():
            if nid in live or not p.is_file():
                continue
            meta, before, after = _split(p.read_text(encoding="utf-8"))
            if self.prune and not _user_text(before, after):
                p.unlink()
                st.deleted += 1
                continue
            if meta.get("domaingraph_status") != "removed":
                meta["domaingraph_status"] = "removed"
                fm = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=1000).strip()
                body = p.read_text(encoding="utf-8").split("\n---\n", 1)[1]
                p.write_text(f"---\n{fm}\n---\n{body}", encoding="utf-8", newline="\n")
                st.removed += 1

        self._graph_settings()
        return st

    def _graph_settings(self) -> None:
        p = self.vault / ".obsidian" / "graph.json"
        if p.exists():
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        groups = [
            {"query": f"tag:#type/{t}", "color": {"a": 1, "rgb": rgb}}
            for t, rgb in TYPE_COLOURS.items()
        ]
        groups.append({"query": "path:Sources", "color": {"a": 1, "rgb": 0x222222}})
        p.write_text(
            json.dumps(
                {"colorGroups": groups, "showTags": False, "hideUnresolved": True}, indent=2
            ),
            encoding="utf-8",
        )


def export_vault(
    store: GraphStore, vault: Path, *, min_chunks: int = 1, prune: bool = False
) -> ExportStats:
    return VaultExporter(vault, min_chunks=min_chunks, prune=prune).export(read_graph(store))
