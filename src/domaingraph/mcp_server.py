"""D5: DomainGraph as an MCP server (stdio).

Run it with ``domaingraph mcp`` from the repo root (so ``.env`` and ``data/`` are found), or
point an MCP client at ``python -m domaingraph.mcp_server``. Tools:

* ``search``: passages that answer a question, each with its source title and timestamp.
* ``get_concept`` / ``related_concepts``: what the graph knows about a concept.
* ``add_fact`` / ``recall_facts`` / ``forget_fact`` / ``list_facts``: agent memory (the
  AgentOS long-term memory backend). Memories live in a ``scope``.
* ``ingest_source``: add a file (video, audio, text, Markdown, PDF). Off unless the server
  is started with ``--allow-ingest <dir>``, and limited to files under those folders.

Expected failures raise the SDK's ``ToolError`` so their message reaches the client.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from domaingraph.graph import GraphConfig, GraphError, GraphStore
from domaingraph.llm import LLMError, OllamaEmbedder
from domaingraph.service import DomainGraphService, ServiceError

INSTRUCTIONS = (
    "DomainGraph: a knowledge graph built from lecture recordings and documents. Use search "
    "to find the passages that answer a question; each result has the source title and the "
    "time range ('at') where it was said, which you should cite. Use get_concept for a "
    "definition and how a concept relates to others. add_fact / recall_facts store and look "
    "up your own long-term memory."
)


def build_server(service: DomainGraphService) -> MCPServer:
    server = MCPServer("domaingraph", instructions=INSTRUCTIONS)

    def guard(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ServiceError, GraphError, LLMError, ValueError) as exc:
            raise ToolError(str(exc)) from exc

    @server.tool()
    def search(query: str, k: int = 5, mode: str = "vector") -> dict[str, Any]:
        """Find the lecture passages that best answer a question. Returns up to k passages,
        best first, each with 'source' (lecture title), 'at' (time range, e.g. 41:12-42:30)
        and 'text'. Cite the source and time when you answer. mode: vector (default),
        hybrid or graph."""
        return {"results": guard(service.search, query, k, mode)}

    @server.tool()
    def get_concept(name: str) -> dict[str, Any]:
        """Look up one concept by name or alias (e.g. 'radix sort', 'BST'): its definition,
        related concepts, where it is mentioned (source and time) and facts about it."""
        return guard(service.get_concept, name)

    @server.tool()
    def related_concepts(name: str, predicate: str | None = None, k: int = 10) -> dict[str, Any]:
        """Concepts linked to a concept in the graph, most-mentioned first. predicate filters
        to one of: is_a, part_of, uses, solves, has_property, contrasts_with."""
        return {"related": guard(service.related_concepts, name, predicate, k)}

    @server.tool()
    def add_fact(
        statement: str,
        scope: str = "default",
        kind: str = "fact",
        details: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Save one self-contained fact to long-term memory, e.g. 'The deploy server is
        build-07, port 8443.' Returns its id; 'new' is false when the same text was already
        stored (its existing id is returned)."""
        return guard(service.add_fact, statement, scope, kind, **(details or {}))

    @server.tool()
    def recall_facts(
        query: str, scope: str = "default", k: int = 5, kind: str | None = None
    ) -> dict[str, Any]:
        """Search long-term memory by meaning (not just keywords). Returns matching facts,
        newest first; when two conflict, the newer one is correct. kind: fact or episode."""
        return {"memories": guard(service.recall_facts, query, scope, k, kind)}

    @server.tool()
    def forget_fact(id: int, scope: str = "default") -> dict[str, Any]:
        """Delete a stored fact by the id recall_facts showed, e.g. when it is wrong."""
        return guard(service.forget_fact, id, scope)

    @server.tool()
    def list_facts(scope: str = "default", kind: str | None = None, limit: int = 100):
        """List stored memories, newest first."""
        return {"memories": guard(service.list_facts, scope, kind, limit)}

    @server.tool()
    def clear_scope(scope: str) -> dict[str, Any]:
        """Delete every memory in a scope (not 'default'). For clients that isolate runs,
        such as benchmarks; leave it out of an agent's tool allowlist."""
        return {"deleted": guard(service.clear_scope, scope)}

    @server.tool()
    def ingest_source(path: str, title: str | None = None, extract: bool = False) -> dict[str, Any]:
        """Add a file (video, audio, .txt, .md or .pdf) to the graph so search can find its
        passages. extract=true also extracts concepts with the local LLM (minutes per hour
        of lecture). Only files under the server's allowed folders can be ingested."""
        return guard(service.ingest_source, path, title, extract)

    return server


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="domaingraph mcp", description=__doc__.split("\n")[0])
    add_arguments(p)
    return run(p.parse_args(argv))


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", type=Path, default=Path("data"), help="Data directory")
    p.add_argument("--embed-model", default="bge-m3")
    p.add_argument(
        "--allow-ingest",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help="Let ingest_source read files under DIR (repeatable). Off by default.",
    )
    p.add_argument("--env-file", type=Path, default=Path(".env"))


def run(args: argparse.Namespace) -> int:
    import logging

    # stderr is the client's log; one line per embedding request is noise there.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        store = GraphStore(GraphConfig.from_env(args.env_file))
    except GraphError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    service = DomainGraphService(
        store,
        OllamaEmbedder(args.embed_model),
        out=args.data,
        ingest_roots=args.allow_ingest,
    )
    try:
        build_server(service).run()  # stdio
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
