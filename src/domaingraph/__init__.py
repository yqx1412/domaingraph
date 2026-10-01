"""domaingraph: domain-aware multimodal knowledge graph.

Whisper -> Neo4j -> hybrid search, with an MCP server and Obsidian export.
"""

__version__ = "0.1.0"


def main() -> None:
    from domaingraph.cli import main as cli_main

    raise SystemExit(cli_main())
