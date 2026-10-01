"""Text inputs as blocks: plain text and Markdown by paragraph, PDF by page and paragraph."""

from __future__ import annotations

import re
from pathlib import Path

from domaingraph.models import Block

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def load_text(path: Path) -> list[Block]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Block(text=" ".join(p.split())) for p in _paragraphs(text)]


def load_markdown(path: Path) -> tuple[list[Block], str | None]:
    """Blocks with their heading path; returns the first H1 as the title, if any.

    Front matter is dropped, code fences stay one block each (blank lines inside a fence
    do not split it), and every block carries the headings above it.
    """
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            text = text[end + 4 :]

    blocks: list[Block] = []
    headings: list[str] = []
    title: str | None = None
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if body:
            blocks.append(Block(text=body, heading=list(headings)))

    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            buf.append(line)
            if not in_fence:
                flush()
            continue
        if in_fence:
            buf.append(line)
            continue
        m = _HEADING.match(line)
        if m:
            flush()
            level, name = len(m.group(1)), m.group(2).strip()
            headings[:] = [*headings[: level - 1], name]
            if level == 1 and title is None:
                title = name
            continue
        if not line.strip():
            flush()
        else:
            buf.append(line)
    flush()
    # Join soft-wrapped lines in prose; keep code blocks as written.
    for b in blocks:
        if not _FENCE.match(b.text):
            b.text = " ".join(b.text.split())
    return blocks, title


def load_pdf(path: Path) -> tuple[list[Block], int, str | None]:
    """Blocks with 1-based page numbers; returns ``(blocks, n_pages, title)``."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    blocks: list[Block] = []
    for number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        raw = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)  # re-join words hyphenated at line ends
        for para in _paragraphs(raw):
            blocks.append(Block(text=" ".join(para.split()), page=number))
    meta_title = (reader.metadata.title if reader.metadata else None) or None
    return blocks, len(reader.pages), meta_title
