"""Pack blocks into chunks of about ``target_words``, keeping pointers back to the source.

Rules:

- Blocks are packed whole, in order, until adding the next one would pass ``target_words``.
- A block longer than ``max_words`` is split first: at sentence ends where possible, then
  by word count. Split transcript segments get their time range divided in proportion to
  the words in each part.
- A Markdown heading change always starts a new chunk, so a chunk never mixes sections.
- Consecutive chunks overlap by up to ``overlap_words``: the next chunk starts with the
  last whole block(s) of the previous one. A fact that straddles a boundary is then
  still found whole in one chunk.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, model_validator

from domaingraph.models import Block, Chunk

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class ChunkConfig(BaseModel):
    target_words: int = Field(default=200, ge=20)
    max_words: int = Field(default=300, ge=20)
    overlap_words: int = Field(default=40, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> ChunkConfig:
        if self.max_words < self.target_words:
            raise ValueError("max_words must be >= target_words")
        if self.overlap_words >= self.target_words:
            raise ValueError("overlap_words must be < target_words")
        return self


def _words(text: str) -> int:
    return len(text.split())


def split_block(block: Block, max_words: int) -> list[Block]:
    """Split a block over ``max_words`` into parts of at most ``max_words``."""
    if _words(block.text) <= max_words:
        return [block]
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(block.text):
        words = sentence.split()
        while len(words) > max_words:  # a single run-on "sentence"
            pieces.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if words:
            pieces.append(" ".join(words))
    # Greedily re-merge sentences up to max_words.
    parts: list[str] = []
    for p in pieces:
        if parts and _words(parts[-1]) + _words(p) <= max_words:
            parts[-1] = f"{parts[-1]} {p}"
        else:
            parts.append(p)

    out: list[Block] = []
    if block.start is not None and block.end is not None:
        total = sum(_words(p) for p in parts)
        t, span = block.start, block.end - block.start
        for p in parts:
            dt = span * _words(p) / total
            out.append(block.model_copy(update={"text": p, "start": t, "end": t + dt}))
            t += dt
        out[-1].end = block.end  # no rounding drift at the end
    else:
        out = [block.model_copy(update={"text": p}) for p in parts]
    return out


def chunk_blocks(
    source_id: str, blocks: list[Block], cfg: ChunkConfig | None = None
) -> list[Chunk]:
    cfg = cfg or ChunkConfig()
    units = [part for b in blocks for part in split_block(b, cfg.max_words)]
    chunks: list[Chunk] = []
    current: list[Block] = []

    def emit() -> None:
        chunks.append(_make_chunk(source_id, len(chunks), current))

    for unit in units:
        if current and unit.heading != current[-1].heading:
            emit()
            current = []  # new section: no overlap across headings
        elif (
            current and sum(_words(u.text) for u in current) + _words(unit.text) > cfg.target_words
        ):
            emit()
            current = _overlap(current, cfg.overlap_words)
        current.append(unit)
    if current:
        emit()
    return chunks


def _overlap(units: list[Block], budget: int) -> list[Block]:
    """The trailing whole units of a chunk that fit in ``budget`` words."""
    kept: list[Block] = []
    used = 0
    for u in reversed(units):
        n = _words(u.text)
        if used + n > budget:
            break
        kept.insert(0, u)
        used += n
    # Never carry the whole chunk over; that would repeat it.
    return kept if len(kept) < len(units) else []


def _make_chunk(source_id: str, index: int, units: list[Block]) -> Chunk:
    starts = [u.start for u in units if u.start is not None]
    ends = [u.end for u in units if u.end is not None]
    pages = [u.page for u in units if u.page is not None]
    # Prose joins with spaces; anything with line structure (code blocks) keeps it.
    sep = "\n\n" if any("\n" in u.text for u in units) else " "
    text = sep.join(u.text for u in units)
    return Chunk(
        id=f"{source_id}:{index:04d}",
        source_id=source_id,
        index=index,
        text=text,
        n_words=_words(text),
        start=min(starts) if starts else None,
        end=max(ends) if ends else None,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        heading=list(units[0].heading),
    )
