"""Data model for ingested sources and their chunks.

A *source* is one input file. Its *chunks* are the passages later stages work with:
extraction (D2) reads them, the graph (D3) stores them as ``Source`` passages, search (D4)
returns them. Every chunk keeps a pointer back to where it came from: a time range for
audio and video, a page range for PDFs, a heading path for Markdown.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

SourceKind = Literal["video", "audio", "text", "markdown", "pdf"]

KIND_BY_SUFFIX: dict[str, SourceKind] = {
    **dict.fromkeys((".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"), "video"),
    **dict.fromkeys((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"), "audio"),
    **dict.fromkeys((".txt",), "text"),
    **dict.fromkeys((".md", ".markdown"), "markdown"),
    **dict.fromkeys((".pdf",), "pdf"),
}


def kind_of(path: Path) -> SourceKind:
    try:
        return KIND_BY_SUFFIX[path.suffix.lower()]
    except KeyError:
        known = ", ".join(sorted(KIND_BY_SUFFIX))
        raise ValueError(f"unsupported file type {path.suffix!r}; supported: {known}") from None


class Segment(BaseModel):
    """One timed piece of a transcript, as the speech recognizer returned it."""

    start: float
    end: float
    text: str


class Block(BaseModel):
    """One unit of a source before chunking: a transcript segment or a paragraph."""

    text: str
    start: float | None = None
    end: float | None = None
    page: int | None = None
    heading: list[str] = Field(default_factory=list)


class Source(BaseModel):
    id: str
    path: str
    kind: SourceKind
    title: str
    sha256: str
    bytes: int
    url: str | None = None  # where the original lives, e.g. a YouTube video (D6 links times)
    duration: float | None = None  # seconds, audio/video
    pages: int | None = None  # PDF
    language: str | None = None
    asr: dict[str, Any] | None = None  # model, device, compute type, timing
    chunking: dict[str, Any] = Field(default_factory=dict)
    n_chunks: int = 0
    ingested_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )


class Chunk(BaseModel):
    id: str  # "<source id>:<index>"
    source_id: str
    index: int
    text: str
    n_words: int
    start: float | None = None
    end: float | None = None
    page_start: int | None = None
    page_end: int | None = None
    heading: list[str] = Field(default_factory=list)

    def locator(self) -> str:
        """Human-readable pointer back into the source, e.g. "12:05-13:40" or "p. 3-4"."""
        if self.start is not None and self.end is not None:
            return f"{timestamp(self.start)}-{timestamp(self.end)}"
        if self.page_start is not None:
            if self.page_end in (None, self.page_start):
                return f"p. {self.page_start}"
            return f"p. {self.page_start}-{self.page_end}"
        if self.heading:
            return " > ".join(self.heading)
        return f"chunk {self.index}"


def timestamp(seconds: float) -> str:
    """``75.4`` -> ``"01:15"``; ``3725`` -> ``"1:02:05"``."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
