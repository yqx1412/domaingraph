"""Ingest one file: detect its kind, turn it into blocks, chunk them, write the result.

Output layout (``out`` defaults to ``data/``)::

    data/sources/<source id>/source.json      metadata, ASR settings and timing
    data/sources/<source id>/transcript.json  timed segments (audio and video only)
    data/sources/<source id>/chunks.jsonl     one chunk per line

The source id is derived from the file's content (SHA-256), so ingesting the same file
again, even renamed or moved, is recognized and skipped. Transcripts are the expensive
part: a re-ingest with ``force`` re-chunks from the saved transcript when the ASR model is
the same, instead of transcribing again.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from domaingraph import documents, media
from domaingraph.asr import Transcriber
from domaingraph.chunking import ChunkConfig, chunk_blocks
from domaingraph.models import Block, Chunk, Segment, Source, kind_of

TranscriberFactory = Callable[[], Transcriber]


@dataclass
class IngestResult:
    source: Source
    chunks: list[Chunk]
    skipped: bool = False  # already ingested
    reused_transcript: bool = False


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_dir(out: Path, source_id: str) -> Path:
    return out / "sources" / source_id


def load_source(out: Path, source_id: str) -> tuple[Source, list[Chunk]]:
    d = source_dir(out, source_id)
    source = Source.model_validate_json((d / "source.json").read_text(encoding="utf-8"))
    lines = (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    return source, [Chunk.model_validate_json(line) for line in lines if line.strip()]


def list_sources(out: Path) -> list[Source]:
    root = out / "sources"
    if not root.is_dir():
        return []
    found = []
    for d in sorted(root.iterdir()):
        f = d / "source.json"
        if f.is_file():
            found.append(Source.model_validate_json(f.read_text(encoding="utf-8")))
    return sorted(found, key=lambda s: s.ingested_at)


def ingest(
    path: Path,
    *,
    out: Path,
    transcriber: TranscriberFactory | None = None,
    chunking: ChunkConfig | None = None,
    language: str | None = None,
    force: bool = False,
    asr_model: str | None = None,
) -> IngestResult:
    """``transcriber`` is only called when audio actually has to be transcribed, so the
    model is never loaded for text inputs or for a reused transcript. ``asr_model`` names
    the model it would load; a saved transcript from the same model is reused."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    kind = kind_of(path)
    cfg = chunking or ChunkConfig()
    sha = file_sha256(path)
    source_id = sha[:16]
    d = source_dir(out, source_id)

    if (d / "source.json").is_file() and not force:
        source, chunks = load_source(out, source_id)
        return IngestResult(source, chunks, skipped=True)

    source = Source(
        id=source_id,
        path=str(path),
        kind=kind,
        title=path.stem.replace("_", " ").replace("-", " ").strip(),
        sha256=sha,
        bytes=path.stat().st_size,
        chunking=cfg.model_dump(),
    )
    reused = False
    if kind in ("audio", "video"):
        blocks, reused = _media_blocks(path, source, d, transcriber, language, asr_model)
    elif kind == "pdf":
        blocks, source.pages, title = documents.load_pdf(path)
        source.title = title or source.title
    elif kind == "markdown":
        blocks, title = documents.load_markdown(path)
        source.title = title or source.title
    else:
        blocks = documents.load_text(path)

    chunks = chunk_blocks(source_id, blocks, cfg)
    source.n_chunks = len(chunks)
    d.mkdir(parents=True, exist_ok=True)
    with (d / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    # source.json last: its presence marks a complete ingest.
    (d / "source.json").write_text(source.model_dump_json(indent=2), encoding="utf-8")
    return IngestResult(source, chunks, reused_transcript=reused)


def _media_blocks(
    path: Path,
    source: Source,
    d: Path,
    transcriber: TranscriberFactory | None,
    language: str | None,
    asr_model: str | None,
) -> tuple[list[Block], bool]:
    info = media.probe(path)
    source.duration = round(info["duration"], 2)
    cached = d / "transcript.json"
    if cached.is_file() and asr_model is not None:
        saved = json.loads(cached.read_text(encoding="utf-8"))
        if saved.get("asr", {}).get("model") == asr_model:
            source.asr, source.language = saved["asr"], saved.get("language")
            segments = [Segment.model_validate(s) for s in saved["segments"]]
            return [Block(text=s.text, start=s.start, end=s.end) for s in segments], True
    if transcriber is None:
        raise ValueError(f"{path.name} is {source.kind}; a transcriber is needed")
    audio = media.decode(path)
    result = transcriber().transcribe(audio, language=language)
    source.asr, source.language = result.info, result.language
    d.mkdir(parents=True, exist_ok=True)
    cached.write_text(
        json.dumps(
            {
                "asr": result.info,
                "language": result.language,
                "segments": [s.model_dump() for s in result.segments],
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return [Block(text=s.text, start=s.start, end=s.end) for s in result.segments], False
