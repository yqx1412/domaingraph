"""``domaingraph ingest|sources|show|wer``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from domaingraph import __version__
from domaingraph.asr import DEFAULT_MODEL
from domaingraph.chunking import ChunkConfig
from domaingraph.evaluate import word_error_rate
from domaingraph.media import MediaError
from domaingraph.models import KIND_BY_SUFFIX, timestamp
from domaingraph.pipeline import ingest, list_sources, load_source

DEFAULT_OUT = Path("data")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="domaingraph", description="Domain knowledge graph.")
    p.add_argument("--version", action="version", version=f"domaingraph {__version__}")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Data directory (default: data)")
    sub = p.add_subparsers(dest="command")

    ing = sub.add_parser("ingest", help="Ingest files or folders (video, audio, txt, md, pdf)")
    ing.add_argument("paths", nargs="+", type=Path)
    ing.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Whisper model (default: {DEFAULT_MODEL})"
    )
    ing.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ing.add_argument("--compute-type", default="auto")
    ing.add_argument("--language", default=None, help="e.g. en; default: detect")
    ing.add_argument("--force", action="store_true", help="Re-ingest files already ingested")
    ing.add_argument("--chunk-words", type=int, default=200)
    ing.add_argument("--max-words", type=int, default=300)
    ing.add_argument("--overlap-words", type=int, default=40)

    sub.add_parser("sources", help="List ingested sources")

    show = sub.add_parser("show", help="Print a source's chunks")
    show.add_argument("source_id", help="Source id or a unique prefix")
    show.add_argument("--limit", type=int, default=None)

    wer = sub.add_parser("wer", help="Word error rate of a transcript against a reference")
    wer.add_argument("source_id")
    wer.add_argument("reference", type=Path, help="Reference text file")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        return _ingest(args)
    if args.command == "sources":
        return _sources(args)
    if args.command == "show":
        return _show(args)
    if args.command == "wer":
        return _wer(args)
    parser.print_help()
    return 0


def _expand(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in KIND_BY_SUFFIX)
        else:
            files.append(p)
    return files


def _ingest(args: argparse.Namespace) -> int:
    try:
        cfg = ChunkConfig(
            target_words=args.chunk_words,
            max_words=args.max_words,
            overlap_words=args.overlap_words,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    loaded = {}

    def transcriber():
        if "t" not in loaded:  # load the model once, and only if something needs it
            from domaingraph.asr import WhisperTranscriber

            print(f"loading {args.model} ...", file=sys.stderr, flush=True)
            loaded["t"] = WhisperTranscriber(
                args.model, device=args.device, compute_type=args.compute_type
            )
            t = loaded["t"]
            print(f"  on {t.device} ({t.compute_type})", file=sys.stderr, flush=True)
        return loaded["t"]

    failures = 0
    for path in _expand(args.paths):
        try:
            r = ingest(
                path,
                out=args.out,
                transcriber=transcriber,
                chunking=cfg,
                language=args.language,
                force=args.force,
                asr_model=args.model,
            )
        except (MediaError, ValueError, FileNotFoundError) as exc:
            print(f"FAIL {path}: {exc}", file=sys.stderr)
            failures += 1
            continue
        s = r.source
        what = "skipped (already ingested)" if r.skipped else "ingested"
        extra = ""
        if s.duration:
            extra += f", {timestamp(s.duration)} of audio"
        if s.asr and not r.skipped:
            how = (
                "reused transcript"
                if r.reused_transcript
                else (
                    f"{s.asr['seconds']} s on {s.asr['device']}, "
                    f"{s.asr['realtime_factor']}x realtime"
                )
            )
            extra += f", {how}"
        if s.pages:
            extra += f", {s.pages} pages"
        print(f"{what}: {s.id}  {s.kind:<8} {len(r.chunks)} chunks{extra}  {path.name}")
    return 1 if failures else 0


def _sources(args: argparse.Namespace) -> int:
    for s in list_sources(args.out):
        size = timestamp(s.duration) if s.duration else (f"{s.pages} p." if s.pages else "")
        print(f"{s.id}  {s.kind:<8} {s.n_chunks:>4} chunks  {size:>8}  {s.title}")
    return 0


def _resolve(out: Path, prefix: str) -> str | None:
    ids = [s.id for s in list_sources(out) if s.id.startswith(prefix)]
    if len(ids) != 1:
        print(
            f"error: {'no source' if not ids else 'several sources'} match {prefix!r}",
            file=sys.stderr,
        )
        return None
    return ids[0]


def _show(args: argparse.Namespace) -> int:
    sid = _resolve(args.out, args.source_id)
    if sid is None:
        return 2
    source, chunks = load_source(args.out, sid)
    print(f"# {source.title} ({source.kind}, {len(chunks)} chunks)\n")
    for c in chunks[: args.limit]:
        print(f"[{c.index}] {c.locator()}  ({c.n_words} words)")
        print(f"    {c.text[:300]}{'...' if len(c.text) > 300 else ''}\n")
    return 0


def _wer(args: argparse.Namespace) -> int:
    import json

    sid = _resolve(args.out, args.source_id)
    if sid is None:
        return 2
    transcript = args.out / "sources" / sid / "transcript.json"
    if not transcript.is_file():
        print("error: that source has no transcript", file=sys.stderr)
        return 2
    segs = json.loads(transcript.read_text(encoding="utf-8"))["segments"]
    hyp = " ".join(s["text"] for s in segs)
    r = word_error_rate(args.reference.read_text(encoding="utf-8"), hyp)
    print(
        f"WER {r.wer:.2%}  ({r.substitutions} substitutions, {r.deletions} deletions, "
        f"{r.insertions} insertions; {r.reference_words} reference words)"
    )
    return 0
