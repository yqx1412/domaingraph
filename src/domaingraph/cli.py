"""``domaingraph ingest|sources|show|wer|extract|merge|concepts|eval-extract``."""

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
DEFAULT_EXTRACT_MODEL = "qwen3:14b"
DEFAULT_MERGE_THRESHOLD = 0.85
DEFAULT_GRAPH_MODEL = "qwen3:8b"
SEARCH_MODES = ("bm25", "vector", "graph", "hybrid", "hybrid-rrf")
# Diagnostics for eval-search only: graph search over the D2 gold sets instead of the
# extracted graph, i.e. what perfect extraction would give.
EVAL_ONLY_MODES = ("graph-gold", "hybrid-gold")


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

    ex = sub.add_parser("extract", help="Extract concepts, relations and facts with an LLM")
    ex.add_argument("source_ids", nargs="*", help="Source ids or prefixes (default: all)")
    ex.add_argument("--model", default=DEFAULT_EXTRACT_MODEL)
    ex.add_argument("--force", action="store_true", help="Re-extract chunks already done")
    ex.add_argument(
        "--no-known", action="store_true", help="Don't show earlier concept names in prompts"
    )
    _merge_args(ex)

    mg = sub.add_parser("merge", help="Re-merge saved extractions into one knowledge file")
    mg.add_argument("--model", default=DEFAULT_EXTRACT_MODEL)
    _merge_args(mg)

    cs = sub.add_parser("concepts", help="List merged concepts")
    cs.add_argument("--model", default=DEFAULT_EXTRACT_MODEL)
    cs.add_argument("--source", default=None, help="Only concepts mentioned in this source")
    cs.add_argument("--limit", type=int, default=None)

    ev = sub.add_parser("eval-extract", help="Score extractions against gold sets")
    ev.add_argument("--models", default=DEFAULT_EXTRACT_MODEL, help="Comma-separated")
    ev.add_argument("--gold", type=Path, default=Path("benchmarks/extraction/gold"))
    ev.add_argument("--lenient-threshold", type=float, default=0.85)
    ev.add_argument("--report", type=Path, default=None, help="Write a Markdown report here")
    ev.add_argument("--details", action="store_true", help="Print matches, misses, FPs")
    ev.add_argument(
        "--min-chunks",
        default="1",
        help="Count a concept only if this many chunks of the lecture mention it; "
        "comma-separated to compare (e.g. 1,2,3)",
    )
    ev.add_argument(
        "--sweep",
        default=None,
        help="Comma-separated merge thresholds to compare instead (e.g. 0.8,0.85,0.9,1.01)",
    )
    _merge_args(ev)

    gr = sub.add_parser("graph", help="Neo4j graph store (D3)")
    gsub = gr.add_subparsers(dest="graph_command")
    gl = gsub.add_parser("load", help="Load a knowledge file and its sources into Neo4j")
    gl.add_argument(
        "--model",
        default=DEFAULT_GRAPH_MODEL,
        help=f"Which extraction to load (default: {DEFAULT_GRAPH_MODEL}, the best in D2)",
    )
    gl.add_argument("--domain", default="algorithms", help="Domain the sources belong to")
    gl.add_argument("--embed-model", default="bge-m3")
    gl.add_argument("--no-embed", action="store_true", help="Skip embeddings")
    gl.add_argument("--replace", action="store_true", help="Replace knowledge from another model")
    gsub.add_parser("stats", help="Node and edge counts")
    gt = gsub.add_parser("trace", help="Walk from a concept to its sources and timestamps")
    gt.add_argument("concept", help="Concept name or alias")
    gt.add_argument("--relations", type=int, default=10, help="How many relations to show")
    gs = gsub.add_parser("similar", help="Vector-index search over concepts or chunks")
    gs.add_argument("text")
    gs.add_argument("--chunks", action="store_true", help="Search passages, not concepts")
    gs.add_argument("-k", type=int, default=5)
    gs.add_argument("--embed-model", default="bge-m3")
    grs = gsub.add_parser("reset", help="Delete everything in the graph")
    grs.add_argument("--yes", action="store_true", help="Confirm")

    se = sub.add_parser("search", help="Search passages (D4)")
    se.add_argument("query")
    se.add_argument("--mode", default="hybrid", choices=SEARCH_MODES)
    se.add_argument("-k", type=int, default=5)
    se.add_argument("--embed-model", default="bge-m3")

    es = sub.add_parser("eval-search", help="Score the search modes on the query set (D4)")
    es.add_argument("--queries", type=Path, default=Path("benchmarks/search/queries"))
    es.add_argument("--modes", default=",".join(SEARCH_MODES), help="Comma-separated")
    es.add_argument("--embed-model", default="bge-m3")
    es.add_argument("--report", type=Path, default=None, help="Write a Markdown report here")
    es.add_argument("--details", action="store_true", help="Print each query's first hit")
    es.add_argument(
        "--split", default="all", choices=["all", "dev", "test"], help="Score one half only"
    )
    return p


def _merge_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--embed-model", default="bge-m3")
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_MERGE_THRESHOLD,
        help=f"Embedding merge threshold (default: {DEFAULT_MERGE_THRESHOLD})",
    )
    p.add_argument("--no-embed", action="store_true", help="Merge by names only")


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
    if args.command in ("extract", "merge", "concepts", "eval-extract"):
        from domaingraph.llm import LLMError

        try:
            return {
                "extract": _extract,
                "merge": _merge,
                "concepts": _concepts,
                "eval-extract": _eval_extract,
            }[args.command](args)
        except LLMError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if args.command in ("graph", "search", "eval-search"):
        from domaingraph.graph import GraphError
        from domaingraph.llm import LLMError

        handler = {"graph": _graph, "search": _search, "eval-search": _eval_search}
        try:
            return handler[args.command](args, parser)
        except (GraphError, LLMError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
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


# --- D2: extraction -----------------------------------------------------------------------


def _knowledge_path(out: Path, model: str) -> Path:
    from domaingraph.extraction import model_slug

    return out / "knowledge" / f"{model_slug(model)}.json"


def _embedder(args: argparse.Namespace):
    if args.no_embed:
        return None
    from domaingraph.llm import OllamaEmbedder

    return OllamaEmbedder(args.embed_model)


def _merged(out: Path, model: str, args: argparse.Namespace, embedder=None):
    """Merge every source's saved extraction for ``model``."""
    from domaingraph.extraction import load_results
    from domaingraph.merge import merge

    results = []
    for s in list_sources(out):
        results += [r for r in load_results(out, s.id, model) if r.error is None]
    emb = embedder if embedder is not None else _embedder(args)
    return merge(results, embedder=emb, threshold=args.threshold, model=model)


def _write_knowledge(out: Path, kn) -> Path:
    p = _knowledge_path(out, kn.model)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(kn.model_dump_json(indent=1), encoding="utf-8")
    return p


def _extract(args: argparse.Namespace) -> int:
    from domaingraph.extraction import extract_source
    from domaingraph.llm import OllamaStructured

    if args.source_ids:
        ids = [_resolve(args.out, x) for x in args.source_ids]
        if None in ids:
            return 2
    else:
        ids = [s.id for s in list_sources(args.out)]
    if not ids:
        print("error: nothing ingested yet", file=sys.stderr)
        return 2
    llm = OllamaStructured(args.model)
    errors = 0
    try:
        for sid in ids:
            source, chunks = load_source(args.out, sid)
            print(f"# {source.title} ({len(chunks)} chunks) with {args.model}", flush=True)

            def show(r, reused):
                nonlocal errors
                if r.error:
                    errors += 1
                    print(f"  [{r.index:>3}] {r.locator}  ERROR {r.error}", flush=True)
                elif not reused:
                    e = r.extraction
                    print(
                        f"  [{r.index:>3}] {r.locator}  {len(e.concepts)} concepts, "
                        f"{len(e.relations)} relations, {len(e.facts)} facts  ({r.seconds} s)",
                        flush=True,
                    )

            extract_source(
                llm,
                source,
                chunks,
                out=args.out,
                force=args.force,
                known_names=not args.no_known,
                on_chunk=show,
            )
    finally:
        llm.unload()  # free VRAM for the embedding model and the next run
        llm.close()
    kn = _merged(args.out, args.model, args)
    path = _write_knowledge(args.out, kn)
    st = kn.stats
    print(
        f"merged {st['mentions']} concept mentions -> {st['concepts']} concepts "
        f"({st['embedding_merges']} by embedding), {st['relations']} relations, "
        f"{st['facts']} facts -> {path}"
    )
    if errors:
        print(f"{errors} chunks failed; run extract again to retry them", file=sys.stderr)
    return 1 if errors else 0


def _merge(args: argparse.Namespace) -> int:
    kn = _merged(args.out, args.model, args)
    if not kn.sources:
        print(f"error: no extractions for {args.model}", file=sys.stderr)
        return 2
    path = _write_knowledge(args.out, kn)
    print(f"{kn.stats['concepts']} concepts, {kn.stats['relations']} relations -> {path}")
    return 0


def _concepts(args: argparse.Namespace) -> int:
    from domaingraph.merge import Knowledge

    p = _knowledge_path(args.out, args.model)
    if not p.is_file():
        print(f"error: no knowledge file for {args.model}; run extract first", file=sys.stderr)
        return 2
    kn = Knowledge.model_validate_json(p.read_text(encoding="utf-8"))
    sid = _resolve(args.out, args.source) if args.source else None
    if args.source and sid is None:
        return 2
    cs = [c for c in kn.concepts if sid is None or any(m.source_id == sid for m in c.mentions)]
    cs.sort(key=lambda c: -len(c.mentions))
    for c in cs[: args.limit]:
        ms = [m for m in c.mentions if sid is None or m.source_id == sid]
        where = ", ".join(m.locator for m in ms[:4]) + (" ..." if len(ms) > 4 else "")
        alias = f"  (aka {', '.join(c.aliases[:4])})" if c.aliases else ""
        print(f"{c.name}{alias}  [{c.type}, {len(ms)} mentions, conf {c.confidence:.2f}]")
        print(f"    {c.definition}")
        print(f"    at {where}")
    return 0


def _eval_extract(args: argparse.Namespace) -> int:
    from domaingraph.evaluate_extraction import (
        SUMMARY_HEADER,
        load_gold_dir,
        markdown_table,
        score_source,
        summary_rows,
    )

    golds = load_gold_dir(args.gold)
    if not golds:
        print(f"error: no gold files in {args.gold}", file=sys.stderr)
        return 2
    if args.sweep:
        return _sweep(args, golds)
    embedder = _embedder(args)
    lenient = embedder
    if lenient is None:
        from domaingraph.llm import OllamaEmbedder

        lenient = OllamaEmbedder(args.embed_model)
    scores = {}
    min_chunks = [int(x) for x in args.min_chunks.split(",") if x.strip()]
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        kn = _merged(args.out, model, args, embedder=embedder)
        missing = [g.source_id for g in golds if g.source_id not in kn.sources]
        if missing:
            print(f"error: {model} has no extraction for {', '.join(missing)}", file=sys.stderr)
            return 2
        for k in min_chunks:
            key = model if min_chunks == [1] else f"{model}, >= {k} chunks"
            scores[key] = [
                score_source(
                    kn,
                    g,
                    embedder=lenient,
                    lenient_threshold=args.lenient_threshold,
                    min_chunks=k,
                )
                for g in golds
            ]
        if args.details:
            for s in scores[key]:
                print(f"\n## {key}: {s.title}")
                print(
                    f"  matched ({len(s.matched)}): "
                    + "; ".join(
                        f"{k} -> {v}" if k != v else k for k, v in sorted(s.matched.items())
                    )
                )
                print(f"  not in gold ({len(s.false_positives)}): " + ", ".join(s.false_positives))
                print(f"  missed ({len(s.missed)}): " + ", ".join(s.missed))
    merge_desc = "names only" if args.no_embed else f"{args.embed_model} >= {args.threshold}"
    table = markdown_table(SUMMARY_HEADER, summary_rows(scores))
    print(f"\nGold: {len(golds)} sources, merge: {merge_desc}\n")
    print(table)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        per_source = []
        for model, ss in scores.items():
            for s in ss:
                per_source.append(
                    [
                        model,
                        s.title.split(": ", 1)[-1],
                        f"{s.concepts.precision:.0%} / {s.concepts.recall:.0%}",
                        f"{s.core_recall.recall:.0%}",
                        f"{s.relations.precision:.0%} / {s.relations.recall:.0%}",
                        ", ".join(s.missed_core) or "-",
                    ]
                )
        body = (
            f"# D2 extraction results\n\nGold: {len(golds)} sources; merge: {merge_desc}; "
            f"lenient match: {args.embed_model} >= {args.lenient_threshold}\n\n{table}\n\n"
            "## Per source\n\n"
            + markdown_table(
                ["Model", "Lecture", "Concepts P / R", "Core R", "Relations P / R", "Missed core"],
                per_source,
            )
            + "\n"
        )
        args.report.write_text(body, encoding="utf-8")
        print(f"\nreport -> {args.report}")
    return 0


def _sweep(args: argparse.Namespace, golds) -> int:
    """Merge quality and concept scores across embedding thresholds (names-only first)."""
    from domaingraph.evaluate_extraction import PRF, markdown_table, merge_quality, score_source
    from domaingraph.llm import OllamaEmbedder

    class Cached:
        """Same embedder, memoized: a sweep embeds the same names many times."""

        def __init__(self, inner):
            self.inner, self.model, self.memo = inner, inner.model, {}

        def embed(self, texts):
            todo = [t for t in dict.fromkeys(texts) if t not in self.memo]
            self.memo.update(zip(todo, self.inner.embed(todo), strict=True))
            return [self.memo[t] for t in texts]

    emb = Cached(OllamaEmbedder(args.embed_model))
    rows = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        for t in ["names", *[x.strip() for x in args.sweep.split(",")]]:
            args.no_embed, args.threshold = t == "names", 1.0 if t == "names" else float(t)
            kn = _merged(args.out, model, args, embedder=None if args.no_embed else emb)
            ss = [score_source(kn, g, embedder=None) for g in golds]
            c = sum((s.concepts for s in ss), PRF())
            mq = merge_quality(kn, golds)
            rows.append(
                [
                    model,
                    t,
                    str(kn.stats["concepts"]),
                    str(kn.stats["embedding_merges"]),
                    f"{mq.precision:.1%} / {mq.recall:.1%}",
                    f"{c.precision:.0%} / {c.recall:.0%}",
                    str(sum(s.duplicates for s in ss)),
                ]
            )
    print(
        markdown_table(
            [
                "Model",
                "Threshold",
                "Concepts",
                "Emb. merges",
                "Merge P / R",
                "Concepts P / R",
                "Dups",
            ],
            rows,
        )
    )
    return 0


# --- D3: graph store -----------------------------------------------------------------------


def _graph(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from domaingraph.graph import GraphConfig, GraphStore

    if args.graph_command is None:
        parser.parse_args(["graph", "--help"])
        return 0
    with GraphStore(GraphConfig.from_env()) as store:
        return {
            "load": _graph_load,
            "stats": _graph_stats,
            "trace": _graph_trace,
            "similar": _graph_similar,
            "reset": _graph_reset,
        }[args.graph_command](args, store)


def _graph_load(args: argparse.Namespace, store) -> int:
    from domaingraph.graph import build_rows
    from domaingraph.llm import OllamaEmbedder
    from domaingraph.merge import Knowledge

    p = _knowledge_path(args.out, args.model)
    if not p.is_file():
        print(f"error: no knowledge file for {args.model}; run extract first", file=sys.stderr)
        return 1
    kn = Knowledge.model_validate_json(p.read_text(encoding="utf-8"))
    known = {s.id for s in list_sources(args.out)}
    missing = [s for s in kn.sources if s not in known]
    if missing:
        print(f"error: knowledge refers to sources not in {args.out}: {missing}", file=sys.stderr)
        return 1
    sources = [load_source(args.out, s) for s in kn.sources]
    rows = build_rows(kn, sources, args.domain)
    embedder = None if args.no_embed else OllamaEmbedder(args.embed_model)
    st = store.load(rows, model=kn.model, embedder=embedder, replace=args.replace)
    print(
        f"loaded {kn.model} into domain {args.domain!r}: {st.sources} sources, "
        f"{st.chunks} chunks, {st.concepts} concepts, {st.mentions} mentions, "
        f"{st.relations} RELATED_TO, {st.part_of} PART_OF, {st.facts} facts; "
        f"embedded {st.embedded}; removed {st.removed}"
    )
    return 0


def _graph_stats(args: argparse.Namespace, store) -> int:
    meta = store.meta()
    if meta:
        print(
            f"model {meta.get('model')}, embeddings {meta.get('embed_model')} "
            f"({meta.get('dims')} dims), loaded {meta.get('loaded_at')}"
        )
    for k, v in store.counts().items():
        print(f"{k:>18}  {v}")
    return 0


def _graph_trace(args: argparse.Namespace, store) -> int:
    c = store.find_concept(args.concept)
    if c is None:
        print(f"error: no concept named {args.concept!r}", file=sys.stderr)
        return 1
    aka = f" (also: {', '.join(c['aliases'])})" if c["aliases"] else ""
    print(f"{c['name']} [{c['type']}]{aka}")
    if c["definition"]:
        print(f"  {c['definition']}")
    rels = store.neighbours(c["id"])
    if rels:
        print(f"\nrelations ({len(rels)}, most-mentioned first):")
        for r in rels[: args.relations]:
            arrow = f"-[{r['predicate']}]->" if r["dir"] == "out" else f"<-[{r['predicate']}]-"
            print(f"  {arrow} {r['other']}  (x{r['n']})")
        if len(rels) > args.relations:
            print(f"  ... {len(rels) - args.relations} more")
    print("\nmentioned in:")
    current = None
    for r in store.trace(c["id"]):
        if r["source"] != current:
            current = r["source"]
            print(f"  {current}")
        said = ", ".join(r["said_as"]) if r["said_as"] else ""
        print(f"    {r['locator']:>13}  chunk {r['chunk']:>2}  {said}")
    return 0


def _graph_similar(args: argparse.Namespace, store) -> int:
    from domaingraph.llm import OllamaEmbedder

    meta = store.meta()
    if meta.get("embed_model") and meta["embed_model"] != args.embed_model:
        print(f"error: the graph was embedded with {meta['embed_model']}", file=sys.stderr)
        return 1
    vec = OllamaEmbedder(args.embed_model).embed([args.text])[0]
    label = "Chunk" if args.chunks else "Concept"
    for r in store.similar(vec, label, args.k):
        where = f"  {r['locator']}" if r["locator"] else ""
        print(f"{r['score']:.3f}  {r['name']}{where}")
    return 0


def _graph_reset(args: argparse.Namespace, store) -> int:
    if not args.yes:
        print("error: this deletes the whole graph; pass --yes to confirm", file=sys.stderr)
        return 1
    store.reset()
    print("graph emptied")
    return 0


# --- D4: search ----------------------------------------------------------------------------


def _searchers(store, out: Path, modes) -> dict:
    from domaingraph.search import BM25Search, GraphSearch, HybridSearch, VectorSearch

    built: dict = {}
    if "bm25" in modes:
        chunks = [(c.id, c.text) for s in list_sources(out) for c in load_source(out, s.id)[1]]
        built["bm25"] = BM25Search(chunks)
    if {"vector", "hybrid", "hybrid-rrf"} & set(modes):
        built["vector"] = VectorSearch(store)
    if {"graph", "hybrid", "hybrid-rrf"} & set(modes):
        built["graph"] = GraphSearch(store)
    if "hybrid" in modes:
        built["hybrid"] = HybridSearch(built["vector"], built["graph"])
    if "hybrid-rrf" in modes:
        built["hybrid-rrf"] = HybridSearch(
            built["vector"], built["graph"], fusion="rrf", bonus=0.01
        )
    if {"graph-gold", "hybrid-gold"} & set(modes):
        from domaingraph.evaluate_extraction import load_gold_dir
        from domaingraph.llm import OllamaEmbedder
        from domaingraph.search import gold_graph

        srcs = sorted(list_sources(out), key=lambda s: s.id)
        loaded = {s.id: load_source(out, s.id)[1] for s in srcs}
        by_pos = {(sid, c.index): c.id for sid, cs in loaded.items() for c in cs}
        built["graph-gold"] = gold_graph(
            load_gold_dir(Path("benchmarks/extraction/gold")),
            [c.id for cs in loaded.values() for c in cs],
            lambda s, i: by_pos.get((s, i)),
            OllamaEmbedder("bge-m3"),
        )
        built["hybrid-gold"] = HybridSearch(VectorSearch(store), built["graph-gold"])
    return {m: built[m] for m in modes}


def _check_embed_model(store, model: str) -> None:
    from domaingraph.graph import GraphError

    meta = store.meta()
    if meta.get("embed_model") not in (None, model):
        raise GraphError(f"the graph was embedded with {meta['embed_model']}, not {model}")


def _chunk_index(out: Path) -> dict[str, tuple[str, str]]:
    """chunk id -> (source title, locator)."""
    idx = {}
    for s in list_sources(out):
        for c in load_source(out, s.id)[1]:
            idx[c.id] = (s.title, c.locator())
    return idx


def _search(args: argparse.Namespace, parser) -> int:
    from domaingraph.graph import GraphConfig, GraphStore
    from domaingraph.llm import OllamaEmbedder

    with GraphStore(GraphConfig.from_env()) as store:
        _check_embed_model(store, args.embed_model)
        searcher = _searchers(store, args.out, [args.mode])[args.mode]
        vec = OllamaEmbedder(args.embed_model).embed([args.query])[0]
        where = _chunk_index(args.out)
        texts = {
            c.id: c.text for s in list_sources(args.out) for c in load_source(args.out, s.id)[1]
        }
        for rank, (cid, score) in enumerate(searcher.search(args.query, vec, args.k), 1):
            title, loc = where.get(cid, ("?", "?"))
            print(f"{rank}. {score:.3f}  {title}  {loc}")
            print(f"   {texts.get(cid, '')[:160]}...")
    return 0


def _eval_search(args: argparse.Namespace, parser) -> int:
    from domaingraph.evaluate_search import (
        METRICS,
        QueryResult,
        by_type,
        load_queries,
        paired_bootstrap,
        split_of,
        summarize,
    )
    from domaingraph.graph import GraphConfig, GraphStore
    from domaingraph.llm import OllamaEmbedder

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in SEARCH_MODES + EVAL_ONLY_MODES]
    if bad:
        print(
            f"error: unknown modes {bad}; choose from {SEARCH_MODES + EVAL_ONLY_MODES}",
            file=sys.stderr,
        )
        return 1
    by_pos = {}
    for s in list_sources(args.out):
        for c in load_source(args.out, s.id)[1]:
            by_pos[(s.id, c.index)] = c.id
    queries = load_queries(args.queries, lambda sid, i: by_pos.get((sid, i)))
    if args.split != "all":
        queries = [q for q in queries if split_of(q.id) == args.split]

    with GraphStore(GraphConfig.from_env()) as store:
        _check_embed_model(store, args.embed_model)
        searchers = _searchers(store, args.out, modes)
        vecs = OllamaEmbedder(args.embed_model).embed([q.text for q in queries])
        results = {
            m: [
                QueryResult(q, [cid for cid, _ in s.search(q.text, v, 10)])
                for q, v in zip(queries, vecs, strict=True)
            ]
            for m, s in searchers.items()
        }

    cols = list(METRICS)
    type_counts = ", ".join(f"{t} {len(r)}" for t, r in sorted(by_type(results[modes[0]]).items()))
    lines = [
        f"Queries: {len(queries)}, split {args.split} ({type_counts})",
        "",
    ]
    lines += ["| Mode | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for m in modes:
        s = summarize(results[m])
        lines.append(f"| {m} | " + " | ".join(f"{s[c]:.3f}" for c in cols) + " |")

    lines += ["", "MRR@10 by query type:", ""]
    types = sorted(by_type(results[modes[0]]))
    lines += ["| Mode | " + " | ".join(types) + " |", "|---" * (len(types) + 1) + "|"]
    for m in modes:
        bt = by_type(results[m])
        lines.append(
            f"| {m} | " + " | ".join(f"{summarize(bt[t])['MRR@10']:.3f}" for t in types) + " |"
        )

    if "vector" in modes:
        lines += ["", "Paired difference vs vector (95% bootstrap interval over queries):", ""]
        lines += ["| Mode | MRR@10 | R@5 |", "|---|---|---|"]
        for m in modes:
            if m == "vector":
                continue
            cells = []
            for metric in ("MRR@10", "R@5"):
                f = METRICS[metric]
                d, lo, hi = paired_bootstrap(
                    [f(r) for r in results["vector"]], [f(r) for r in results[m]]
                )
                cells.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")

    text = "\n".join(lines)
    print(text)
    if args.details:
        print()
        for i, q in enumerate(queries):
            hits = "  ".join(f"{m}={results[m][i].first_hit() or '-'}" for m in modes)
            print(f"{q.id:6} {q.type[:5]:5} {hits}  {q.text}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("# D4 search results\n\n" + text + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0
