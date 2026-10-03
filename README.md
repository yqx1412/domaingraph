# domaingraph

Domain-aware multimodal knowledge graph: Whisper -> Neo4j -> hybrid search, with an MCP server and Obsidian export.

**Question this project answers:** How can an agent keep structured, domain-aware, multimodal long-term knowledge, and does a graph beat vector-only search?

Part of the local AI agent ecosystem; see `../ROADMAP.md`.

## D1: ingest pipeline

Turns lectures and documents into timestamped (or page/heading-tagged) chunks of ~200 words.

```powershell
uv sync --extra gpu                     # CUDA 12 libraries for faster-whisper on an NVIDIA GPU
domaingraph ingest lecture.mp4 notes/   # video, audio, .txt, .md, .pdf; folders are walked
domaingraph sources                     # what's ingested
domaingraph show <source-id>            # its chunks
domaingraph wer <source-id> ref.txt     # word error rate against a reference transcript
```

- **Video / audio:** ffmpeg decodes to 16 kHz mono, then faster-whisper (`large-v3-turbo`, VAD on, beam 5) produces timestamped segments. ffmpeg must be on PATH (`winget install Gyan.FFmpeg`).
- **Documents:** text and Markdown split into paragraphs (Markdown keeps its heading path), PDF keeps page numbers.
- **Chunking:** segments/paragraphs are packed into ~200-word chunks (max 300) with a 40-word overlap; each chunk keeps its time range, page range or heading.
- **Idempotent:** sources are keyed by content hash, so the same file is skipped even if renamed. `--force` re-chunks from the saved transcript instead of transcribing again.
- Output goes to `data/sources/<id>/` (`source.json`, `transcript.json`, `chunks.jsonl`); `data/` is git-ignored.

**Real-lecture result** (RTX 5060 Ti 16 GB, CUDA float16): MIT 6.006 Fall 2011, Lecture 6 "AVL Trees, AVL Sort" ([YouTube](https://www.youtube.com/watch?v=FNeL18KsWPc), CC BY-NC-SA).

| Audio | Transcription time | Speed | Chunks | WER vs MIT's human captions |
|---|---|---|---|---|
| 51:58 | 58.8 s | 53x realtime | 43 | 6.1% (97 sub, 230 del, 105 ins / 7,059 words) |

Most errors are deletions: the human captions keep repeated words and false starts that Whisper drops. WER normalizes case, punctuation and digits (`6` vs `six`); caption sound tags like `[LAUGHTER]` were removed from the reference.

## D2: knowledge extraction

A local LLM (through Ollama) reads each chunk and returns concepts, relations and facts as JSON constrained to a Pydantic schema. Duplicate concepts are then merged across chunks and sources.

```powershell
ollama pull qwen3:8b; ollama pull bge-m3
domaingraph extract --model qwen3:8b            # all sources; resumes, retries failed chunks
domaingraph concepts --model qwen3:8b --limit 20
domaingraph eval-extract --models qwen3:8b,qwen3:14b --min-chunks 1,2
domaingraph eval-extract --models qwen3:8b --sweep 0.75,0.85,0.9   # merge thresholds
```

- **Extraction** (`extraction.py`): one call per chunk, temperature 0, with the reply constrained by Ollama's `format` to a JSON schema. That schema caps a chunk at 12 concepts, 12 relations and 8 facts.
  - Concepts have a name, aliases, a type, a one-line definition and a confidence score.
  - Relations use 6 predicates: `is_a`, `part_of`, `uses`, `solves`, `has_property`, `contrasts_with`.
  - Facts are self-contained claims, linked to the concepts they involve.
  - Each prompt lists the concept names earlier chunks used, so the same idea keeps the same name.
  - Every item keeps a pointer back to its chunk and time range.
  - Output goes to `data/extractions/<source>/<model>.jsonl`, written after each chunk, so an interrupted run resumes.
- **Merging** (`merge.py`), in two passes:
  1. Normalized names (case, hyphens, spaces, articles, plurals, "operation"), plus "this name is another concept's alias", e.g. `BST` -> `binary search tree`.
  2. bge-m3 similarity of names, at >= 0.85 by default and only between concepts of the same type. A contrast guard blocks look-alike pairs with different meanings, such as `left rotate` / `right rotate` or `binary tree` / `binary search tree`.

  The merged result is written to `data/knowledge/<model>.json`. Every concept, relation and fact keeps all its mentions, with chunk, time range and confidence.
- **Evaluation** (`evaluate_extraction.py`): the extracted concepts and relations are compared with gold sets in `benchmarks/extraction/gold/`. It also measures pairwise merge quality.

**Result:** gold sets for 3 lectures (MIT 6.006 Lectures 5-7): 82 concepts and 80 relations. Strict name/alias matching:

| Model | Concepts P / R | Core concepts R | Relations P / R | Rel P, ends in gold |
|---|---|---|---|---|
| qwen3:8b | 28% / 87% | 95% | 6% / 51% | 20% |
| qwen3:14b | 23% / 88% | 93% | 5% / 41% | 20% |
| llama3.1:8b | 30% / 16% | 26% | 12% / 4% | 3 of 4 |

- **Recall is high, precision is low.** The extractor finds almost every concept the lecture teaches, but also lists generic words (`node`, `pointer`) and minor real ones the gold set leaves out.
- **Relations are the weakest part.** Only about 1 in 5 relations between two gold concepts is right; the rest have the wrong predicate or direction.
- **The model's confidence carries no signal:** 96% of mentions are scored 0.9.
- **qwen3:8b beats qwen3:14b here**, because 14b lists more concepts at the same recall.
- **llama3.1:8b returns empty output for 125 of 131 chunks.**

**Caveat:** the gold sets were drafted by AI annotators (one per lecture, without seeing any extractor output) and then corrected in review. No human has checked them yet, so these numbers measure agreement with that annotation.

Full write-up, including the merge-threshold sweep and the prompt v1 -> v2 change: [`benchmarks/extraction/results/d2-extraction.md`](benchmarks/extraction/results/d2-extraction.md).

## Development

```powershell
uv sync
uv run pre-commit install
uv run pytest
uv run ruff check .
```
