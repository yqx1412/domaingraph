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

## Development

```powershell
uv sync
uv run pre-commit install
uv run pytest
uv run ruff check .
```
