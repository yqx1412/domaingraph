import json
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from domaingraph.asr import Transcript
from domaingraph.chunking import ChunkConfig, chunk_blocks, split_block
from domaingraph.cli import main
from domaingraph.documents import load_markdown, load_pdf, load_text
from domaingraph.evaluate import normalize, number_words, word_error_rate
from domaingraph.media import SAMPLE_RATE, MediaError, decode, probe
from domaingraph.models import Block, Chunk, Segment, kind_of, timestamp
from domaingraph.pipeline import ingest, list_sources, load_source

FIXTURES = Path(__file__).parent / "fixtures"
HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")

# -- models ----------------------------------------------------------------------------


def test_timestamp_and_locator() -> None:
    assert timestamp(75.4) == "01:15"
    assert timestamp(3725) == "1:02:05"
    c = Chunk(id="s:0", source_id="s", index=0, text="x", n_words=1, start=61, end=125)
    assert c.locator() == "01:01-02:05"
    p = Chunk(id="s:1", source_id="s", index=1, text="x", n_words=1, page_start=3, page_end=4)
    assert p.locator() == "p. 3-4"


def test_kind_of() -> None:
    assert kind_of(Path("a.MP4")) == "video"
    assert kind_of(Path("a.m4a")) == "audio"
    assert kind_of(Path("notes.md")) == "markdown"
    with pytest.raises(ValueError, match="unsupported"):
        kind_of(Path("a.docx"))


# -- chunking --------------------------------------------------------------------------


def words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def test_packs_blocks_to_target_with_overlap_and_time_ranges() -> None:
    blocks = [Block(text=words(30, f"b{i}_"), start=i * 10.0, end=i * 10.0 + 9) for i in range(10)]
    chunks = chunk_blocks("s", blocks, ChunkConfig(target_words=100, overlap_words=30))
    assert all(c.n_words <= 100 for c in chunks)
    # Each chunk after the first starts with the last block of the previous one.
    for prev, nxt in pairwise(chunks):
        assert nxt.text.split()[0] == prev.text.split()[-30]
        assert nxt.start == prev.end - 9
    assert chunks[0].start == 0.0 and chunks[-1].end == 99.0
    # Every block appears in some chunk.
    joined = " ".join(c.text for c in chunks)
    assert all(b.text in joined for b in blocks)
    assert [c.id for c in chunks[:2]] == ["s:0000", "s:0001"]


def test_no_overlap_when_disabled_or_block_too_big() -> None:
    blocks = [Block(text=words(60, f"b{i}_")) for i in range(4)]
    chunks = chunk_blocks("s", blocks, ChunkConfig(target_words=100, overlap_words=40))
    assert len(chunks) == 4  # 60 + 60 > 100, and a 60-word block cannot be the overlap
    assert sum(c.n_words for c in chunks) == 240


def test_long_block_is_split_with_interpolated_times() -> None:
    text = ". ".join(words(10, f"s{i}_") for i in range(10)) + "."
    block = Block(text=text, start=100.0, end=200.0)
    parts = split_block(block, max_words=30)
    assert all(len(p.text.split()) <= 30 for p in parts)
    assert parts[0].start == 100.0 and parts[-1].end == 200.0
    for a, b in pairwise(parts):
        assert a.end == pytest.approx(b.start)
    run_on = split_block(Block(text=words(95)), max_words=30)  # no sentence ends at all
    assert [len(p.text.split()) for p in run_on] == [30, 30, 30, 5]


def test_heading_change_starts_a_new_chunk_without_overlap() -> None:
    blocks = [
        Block(text=words(10, "a"), heading=["Raft"]),
        Block(text=words(10, "b"), heading=["Raft"]),
        Block(text=words(10, "c"), heading=["Paxos"]),
    ]
    chunks = chunk_blocks("s", blocks, ChunkConfig(target_words=100, overlap_words=10))
    assert [c.heading for c in chunks] == [["Raft"], ["Paxos"]]
    assert "b0" not in chunks[1].text


def test_chunk_config_validation() -> None:
    with pytest.raises(ValueError):
        ChunkConfig(target_words=100, max_words=50)
    with pytest.raises(ValueError):
        ChunkConfig(target_words=100, overlap_words=100)


# -- documents -------------------------------------------------------------------------


def test_markdown_headings_front_matter_and_code(tmp_path: Path) -> None:
    md = tmp_path / "notes.md"
    md.write_text(
        "---\ntags: [x]\n---\n# Consensus\n\nIntro line one\nwraps here.\n\n"
        "## Raft\n\nLeaders win elections.\n\n```python\ndef f():\n\n    return 1\n```\n\n"
        "### Terms\n\nTerms are numbered.\n\n## Paxos\n\nAcceptors vote.\n",
        encoding="utf-8",
    )
    blocks, title = load_markdown(md)
    assert title == "Consensus"
    assert [(b.heading, b.text[:20]) for b in blocks] == [
        (["Consensus"], "Intro line one wraps"),
        (["Consensus", "Raft"], "Leaders win election"),
        (["Consensus", "Raft"], "```python\ndef f():\n\n"),
        (["Consensus", "Raft", "Terms"], "Terms are numbered."),
        (["Consensus", "Paxos"], "Acceptors vote."),
    ]
    assert "\n\n    return 1" in blocks[2].text  # code block kept whole, blank line included
    assert "tags" not in " ".join(b.text for b in blocks)


def test_text_paragraphs(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("one\ntwo\n\n\nthree\n", encoding="utf-8")
    assert [b.text for b in load_text(f)] == ["one two", "three"]


def _tiny_pdf(path: Path, pages: list[str]) -> None:
    """A minimal valid PDF with one line of Helvetica text per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")
    font_id = 3 + 2 * len(pages)
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {4 + 2 * i} 0 R >>"
        )
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offsets = b"%PDF-1.4\n", []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)


def test_pdf_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    _tiny_pdf(pdf, ["Raft elects a leader.", "Paxos uses acceptors.", "Logs replicate."])
    blocks, n, _ = load_pdf(pdf)
    assert n == 3
    assert [(b.page, b.text) for b in blocks] == [
        (1, "Raft elects a leader."),
        (2, "Paxos uses acceptors."),
        (3, "Logs replicate."),
    ]
    chunks = chunk_blocks("p", blocks)
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 3)


# -- WER -------------------------------------------------------------------------------


def test_normalize_and_numbers() -> None:
    assert number_words(150) == "one hundred fifty"
    assert number_words(1287) == "one thousand two hundred eighty seven"
    assert normalize("Lecture 7: Raft's up-to-date logs!") == [
        "lecture",
        "seven",
        "raft's",
        "up",
        "to",
        "date",
        "logs",
    ]


def test_word_error_rate() -> None:
    assert word_error_rate("the cat sat", "the cat sat").wer == 0
    r = word_error_rate("the cat sat on the mat", "the bat sat on mat today")
    assert (r.substitutions, r.deletions, r.insertions) == (1, 1, 1)
    assert r.wer == pytest.approx(3 / 6)
    assert word_error_rate("lecture seven", "Lecture 7.").wer == 0


# -- media -----------------------------------------------------------------------------


def _tone(path: Path, seconds: float = 2.0, video: bool = False) -> None:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={seconds}",
    ]
    if video:
        cmd += [
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=64x64:d={seconds}",
            "-shortest",
            "-c:v",
            "libx264",
        ]
    subprocess.run([*cmd, str(path)], check=True)


@needs_ffmpeg
def test_probe_and_decode_video(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _tone(clip, 2.0, video=True)
    info = probe(clip)
    assert info["duration"] == pytest.approx(2.0, abs=0.1)
    audio = decode(clip)
    assert audio.dtype == np.float32
    assert len(audio) / SAMPLE_RATE == pytest.approx(2.0, abs=0.1)
    assert 0.1 < np.abs(audio).max() <= 1.0


@needs_ffmpeg
def test_video_without_audio_is_a_clear_error(tmp_path: Path) -> None:
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
         "-c:v", "libx264", str(silent)],
        check=True,
    )  # fmt: skip
    with pytest.raises(MediaError, match="no audio"):
        probe(silent)


# -- pipeline (fake transcriber; real ffmpeg) ------------------------------------------


class FakeTranscriber:
    name = model_name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio: np.ndarray, *, language: str | None = None) -> Transcript:
        self.calls += 1
        dur = len(audio) / SAMPLE_RATE
        segs = [
            Segment(start=i * dur / 4, end=(i + 1) * dur / 4, text=f"Sentence {i} about Raft.")
            for i in range(4)
        ]
        return Transcript(segs, "en", {"model": "fake", "device": "cpu", "seconds": 0.1,
                                       "realtime_factor": 20.0})  # fmt: skip


@needs_ffmpeg
def test_ingest_video_end_to_end_and_idempotent(tmp_path: Path) -> None:
    clip = tmp_path / "lecture_07.mp4"
    _tone(clip, 2.0, video=True)
    fake = FakeTranscriber()
    out = tmp_path / "data"

    r = ingest(clip, out=out, transcriber=lambda: fake, asr_model="fake")
    assert not r.skipped and fake.calls == 1
    assert r.source.kind == "video" and r.source.title == "lecture 07"
    assert r.source.duration == pytest.approx(2.0, abs=0.1)
    assert r.chunks[0].start == 0.0 and r.chunks[-1].end == pytest.approx(2.0, abs=0.1)
    d = out / "sources" / r.source.id
    assert {p.name for p in d.iterdir()} == {"source.json", "transcript.json", "chunks.jsonl"}
    assert json.loads((d / "transcript.json").read_text())["segments"][0]["text"].startswith(
        "Sentence"
    )

    # Same content under another name: recognized, nothing redone.
    copy = tmp_path / "renamed.mp4"
    shutil.copy(clip, copy)
    again = ingest(copy, out=out, transcriber=lambda: fake, asr_model="fake")
    assert again.skipped and fake.calls == 1

    # --force with the same model re-chunks from the saved transcript.
    forced = ingest(clip, out=out, transcriber=lambda: fake, asr_model="fake", force=True,
                    chunking=ChunkConfig(target_words=20, overlap_words=5))  # fmt: skip
    assert forced.reused_transcript and fake.calls == 1
    assert [s.id for s in list_sources(out)] == [r.source.id]
    assert load_source(out, r.source.id)[1] == forced.chunks


def test_ingest_text_never_loads_a_transcriber(tmp_path: Path) -> None:
    def boom():
        raise AssertionError("transcriber should not be loaded for text")

    md = tmp_path / "notes.md"
    md.write_text("# Notes\n\nRaft has leaders.\n", encoding="utf-8")
    r = ingest(md, out=tmp_path / "data", transcriber=boom)
    assert r.source.title == "Notes" and r.chunks[0].heading == ["Notes"]


def test_cli_ingest_sources_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = str(tmp_path / "data")
    assert main(["--out", out, "ingest", str(FIXTURES / "raft_lecture.txt")]) == 0
    line = capsys.readouterr().out
    assert "ingested" in line and "text" in line
    assert main(["--out", out, "sources"]) == 0
    sid = capsys.readouterr().out.split()[0]
    assert main(["--out", out, "show", sid[:6], "--limit", "1"]) == 0
    assert "[0]" in capsys.readouterr().out
    assert main(["--out", out, "ingest", str(tmp_path / "missing.mp4")]) == 1
