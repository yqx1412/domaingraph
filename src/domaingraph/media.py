"""Audio and video input through ffmpeg: probe the duration, decode to 16 kHz mono PCM.

Decoding is done here rather than inside faster-whisper so that every container ffmpeg
reads (mp4, mkv, webm, m4a, ...) works the same way, and so that faster-whisper's own
decoder dependency is never on the path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


class MediaError(RuntimeError):
    """ffmpeg is missing, or the file has no decodable audio."""


def _tool(name: str) -> str:
    override = os.environ.get(f"DOMAINGRAPH_{name.upper()}")
    if override:
        return override
    found = shutil.which(name)
    if found is None:
        raise MediaError(
            f"{name} not found on PATH. Install ffmpeg (Windows: winget install Gyan.FFmpeg, "
            f"then open a new terminal) or set DOMAINGRAPH_{name.upper()} to its path."
        )
    return found


def probe(path: Path) -> dict:
    """Duration and audio stream info from ffprobe. Raises MediaError without audio."""
    proc = subprocess.run(
        [
            _tool("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise MediaError(f"ffprobe could not read {path.name}: {proc.stderr.strip()[:300]}")
    info = json.loads(proc.stdout or "{}")
    audio = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        raise MediaError(f"{path.name} has no audio stream")
    duration = float(info.get("format", {}).get("duration") or 0.0)
    return {"duration": duration, "audio_codec": audio[0].get("codec_name")}


def decode(path: Path) -> np.ndarray:
    """The first audio stream as float32 samples in [-1, 1], 16 kHz mono."""
    proc = subprocess.run(
        [
            _tool("ffmpeg"),
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", errors="replace").strip()[:300]
        raise MediaError(f"ffmpeg could not decode {path.name}: {msg}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
