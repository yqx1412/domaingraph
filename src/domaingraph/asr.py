"""Speech recognition with faster-whisper, on the GPU when one is usable.

GPU inference needs NVIDIA's cuBLAS and cuDNN DLLs. They come from the optional ``gpu``
extra (``uv sync --extra gpu``) as pip packages; :func:`_cuda_dll_dirs` puts them on the
DLL search path before CTranslate2 loads. If CUDA is not usable after all, the model falls
back to the CPU (int8), which is several times slower but gives the same kind of output.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from domaingraph.media import SAMPLE_RATE
from domaingraph.models import Segment

log = logging.getLogger(__name__)

DEFAULT_MODEL = "large-v3-turbo"


@dataclass
class Transcript:
    segments: list[Segment]
    language: str | None
    info: dict[str, Any] = field(default_factory=dict)


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, *, language: str | None = None) -> Transcript: ...


def _cuda_dll_dirs() -> list[str]:
    """Register the pip-installed NVIDIA DLL folders (Windows). Harmless elsewhere."""
    try:
        import nvidia  # namespace package from nvidia-cublas-cu12 / nvidia-cudnn-cu12
    except ImportError:
        return []
    added = []
    for root in getattr(nvidia, "__path__", []):
        for sub in ("cublas", "cudnn", "cuda_nvrtc"):
            d = os.path.join(root, sub, "bin")
            if os.path.isdir(d) and d not in added:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(d)
                # CTranslate2 loads cuBLAS by name, which only searches PATH on Windows.
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                added.append(d)
    return added


class WhisperTranscriber:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 5,
        vad: bool = True,
    ) -> None:
        self.model_name = model
        self.beam_size = beam_size
        self.vad = vad
        self.device, self.compute_type = self._load(model, device, compute_type)
        self.name = f"faster-whisper/{model}"

    def _load(self, model: str, device: str, compute_type: str) -> tuple[str, str]:
        import ctranslate2
        from faster_whisper import WhisperModel

        _cuda_dll_dirs()
        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        if device == "cuda":
            try:  # cuBLAS/cuDNN load lazily, on the first encode: find out now
                list(self._model.transcribe(np.zeros(SAMPLE_RATE, np.float32), language="en")[0])
            except RuntimeError as exc:
                log.warning("CUDA unusable (%s); falling back to CPU int8", exc)
                device, compute_type = "cpu", "int8"
                self._model = WhisperModel(model, device=device, compute_type=compute_type)
        return device, compute_type

    def transcribe(self, audio: np.ndarray, *, language: str | None = None) -> Transcript:
        start = time.perf_counter()
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            vad_filter=self.vad,
            condition_on_previous_text=False,  # limits runaway repetition on long audio
        )
        segs = [
            Segment(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip())
            for s in segments
            if s.text.strip()
        ]
        seconds = time.perf_counter() - start
        duration = len(audio) / SAMPLE_RATE
        return Transcript(
            segments=segs,
            language=info.language,
            info={
                "model": self.model_name,
                "device": self.device,
                "compute_type": self.compute_type,
                "beam_size": self.beam_size,
                "vad": self.vad,
                "seconds": round(seconds, 1),
                "realtime_factor": round(duration / seconds, 1) if seconds else None,
                "language_probability": round(info.language_probability, 3),
            },
        )
