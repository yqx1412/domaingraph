"""Minimal Ollama client: structured (JSON-schema) chat and embeddings.

Kept separate from AgentOS on purpose: DomainGraph must run without it. Settings match the
lessons from AgentOS A3: a larger context than Ollama's 4096-token default (which truncates
silently) and a cap on reply length (a degenerate generation otherwise runs to the timeout).
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class LLMError(RuntimeError):
    """The model backend could not produce a usable response."""


@dataclass
class StructuredReply:
    data: dict[str, Any]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


class StructuredLLM(Protocol):
    model: str

    def chat_json(self, system: str, user: str, schema: dict[str, Any]) -> StructuredReply: ...


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class _OllamaBase:
    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._client.post(path, json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()[:500]  # Ollama puts the reason in the body
            raise LLMError(
                f"Ollama returned HTTP {exc.response.status_code}: {detail or '<empty body>'}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        return resp.json()

    def unload(self) -> None:
        """Free the model's VRAM; errors are ignored because nothing depends on it."""
        with contextlib.suppress(httpx.HTTPError):
            self._client.post("/api/generate", json={"model": self.model, "keep_alive": 0})

    def close(self) -> None:
        self._client.close()


class OllamaStructured(_OllamaBase):
    """Chat whose reply is constrained to a JSON schema (Ollama ``format``)."""

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        num_predict: int = 3072,
        think: bool = False,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(model, base_url, timeout=timeout, transport=transport)
        self.options = {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict}
        self.think = think

    def chat_json(self, system: str, user: str, schema: dict[str, Any]) -> StructuredReply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": schema,
            "stream": False,
            "options": self.options,
        }
        # Only qwen3-style models accept "think"; others reject the field.
        if self.model.startswith("qwen3"):
            payload["think"] = self.think
        data = self._post("/api/chat", payload)
        content = (data.get("message") or {}).get("content") or ""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            # Usually a reply cut off by num_predict.
            raise LLMError(f"reply is not valid JSON ({exc}); starts {content[:120]!r}") from exc
        if not isinstance(parsed, dict):
            raise LLMError(f"reply is JSON but not an object: {content[:120]!r}")
        return StructuredReply(
            data=parsed,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            seconds=round(int(data.get("total_duration") or 0) / 1e9, 2),
        )


class OllamaEmbedder(_OllamaBase):
    """``/api/embed``; vectors come back L2-normalized from Ollama."""

    def __init__(self, model: str = "bge-m3", base_url: str = DEFAULT_BASE_URL, **kw: Any):
        super().__init__(model, base_url, **kw)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            data = self._post("/api/embed", {"model": self.model, "input": texts[i : i + 64]})
            out += data["embeddings"]
        return out
