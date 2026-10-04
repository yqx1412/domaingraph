"""D8 benchmark: train/tune on train+dev, then score every method once on the held-out sets.

    uv run --extra train --extra classify python benchmarks/classify/run.py [--only NAME,...]

Writes ``results/d8-results.json`` (scores, timings, chosen settings) and
``results/d8-predictions.json`` (per-item predictions on the scored sets).

Protocol, fixed before any test score existed:
- Hyperparameters are picked on ``dev`` (lecture 10 of each course) only: C for the two
  logistic regressions, the epoch count for the encoder. Zero-shot methods have none.
- Scored sets: ``test`` (lectures 3, 6, 9 of every course), ``test-short`` (the first 12
  words of each test chunk), ``ooc`` (5 lectures of 6.046J, a different algorithms course)
  and ``queries`` (D4's 102 search questions, all algorithms).
- Accuracy intervals resample whole lectures, since chunks of one lecture are correlated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from domaingraph.classify import (
    EmbedLogReg,
    EncoderClassifier,
    TfidfLogReg,
    ZeroShotEmbed,
    ZeroShotLLM,
    accuracy,
    clean_text,
    lecture_vote,
    load_corpus,
    score,
    short_text,
)
from domaingraph.llm import OllamaEmbedder, OllamaStructured

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
C_GRID = (1.0, 10.0, 100.0)
MAX_EPOCHS = 4


class CachedEmbedder:
    """bge-m3 through Ollama, cached so the C sweep does not re-embed."""

    def __init__(self) -> None:
        self.inner = OllamaEmbedder("bge-m3")
        self.model = self.inner.model
        self.cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for t, v in zip(todo, self.inner.embed(todo), strict=True):
            self.cache[t] = v
        return [self.cache[t] for t in texts]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="Comma list of methods to run")
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}

    ex = load_corpus(HERE / "corpus.yaml", [REPO / "data", REPO / "data" / "d8"])
    split = {s: [e for e in ex if e.split == s] for s in ("train", "dev", "test", "ooc")}
    queries = []
    for f in sorted((REPO / "benchmarks/search/queries").glob("*.yaml")):
        queries += [q["query"] for q in yaml.safe_load(f.read_text("utf-8"))["queries"]]
    sets = {
        "test": (
            [e.text for e in split["test"]],
            [e.domain for e in split["test"]],
            [e.group for e in split["test"]],
        ),
        "test-short": (
            [short_text(e.text) for e in split["test"]],
            [e.domain for e in split["test"]],
            [e.group for e in split["test"]],
        ),
        "ooc": (
            [e.text for e in split["ooc"]],
            [e.domain for e in split["ooc"]],
            [e.group for e in split["ooc"]],
        ),
        "queries": (
            [clean_text(q) for q in queries],
            ["algorithms"] * len(queries),
            [str(i) for i in range(len(queries))],
        ),
    }
    tr_x, tr_y = [e.text for e in split["train"]], [e.domain for e in split["train"]]
    dv_x, dv_y = [e.text for e in split["dev"]], [e.domain for e in split["dev"]]
    log(
        f"train {len(tr_x)}, dev {len(dv_x)}, "
        + ", ".join(f"{k} {len(v[0])}" for k, v in sets.items())
    )

    out_path = HERE / "results" / "d8-results.json"
    pred_path = HERE / "results" / "d8-predictions.json"
    out_path.parent.mkdir(exist_ok=True)
    results = json.loads(out_path.read_text("utf-8")) if out_path.exists() else {}
    preds = json.loads(pred_path.read_text("utf-8")) if pred_path.exists() else {}
    embedder = CachedEmbedder()

    def want(name: str) -> bool:
        return not only or name in only

    def pick_c(make):
        dev = {}
        for c in C_GRID:
            m = make(c)
            m.fit(tr_x, tr_y)
            dev[c] = round(accuracy(dv_y, m.predict(dv_x)), 4)
        best = max(C_GRID, key=lambda c: (dev[c], -c))  # ties go to the stronger penalty
        return best, {str(k): v for k, v in dev.items()}

    methods = []
    if want("tfidf-lr"):
        c, dev = pick_c(lambda c: TfidfLogReg(c))
        methods.append((TfidfLogReg(c), {"C": c, "dev_by_C": dev}, "cpu"))
    if want("embed-lr"):
        c, dev = pick_c(lambda c: EmbedLogReg(embedder, c))
        methods.append((EmbedLogReg(embedder, c), {"C": c, "dev_by_C": dev}, "cuda (Ollama)"))
    if want("zeroshot-embed"):
        methods.append((ZeroShotEmbed(embedder), {}, "cuda (Ollama)"))
    if want("zeroshot-llm"):
        llm = OllamaStructured("qwen3:8b", num_predict=64, num_ctx=4096)
        methods.append((ZeroShotLLM(llm), {"model": "qwen3:8b"}, "cuda (Ollama)"))
    if want("finetuned-encoder"):
        # Epoch count chosen on dev: train MAX_EPOCHS, record dev each epoch, retrain to best.
        probe = EncoderClassifier(epochs=MAX_EPOCHS)
        probe.fit(
            tr_x, tr_y, on_epoch=lambda ep: {"dev": round(accuracy(dv_y, probe.predict(dv_x)), 4)}
        )
        best = max(probe.history, key=lambda r: (r["dev"], -r["epoch"]))["epoch"]
        log(f"encoder dev by epoch: {probe.history} -> {best}")
        del probe
        methods.append((EncoderClassifier(epochs=best), {"epochs": best}, "cuda"))

    for m, settings, device in methods:
        log(f"{m.name}: fitting")
        t0 = time.perf_counter()
        m.fit(tr_x, tr_y)
        fit_s = time.perf_counter() - t0
        row = {"settings": settings, "device": device, "fit_seconds": round(fit_s, 1), "sets": {}}
        if isinstance(m, EncoderClassifier):
            row["settings"]["history"] = m.history
        preds[m.name] = {}
        for name, (x, y, g) in sets.items():
            if isinstance(m, EmbedLogReg | ZeroShotEmbed):
                embedder.cache.clear()  # time the embedding too, not a cache hit
            t0 = time.perf_counter()
            p = m.predict(x)
            secs = time.perf_counter() - t0
            s = asdict(score(y, p, g))
            s["ms_per_item"] = round(1000 * secs / len(x), 2)
            if name in ("test", "ooc"):
                s["lecture_vote"] = round(lecture_vote(split[name.split("-")[0]], p), 4)
            row["sets"][name] = s
            preds[m.name][name] = p
            log(
                f"  {name}: acc {s['accuracy']} {s['ci']} f1 {s['macro_f1']} "
                f"{s['ms_per_item']} ms/item"
            )
        if isinstance(m, ZeroShotLLM):
            row["invalid_replies"] = m.failures
        results[m.name] = row
        out_path.write_text(json.dumps(results, indent=2) + "\n", "utf-8", newline="\n")
        pred_path.write_text(json.dumps(preds) + "\n", "utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
