"""D8: domain classification of passages and short queries.

Five ways to say which domain a text belongs to, compared on the same lecture-level splits:

- ``tfidf-lr``: TF-IDF word/bigram features + logistic regression (scikit-learn).
- ``embed-lr``: frozen bge-m3 embeddings (Ollama) + logistic regression.
- ``zeroshot-embed``: nearest domain description by bge-m3 cosine; no training data.
- ``zeroshot-llm``: the local LLM picks a domain from the descriptions; no training data.
- ``finetuned-encoder``: a small encoder fine-tuned end to end (``train`` extra; see
  ``EncoderClassifier``).

Labels come from course membership (``benchmarks/classify/corpus.yaml``), and splits are by
lecture, so no two chunks of one lecture end up on both sides.
"""

from __future__ import annotations

import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from domaingraph.llm import Embedder, LLMError, StructuredLLM

# Written from what each FIELD covers, not from the courses' syllabi, and fixed before any
# score existed. The zero-shot methods see nothing else.
DOMAINS: dict[str, str] = {
    "algorithms": (
        "Algorithms and data structures: designing and analysing algorithms, running time "
        "and asymptotic complexity, sorting, searching, trees, heaps, hashing, graphs."
    ),
    "linear_algebra": (
        "Linear algebra: vectors, matrices, systems of linear equations, elimination, "
        "vector spaces and subspaces, rank, determinants, eigenvalues."
    ),
    "artificial_intelligence": (
        "Artificial intelligence: building systems that reason, plan, search for solutions, "
        "satisfy constraints, recognise things and learn from examples."
    ),
    "discrete_math": (
        "Discrete mathematics for computer science: proofs, induction, logic, number "
        "theory, graph theory, counting and combinatorics, probability."
    ),
    "distributed_systems": (
        "Distributed systems: programs running on many networked machines, RPC, "
        "replication, fault tolerance, consensus, consistency, distributed storage."
    ),
}

# Every OpenCourseWare upload opens with this notice, and 6.824 (not an OCW upload) never
# does. Left in, it would let a model separate 6.824 from the rest by boilerplate.
_OCW_NOTICE = re.compile(
    r"The following content is provided under a Creative Commons license\..*?"
    r"(?:ocw\.mit\.edu\.?|OCW at MIT\.?)",
    re.IGNORECASE | re.DOTALL,
)
# Lecturers say their course number ("in 006 we..."). It names the course, not the field,
# so it is masked. Lecturer names are not; the 6.046 out-of-course set covers that.
_COURSE_NUMBER = re.compile(
    r"\b(?:(?:6\.)?006|6006|18\.06|(?:6\.)?034|6\.042J?|(?:6\.)?824|6\.046J?)\b"
)


def clean_text(text: str) -> str:
    text = _COURSE_NUMBER.sub("this course", _OCW_NOTICE.sub(" ", text))
    text = re.sub(r"^\s*MIT OpenCourseWare\b", " ", text)  # a notice Whisper cut short
    return re.sub(r"\s+", " ", text).strip()


def short_text(text: str, words: int = 12) -> str:
    """The first ``words`` words: a stand-in for a short note or query."""
    return " ".join(text.split()[:words])


@dataclass(frozen=True)
class Example:
    text: str
    domain: str
    course: str
    lecture: int
    split: str  # train | dev | test | ooc
    chunk: int = 0

    @property
    def group(self) -> str:
        """Clustering unit for the bootstrap: chunks of one lecture are not independent."""
        return f"{self.course}#{self.lecture}"


def _video_id(meta: dict[str, Any]) -> str | None:
    url = meta.get("url") or ""
    m = re.search(r"[?&]v=([\w-]{11})", url)
    if m:
        return m.group(1)
    stem = Path(meta.get("path") or "").stem  # <course tag>-<video id>
    return stem[-11:] if len(stem) >= 11 else None


def load_corpus(manifest_path: Path, roots: Sequence[Path]) -> list[Example]:
    """Join ingested sources under ``roots`` to the manifest by YouTube video id."""
    import json

    manifest = yaml.safe_load(manifest_path.read_text("utf-8"))
    by_split = {n: s for s, ns in manifest["splits"].items() for n in ns}
    index: dict[str, tuple[str, str, int, str]] = {}
    for c in manifest["courses"]:
        for n, vid in enumerate(c["lectures"], 1):
            if vid is None:  # a lecture deliberately left out; numbering stays fixed
                continue
            split = c.get("split") or by_split.get(n)
            if split is None:
                raise ValueError(f"{c['course']} lecture {n} has no split")
            index[vid] = (c["course"], c["domain"], n, split)

    out: list[Example] = []
    seen: set[str] = set()
    for root in roots:
        for meta_path in sorted((root / "sources").glob("*/source.json")):
            meta = json.loads(meta_path.read_text("utf-8"))
            vid = _video_id(meta)
            if vid not in index or vid in seen:
                continue
            seen.add(vid)
            course, domain, n, split = index[vid]
            for line in (meta_path.parent / "chunks.jsonl").read_text("utf-8").splitlines():
                chunk = json.loads(line)
                text = clean_text(chunk["text"])
                if len(text.split()) >= 5:
                    out.append(Example(text, domain, course, n, split, int(chunk["index"])))
    missing = sorted(set(index) - seen)
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest lectures not ingested: {missing[:5]}")
    return out


# ---------------------------------------------------------------------------- classifiers


class Classifier(Protocol):
    name: str

    def fit(self, texts: list[str], labels: list[str]) -> None: ...

    def predict(self, texts: list[str]) -> list[str]: ...


class TfidfLogReg:
    name = "tfidf-lr"

    def __init__(self, c: float = 10.0) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.c = c
        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(C=c, max_iter=2000, class_weight="balanced"),
        )

    def fit(self, texts: list[str], labels: list[str]) -> None:
        self.model.fit(texts, labels)

    def predict(self, texts: list[str]) -> list[str]:
        return [str(p) for p in self.model.predict(texts)]


class EmbedLogReg:
    """Logistic regression on frozen sentence embeddings."""

    name = "embed-lr"

    def __init__(self, embedder: Embedder, c: float = 10.0) -> None:
        from sklearn.linear_model import LogisticRegression

        self.embedder = embedder
        self.c = c
        self.model = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")

    def fit(self, texts: list[str], labels: list[str]) -> None:
        self.model.fit(self.embedder.embed(texts), labels)

    def predict(self, texts: list[str]) -> list[str]:
        return [str(p) for p in self.model.predict(self.embedder.embed(texts))]


class ZeroShotEmbed:
    """Nearest domain description by cosine. Needs no labelled data."""

    name = "zeroshot-embed"

    def __init__(self, embedder: Embedder, domains: dict[str, str] = DOMAINS) -> None:
        self.embedder = embedder
        self.labels = list(domains)
        self._centroids = embedder.embed(list(domains.values()))

    def fit(self, texts: list[str], labels: list[str]) -> None:  # nothing to learn
        return None

    def predict(self, texts: list[str]) -> list[str]:
        out = []
        for v in self.embedder.embed(texts):
            scores = [sum(a * b for a, b in zip(v, c, strict=True)) for c in self._centroids]
            out.append(self.labels[max(range(len(scores)), key=scores.__getitem__)])
        return out


ZEROSHOT_SYSTEM = (
    "You classify a passage from a university lecture transcript, or a short question, "
    "into exactly one academic domain. Transcripts are spoken and can be informal or "
    "start mid-sentence. Pick the domain the text is most about.\n\nDomains:\n"
)


class ZeroShotLLM:
    name = "zeroshot-llm"

    def __init__(self, llm: StructuredLLM, domains: dict[str, str] = DOMAINS) -> None:
        self.llm = llm
        self.labels = list(domains)
        self.system = ZEROSHOT_SYSTEM + "\n".join(f"- {k}: {v}" for k, v in domains.items())
        self.schema = {
            "type": "object",
            "properties": {"domain": {"type": "string", "enum": self.labels}},
            "required": ["domain"],
        }
        self.failures = 0

    def fit(self, texts: list[str], labels: list[str]) -> None:
        return None

    def predict(self, texts: list[str]) -> list[str]:
        out = []
        for t in texts:
            try:
                d = str(self.llm.chat_json(self.system, f"Text:\n{t}", self.schema).data["domain"])
            except (LLMError, KeyError):
                d = ""
            if d not in self.labels:
                self.failures += 1
                d = "<invalid>"
            out.append(d)
        return out


class EncoderClassifier:
    """A small encoder fine-tuned end to end with a linear head (``train`` extra)."""

    name = "finetuned-encoder"

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        epochs: int = 3,
        lr: float = 5e-5,
        batch_size: int = 16,
        max_length: int = 256,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.max_length, self.seed = max_length, seed
        self.device = device
        self.labels: list[str] = []
        self.history: list[dict[str, float]] = []

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        on_epoch: Callable[[int], dict[str, float]] | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(self.seed)
        rng = random.Random(self.seed)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = sorted(set(labels))
        ids = {lab: i for i, lab in enumerate(self.labels)}
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.labels)
        ).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.epochs * ((len(texts) + self.batch_size - 1) // self.batch_size)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, 1 - s / steps)
        )
        order = list(range(len(texts)))
        for epoch in range(self.epochs):
            self.model.train()
            rng.shuffle(order)
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                batch = order[i : i + self.batch_size]
                enc = self._encode([texts[j] for j in batch])
                y = torch.tensor([ids[labels[j]] for j in batch], device=self.device)
                loss = self.model(**enc, labels=y).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                total += loss.item() * len(batch)
            row = {"epoch": epoch + 1, "loss": round(total / len(texts), 4)}
            if on_epoch:
                row |= on_epoch(epoch + 1)
            self.history.append(row)

    def _encode(self, texts: list[str]) -> Any:
        enc = self.tok(
            texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def predict(self, texts: list[str]) -> list[str]:
        import torch

        self.model.eval()
        out: list[str] = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                logits = self.model(**self._encode(texts[i : i + 64])).logits
                out += [self.labels[j] for j in logits.argmax(-1).tolist()]
        return out


# -------------------------------------------------------------------------------- metrics


@dataclass
class Scores:
    n: int
    accuracy: float
    macro_f1: float
    ci: tuple[float, float]
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold) if gold else 0.0


def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    labels = sorted(set(gold))
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred, strict=True))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred, strict=True))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred, strict=True))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def cluster_bootstrap(
    gold: Sequence[str],
    pred: Sequence[str],
    groups: Sequence[str],
    *,
    n: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% interval for accuracy, resampling whole lectures rather than chunks."""
    by: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    keys = sorted(by)
    rng = random.Random(seed)
    accs = []
    for _ in range(n):
        idx = [i for k in rng.choices(keys, k=len(keys)) for i in by[k]]
        accs.append(sum(gold[i] == pred[i] for i in idx) / len(idx))
    accs.sort()
    return accs[int(0.025 * n)], accs[int(0.975 * n) - 1]


def score(gold: Sequence[str], pred: Sequence[str], groups: Sequence[str]) -> Scores:
    conf: dict[str, dict[str, int]] = {}
    for g, p in zip(gold, pred, strict=True):
        conf.setdefault(g, Counter())[p] += 1  # type: ignore[assignment]
    return Scores(
        n=len(gold),
        accuracy=round(accuracy(gold, pred), 4),
        macro_f1=round(macro_f1(gold, pred), 4),
        ci=tuple(round(x, 4) for x in cluster_bootstrap(gold, pred, groups)),  # type: ignore[arg-type]
        confusion={g: dict(c) for g, c in conf.items()},
    )


def lecture_vote(examples: Sequence[Example], pred: Sequence[str]) -> float:
    """Accuracy when each lecture gets the majority label of its chunks."""
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    gold: dict[str, str] = {}
    for ex, p in zip(examples, pred, strict=True):
        votes[ex.group][p] += 1
        gold[ex.group] = ex.domain
    return sum(votes[g].most_common(1)[0][0] == gold[g] for g in votes) / len(votes)


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0
