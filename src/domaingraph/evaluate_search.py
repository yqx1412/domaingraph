"""D4 evaluation: recall@k, hit@k and MRR of each search mode on a labeled query set.

Query files live in ``benchmarks/search/queries/``. A per-lecture file names one
``source_id`` and lists ``relevant`` chunk indices; ``cross.yaml`` maps lecture names to
source ids and lists indices per lecture. Every relevant passage is resolved to a chunk id
up front, so a typo in a label fails loudly instead of scoring as a miss.

* **recall@k**: share of a query's relevant passages in the top k, averaged over queries.
* **hit@k**: share of queries with at least one relevant passage in the top k.
* **MRR@10**: mean of 1 / rank of the first relevant passage (0 if none in the top 10).

Differences between two modes come with a paired bootstrap 95% interval over queries, since
~100 queries leave real uncertainty.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Query:
    id: str
    text: str
    type: str
    relevant: frozenset[str]  # chunk ids
    answer: str = ""


def load_queries(d: Path, chunk_id: Callable[[str, int], str | None]) -> list[Query]:
    """``chunk_id(source_id, index)`` maps a label to a chunk id, or None if it doesn't
    exist."""
    out: list[Query] = []
    seen: set[str] = set()
    for f in sorted(d.glob("*.yaml")):
        doc = yaml.safe_load(f.read_text(encoding="utf-8"))
        for q in doc["queries"]:
            if q["id"] in seen:
                raise ValueError(f"{f.name}: duplicate query id {q['id']}")
            seen.add(q["id"])
            if "sources" in doc:
                pairs = [
                    (doc["sources"][lec], i) for lec, idx in q["relevant"].items() for i in idx
                ]
            else:
                pairs = [(doc["source_id"], i) for i in q["relevant"]]
            ids = set()
            for sid, i in pairs:
                cid = chunk_id(sid, i)
                if cid is None:
                    raise ValueError(f"{f.name} {q['id']}: no chunk {i} in source {sid}")
                ids.add(cid)
            if not ids:
                raise ValueError(f"{f.name} {q['id']}: no relevant passages")
            out.append(Query(q["id"], q["query"], q["type"], frozenset(ids), q.get("answer", "")))
    return out


@dataclass
class QueryResult:
    query: Query
    ranked: list[str]

    def first_hit(self) -> int | None:
        for i, cid in enumerate(self.ranked, 1):
            if cid in self.query.relevant:
                return i
        return None

    def recall(self, k: int) -> float:
        return len(self.query.relevant & set(self.ranked[:k])) / len(self.query.relevant)

    def hit(self, k: int) -> float:
        r = self.first_hit()
        return 1.0 if r is not None and r <= k else 0.0

    def rr(self, k: int = 10) -> float:
        r = self.first_hit()
        return 1.0 / r if r is not None and r <= k else 0.0


METRICS: dict[str, Callable[[QueryResult], float]] = {
    "R@1": lambda r: r.recall(1),
    "R@5": lambda r: r.recall(5),
    "R@10": lambda r: r.recall(10),
    "Hit@5": lambda r: r.hit(5),
    "MRR@10": lambda r: r.rr(10),
}


def summarize(results: Sequence[QueryResult]) -> dict[str, float]:
    n = len(results)
    return {m: sum(f(r) for r in results) / n for m, f in METRICS.items()} if n else {}


def by_type(results: Sequence[QueryResult]) -> dict[str, list[QueryResult]]:
    out: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        out[r.query.type].append(r)
    return dict(out)


def split_of(query_id: str) -> str:
    """A fixed half/half split by hash of the id: ``dev`` for tuning, ``test`` for reporting
    anything tuned. Defined before any variant was tried."""
    return "dev" if hashlib.sha1(query_id.encode()).digest()[0] % 2 == 0 else "test"


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of ``b - a`` and its 95% percentile interval, resampling queries."""
    assert len(a) == len(b) and a
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    m = len(diffs)
    means = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return sum(diffs) / m, means[int(0.025 * n)], means[int(0.975 * n) - 1]
