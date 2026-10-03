"""D7 evaluation: a sentence-transformers model on the D4 query set and on general search.

* **D4** (in-domain): the 102 labeled queries against the passages of chosen lectures, scored
  exactly as D4 scores vector search (recall@k, hit@5, MRR@10), with each query's metrics
  kept for paired bootstrap comparisons. Two corpora: Lectures 5-7 only (D4's setting) and
  all 10 lectures (the 7 others as distractors).
* **General search:** NanoBEIR, 13 small public retrieval sets (50 queries each, from
  BEIR: SciFact, NFCorpus, FiQA, ...), nDCG@10. This is the "didn't get worse on general
  search" check: fine-tuning on lecture transcripts can make a model forget other domains.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from domaingraph.evaluate_search import METRICS, QueryResult, load_queries, summarize
from domaingraph.pipeline import list_sources, load_source


def d4_results(
    encode: Callable[[list[str]], np.ndarray],
    out: Path,
    queries_dir: Path,
    corpus_sources: set[str] | None = None,
    k: int = 10,
) -> list[QueryResult]:
    """Dense retrieval of each D4 query over the chunks of ``corpus_sources`` (all if None)."""
    chunks = []
    by_pos = {}
    for s in list_sources(out):
        for c in load_source(out, s.id)[1]:
            by_pos[(s.id, c.index)] = c.id
            if corpus_sources is None or s.id in corpus_sources:
                chunks.append(c)
    queries = load_queries(queries_dir, lambda sid, i: by_pos.get((sid, i)))
    ids = [c.id for c in chunks]
    doc = encode([c.text for c in chunks])
    qv = encode([q.text for q in queries])
    scores = qv @ doc.T
    results = []
    for qi, q in enumerate(queries):
        top = np.argsort(-scores[qi])[:k]
        results.append(QueryResult(q, [ids[i] for i in top]))
    return results


def per_query(results: Sequence[QueryResult]) -> dict[str, list[float]]:
    return {m: [f(r) for r in results] for m, f in METRICS.items()}


def summary(results: Sequence[QueryResult]) -> dict[str, float]:
    return summarize(results)


def nanobeir(model: Any, datasets: Sequence[str] | None = None) -> dict[str, float]:
    """nDCG@10 per NanoBEIR dataset, plus their mean."""
    from sentence_transformers.evaluation import NanoBEIREvaluator

    ev = NanoBEIREvaluator(
        dataset_names=list(datasets) if datasets else None, show_progress_bar=False
    )
    res = ev(model)
    # Keys look like "NanoSciFact_cosine_ndcg@10"; the mean is "NanoBEIR_mean_cosine_ndcg@10".
    out = {
        k.split("_", 1)[0].removeprefix("Nano"): float(v)
        for k, v in res.items()
        if k.endswith("_cosine_ndcg@10") and not k.startswith("NanoBEIR_mean")
    }
    out["mean"] = float(np.mean(list(out.values())))
    return out
