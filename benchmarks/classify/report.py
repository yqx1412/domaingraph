"""Tables for the D8 write-up from ``results/d8-results.json`` and the predictions file.

    uv run python benchmarks/classify/report.py

Paired differences resample whole lectures (test, ooc) or single queries, with the same
resample applied to both methods.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

from domaingraph.classify import load_corpus

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ORDER = ["tfidf-lr", "finetuned-encoder", "embed-lr", "zeroshot-llm", "zeroshot-embed"]
BASE = "tfidf-lr"


def paired(gold, a, b, groups, n=2000, seed=0):
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    keys = sorted(by)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        idx = [i for k in rng.choices(keys, k=len(keys)) for i in by[k]]
        diffs.append(sum((a[i] == gold[i]) - (b[i] == gold[i]) for i in idx) / len(idx))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]


def main() -> None:
    res = json.loads((HERE / "results/d8-results.json").read_text("utf-8"))
    pred = json.loads((HERE / "results/d8-predictions.json").read_text("utf-8"))
    ex = load_corpus(HERE / "corpus.yaml", [REPO / "data", REPO / "data" / "d8"])
    test = [e for e in ex if e.split == "test"]
    ooc = [e for e in ex if e.split == "ooc"]
    nq = sum(
        len(yaml.safe_load(f.read_text("utf-8"))["queries"])
        for f in (REPO / "benchmarks/search/queries").glob("*.yaml")
    )
    gold = {
        "test": ([e.domain for e in test], [e.group for e in test]),
        "test-short": ([e.domain for e in test], [e.group for e in test]),
        "ooc": ([e.domain for e in ooc], [e.group for e in ooc]),
        "queries": (["algorithms"] * nq, [str(i) for i in range(nq)]),
    }
    names = [m for m in ORDER if m in res]

    print(
        "| Method | Test acc (95% CI) | Macro-F1 | Lecture vote | 12-word snippets | "
        "6.046J (other course) | D4 queries | ms/chunk | Fit |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for m in names:
        s = res[m]["sets"]
        t = s["test"]
        print(
            f"| {m} | {t['accuracy']:.3f} [{t['ci'][0]:.2f}, {t['ci'][1]:.2f}] "
            f"| {t['macro_f1']:.3f} | {t['lecture_vote']:.2f} | {s['test-short']['accuracy']:.3f} "
            f"| {s['ooc']['accuracy']:.3f} | {s['queries']['accuracy']:.3f} "
            f"| {t['ms_per_item']:g} ({res[m]['device']}) | {res[m]['fit_seconds']:g} s |"
        )
    print(f"\nAccuracy difference vs {BASE} (95% CI, paired):\n")
    print("| Method | Test | 12-word snippets | 6.046J | D4 queries |")
    print("|---|---|---|---|---|")
    for m in names:
        if m == BASE:
            continue
        cells = []
        for k, (g, grp) in gold.items():
            a, b = pred[m][k], pred[BASE][k]
            d = (
                sum(x == y for x, y in zip(a, g, strict=True))
                - sum(x == y for x, y in zip(b, g, strict=True))
            ) / len(g)
            lo, hi = paired(g, a, b, grp)
            cells.append(f"{d:+.3f} [{lo:+.2f}, {hi:+.2f}]")
        print(f"| {m} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
