import json
from collections import defaultdict
from pathlib import Path

import yaml

from domaingraph.evaluate_search import paired_bootstrap

base = json.loads(Path("benchmarks/finetune/results/base.json").read_text(encoding="utf-8"))
tuned = json.loads(Path("benchmarks/finetune/results/tuned.json").read_text(encoding="utf-8"))
qtype = {}
for f in Path("benchmarks/search/queries").glob("*.yaml"):
    for q in yaml.safe_load(f.read_text(encoding="utf-8"))["queries"]:
        qtype[q["id"]] = q["type"]
for corpus in ("d4_l5-7", "d4_all10"):
    b, t = base[corpus], tuned[corpus]
    assert b["ids"] == t["ids"]
    print(f"\n{corpus}")
    for m in ("R@1", "R@5", "R@10", "Hit@5", "MRR@10"):
        d, lo, hi = paired_bootstrap(b["per_query"][m], t["per_query"][m])
        print(f"  {m:7} base {b['summary'][m]:.3f} tuned {t['summary'][m]:.3f}  "
              f"diff {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")  # fmt: skip
    by = defaultdict(lambda: [[], []])
    for i, qid in enumerate(b["ids"]):
        by[qtype[qid]][0].append(b["per_query"]["MRR@10"][i])
        by[qtype[qid]][1].append(t["per_query"]["MRR@10"][i])
    print("  MRR by type:", {k: f"{sum(v[0])/len(v[0]):.3f}->{sum(v[1])/len(v[1]):.3f}"
                             for k, v in sorted(by.items())})  # fmt: skip
    pairs_mrr = list(zip(b["per_query"]["MRR@10"], t["per_query"]["MRR@10"], strict=True))
    better = sum(y > x for x, y in pairs_mrr)
    worse = sum(y < x for x, y in pairs_mrr)
    print(f"  queries better {better}, worse {worse}")
nb, nt = base["nanobeir"], tuned["nanobeir"]
print("\nNanoBEIR nDCG@10:")
for k in nb:
    print(f"  {k:15} {nb[k]:.3f} -> {nt[k]:.3f}  ({nt[k] - nb[k]:+.3f})")
