"""Score one model: D4 queries on Lectures 5-7 and on all 10, plus NanoBEIR.
Usage: python eval_model.py <model name or path> <out.json> [--no-beir]"""

import json
import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from domaingraph.evaluate_embeddings import d4_results, nanobeir, per_query, summary

TEST = {"1c28a3b4720dda55", "9ef0cbfe913412e8", "9ec4d790bec6602e"}
name, out = sys.argv[1], Path(sys.argv[2])
model = SentenceTransformer(name, device="cuda")
model.max_seq_length = 512


def encode(texts):
    return model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)


report = {"model": name}
t = time.time()
for label, corpus in (("d4_l5-7", TEST), ("d4_all10", None)):
    res = d4_results(encode, Path("data"), Path("benchmarks/search/queries"), corpus)
    report[label] = {"summary": summary(res), "per_query": per_query(res),
                     "ids": [r.query.id for r in res]}  # fmt: skip
    print(label, {k: round(v, 3) for k, v in report[label]["summary"].items()}, flush=True)
print(f"D4 in {time.time() - t:.0f}s", flush=True)
if "--no-beir" not in sys.argv:
    t = time.time()
    report["nanobeir"] = nanobeir(model)
    print("nanobeir", {k: round(v, 3) for k, v in report["nanobeir"].items()})
    print(f"NanoBEIR in {time.time() - t:.0f}s")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=1), encoding="utf-8")
