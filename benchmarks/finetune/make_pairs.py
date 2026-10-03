"""Build the D7 training pairs and report what they contain (no training)."""

import json
from collections import Counter
from pathlib import Path

from domaingraph.finetune import build_pairs, save_pairs
from domaingraph.graph import GraphConfig, GraphStore
from domaingraph.pipeline import list_sources

TEST = {"1c28a3b4720dda55", "9ef0cbfe913412e8", "9ec4d790bec6602e"}  # Lectures 5, 6, 7
DEV = {"f0e05624d768b1e8"}  # Lecture 10
srcs = {s.id: s.title for s in list_sources(Path("data"))}
train = set(srcs) - TEST - DEV
print("train:", sorted(srcs[s].split(", ", 1)[1].split(":")[0] for s in train))
print("dev:  ", [srcs[s].split(", ", 1)[1].split(":")[0] for s in DEV])

with GraphStore(GraphConfig.from_env()) as store:
    pairs = build_pairs(store, train)
    dev_pairs = build_pairs(store, DEV)
save_pairs(pairs, Path("data/finetune/train_pairs.jsonl"))
print(len(pairs), Counter(p.kind for p in pairs))
print("by source:", Counter(srcs[p.source].split(", ", 1)[1].split(":")[0] for p in pairs))
leak = [p for p in pairs if p.source in TEST | DEV]
print("pairs from test/dev sources:", len(leak))
# Text-level leak check: no test or dev passage text appears as a positive.
test_texts = set()
for sid in TEST | DEV:
    text = Path(f"data/sources/{sid}/chunks.jsonl").read_text(encoding="utf-8")
    for line in text.splitlines():
        test_texts.add(json.loads(line)["text"])
print("positives that are test/dev passages:", sum(p.positive in test_texts for p in pairs))
for kind in ("fact", "concept", "relation"):
    p = next(p for p in pairs if p.kind == kind)
    print(f"\n[{kind}] {p.anchor[:120]!r}\n   -> {p.positive[:160]!r}")
print("\ndev pairs (not used for training):", len(dev_pairs))
