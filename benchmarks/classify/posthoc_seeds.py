"""Post-hoc: encoder seed spread (epochs fixed at the dev-chosen 3)."""

import json
from pathlib import Path

from domaingraph.classify import EncoderClassifier, accuracy, load_corpus, short_text

REPO = Path(__file__).resolve().parents[2]
ex = load_corpus(REPO / "benchmarks/classify/corpus.yaml", [REPO / "data", REPO / "data/d8"])
s = {k: [e for e in ex if e.split == k] for k in ("train", "test", "ooc")}
out = {}
for seed in (0, 1, 2):
    m = EncoderClassifier(epochs=3, seed=seed)
    m.fit([e.text for e in s["train"]], [e.domain for e in s["train"]])
    row = {}
    for k in ("test", "ooc"):
        row[k] = round(accuracy([e.domain for e in s[k]], m.predict([e.text for e in s[k]])), 4)
    row["test-short"] = round(
        accuracy([e.domain for e in s["test"]], m.predict([short_text(e.text) for e in s["test"]])),
        4,
    )
    out[seed] = row
    print(seed, row, flush=True)
(REPO / "benchmarks/classify/results/posthoc_encoder_seeds.json").write_text(
    json.dumps(out, indent=2) + "\n", "utf-8", newline="\n"
)
