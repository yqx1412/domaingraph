"""D7b: train on generated questions. Usage: python train_q.py <variant> <out dir>

Variants (all fixed before any score; the winner is picked on the Lecture 10 dev set only):
  q      generated (question, passage) pairs, in-batch negatives
  qneg   the same questions as (question, passage, hard negative) triples
  q+d7   the questions plus D7's fact/concept/relation pairs
Shared settings: lr 2e-5, batch 32, 3 epochs, dev scored every 10 steps.
Dev: Lecture 10's generated questions, searched over Lecture 10 plus the training lectures
(no test-lecture text), and D7's Lecture 10 fact -> passage set for continuity.
"""

import json
import sys
from pathlib import Path

from sentence_transformers.evaluation import SequentialEvaluator

from domaingraph.finetune import BASE_MODEL, TrainConfig, dev_evaluator, load_pairs, train
from domaingraph.graph import GraphConfig, GraphStore
from domaingraph.pipeline import list_sources, load_source
from domaingraph.questions import (
    load_questions,
    mine_hard_negatives,
    question_dev_evaluator,
    question_pairs,
)

TEST = {"1c28a3b4720dda55", "9ef0cbfe913412e8", "9ec4d790bec6602e"}  # Lectures 5, 6, 7
DEV = {"f0e05624d768b1e8"}  # Lecture 10
variant, out = sys.argv[1], Path(sys.argv[2])
assert variant in ("q", "qneg", "q+d7"), variant

chunks = {}
for s in list_sources(Path("data")):
    if s.id not in TEST:
        chunks.update({c.id: c for c in load_source(Path("data"), s.id)[1]})
assert not any(c.source_id in TEST for c in chunks.values())
train_chunks = {k: c for k, c in chunks.items() if c.source_id not in DEV}

train_q = load_questions(Path("data/finetune/questions_train.jsonl"))
dev_q = load_questions(Path("data/finetune/questions_dev.jsonl"))
assert {q.source for q in train_q}.isdisjoint(TEST | DEV)
assert {q.source for q in dev_q} == DEV

if variant == "qneg":
    from sentence_transformers import SentenceTransformer

    base = SentenceTransformer(BASE_MODEL, device="cuda")
    base.max_seq_length = 512

    def encode(texts):
        return base.encode(texts, batch_size=32, normalize_embeddings=True)

    pairs = mine_hard_negatives(train_q, train_chunks, encode)
    del base
    import torch

    torch.cuda.empty_cache()
else:
    pairs = question_pairs(train_q, train_chunks)
    if variant == "q+d7":
        pairs += load_pairs(Path("data/finetune/train_pairs.jsonl"))
print(f"{variant}: {len(pairs)} pairs", flush=True)

with GraphStore(GraphConfig.from_env()) as store:
    fact_dev = dev_evaluator(store, DEV, name="lecture10")
dev = SequentialEvaluator([question_dev_evaluator(dev_q, chunks, name="l10q"), fact_dev])
cfg = TrainConfig(out=out, lr=2e-5, batch_size=32, epochs=3, eval_steps=10)
log = train(pairs, dev, cfg)
keys = ("l10q_cosine_mrr@10", "l10q_cosine_recall@10", "lecture10_cosine_mrr@10")
for when in ("dev_before", "dev_after"):
    print(when, {k: round(log[when][k], 4) for k in keys})
curve = [(h["step"], round(h["eval_l10q_cosine_mrr@10"], 4))
         for h in log["history"] if "eval_l10q_cosine_mrr@10" in h]  # fmt: skip
print("l10q curve:", json.dumps(curve))
