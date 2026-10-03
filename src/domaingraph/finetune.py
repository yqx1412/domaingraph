"""D7: fine-tune the embedding model on pairs generated from the graph.

Pairs (anchor, positive), all from the TRAINING lectures only:

* **fact -> passage:** an extracted fact and the passage it was stated in. Facts read like
  short answers, so they are the closest thing to (question, passage) pairs the graph has.
* **concept -> passage:** a concept's name and up to ``per_concept`` passages that state
  facts about it (the places it is explained, not just named). Only the name: a concept
  merged across lectures may have its definition from a test lecture.
* **concept -> concept:** the two ends of a relation said in a training lecture.

Lectures 5-7 are excluded completely: the D4 query set was labeled on them, so any of their
passages, facts or relations in training would leak the test set. Lecture 10 is held out
too, as the dev set that picks the checkpoint (``dev_pairs``), so the test set is never used
for any choice.

Training uses in-batch negatives (``CachedMultipleNegativesRankingLoss``): for each anchor,
every other positive in the batch is a negative. GradCache keeps a large batch (more
negatives) within 16 GB of VRAM. Batches never hold two pairs with the same text, which
would turn a true positive into a negative.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from domaingraph.graph import GraphStore

BASE_MODEL = "BAAI/bge-m3"


@dataclass
class Pair:
    anchor: str
    positive: str
    kind: str  # fact | concept | relation
    source: str  # source id the pair comes from


def build_pairs(
    store: GraphStore,
    sources: set[str],
    *,
    per_concept: int = 3,
    max_relations: int = 4000,
    seed: int = 0,
) -> list[Pair]:
    """Every pair whose evidence lies entirely in ``sources``."""
    rng = random.Random(seed)
    src = sorted(sources)
    pairs: list[Pair] = []

    for r in store.run(
        """MATCH (f:Fact)-[:MENTIONED_IN]->(ch:Chunk)-[:PART_OF]->(s:Source)
        WHERE NOT f:Memory AND s.id IN $src
        RETURN f.statement AS fact, ch.text AS text, s.id AS sid
        ORDER BY s.id, ch.index, f.id""",
        src=src,
    ):
        pairs.append(Pair(r["fact"], r["text"], "fact", r["sid"]))

    rows = store.run(
        """MATCH (c:Concept)<-[:ABOUT]-(f:Fact)-[:MENTIONED_IN]->(ch:Chunk)-[:PART_OF]->(s:Source)
        WHERE NOT f:Memory AND s.id IN $src
        WITH c, ch, s, count(f) AS facts
        RETURN c.id AS id, c.name AS name, ch.text AS text, s.id AS sid, facts
        ORDER BY c.id, facts DESC, ch.id""",
        src=src,
    )
    taken: dict[str, int] = defaultdict(int)
    for r in rows:
        if taken[r["id"]] < per_concept:
            taken[r["id"]] += 1
            # The name only: a merged concept's definition may come from a test lecture.
            pairs.append(Pair(r["name"], r["text"], "concept", r["sid"]))

    # A relation is in training only if every chunk that states it is a training chunk.
    # (Filtered here rather than in Cypher, where the list comprehension hit a type error.)
    rels = []
    for r in store.run(
        """MATCH (x:Concept)-[e:RELATED_TO|PART_OF]->(y:Concept)
        RETURN x.name AS a, y.name AS b, e.chunks AS chunks ORDER BY x.id, y.id"""
    ):
        srcs = {str(c).split(":", 1)[0] for c in (r["chunks"] or [])}
        if srcs and srcs <= sources:
            rels.append({"a": r["a"], "b": r["b"], "sid": min(srcs)})
    rng.shuffle(rels)
    for r in rels[:max_relations]:
        pairs.append(Pair(r["a"], r["b"], "relation", r["sid"]))
    return pairs


def save_pairs(pairs: list[Pair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def load_pairs(path: Path) -> list[Pair]:
    return [Pair(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]


def dev_evaluator(store: GraphStore, dev_sources: set[str], name: str = "dev"):
    """Facts of the dev lectures as queries, their passages as the corpus."""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    rows = store.run(
        """MATCH (f:Fact)-[:MENTIONED_IN]->(ch:Chunk)-[:PART_OF]->(s:Source)
        WHERE NOT f:Memory AND s.id IN $src
        RETURN f.id AS fid, f.statement AS q, ch.id AS cid, ch.text AS text""",
        src=sorted(dev_sources),
    )
    queries = {r["fid"]: r["q"] for r in rows}
    corpus = {r["cid"]: r["text"] for r in rows}
    relevant: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        relevant[r["fid"]].add(r["cid"])
    return InformationRetrievalEvaluator(
        queries, corpus, relevant, name=name, mrr_at_k=[10], ndcg_at_k=[10],
        accuracy_at_k=[1, 5], precision_recall_at_k=[5], show_progress_bar=False,
    )  # fmt: skip


@dataclass
class TrainConfig:
    base: str = BASE_MODEL
    out: Path = Path("models/bge-m3-6006")
    epochs: int = 1
    lr: float = 1e-5
    batch_size: int = 64
    mini_batch_size: int = 8
    max_seq_length: int = 512
    warmup_ratio: float = 0.1
    eval_steps: int = 50
    seed: int = 0


def train(pairs: list[Pair], dev, cfg: TrainConfig) -> dict[str, Any]:
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    model = SentenceTransformer(cfg.base, device="cuda")
    model.max_seq_length = cfg.max_seq_length
    before = dev(model)

    data = Dataset.from_dict(
        {"anchor": [p.anchor for p in pairs], "positive": [p.positive for p in pairs]}
    ).shuffle(seed=cfg.seed)
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=cfg.mini_batch_size)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(cfg.out / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        # No intermediate checkpoints: each one copies 2.2 GB of weights to CPU memory,
        # and on this 32 GB Windows host (with Neo4j, Ollama and Docker resident) the first
        # one hit the commit limit. The dev curve is logged instead; the number of steps is
        # chosen from it (on Lecture 10, never on the test lectures).
        save_strategy="no",
        dataloader_pin_memory=False,
        logging_steps=10,
        seed=cfg.seed,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=data, loss=loss, evaluator=dev
    )
    trainer.train()
    after = dev(model)
    import torch

    torch.cuda.empty_cache()
    model.save_pretrained(str(cfg.out / "final"))
    log = {
        "config": {k: str(v) for k, v in asdict(cfg).items()},
        "pairs": len(pairs),
        "dev_before": before,
        "dev_after": after,
        "history": trainer.state.log_history,
    }
    (cfg.out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return log
