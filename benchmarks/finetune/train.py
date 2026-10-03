"""D7 training run: pairs from Lectures 1-4, 8, 9; checkpoint chosen on Lecture 10."""

import json
import sys
from pathlib import Path

from domaingraph.finetune import TrainConfig, dev_evaluator, load_pairs, train
from domaingraph.graph import GraphConfig, GraphStore

DEV = {"f0e05624d768b1e8"}  # Lecture 10
pairs = load_pairs(Path("data/finetune/train_pairs.jsonl"))
cfg = TrainConfig(out=Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/bge-m3-6006"))
with GraphStore(GraphConfig.from_env()) as store:
    dev = dev_evaluator(store, DEV, name="lecture10")
log = train(pairs, dev, cfg)
show = {k: round(v, 4) for k, v in log["dev_before"].items() if "mrr" in k or "ndcg" in k}
print("dev before:", show)
show = {k: round(v, 4) for k, v in log["dev_after"].items() if "mrr" in k or "ndcg" in k}
print("dev after: ", show)
print("steps:", log["history"][-1].get("step"))
evals = [h for h in log["history"] if "eval_lecture10_cosine_mrr@10" in h]
print(json.dumps([(h["step"], round(h["eval_lecture10_cosine_mrr@10"], 4)) for h in evals]))
