# D7: fine-tuning the embedding model on pairs from the graph

**Result: no measurable gain on the D4 test questions.** Fine-tuning bge-m3 on 4,203 pairs
generated from the graph improved the in-domain dev set (Lecture 10: MRR@10 0.404 -> 0.478).
It did not improve the D4 query set: MRR@10 moved -0.02, with a 95% interval of
[-0.07, +0.03]. General search held: the NanoBEIR mean nDCG@10 was 0.604 before and 0.605
after. The training pairs teach the model to match a fact *statement* to its passage, and the
dev set measures exactly that. The D4 queries are questions, and on paraphrased questions the
tuned model got worse.

Reproduce (needs the 10-lecture graph from D6, a CUDA GPU and `uv sync --extra train`):

```powershell
uv run python benchmarks/finetune/make_pairs.py      # -> data/finetune/train_pairs.jsonl
uv run python benchmarks/finetune/train.py           # -> models/bge-m3-6006/final (~3 min)
uv run python benchmarks/finetune/eval_model.py BAAI/bge-m3 benchmarks/finetune/results/base.json
uv run python benchmarks/finetune/eval_model.py models/bge-m3-6006/final benchmarks/finetune/results/tuned.json
uv run python benchmarks/finetune/compare.py
```

## Setup

- **Model:** `BAAI/bge-m3` (568M parameters, XLM-RoBERTa large, CLS pooling) in
  sentence-transformers 6.1 on PyTorch 2.11 with CUDA 12.8, RTX 5060 Ti 16 GB. D3/D4 used
  Ollama's build of the same model, which can't be trained. So both the base and the tuned
  model are scored here through sentence-transformers, the same way. On Lectures 5-7 the HF
  base scores MRR 0.582, against 0.572 for Ollama's in D4. The two builds agree closely.
- **Splits by lecture.** Test is **Lectures 5-7**: the D4 queries were labeled on them, so
  none of their passages, facts or relations go into training. Dev is **Lecture 10**, the
  only data used for any choice during training. Train is **Lectures 1-4, 8 and 9**.
- **Pairs** (`src/domaingraph/finetune.py`), all from the training lectures:

  | Kind | Anchor -> positive | Pairs |
  |---|---|---|
  | fact | an extracted fact -> the passage it was stated in | 1,978 |
  | concept | a concept's name -> up to 3 passages that state facts about it | 903 |
  | relation | concept name -> related concept name (relation said only in training lectures) | 1,322 |

  Concept anchors are names only. A concept merged across lectures may carry a definition
  written from a test-lecture passage, and that would leak. I checked that 0 pairs come from
  a test or dev source and that 0 positives are test or dev passages.
- **Training:**
  - `CachedMultipleNegativesRankingLoss` (in-batch negatives with GradCache), batch 64 in
    mini-batches of 8, with no duplicate texts in a batch.
  - 1 epoch (66 steps), lr 1e-5, 10% warmup, bf16, max 512 tokens. It took 3 minutes.
  - The logged loss fell from 2.8 at step 10 to between 2.2 and 2.4.
  - Dev MRR at step 50 was 0.4795 and at the end 0.4777, a negligible difference, so the
    final weights are used.
- **Evaluation:** both models are scored the same way.
  - **D4:** the 102 reviewed queries, as dense search over Lectures 5-7 (131 passages, D4's
    setting) and over all 10 lectures (430 passages, the other 7 as distractors). Intervals
    are paired bootstraps over queries.
  - **General:** NanoBEIR (sentence-transformers' `NanoBEIREvaluator`), 13 small public
    BEIR retrieval sets of 50 queries each, scored by nDCG@10.

## Results

### D4 queries (in-domain, held out)

| Corpus | Metric | base | tuned | diff (95% CI) |
|---|---|---|---|---|
| Lectures 5-7 | MRR@10 | 0.582 | 0.564 | -0.018 [-0.069, +0.032] |
| | R@1 | 0.314 | 0.291 | -0.023 [-0.083, +0.036] |
| | R@5 | 0.635 | 0.636 | +0.001 [-0.054, +0.056] |
| | R@10 | 0.738 | 0.765 | +0.027 [-0.018, +0.072] |
| | Hit@5 | 0.765 | 0.775 | +0.010 [-0.049, +0.069] |
| All 10 lectures | MRR@10 | 0.548 | 0.520 | -0.029 [-0.083, +0.026] |
| | R@1 | 0.290 | 0.257 | -0.033 [-0.093, +0.027] |
| | R@5 | 0.569 | 0.607 | +0.038 [-0.025, +0.100] |
| | R@10 | 0.706 | 0.717 | +0.011 [-0.044, +0.065] |
| | Hit@5 | 0.706 | 0.745 | +0.039 [-0.029, +0.108] |

MRR@10 by query type, base -> tuned:

| Corpus | definition | fact | procedure | relation | paraphrase | cross |
|---|---|---|---|---|---|---|
| Lectures 5-7 | 0.632 -> 0.672 | 0.602 -> 0.575 | 0.397 -> 0.382 | 0.817 -> 0.792 | **0.516 -> 0.441** | 0.503 -> 0.503 |
| All 10 | 0.604 -> 0.644 | 0.552 -> 0.548 | 0.380 -> 0.380 | 0.785 -> 0.743 | **0.453 -> 0.295** | 0.500 -> 0.503 |

### General search (NanoBEIR nDCG@10)

| Set | base | tuned | | Set | base | tuned |
|---|---|---|---|---|---|---|
| SciFact | 0.644 | 0.699 | | FEVER | 0.903 | 0.856 |
| DBPedia | 0.604 | 0.630 | | FiQA2018 | 0.574 | 0.535 |
| ClimateFEVER | 0.368 | 0.385 | | NQ | 0.687 | 0.664 |
| ArguAna | 0.514 | 0.524 | | QuoraRetrieval | 0.954 | 0.951 |
| Touche2020 | 0.471 | 0.478 | | SCIDOCS | 0.364 | 0.364 |
| NFCorpus | 0.336 | 0.340 | | HotpotQA | 0.819 | 0.822 |
| MSMARCO | 0.613 | 0.615 | | **mean** | **0.604** | **0.605** |

## Findings

- **The model learned what the pairs teach, and the pairs teach the wrong task.** The dev
  set (fact -> passage, from an unseen lecture) gained 0.07 MRR, so the training worked. But
  a fact is a declarative sentence that repeats the passage's wording ("Radix sort uses
  counting sort as a subroutine"). A student's question doesn't ("why does radix sort need a
  stable sort?"). On the D4 questions the gain is gone: 19 queries improve and 20 get worse.
- **Paraphrases get worse.** Those queries avoid the lecture's vocabulary on purpose, and
  the tuned model leans harder on that vocabulary. With the 7 distractor lectures in the
  corpus, paraphrase MRR drops from 0.45 to 0.30. Definition questions, which name the
  concept, gain 0.04.
- **General search didn't get worse.** The NanoBEIR mean is unchanged (+0.001). It gained on
  scientific text (SciFact +0.055) and lost on FEVER (-0.047) and FiQA (-0.039), so the
  check passes only on average.
- **The recall-side gains are small and within noise:** R@10 +0.027 on Lectures 5-7, and
  Hit@5 +0.039 on all 10.

## What would likely help

- **Question-shaped training data.** Generate questions for each training passage with the
  local LLM (the GPL / InPars approach) instead of using fact statements. That matches the
  D4 query distribution without touching the test lectures.
- **Hard negatives** from the same lecture. In-batch negatives here mostly come from other
  lectures, which are easy to tell apart.
- **More lectures.** 4,203 pairs from 6 lectures is small for a 568M-parameter model.

## Limits

- One training run (seed 0) and one set of hyperparameters. Nothing was tuned on the test
  lectures, but nothing was tuned much on dev either.
- The D4 labels are AI-drafted and reviewed only where vector search missed (see D4).
