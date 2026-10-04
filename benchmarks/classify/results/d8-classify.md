# D8: domain classifier

Which academic domain is a passage (or a question) from? Five methods, one corpus, one set
of lecture-level splits. Scripts: `benchmarks/classify/run.py` (the run),
`report.py` (these tables) and `posthoc_seeds.py` (encoder seed spread).

**Answer:** TF-IDF + logistic regression wins on every set, at 0.12 ms per chunk on the
CPU. The zero-shot LLM is about 2,000x slower and 13 points worse on full passages. No
method classifies short text well: on 12-word snippets the best is 58%, and 22% of D4's
algorithms questions get routed to the wrong domain even by the winner. Whole lectures,
on the other hand, are easy: majority vote over a lecture's chunks gets all 20 held-out
lectures right with TF-IDF.

## Corpus

`benchmarks/classify/corpus.yaml` lists every lecture by YouTube id.

| Domain | Course | Lectures | Train / dev / test chunks |
|---|---|---|---|
| algorithms | MIT 6.006 Fall 2011 | 1-10 | 261 / 39 / 130 |
| linear_algebra | MIT 18.06 Spring 2005 | 1-10 | 199 / 36 / 102 |
| artificial_intelligence | MIT 6.034 Fall 2010 | 1-10 | 235 / 41 / 122 |
| discrete_math | MIT 6.042J Fall 2010 | 1-3, 5-10 | 267 / 61 / 181 |
| distributed_systems | MIT 6.824 Spring 2020 | 1-10 | 444 / 76 / 221 |
| algorithms (out-of-course test) | MIT 6.046J Spring 2015 | 1-5 | - / - / 317 |

- **Labels are the course,** not an annotation. That is cheap and unambiguous, but a
  passage can be about another field than its course (see "Where they fail").
- **Splits are by lecture:** lectures 1, 2, 4, 5, 7, 8 train, 10 is dev, 3, 6, 9 test, in
  every course. Neighbouring chunks overlap, so a chunk-level split would leak.
- **6.042J lecture 4 is missing.** YouTube's bot check refused the download. It's a train
  lecture, so no test number depends on it.
- **Shortcuts removed:** every OpenCourseWare upload opens with the same Creative Commons
  notice and 6.824 (not an OCW upload) never does, so the notice is stripped. Spoken course
  numbers ("in 006 we...") are masked. Lecturer names are not; the 6.046J set, a different
  course with different lecturers, checks whether "algorithms" was learned or only "6.006".
- **Chunks** are the D1 ones: about 190 words, ~1 min of speech.

## Methods

| Method | What it is | Tuned on dev |
|---|---|---|
| `tfidf-lr` | word + bigram TF-IDF, logistic regression (scikit-learn) | C = 10 of {1, 10, 100} |
| `finetuned-encoder` | bge-small-en-v1.5 (33M) fine-tuned end to end, linear head | 3 of 1-4 epochs |
| `embed-lr` | frozen bge-m3 embeddings (Ollama), logistic regression | C = 100 |
| `zeroshot-llm` | qwen3:8b picks one of five domain descriptions, JSON enum | none |
| `zeroshot-embed` | nearest domain description by bge-m3 cosine | none |

The five domain descriptions describe each field, not the course syllabus, and were
written before any score existed (`DOMAINS` in `src/domaingraph/classify.py`). The code and
this protocol were committed (`292ad38`) before the first scored run.

## Results

Scored sets: **test** (15 lectures, 756 chunks), **12-word snippets** (the first 12
words of each test chunk), **6.046J** (317 chunks, all algorithms) and **D4 queries** (the
102 search questions, all algorithms). Intervals resample whole lectures.

| Method | Test acc (95% CI) | Macro-F1 | Lecture vote | 12-word snippets | 6.046J | D4 queries | ms/chunk | Fit |
|---|---|---|---|---|---|---|---|---|
| **tfidf-lr** | **0.918** [0.85, 0.98] | 0.917 | 15/15 | **0.581** | **0.899** | **0.784** | 0.12 (CPU) | 0.7 s |
| finetuned-encoder | 0.845 [0.72, 0.94] | 0.835 | 14/15 | 0.496 | 0.666 | 0.735 | 1.75 (GPU) | 27 s |
| embed-lr | 0.840 [0.72, 0.94] | 0.829 | 15/15 | 0.429 | 0.565 | 0.676 | 15 (GPU, Ollama) | 0.2 s + embedding |
| zeroshot-llm | 0.792 [0.63, 0.91] | 0.789 | 13/15 | 0.573 | 0.820 | 0.657 | 245 (GPU, Ollama) | none |
| zeroshot-embed | 0.660 [0.53, 0.79] | 0.651 | 12/15 | 0.388 | 0.628 | 0.627 | 15 (GPU, Ollama) | none |

Lecture vote on 6.046J: 5/5 for tfidf-lr, embed-lr and zeroshot-llm, 4/5 for the encoder
and zeroshot-embed. The LLM returned a valid domain for all 1,931 texts.

Accuracy difference vs `tfidf-lr`, paired (same resample for both):

| Method | Test | 12-word snippets | 6.046J | D4 queries |
|---|---|---|---|---|
| finetuned-encoder | -0.073 [-0.18, +0.00] | -0.085 [-0.17, +0.02] | -0.233 [-0.38, -0.13] | -0.049 [-0.14, +0.04] |
| embed-lr | -0.078 [-0.16, -0.01] | -0.152 [-0.27, -0.01] | -0.334 [-0.42, -0.28] | -0.108 [-0.21, -0.01] |
| zeroshot-llm | -0.126 [-0.26, -0.03] | -0.008 [-0.07, +0.05] | -0.079 [-0.15, -0.01] | -0.127 [-0.24, -0.02] |
| zeroshot-embed | -0.258 [-0.35, -0.16] | -0.193 [-0.30, -0.06] | -0.271 [-0.41, -0.16] | -0.157 [-0.26, -0.05] |

**Encoder seed spread** (3 seeds, epochs fixed at 3): test 0.835-0.845, 6.046J 0.666-0.729,
snippets 0.492-0.503. The single-seed row above is not a lucky or unlucky draw.

## What it shows

- **The vocabulary is the signal, and TF-IDF reads it directly.** Its strongest features per
  class are mostly the field's terms: `matrix, row, column, elimination, pivot` for linear
  algebra, `heuristic, search, constraint` for AI, `proof, prove, prime, induction` for
  discrete math, `server, leader, client, lock` for distributed systems. Some are the
  lecturer's habits, though: `you know` and `sort of` for 6.824, `ok` and `let me` for
  18.06, `python` for 6.006. Those can't help on 6.046J, which still scores 0.90.
- **The trained encoder learned the course, not the field.** It's 7 points behind TF-IDF on
  test (borderline: [-0.18, +0.00]), but on 6.046J it falls to 0.67, 23 points behind, the
  clearest gap in the study. Frozen bge-m3 + LR does the same, worse (0.57).
- **The zero-shot LLM is the most robust learned-nothing method** but not a good one: 0.79
  on test, 0.82 on 6.046J, at 245 ms per chunk. It ties TF-IDF only on 12-word snippets,
  where everything is bad.
- **Short text breaks every method.** On 12-word snippets the best is 58% (five classes, so
  chance is 20%). Many snippets carry no domain signal at all: "The parent of C is y, it's
  still y. So in a" (6.006, called distributed_systems by TF-IDF). The D4 questions are real short queries: the best method sends
  22 of 102 algorithms questions elsewhere, mostly to distributed_systems.
- **Whole sources are easy.** Majority vote over a lecture's chunks gets 15/15 test and 5/5
  6.046J lectures right with TF-IDF.

## Where they fail

- **discrete_math is the hard class** (TF-IDF 0.62-0.78 per lecture, every other domain
  0.90-1.00). 6.042J covers proofs, graphs and number theory, which overlap algorithms and AI.
- **Part of that is the labels.** 6.042J lecture 9 is about communication networks and their
  diameter. The LLM calls 53 of its 60 chunks distributed_systems, which
  is a fair reading of the text even though the course label says discrete_math.
- **The LLM's real mistakes:** 18.06 lecture 9 (independence, basis, dimension) goes to
  discrete_math in 19 of 35 chunks, and the three 6.034 test lectures lose 20-32% of their
  chunks, mostly to discrete_math and algorithms (the games lecture: 7 of 42 to algorithms).

## Should DomainGraph use it?

- **For tagging a source's domain at ingest, yes, and TF-IDF is enough:** it was right on
  all 20 held-out lectures by vote, trains in under a second and needs no GPU. Today the
  domain is a `graph load --domain` flag, which works while every source is one course.
- **For routing a query to a domain, no.** 22% of real questions misrouted by the best
  method would hide the right answer more often than searching all domains would.
- **Not wired into the product yet.** The graph holds one domain, so `--domain auto` would
  have nothing to decide until a second course is loaded.

## Limits

- Five domains, one course each (plus 6.046J for algorithms only). The out-of-course test
  is the only evidence about new courses, and it covers one domain.
- Labels are course membership; nobody checked passages individually.
- Speeds are on this machine (RTX 5060 Ti); Ollama timings include HTTP round trips, one
  chunk per call for the LLM.
- One run per method except the encoder seed check.
