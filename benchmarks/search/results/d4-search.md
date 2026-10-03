# D4: hybrid search and evaluation

Does a graph beat vector-only search on lecture passages? **Not on this data.** Vector search
is the strongest single mode. Graph-only search is clearly worse, even when it runs on
hand-corrected gold concepts. Hybrid, which uses the graph to re-rank vector's candidates,
roughly ties vector on rank metrics and finds more relevant passages in the top 10. On the
held-out half its edge is within noise.

Reproduce (needs the D3 graph loaded and Ollama with bge-m3):

```powershell
uv run domaingraph eval-search --split test --report benchmarks/search/results/d4-test.md
uv run domaingraph eval-search --details --report benchmarks/search/results/d4-all.md
```

## Setup

- **Corpus:** 131 passages of about 200 words from MIT 6.006 Fall 2011 Lectures 5-7, as
  loaded in D3. The graph holds qwen3:8b's extraction: 215 concepts, 694 relations and 1,021
  facts.
- **Queries:** 102, in `benchmarks/search/queries/`. There are 30 per lecture (6 each of
  definition, fact, procedure, relation and paraphrase) plus 12 cross-lecture queries. Each
  lists every passage that answers it: 52 queries have 1 relevant passage, 37 have 2 and 13
  have 3-4.
- **The queries are AI-drafted.** Four agents wrote them from the transcripts alone, one per
  lecture plus one for cross-lecture queries. None of them saw the extracted graph, the
  search code or any search result. I spot-checked 10 labels against their passages and all
  10 hold up. As with D2's gold sets, nobody has reviewed the full set by hand.
- **Paraphrase queries avoid the lecture's terms on purpose**, such as "how do I keep track
  of how many things sit below each node" for subtree size. They are the hard case for
  keyword and name matching.

## The modes

| Mode | What it does |
|---|---|
| `bm25` | Keyword search over passage text. A reference point, not one of the roadmap's three modes. |
| `vector` | bge-m3 query embedding against Neo4j's `chunk_embedding` index. |
| `graph` | No passage embeddings. Finds the query's concepts by name/alias in the text and by the `concept_embedding` index (top 3, cosine >= 0.55). Ranks passages by the concepts they mention, weighted by idf and by facts stated there. Concepts one hop away by `RELATED_TO`/`PART_OF` count at 0.25x. |
| `hybrid` | Takes vector's top 50 and re-scores each as `cosine + 0.05 x graph score / best graph score among the 50`. The graph only reorders what vector found. |
| `hybrid-rrf` | Reciprocal rank fusion of vector's and graph's top 50, plus a small bonus for passages that mention the query's concepts. |
| `graph-gold`, `hybrid-gold` | Diagnostics: the same graph and hybrid scoring, run on the D2 gold sets (82 hand-corrected concepts with their labeled passages, 80 relations) instead of the extracted graph. They show what better extraction alone could do. |

**How the hybrid was chosen.**
- `hybrid-rrf` was the hybrid I wrote first. I fixed its weights and committed the code
  (`c03eebc`) before scoring anything.
- On the full query set it lost to vector (MRR 0.473 vs 0.566). So I split the queries in two
  by a hash of their id: `dev` (52) and `test` (50).
- On `dev` only, I tried a re-rank hybrid with bonuses of 0.02, 0.05 and 0.1, RRF weights of
  1:0.5, and a degree-normalized hop.
- The re-rank hybrid with a 0.05 bonus won on `dev`, so it became `hybrid` (`9333653`).
- **`test` is the honest number for it.** The other modes were never tuned, so their
  full-set numbers are fair too.

## Results

### Held-out `test` half (50 queries)

| Mode | R@1 | R@5 | R@10 | Hit@5 | MRR@10 |
|---|---|---|---|---|---|
| bm25 | 0.363 | 0.657 | 0.747 | 0.760 | 0.616 |
| **vector** | **0.383** | **0.717** | **0.827** | 0.840 | **0.653** |
| graph | 0.187 | 0.483 | 0.653 | 0.560 | 0.397 |
| hybrid | 0.367 | 0.697 | 0.787 | 0.840 | 0.620 |
| hybrid-rrf | 0.333 | 0.637 | 0.707 | 0.760 | 0.560 |
| graph-gold | 0.123 | 0.410 | 0.520 | 0.500 | 0.322 |
| hybrid-gold | 0.313 | 0.737 | 0.807 | **0.860** | 0.612 |

Paired difference vs vector, with 95% bootstrap intervals over queries:

| Mode | MRR@10 | R@5 |
|---|---|---|
| bm25 | -0.037 [-0.136, +0.067] | -0.060 [-0.180, +0.060] |
| graph | **-0.256 [-0.388, -0.125]** | **-0.233 [-0.363, -0.110]** |
| hybrid | -0.033 [-0.106, +0.035] | -0.020 [-0.080, +0.040] |
| hybrid-rrf | -0.093 [-0.223, +0.035] | -0.080 [-0.180, +0.010] |
| graph-gold | **-0.331 [-0.461, -0.200]** | **-0.307 [-0.443, -0.170]** |
| hybrid-gold | -0.041 [-0.104, +0.016] | +0.020 [+0.000, +0.050] |

### All 102 queries

`hybrid` was tuned on `dev`, which is half of this set.

| Mode | R@1 | R@5 | R@10 | Hit@5 | MRR@10 |
|---|---|---|---|---|---|
| bm25 | 0.287 | 0.607 | 0.718 | 0.706 | 0.549 |
| vector | 0.308 | 0.639 | 0.732 | 0.755 | 0.566 |
| graph | 0.150 | 0.445 | 0.620 | 0.529 | 0.343 |
| hybrid | 0.319 | 0.666 | 0.777 | 0.775 | 0.583 |
| hybrid-rrf | 0.225 | 0.595 | 0.727 | 0.706 | 0.473 |
| graph-gold | 0.137 | 0.406 | 0.548 | 0.510 | 0.347 |
| hybrid-gold | 0.252 | 0.699 | 0.762 | 0.814 | 0.542 |

MRR@10 by query type (all 102):

| Mode | cross (12) | definition | fact | paraphrase | procedure | relation |
|---|---|---|---|---|---|---|
| bm25 | **0.581** | 0.545 | 0.545 | 0.407 | 0.494 | 0.730 |
| vector | 0.419 | 0.632 | 0.597 | **0.511** | 0.397 | 0.789 |
| graph | 0.238 | 0.385 | 0.431 | 0.252 | 0.377 | 0.341 |
| hybrid | 0.400 | **0.706** | 0.589 | 0.413 | **0.496** | **0.837** |
| hybrid-rrf | 0.349 | 0.514 | **0.644** | 0.314 | 0.392 | 0.585 |
| graph-gold | 0.510 | 0.438 | 0.434 | 0.022 | 0.270 | 0.463 |
| hybrid-gold | 0.513 | 0.590 | 0.583 | 0.391 | 0.417 | 0.750 |

## Findings

- **Graph-only search can't pick the passage within a concept.** Ranking by mentioned
  concepts gets the topic right but not the spot. `AVL tree` is mentioned in 21 of Lecture
  6's 43 passages, and a question about the AVL property needs one or two of them. Concept
  weights rank all 21 almost alike. Vector compares the question with each passage's text,
  so it can find the passage that answers it. The graph does better than vector on 27 of
  102 queries but worse on 53.
- **Better extraction does not fix that.** On the gold concepts, graph-only scores no
  better (MRR 0.347 vs 0.343). Gold labels mark where a concept is taught, which narrows the
  passages, but they cover only 82 concepts. A query about anything else finds no seed:
  paraphrase MRR falls to 0.02. The extracted graph is noisier, but it covers more terms.
- **The graph helps most as a re-ranker.** `hybrid` can only reorder vector's top 50. It
  moves relevant passages up from vector's ranks 11-50, so fewer queries miss in the top 10
  (11 vs 18) and R@10 rises by 4.5 points on all queries. It helps most on procedure,
  definition and relation questions (+0.05 to +0.10 MRR). It hurts paraphrases (-0.10), where the concepts it matches are often the
  wrong ones. On `test` the net effect on MRR is -0.03 with an interval that spans zero.
  The graph adds signal, but on rank metrics it isn't measurably better than vector alone.
- **Rank fusion with a weak ranker hurts.** `hybrid-rrf` gives graph's ranking equal say,
  and graph is much weaker, so it drags the good vector ranking down (-0.09 MRR on all
  queries, with an interval that excludes zero).
- **Cross-lecture queries are where structure shows up,** but there are only 12 of them.
  `graph-gold` and `hybrid-gold` reach 0.51 MRR on them, against 0.42 for vector. bm25 does
  well too (0.58), because these queries name their concepts outright. That's too few
  queries to conclude anything, but this is the case to grow if the graph is to earn its
  place.
- **Keywords are a strong baseline here.** bm25 trails vector by only 0.02-0.04 MRR, well
  within noise. Lecture questions use the lecture's vocabulary, except the paraphrases,
  where bm25 is weakest.

## Limits

- **102 queries over 131 passages from 3 lectures.** The intervals above are wide, and only
  differences of about 0.1 MRR are clearly real.
- **AI-drafted labels.** Partial-answer passages are a judgment call, so recall for
  multi-passage queries is noisy.
- **Graph weights were set once, without tuning.** The hop weight was 0.25, and the fact bonus
  was 0.5 per fact up to 3 facts. Turning hops off scored about the same.
- **One embedding model**, bge-m3, for both passages and concepts. D7 fine-tunes it, and
  will report on this same query set.
