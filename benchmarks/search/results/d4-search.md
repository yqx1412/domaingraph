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
  lists every passage that answers it: 50 queries have 1 relevant passage, 40 have 2, 11
  have 3-4 and one has 8.
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
| bm25 | 0.353 | 0.647 | 0.737 | 0.740 | 0.596 |
| **vector** | **0.383** | **0.717** | **0.827** | 0.840 | **0.653** |
| graph | 0.187 | 0.483 | 0.643 | 0.560 | 0.394 |
| hybrid | 0.367 | 0.697 | 0.777 | 0.840 | 0.618 |
| hybrid-rrf | 0.333 | 0.637 | 0.697 | 0.760 | 0.557 |
| graph-gold | 0.123 | 0.410 | 0.520 | 0.500 | 0.322 |
| hybrid-gold | 0.313 | 0.737 | 0.807 | **0.860** | 0.612 |

Paired difference vs vector, with 95% bootstrap intervals over queries:

| Mode | MRR@10 | R@5 |
|---|---|---|
| bm25 | -0.057 [-0.150, +0.037] | -0.070 [-0.190, +0.047] |
| graph | **-0.259 [-0.391, -0.129]** | **-0.233 [-0.363, -0.110]** |
| hybrid | -0.035 [-0.108, +0.032] | -0.020 [-0.080, +0.040] |
| hybrid-rrf | -0.095 [-0.225, +0.032] | -0.080 [-0.180, +0.010] |
| graph-gold | **-0.331 [-0.461, -0.200]** | **-0.307 [-0.443, -0.170]** |
| hybrid-gold | -0.041 [-0.104, +0.016] | +0.020 [+0.000, +0.050] |

### All 102 queries

`hybrid` was tuned on `dev`, which is half of this set.

| Mode | R@1 | R@5 | R@10 | Hit@5 | MRR@10 |
|---|---|---|---|---|---|
| bm25 | 0.278 | 0.597 | 0.710 | 0.696 | 0.546 |
| vector | 0.308 | 0.640 | 0.738 | 0.765 | 0.572 |
| graph | 0.150 | 0.453 | 0.623 | 0.549 | 0.350 |
| hybrid | 0.321 | 0.673 | 0.773 | 0.794 | 0.594 |
| hybrid-rrf | 0.230 | 0.593 | 0.721 | 0.716 | 0.482 |
| graph-gold | 0.139 | 0.409 | 0.549 | 0.510 | 0.347 |
| hybrid-gold | 0.252 | 0.701 | 0.769 | 0.824 | 0.547 |

MRR@10 by query type (all 102):

| Mode | cross (12) | definition | fact | paraphrase | procedure | relation |
|---|---|---|---|---|---|---|
| bm25 | **0.637** | 0.545 | 0.548 | 0.352 | 0.494 | 0.730 |
| vector | 0.461 | 0.632 | 0.602 | **0.511** | 0.397 | 0.789 |
| graph | 0.268 | 0.385 | 0.449 | 0.243 | 0.386 | 0.341 |
| hybrid | 0.483 | **0.706** | 0.599 | 0.406 | **0.496** | **0.837** |
| hybrid-rrf | 0.383 | 0.514 | **0.681** | 0.306 | 0.392 | 0.585 |
| graph-gold | 0.510 | 0.438 | 0.434 | 0.022 | 0.270 | 0.463 |
| hybrid-gold | 0.544 | 0.590 | 0.589 | 0.391 | 0.417 | 0.750 |

## Findings

- **Graph-only search can't pick the passage within a concept.** Ranking by mentioned
  concepts gets the topic right but not the spot. `AVL tree` is mentioned in 21 of Lecture
  6's 43 passages, and a question about the AVL property needs one or two of them. Concept
  weights rank all 21 almost alike. Vector compares the question with each passage's text,
  so it can find the passage that answers it. The graph does better than vector on 26 of
  102 queries but worse on 53.
- **Better extraction does not fix that.** On the gold concepts, graph-only scores no
  better (MRR 0.347 vs 0.350). Gold labels mark where a concept is taught, which narrows the
  passages, but they cover only 82 concepts. A query about anything else finds no seed:
  paraphrase MRR falls to 0.02. The extracted graph is noisier, but it covers more terms.
- **The graph helps most as a re-ranker.** `hybrid` can only reorder vector's top 50. It
  moves relevant passages up from vector's ranks 11-50, so fewer queries miss in the top 10
  (11 vs 16) and R@10 rises by 3.5 points on all queries. It helps most on procedure,
  definition and relation questions (+0.05 to +0.10 MRR). It hurts paraphrases (-0.11), where the concepts it matches are often the
  wrong ones. On `test` the net effect on MRR is -0.035 with an interval that spans zero.
  The graph adds signal, but on rank metrics it isn't measurably better than vector alone.
- **Rank fusion with a weak ranker hurts.** `hybrid-rrf` gives graph's ranking equal say,
  and graph is much weaker, so it drags the good vector ranking down (-0.09 MRR on all
  queries, with an interval that excludes zero).
- **Cross-lecture queries are where structure shows up,** but there are only 12 of them.
  `graph-gold` and `hybrid-gold` reach 0.51-0.54 MRR on them, against 0.46 for vector. bm25
  does better still (0.64), because these queries name their concepts outright. That's too few
  queries to conclude anything, but this is the case to grow if the graph is to earn its
  place.
- **Keywords are a strong baseline here.** bm25 trails vector by only 0.03-0.06 MRR, well
  within noise. Lecture questions use the lecture's vocabulary, except the paraphrases,
  where bm25 is weakest.

## Label review

After the first scoring, I reviewed the 18 queries where vector found no relevant passage in its
top 10. For each one I read the labeled passages and the top 3 of vector, graph and hybrid. A
label changed only where a passage clearly states the answer and the label had left it out, or
where a labeled passage doesn't answer the query. 8 of the 18 queries changed:

| Query | Change | Why |
|---|---|---|
| l5-15 | + L5 #28 | "Go to the left till you hit a leaf" is the answer itself |
| l6-09 | + L6 #3 | the overlap repeats "you can solve them in order h time" |
| l5-26 | - L5 #36 | it only asks how to compute rank, the answer is in #38-39 (now the same as l5-18) |
| x-02 | + L5 #27, #28, #42, L6 #4 | O(h) for the other operations, and the list-shaped worst case |
| x-05 | - L5 #27 | about find min, not sorting; Lecture 5 never covers BST sort, so this query needs only Lecture 6 |
| x-09 | + L6 #21 | states the h < 1.44 log n bound the answer gives |
| x-12 | - L5 #17 | about pointers, not rotations or order |
| l5-11 | (none) | Lecture 6 #4 answers it too, but a per-lecture query can only label its own lecture |

The other 10 are genuine misses: the labeled passages are right and vector ranked them 11th to 42nd.

**Effect:** 2 of the 18 became hits for vector (x-02, l6-09). On `test`, vector's numbers didn't
move, and no other mode moved by more than 0.02 MRR. Every conclusion above holds before and after.
The review started from vector's misses, so it could favor vector, which is why the graph and hybrid
top 3 were read too. The tables above use the reviewed labels; the pre-review numbers are in the
history of this file (`f496661`).

## Limits

- **102 queries over 131 passages from 3 lectures.** The intervals above are wide, and only
  differences of about 0.1 MRR are clearly real.
- **AI-drafted labels.** Partial-answer passages are a judgment call, so recall for
  multi-passage queries is noisy.
- **Graph weights were set once, without tuning.** The hop weight was 0.25, and the fact bonus
  was 0.5 per fact up to 3 facts. Turning hops off scored about the same.
- **One embedding model**, bge-m3, for both passages and concepts. D7 fine-tunes it, and
  will report on this same query set.
