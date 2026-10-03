# D2 extraction results

Three MIT 6.006 (Fall 2011) lectures, transcribed in D1: Lecture 5 (binary search trees,
43 chunks), Lecture 6 (AVL trees, 43) and Lecture 7 (counting/radix sort, lower bounds,
45). 131 chunks in total, one structured-output call per chunk, temperature 0, prompt v2.

Reproduce: `domaingraph extract --model <m>` for each model, then
`domaingraph eval-extract --models qwen3:8b,qwen3:14b,llama3.1:8b --min-chunks 1,2`.

## How to read this

- **The gold sets were drafted by an AI annotator, not by a person.** Three agents each
  read one transcript and wrote `benchmarks/extraction/gold/lecture{5,6,7}.yaml` without
  seeing any extractor output. A review then made these corrections:
  - **Lecture 5:** `BST is_a sorted array` became `contrasts_with`, and `BST solves rank`
    was removed (rank is an operation, not a problem).
  - **Lecture 6:** the alias "balance factor" was dropped, since the lecture never says it.
    `balanced BST` was retyped as a data structure. `double rotation part_of rotation`
    became `uses`, and `successor part_of abstract data type` was removed.
  - **Lecture 7:** the alias "heap sort" was dropped from `heap`. `stable sort` was added,
    with 2 relations: the lecture demonstrates stability without naming it, and radix sort
    depends on it.

  A second pass, made with the user, applied two predicate rules and tightened aliases:
  - **`has_property`** means an intrinsic property or invariant of the subject. It does not
    cover "runs in O(h)" or "preserves X". **`uses`** needs an algorithm, data structure,
    operation or technique as its subject. **`solves`** also covers implementing an ADT
    (`heap solves priority queue`).
  - **Removed relations:** in Lecture 5, `insert` / `find min` / `rank` `has_property tree
    height` and `BST uses binary search`. In Lecture 6, `rotation has_property BST property`
    and `logarithmic height uses recurrence`. In Lecture 7, `height of a tree` / `binary
    search` `has_property worst case running time`, plus both `uses key`.
  - **Changed relations:** in Lecture 6, `height uses data structure augmentation` became
    `AVL tree uses data structure augmentation`. In Lecture 7, `part_of` became `is_a` for
    `comparison model` / `RAM model` → model of computation and `integer sorting` → sorting.
  - **Added:** in Lecture 6, `left rotate` and `right rotate` became their own concepts
    (each `is_a rotation`) instead of aliases of `rotation`. In Lecture 7, `lower bound uses
    permutation` and `lower bound uses Stirling's approximation` cover the proof chain.
  - **Aliases dropped** as too generic or wrong: `safety constraint`, `size`, `h`, `array`,
    `list`, `linked list`, `key`, `constant`, `linear` (L5); `find the min`, `interface`
    (L6); `comparison sort(ing)` on `comparison model`, `ram`, `word`, `search` (L7).
  - **Types made consistent:** `tree height` → property (L5), `priority queue` and
    `abstract data type` → other (L6, L7). Names in L7 are cased like the other lectures.

  Until a
  human has checked them, these numbers measure agreement with that annotator.
- **Gold: 82 concepts (42 core) and 80 relations.** "Core" means a student must learn it
  from that lecture.
- **Precision is a lower bound.** A real concept the annotator left out (`heap sort`,
  `left subtree`) counts as a false positive. Strict matching is by normalized name or
  alias; "lenient" adds bge-m3 name similarity >= 0.85.
- **`>= 2 chunks`** counts a concept only if at least two chunks of the lecture mention it.
- **Dups** are extracted concepts that match a gold concept another extracted concept
  already matched, i.e. duplicates the merger missed.
- **Rel P, ends in gold**: relation precision over only the relations whose two ends are
  both gold concepts. This separates "wrong relation" from "relation between concepts the
  gold set doesn't have".

## Results

Merge: names, then bge-m3 >= 0.85 with the contrast guard (see below).

| Model | Concepts P / R / F1 | Lenient P / R | Core R | Dups | Relations P / R | Any-pred P / R | Rel P, ends in gold | #concepts / #rels |
|---|---|---|---|---|---|---|---|---|
| qwen3:8b, >= 1 chunks | 28% / 87% / 43% | 29% / 88% | 95% | 12 | 6% / 51% | 7% / 57% | 20% | 250 / 700 |
| qwen3:8b, >= 2 chunks | 31% / 72% / 44% | 33% / 76% | 88% | 9 | 6% / 46% | 7% / 52% | 20% | 189 / 629 |
| qwen3:14b, >= 1 chunks | 23% / 88% / 37% | 23% / 88% | 93% | 17 | 5% / 41% | 6% / 57% | 20% | 312 / 726 |
| qwen3:14b, >= 2 chunks | 26% / 77% / 39% | 28% / 80% | 86% | 14 | 5% / 39% | 7% / 54% | 20% | 240 / 657 |
| llama3.1:8b, >= 1 chunks | 30% / 16% / 21% | 40% / 21% | 26% | 0 | 12% / 4% | 17% / 5% | 75% (3 of 4) | 43 / 24 |
| llama3.1:8b, >= 2 chunks | 36% / 6% / 11% | 50% / 9% | 7% | 0 | 0% / 0% | 0% / 0% | 0% | 14 / 4 |

## Findings

- **Recall is high, precision is low.** qwen3:8b finds 87% of the gold concepts and 95% of
  the core ones. But only about 3 in 10 of its concepts are in the gold set. A sample of
  the false positives shows two kinds:
  - generic words (`node`, `pointer`, `set`, `time complexity`, `log n`)
  - real but minor concepts the annotator left out (`heap sort`, `left subtree`,
    `base case`)

  The prompt already tells the model to skip the first kind, and the schema caps a chunk at
  12 concepts. Neither was enough.
- **qwen3:8b is the better extractor, not qwen3:14b.** Both reach the same recall, but 14b
  lists about 25% more concepts, so its precision is lower (23% vs 28%). Its relation
  recall on Lecture 5 is also lower (40% vs 60%).
- **Relations are the weak part.** Only 5-6% of extracted relations are exactly in gold.
  Of the relations between two gold concepts, about 1 in 5 is right. Samples of the rest:
  - plausible relations with the wrong predicate: `rotation part_of AVL tree` where gold
    has `AVL tree uses rotation`
  - reversed relations: `integer sorting solves counting sort`
  - weakly supported relations: `successor has_property height`

  Some of these are defensible, and the gold set has only 80 relations. Treat relation
  precision as the least reliable number here. The second gold pass (predicate rules, 10
  relations removed or changed) moved relation precision by at most 2 points, so the low
  number is not an artifact of a few borderline gold choices.
- **The model's confidence is useless.** qwen3:8b gave 96% of its concept mentions exactly
  0.9, and the rest 0.8 or 1.0. Confidence can't be used as a filter.
- **Support works better as a filter, but costs core recall.** Requiring 2+ chunks cuts the
  concept list by about a quarter and raises precision by 3 points (qwen3:8b: 28% -> 31%),
  but core recall drops by 7 points (95% -> 88%). Concepts taught in one stretch of the
  lecture, like `AVL sort`, fall out. The knowledge file keeps every concept with all its
  mentions, so D3/D4 can rank on support rather than drop anything.
- **llama3.1:8b mostly returns nothing.** It returned an empty extraction (about 6 tokens) for
  125 of 131 chunks. The prompt allows empty lists for passages that teach nothing technical,
  and llama takes that way out almost every time. With prompt v1 it gave more (214 concept
  mentions vs 58), but recall was still 36%.

## Merging duplicates

Merge quality is measured over pairs of mentions from the same lecture whose names match
gold concepts. A pair should merge when both name the same gold concept. Precision is the
share of merged pairs that should have merged, and recall the share of should-merge pairs
that did.

**Contrast guard.** Before the guard, embeddings merged names that look alike but mean
different things: `binary tree` / `binary search tree`, `left rotate` / `right rotate`,
`sorted array` / `sorted list`, `log n` / `n log n`. At 0.85 that put merge precision at
88-90%. The guard (`contrast_block` in `merge.py`) blocks an embedding merge when one name
is the other plus extra words, or when the two differ by exactly one swapped word. Names that
share no words (`insert` / `insertion`, `delete` / `remove`) are still decided by similarity.

| Model | Merge | Concepts | Merge P / R | Concepts P / R | Dups |
|---|---|---|---|---|---|
| qwen3:8b | names only | 219 | 100.0% / 79.1% | 28% / 87% | 13 |
| qwen3:8b | bge-m3 >= 0.75 | 202 | 90.1% / 83.0% | 28% / 83% | 11 |
| qwen3:8b | **bge-m3 >= 0.85** | 215 | **100.0% / 82.7%** | 28% / 87% | 12 |
| qwen3:8b | bge-m3 >= 0.90 | 217 | 100.0% / 79.1% | 28% / 87% | 13 |
| qwen3:14b | names only | 278 | 100.0% / 75.0% | 23% / 88% | 18 |
| qwen3:14b | bge-m3 >= 0.75 | 247 | 88.8% / 75.3% | 23% / 83% | 16 |
| qwen3:14b | **bge-m3 >= 0.85** | 271 | **100.0% / 75.1%** | 23% / 88% | 17 |
| qwen3:14b | bge-m3 >= 0.90 | 276 | 100.0% / 75.0% | 23% / 88% | 18 |

- **Names do most of the work.** Normalized names, plus "this name is another concept's
  alias", already merge 1,145 (qwen3:8b) and 1,463 (qwen3:14b) mentions down to 219 and 278
  concepts, without a single wrong merge.
- **Embeddings add little at a safe threshold.** At 0.85 the metric finds no wrong merge,
  and reading the merged groups turns up only one borderline case (`ordering` / `sorted
  order`). The rest are right: `insert` / `insertion`, `delete` / `remove`, `RAM model` /
  `random access machine`. They gain at most 3.6 points of merge recall, though.
- **0.75 merges wrongly.** Now that `left rotate` and `right rotate` are gold concepts of
  their own, the metric sees 0.75 merging them into `rotation` (merge precision 89-90%).
  Before the second gold pass they were aliases of `rotation`, and the metric showed 0.75
  at 99%+. Reading the 0.75 groups for qwen3:14b shows more wrong merges than the metric
  counts:
  - `successor` with `predecessor`
  - `upper bound` with `tight lower bound`
  - `counting sort` with `comparison based sorting`
  - `height of a tree` with `size of a subtree`

  That's why the default is 0.85. At 0.65, merge precision falls to 65-68%.
- **The remaining duplicates need more than names,** such as `BST invariant` vs `ordering
  invariant`. Definitions, or the graph's own relations, could separate
  these. That's a job for D3.

## What changed from the first run (prompt v1)

The first run had no cap on items per chunk. qwen3:14b sometimes listed 50+ loosely related
terms and ran out of its 3,072-token reply budget mid-JSON. That happened on 4 of 131
chunks, and 3 of the 4 failed again on retry. qwen3:8b and llama failed on 1 chunk each.
Prompt v2 adds `maxItems` to the response schema (12 concepts, 12 relations, 8 facts) and a
sharper list of words to skip. No chunk failed in the v2 run.

| Model | v1 concepts P / R | v2 concepts P / R | v1 core R | v2 core R |
|---|---|---|---|---|
| qwen3:8b | 29% / 82% | 28% / 88% | 81% | 95% |
| qwen3:14b | 28% / 75% | 23% / 89% | 81% | 93% |
| llama3.1:8b | 34% / 36% | 30% / 16% | 36% | 26% |

The v1 numbers come from the v1 extractions under the merge settings of that time (bge-m3 >=
0.9, no contrast guard). v2 uses the settings above. Both are scored against the gold sets
as they were before the second review pass, so this table is not directly comparable with
the ones above.

## Per source

| Model | Lecture | Concepts P / R | Core R | Relations P / R | Missed core |
|---|---|---|---|---|---|
| qwen3:8b | Binary Search Trees, BST Sort | 36% / 83% | 93% | 7% / 60% | BST invariant |
| qwen3:8b | AVL Trees, AVL Sort | 29% / 86% | 93% | 7% / 52% | AVL sort |
| qwen3:8b | Counting Sort, Radix Sort, Lower Bounds | 23% / 92% | 100% | 4% / 42% | - |
| qwen3:14b | Binary Search Trees, BST Sort | 29% / 86% | 93% | 4% / 40% | BST invariant |
| qwen3:14b | AVL Trees, AVL Sort | 25% / 89% | 93% | 7% / 55% | AVL sort |
| qwen3:14b | Counting Sort, Radix Sort, Lower Bounds | 17% / 88% | 93% | 3% / 27% | model of computation |
| llama3.1:8b | Binary Search Trees, BST Sort | 30% / 21% | 43% | 18% / 8% | 8 of 14 |
| llama3.1:8b | AVL Trees, AVL Sort | 43% / 21% | 29% | 12% / 3% | 10 of 14 |
| llama3.1:8b | Counting Sort, Radix Sort, Lower Bounds | 11% / 4% | 7% | 0% / 0% | 13 of 14 |

All at `>= 1 chunks`.
