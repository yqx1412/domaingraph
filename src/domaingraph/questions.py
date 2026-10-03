"""D7 follow-up: question-shaped training pairs for the embedding fine-tune.

D7's first run trained on (fact, passage) pairs and improved matching fact statements to
passages, but not the D4 search questions, and paraphrased questions got worse. This module
makes the training data look like the task:

* **Generated questions:** the local LLM reads one TRAINING passage at a time and writes up
  to three questions it answers, always including one paraphrase that avoids the passage's
  key terms. The LLM never sees a test lecture (5-7) or the D4 query set.
* **Hard negatives:** for each question, a passage from the SAME lecture that the base model
  ranks high but that is not the positive. The positive's immediate neighbours are skipped
  (chunks overlap by 40 words, so a neighbour often answers too), and so are the top
  ``skip_top`` candidates, which are the likeliest unlabeled true answers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from domaingraph.finetune import Pair
from domaingraph.llm import LLMError, StructuredLLM
from domaingraph.models import Chunk

QUESTION_TYPES = ("definition", "fact", "procedure", "relation", "paraphrase")
PROMPT_VERSION = "q4"

# Left to choose, qwen3:8b wrote definition/fact/paraphrase on almost every passage (prompt
# q3: 3 relation and 4 procedure questions out of 596). So each passage is asked for one
# paraphrase plus two named types, rotating through every pair of the other four.
TYPE_ROTATION = (
    ("procedure", "relation"),
    ("definition", "fact"),
    ("procedure", "definition"),
    ("relation", "fact"),
    ("procedure", "fact"),
    ("relation", "definition"),
)

QUESTION_SYSTEM = """You write search questions for a retrieval benchmark built from \
lecture transcripts of MIT 6.006 (Introduction to Algorithms).

You get ONE passage and the question types to write. Write questions a student might type \
into a search box, where THIS passage contains the answer.

Rules:
- Each question must be answerable from the passage alone, and must be about the \
algorithms, data structures or analysis the passage teaches.
- Each question must stand on its own. Never refer to "the passage", "the lecture", "the \
professor", "the speaker", "this course", "this class", "this algorithm" or "the \
example": name the algorithm, data structure or idea explicitly.
- Do not copy sentences from the passage. Write like a student who has not read it.
- Question types:
  - definition: what something is
  - fact: a specific claim, number, bound or running time
  - procedure: how something works, or the steps to do something
  - relation: how two things relate, or how they differ
  - paraphrase: a question that describes the idea in everyday words instead of naming \
it. Do not mention that you are avoiding terms.
- Write one question of each requested type. Skip a type only if the passage has nothing \
for it.
- If the passage teaches nothing technical (license text, course logistics, \
administration, small talk), return an empty list."""

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(QUESTION_TYPES)},
                    "question": {"type": "string"},
                },
                "required": ["type", "question"],
            },
        }
    },
    "required": ["questions"],
}


@dataclass
class Question:
    chunk_id: str
    source: str
    type: str
    question: str


# Questions that point at their source instead of standing alone ("...in the passage?"),
# or that announce the paraphrase rule instead of following it.
_META = re.compile(
    r"\b(passage|lecture|lecturer|professor|speaker|this (\w+ )?(course|class|algorithm|"
    r"example|context|case|method|approach|step|process|procedure|data structure|function|"
    r"problem|expression|operation)|the course|explanation|discussed|described|"
    r"(the |this )?(given )?example( given)?|mentioned|technical (terms|jargon)|"
    r"simple terms|without using)\b",
    re.IGNORECASE,
)


def requested_types(chunk: Chunk) -> tuple[str, ...]:
    return (*TYPE_ROTATION[chunk.index % len(TYPE_ROTATION)], "paraphrase")


def _clean(raw: dict, types: Sequence[str] = QUESTION_TYPES) -> list[tuple[str, str]]:
    """Valid items of the requested types, at most one per type; meta questions dropped."""
    out, seen, used = [], set(), set()
    for item in raw.get("questions") or []:
        qtype = str(item.get("type", "")).strip()
        text = " ".join(str(item.get("question", "")).split())
        if qtype not in types or qtype in used or len(text) < 10 or text.lower() in seen:
            continue
        if _META.search(text):
            continue
        seen.add(text.lower())
        used.add(qtype)
        out.append((qtype, text))
    return out[:3]


def generate_questions(
    llm: StructuredLLM,
    chunks: Iterable[Chunk],
    out: Path,
    *,
    log: Callable[[str], None] = print,
) -> list[Question]:
    """Write questions for each chunk to ``out`` (JSONL), resuming chunks already done.

    A chunk is recorded even when it yields no questions (``{"chunk_id", "questions": []}``),
    so a resume does not ask again; a chunk whose call failed is not recorded and is retried.
    """
    done: dict[str, list[dict]] = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["chunk_id"]] = row["questions"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for ch in chunks:
            if ch.id in done:
                continue
            types = requested_types(ch)
            user = f"Question types to write: {', '.join(types)}\n\nPassage:\n{ch.text}"
            try:
                reply = llm.chat_json(QUESTION_SYSTEM, user, QUESTION_SCHEMA)
            except LLMError as exc:
                log(f"{ch.id}: failed ({exc})")
                continue
            qs = [{"type": t, "question": q} for t, q in _clean(reply.data, types)]
            row = {"chunk_id": ch.id, "source": ch.source_id, "prompt": PROMPT_VERSION,
                   "model": llm.model, "questions": qs}  # fmt: skip
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[ch.id] = qs
            log(f"{ch.id}: {len(qs)} questions in {reply.seconds:.1f}s")
    return load_questions(out)


def load_questions(path: Path) -> list[Question]:
    """Saved questions, re-filtered so a tightened ``_META`` also covers older files."""
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for q in row["questions"]:
            if not _META.search(q["question"]):
                found.append(Question(row["chunk_id"], row["source"], q["type"], q["question"]))
    return found


def question_pairs(questions: Sequence[Question], chunks: dict[str, Chunk]) -> list[Pair]:
    return [Pair(q.question, chunks[q.chunk_id].text, "question", q.source) for q in questions]


def mine_hard_negatives(
    questions: Sequence[Question],
    chunks: dict[str, Chunk],
    encode: Callable[[list[str]], object],
    *,
    skip_top: int = 2,
    neighbours: int = 1,
) -> list[Pair]:
    """One (question, positive, negative) triple per question.

    Candidates are the other passages of the question's own lecture, minus the positive and
    its ``neighbours`` on each side. Ranked by the base model's cosine similarity, the top
    ``skip_top`` are skipped (the likeliest unlabeled true answers) and the next one is the
    negative. Deterministic for a given model.
    """
    import numpy as np

    by_source: dict[str, list[Chunk]] = {}
    for ch in chunks.values():
        by_source.setdefault(ch.source_id, []).append(ch)

    ids = sorted(chunks)
    row = {cid: i for i, cid in enumerate(ids)}
    pvec = np.asarray(encode([chunks[c].text for c in ids]))
    qvec = np.asarray(encode([q.question for q in questions]))

    pairs = []
    for q, qv in zip(questions, qvec, strict=True):
        pos = chunks[q.chunk_id]
        cands = [c for c in by_source[pos.source_id] if abs(c.index - pos.index) > neighbours]
        if len(cands) <= skip_top:
            continue
        sims = pvec[[row[c.id] for c in cands]] @ qv
        order = np.argsort(-sims, kind="stable")
        neg = cands[int(order[skip_top])]
        pairs.append(Pair(q.question, pos.text, "question", q.source, negative=neg.text))
    return pairs


def question_dev_evaluator(
    questions: Sequence[Question], corpus: dict[str, Chunk], name: str = "dev_q"
):
    """Generated questions of the dev lecture as queries; ``corpus`` passages to search."""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    queries = {f"q{i}": q.question for i, q in enumerate(questions)}
    relevant = {f"q{i}": {q.chunk_id} for i, q in enumerate(questions)}
    docs = {cid: ch.text for cid, ch in corpus.items()}
    return InformationRetrievalEvaluator(
        queries, docs, relevant, name=name, mrr_at_k=[10], ndcg_at_k=[10],
        accuracy_at_k=[1, 5], precision_recall_at_k=[10], show_progress_bar=False,
    )  # fmt: skip
