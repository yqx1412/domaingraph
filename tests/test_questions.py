"""D7b: generated-question pairs and hard negatives (fakes only: no Ollama, GPU or Neo4j)."""

from __future__ import annotations

import json

import numpy as np

from domaingraph.llm import LLMError, StructuredReply
from domaingraph.models import Chunk
from domaingraph.questions import (
    Question,
    _clean,
    generate_questions,
    load_questions,
    mine_hard_negatives,
    question_pairs,
    requested_types,
)


def _ch(i: int, sid: str = "s1", text: str | None = None) -> Chunk:
    text = text or f"passage {i} of {sid}"
    return Chunk(id=f"{sid}:{i:04d}", source_id=sid, index=i, text=text, n_words=4)


class FakeLLM:
    model = "fake"

    def __init__(self, fail_on: set[str] = frozenset()):
        self.calls: list[str] = []
        self.fail_on = fail_on

    def chat_json(self, system, user, schema):
        self.calls.append(user)
        if any(f in user for f in self.fail_on):
            raise LLMError("boom")
        return StructuredReply(data={"questions": [
            {"type": "procedure", "question": "How do you insert a key into a heap?"},
            {"type": "paraphrase", "question": "How does a pile keep its smallest item on top?"},
            {"type": "definition", "question": "What is a heap?"},  # not requested below
        ]})  # fmt: skip


def test_clean_drops_meta_duplicates_and_unrequested_types():
    raw = {"questions": [
        {"type": "fact", "question": "What running time does the passage give for find?"},
        {"type": "fact", "question": "How many swaps does this sorting algorithm make?"},
        {"type": "fact", "question": "What is the cost of rehashing a table of size m?"},
        {"type": "fact", "question": "What is the height bound of an AVL tree?"},  # 2nd fact
        {"type": "paraphrase", "question": "Explain it in simple terms without using jargon?"},
        {"type": "relation", "question": "How does a heap differ from a binary search tree?"},
        {"type": "bogus", "question": "What is a heap and why does it matter?"},
        {"type": "definition", "question": "short"},
    ]}  # fmt: skip
    assert _clean(raw, ("fact", "relation", "definition")) == [
        ("fact", "What is the cost of rehashing a table of size m?"),
        ("relation", "How does a heap differ from a binary search tree?"),
    ]


def test_requested_types_rotate_and_always_include_paraphrase():
    seen = {requested_types(_ch(i)) for i in range(6)}
    assert len(seen) == 6 and all(t[-1] == "paraphrase" and len(t) == 3 for t in seen)
    assert requested_types(_ch(0)) == requested_types(_ch(6))


def test_generate_resumes_and_retries_failures(tmp_path):
    out = tmp_path / "q.jsonl"
    chunks = [_ch(0), _ch(1), _ch(2)]  # index 0 asks for procedure, relation, paraphrase
    llm = FakeLLM(fail_on={"passage 2 "})
    qs = generate_questions(llm, chunks, out, log=lambda m: None)
    assert len(llm.calls) == 3
    assert "Question types to write: procedure, relation, paraphrase" in llm.calls[0]
    assert {q.chunk_id for q in qs} == {"s1:0000", "s1:0001"}  # chunk 2 failed
    assert [q.type for q in qs if q.chunk_id == "s1:0000"] == ["procedure", "paraphrase"]
    # Resume: done chunks are not asked again; the failed one is retried.
    llm2 = FakeLLM()
    generate_questions(llm2, chunks, out, log=lambda m: None)
    assert len(llm2.calls) == 1 and "passage 2 " in llm2.calls[0]
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["chunk_id"] for r in rows] == ["s1:0000", "s1:0001", "s1:0002"]
    assert load_questions(out) == generate_questions(llm2, chunks, out, log=lambda m: None)


def test_question_pairs_use_passage_text():
    chunks = {c.id: c for c in [_ch(0), _ch(1)]}
    pairs = question_pairs([Question("s1:0001", "s1", "fact", "Q?")], chunks)
    assert [(p.anchor, p.positive, p.kind, p.negative) for p in pairs] == [
        ("Q?", "passage 1 of s1", "question", None)
    ]


def test_hard_negative_skips_positive_neighbours_top_and_other_lectures():
    # Similarity to the question is set per passage by a fake encoder: one dimension per
    # passage, the question vector holds the scores.
    chunks = {c.id: c for c in [_ch(i) for i in range(8)] + [_ch(0, "s2")]}
    ids = sorted(chunks)
    score = {"s1:0003": 1.0, "s1:0002": 0.99, "s1:0004": 0.98,  # positive + neighbours
             "s2:0000": 0.97,  # other lecture
             "s1:0007": 0.9, "s1:0000": 0.8, "s1:0006": 0.7, "s1:0001": 0.6}  # fmt: skip

    def encode(texts):
        if texts[0].startswith("passage"):
            return np.eye(len(ids))
        return np.array([[score.get(c, 0.0) for c in ids]])

    q = Question("s1:0003", "s1", "fact", "Q?")
    (p,) = mine_hard_negatives([q], chunks, encode, skip_top=2)
    # Candidates by score: 7, 0, 6, 1, 5 -> skip 7 and 0 -> negative is passage 6.
    assert p.positive == "passage 3 of s1" and p.negative == "passage 6 of s1"
    (p,) = mine_hard_negatives([q], chunks, encode, skip_top=0)
    assert p.negative == "passage 7 of s1"
