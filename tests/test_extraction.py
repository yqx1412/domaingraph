import json
from pathlib import Path

import httpx
import pytest
import yaml

from domaingraph.cli import main
from domaingraph.evaluate_extraction import PRF, Gold, load_gold_dir, merge_quality, score_source
from domaingraph.extraction import (
    PROMPT_VERSION,
    ChunkExtraction,
    ChunkResult,
    build_prompt,
    clean,
    extract_chunk,
    extract_source,
    extraction_path,
    load_results,
    response_schema,
)
from domaingraph.llm import LLMError, OllamaEmbedder, OllamaStructured, StructuredReply
from domaingraph.merge import contrast_block, merge, name_groups, normalize
from domaingraph.models import Chunk, Source
from domaingraph.pipeline import ingest

GOLD_DIR = Path(__file__).parents[1] / "benchmarks" / "extraction" / "gold"


def _source(sid: str = "s1") -> Source:
    return Source(id=sid, path="x", kind="text", title="Trees", sha256="0" * 64, bytes=1)


def _chunk(i: int, sid: str = "s1", text: str = "text") -> Chunk:
    return Chunk(
        id=f"{sid}:{i}",
        source_id=sid,
        index=i,
        text=text,
        n_words=1,
        start=60.0 * i,
        end=60.0 * i + 50,
    )


def _concept(name, aliases=(), type="data_structure", conf=0.9):
    return {
        "name": name,
        "aliases": list(aliases),
        "type": type,
        "definition": f"{name} def",
        "confidence": conf,
    }


class ScriptedLLM:
    """Returns queued replies; records the prompts it saw."""

    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def chat_json(self, system, user, schema):
        self.prompts.append(user)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return StructuredReply(data=r, prompt_tokens=10, completion_tokens=5, seconds=0.1)


class FakeEmbedder:
    """Vectors chosen per normalized text; unknown texts get orthogonal one-hot vectors."""

    model = "fake-embed"

    def __init__(self, table: dict[str, list[float]] | None = None):
        self.table = table or {}
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            if t in self.table:
                out.append(self.table[t])
            else:
                v = [0.0] * 64
                v[hash(t) % 61 + 3] = 1.0
                out.append(v)
        return out


# -- normalize and cleaning ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("The AVL-Trees", "avl tree"),
        ("binary search trees", "binary search tree"),
        ("Stirling's formula", "stirling formula"),
        ("in-order traversal", "in order traversal"),
        ("priority queues", "priority queue"),
        ("analysis", "analysis"),
        ("binary search", "binary search"),
        ("heaps", "heap"),
        ("vertices", "vertex"),
        ("insert operation", "insert"),
        ("operation", "operation"),
    ],
)
def test_normalize(raw: str, want: str) -> None:
    assert normalize(raw) == want


def test_response_schema_caps_items_but_stored_results_are_uncapped() -> None:
    s = response_schema()
    assert s["properties"]["concepts"]["maxItems"] == 12
    assert s["properties"]["facts"]["maxItems"] == 8
    assert "maxItems" not in ChunkExtraction.model_json_schema()["properties"]["concepts"]
    many = ChunkExtraction.model_validate({"concepts": [_concept(f"c{i}") for i in range(20)]})
    assert len(many.concepts) == 20


def test_clean_splits_parenthesized_alias_and_drops_bad_items() -> None:
    raw = ChunkExtraction.model_validate(
        {
            "concepts": [
                _concept("Binary Search Tree (BST)"),
                _concept("binary search tree"),  # duplicate (case)
                _concept("  "),
                _concept("height", type="property"),
            ],
            "relations": [
                {
                    "subject": "BST",
                    "predicate": "has_property",
                    "object": "height",
                    "confidence": 0.8,
                },
                {
                    "subject": "BST",
                    "predicate": "has_property",
                    "object": "height",
                    "confidence": 0.7,
                },
                {"subject": "BST", "predicate": "uses", "object": "heap", "confidence": 0.8},
                {"subject": "height", "predicate": "uses", "object": "height", "confidence": 0.8},
            ],
            "facts": [
                {
                    "statement": "  A BST  has height h. ",
                    "concepts": ["bst", "nope"],
                    "confidence": 1,
                },
                {"statement": " ", "concepts": [], "confidence": 1},
            ],
        }
    )
    out, dropped = clean(raw)
    assert [c.name for c in out.concepts] == ["Binary Search Tree", "height"]
    assert out.concepts[0].aliases == ["BST"]
    # Relation ends resolved through the alias to the canonical name.
    assert [(r.subject, r.object) for r in out.relations] == [("Binary Search Tree", "height")]
    assert out.facts[0].statement == "A BST has height h."
    assert out.facts[0].concepts == ["Binary Search Tree"]
    assert dropped == {"empty": 2, "duplicate": 2, "dangling_relation": 1, "self_relation": 1}


def test_schema_rejects_unknown_predicate_and_bad_confidence() -> None:
    llm = ScriptedLLM([{"concepts": [_concept("x", conf=1.5)]}])
    r = extract_chunk(llm, _source(), _chunk(0))
    assert r.error is not None and "validation" in r.error
    assert r.extraction.concepts == []


# -- extraction ---------------------------------------------------------------------------


def test_extract_chunk_keeps_pointer_back_to_the_chunk() -> None:
    llm = ScriptedLLM([{"concepts": [_concept("AVL tree")], "relations": [], "facts": []}])
    r = extract_chunk(llm, _source(), _chunk(3))
    assert (r.chunk_id, r.index, r.locator, r.start) == ("s1:3", 3, "03:00-03:50", 180.0)
    assert r.model == "scripted" and r.prompt_version == PROMPT_VERSION
    assert r.prompt_tokens == 10 and r.error is None


def test_extract_chunk_records_llm_error() -> None:
    r = extract_chunk(ScriptedLLM([LLMError("boom")]), _source(), _chunk(0))
    assert r.error == "boom"


def test_build_prompt_lists_known_names() -> None:
    p = build_prompt(_source(), _chunk(1, text="Rotations fix balance."), ["AVL tree", "height"])
    assert "AVL tree, height" in p and "Rotations fix balance." in p and "01:00-01:50" in p


def test_extract_source_passes_names_forward_and_resumes(tmp_path: Path) -> None:
    replies = [
        {"concepts": [_concept("AVL tree")]},
        LLMError("timeout"),
        {"concepts": [_concept("rotation", type="operation")]},
    ]
    llm = ScriptedLLM(replies)
    chunks = [_chunk(i) for i in range(3)]
    res = extract_source(llm, _source(), chunks, out=tmp_path)
    assert [r.error for r in res] == [None, "timeout", None]
    assert "AVL tree" in llm.prompts[1] and "AVL tree" in llm.prompts[2]

    # Second run only redoes the failed chunk.
    llm2 = ScriptedLLM([{"concepts": [_concept("height", type="property")]}])
    res2 = extract_source(llm2, _source(), chunks, out=tmp_path)
    assert len(llm2.prompts) == 1
    # A retried chunk sees names from earlier chunks only, as on the first pass.
    assert "AVL tree" in llm2.prompts[0] and "rotation" not in llm2.prompts[0]
    assert [r.error for r in res2] == [None, None, None]
    saved = load_results(tmp_path, "s1", "scripted")
    assert [r.index for r in saved] == [0, 1, 2]
    assert len(extraction_path(tmp_path, "s1", "scripted").read_text().splitlines()) == 3


def test_extract_source_force_redoes_everything(tmp_path: Path) -> None:
    chunks = [_chunk(0)]
    extract_source(ScriptedLLM([{"concepts": [_concept("a")]}]), _source(), chunks, out=tmp_path)
    llm = ScriptedLLM([{"concepts": [_concept("b")]}])
    res = extract_source(llm, _source(), chunks, out=tmp_path, force=True)
    assert res[0].extraction.concepts[0].name == "b"


# -- merging ------------------------------------------------------------------------------


def _result(i, concepts, relations=(), facts=(), sid="s1") -> ChunkResult:
    c = _chunk(i, sid)
    ex, _ = clean(
        ChunkExtraction.model_validate(
            {"concepts": concepts, "relations": list(relations), "facts": list(facts)}
        )
    )
    return ChunkResult(
        chunk_id=c.id,
        source_id=sid,
        index=i,
        locator=c.locator(),
        start=c.start,
        end=c.end,
        model="m",
        extraction=ex,
    )


def test_name_groups_merge_name_to_alias_but_not_alias_to_alias() -> None:
    names = ["binary search tree", "BST", "Binary Search Trees", "heap", "AVL tree"]
    aliases = [["BST", "tree"], [], [], ["tree"], []]
    g = name_groups(names, aliases)
    assert g[0] == g[1] == g[2]
    assert g[3] != g[0]  # sharing the alias "tree" isn't enough
    assert g[4] not in (g[0], g[3])


def test_merge_names_keeps_mentions_relations_and_facts() -> None:
    results = [
        _result(
            0,
            [_concept("binary search tree", ["BST"]), _concept("height", type="property")],
            [
                {
                    "subject": "binary search tree",
                    "predicate": "has_property",
                    "object": "height",
                    "confidence": 0.6,
                }
            ],
            [
                {
                    "statement": "BSTs have height h.",
                    "concepts": ["binary search tree"],
                    "confidence": 0.9,
                }
            ],
        ),
        _result(
            1,
            [_concept("BST", conf=0.95), _concept("heights", type="property")],
            [
                {
                    "subject": "BST",
                    "predicate": "has_property",
                    "object": "heights",
                    "confidence": 0.8,
                }
            ],
        ),
        _result(2, [_concept("binary search tree")], sid="s2"),
    ]
    kn = merge(results, model="m")
    assert len(kn.concepts) == 2
    bst = next(c for c in kn.concepts if c.id == "binary-search-tree")
    assert bst.name == "binary search tree" and "BST" in bst.aliases
    assert [m.chunk_id for m in bst.mentions] == ["s1:0", "s1:1", "s2:2"]
    assert bst.mentions[1].locator == "01:00-01:50"
    assert bst.confidence == 0.95
    assert len(kn.relations) == 1
    rel = kn.relations[0]
    assert (rel.subject, rel.object, rel.confidence) == ("binary-search-tree", "height", 0.8)
    assert len(rel.mentions) == 2
    assert kn.facts[0].concepts == ["binary-search-tree"]
    assert kn.facts[0].mention.chunk_id == "s1:0"
    assert kn.sources == ["s1", "s2"]
    assert kn.stats["mentions"] == 5 and kn.stats["concepts"] == 2


def test_merge_by_embedding_respects_threshold_and_type() -> None:
    v = [1.0] + [0.0] * 63
    near = [0.95, 0.312] + [0.0] * 62  # cos ~0.95 to v
    emb = FakeEmbedder({"avl property": v, "avl balance condition": near, "avl invariant": near})
    results = [
        _result(
            0,
            [
                _concept("AVL property", type="property"),
                _concept("AVL balance condition", type="property"),
                _concept("AVL invariant", type="operation"),  # same vector, different type
            ],
        )
    ]
    kn = merge(results, embedder=emb, threshold=0.9, model="m")
    names = sorted(c.name for c in kn.concepts)
    assert names == ["AVL invariant", "AVL property"]
    assert kn.stats["embedding_merges"] == 1
    merged = next(c for c in kn.concepts if c.name == "AVL property")
    assert "AVL balance condition" in merged.aliases
    # Higher threshold: nothing merges.
    assert len(merge(results, embedder=emb, threshold=0.99, model="m").concepts) == 3


@pytest.mark.parametrize(
    ("a", "b", "blocked"),
    [
        ("binary tree", "binary search tree", True),  # one adds words
        ("log n", "n log n", True),
        ("rotation", "double rotation", True),
        ("left rotate", "right rotate", True),  # one word swapped
        ("find min", "find max", True),
        ("sorted array", "sorted list", True),
        ("insert", "insertion", False),  # no shared word: left to the threshold
        ("delete", "remove", False),
        ("avl property", "avl balance condition", False),  # differs by more than one word
        ("heap", "heap", False),
    ],
)
def test_contrast_block(a: str, b: str, blocked: bool) -> None:
    assert contrast_block(a, b) is blocked


def test_embedding_merge_skips_contrasting_names_and_spaces_dont_matter() -> None:
    v = [1.0] + [0.0] * 63
    emb = FakeEmbedder({"left rotate": v, "right rotate": v})
    results = [
        _result(
            0,
            [
                _concept("left rotate", type="operation"),
                _concept("right rotate", type="operation"),
                _concept("in-order traversal", type="operation"),
                _concept("inorder traversal", type="operation"),
            ],
        )
    ]
    kn = merge(results, embedder=emb, threshold=0.5, model="m")
    # One concept for both spellings (equal counts: the shorter name wins).
    assert sorted(c.name for c in kn.concepts) == [
        "inorder traversal",
        "left rotate",
        "right rotate",
    ]


def test_merge_drops_relations_made_reflexive_by_merging() -> None:
    results = [
        _result(
            0,
            [_concept("BST"), _concept("binary search tree", ["BST"])],
            [
                {
                    "subject": "BST",
                    "predicate": "is_a",
                    "object": "binary search tree",
                    "confidence": 0.9,
                }
            ],
        )
    ]
    # Within one chunk, clean() already maps the alias to the same concept.
    assert merge(results, model="m").relations == []


# -- gold and scoring ---------------------------------------------------------------------


def test_real_gold_files_are_valid() -> None:
    golds = load_gold_dir(GOLD_DIR)
    assert len(golds) == 3
    for g in golds:
        assert len(g.concepts) >= 15 and len(g.relations) >= 10
        assert any(c.core for c in g.concepts)


def _gold(**over) -> dict:
    g = {
        "source_id": "s1",
        "title": "T",
        "concepts": [
            {
                "name": "binary search tree",
                "aliases": ["BST"],
                "type": "data_structure",
                "core": True,
            },
            {"name": "height", "type": "property", "core": True},
            {"name": "heap", "type": "data_structure"},
        ],
        "relations": [
            {"subject": "binary search tree", "predicate": "has_property", "object": "height"},
            {"subject": "binary search tree", "predicate": "contrasts_with", "object": "heap"},
        ],
    }
    g.update(over)
    return g


def test_gold_validation() -> None:
    Gold.model_validate(_gold())
    with pytest.raises(ValueError, match="not a gold concept"):
        Gold.model_validate(
            _gold(relations=[{"subject": "binary search tree", "predicate": "uses", "object": "x"}])
        )
    with pytest.raises(ValueError, match="unknown predicate"):
        Gold.model_validate(
            _gold(
                relations=[
                    {"subject": "binary search tree", "predicate": "likes", "object": "heap"}
                ]
            )
        )
    bad = _gold()
    bad["concepts"][2]["aliases"] = ["BSTs"]  # normalizes to "bst", owned by another concept
    with pytest.raises(ValueError, match="belongs to both"):
        Gold.model_validate(bad)


def test_score_source() -> None:
    gold = Gold.model_validate(_gold())
    results = [
        _result(
            0,
            [_concept("BSTs"), _concept("tree height", type="property"), _concept("node")],
            [
                {
                    "subject": "BSTs",
                    "predicate": "has_property",
                    "object": "tree height",
                    "confidence": 0.9,
                },
                {"subject": "node", "predicate": "part_of", "object": "BSTs", "confidence": 0.9},
            ],
        ),
        _result(
            1,
            [_concept("binary search tree", type="algorithm"), _concept("heap")],
            [
                {
                    "subject": "heap",
                    "predicate": "contrasts_with",
                    "object": "binary search tree",
                    "confidence": 0.9,
                }
            ],
        ),
        _result(0, [_concept("unrelated")], sid="other"),
    ]
    # Types differ, so "BSTs"/"binary search tree" stay apart: one is a duplicate.
    kn = merge(results, model="m")
    v = [1.0] + [0.0] * 63
    emb = FakeEmbedder({"tree height": v, "height": v})
    s = score_source(kn, gold, embedder=emb, lenient_threshold=0.85)
    assert s.concepts.n_pred == 5  # "unrelated" is another source's
    assert s.concepts.tp == 2 and s.duplicates == 1
    assert sorted(s.false_positives) == ["node", "tree height"]
    assert s.missed == ["height"]
    assert s.core_recall.recall == 0.5
    assert s.concepts_lenient.tp == 3  # "tree height" ~ "height"
    # Only the heap relation matches strictly; the reversed pair still counts as any-pred.
    assert (s.relations.tp, s.relations.n_pred, s.relations.n_gold) == (0, 3, 2)
    assert s.relations_any_predicate.tp == 1
    # Grounded = both ends are gold concepts: only heap -> bst, which is reversed, so no hit.
    assert (s.relations_grounded.tp, s.relations_grounded.n_pred) == (0, 1)

    # Support filter: only concepts mentioned in >= 2 chunks of s1 count. None are.
    s2 = score_source(kn, gold, min_chunks=2)
    assert s2.concepts.n_pred == 0 and s2.relations.n_pred == 0


def test_merge_quality_counts_pairs_within_a_source() -> None:
    gold = Gold.model_validate(_gold())
    results = [
        _result(0, [_concept("BST"), _concept("heap")]),
        _result(1, [_concept("binary search tree"), _concept("heaps")]),
    ]
    # Names-only: "BST" stays apart from "binary search tree" (no alias given), heaps merge.
    q = merge_quality(merge(results, model="m"), [gold])
    # Labelled mentions: BST, heap, binary search tree, heaps -> gold bst, heap, bst, heap.
    # Should-merge pairs: (BST, bst) and (heap, heaps) = 2; merged pairs: only the heap pair.
    assert (q.tp, q.n_pred, q.n_gold) == (1, 1, 2)
    v = [1.0] + [0.0] * 63
    emb = FakeEmbedder({"bst": v, "binary search tree": v, "heap": [0.0, 1.0] + [0.0] * 62})
    q2 = merge_quality(merge(results, embedder=emb, threshold=0.9, model="m"), [gold])
    assert (q2.tp, q2.n_pred, q2.n_gold) == (2, 2, 2)


def test_prf() -> None:
    p = PRF(2, 4, 5) + PRF(1, 1, 1)
    assert (p.precision, p.recall) == (0.6, 0.5)
    assert round(p.f1, 4) == round(2 * 0.6 * 0.5 / 1.1, 4)
    assert PRF().f1 == 0.0


# -- Ollama clients (HTTP mocked) ---------------------------------------------------------


def test_ollama_structured_sends_schema_and_parses() -> None:
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps({"concepts": []})},
                "prompt_eval_count": 7,
                "eval_count": 3,
                "total_duration": 2_500_000_000,
            },
        )

    llm = OllamaStructured("qwen3:8b", transport=httpx.MockTransport(handler))
    r = llm.chat_json("sys", "user", {"type": "object"})
    assert r.data == {"concepts": []} and r.prompt_tokens == 7 and r.seconds == 2.5
    assert seen[0]["format"] == {"type": "object"} and seen[0]["think"] is False
    assert seen[0]["options"]["num_ctx"] == 8192

    # Non-qwen3 models reject the "think" field, so it isn't sent.
    OllamaStructured("llama3.1:8b", transport=httpx.MockTransport(handler)).chat_json("s", "u", {})
    assert "think" not in seen[1]


def test_ollama_structured_bad_json_and_http_error() -> None:
    cut = httpx.MockTransport(lambda r: httpx.Response(200, json={"message": {"content": '{"a'}}))
    with pytest.raises(LLMError, match="not valid JSON"):
        OllamaStructured("m", transport=cut).chat_json("s", "u", {})
    err = httpx.MockTransport(lambda r: httpx.Response(500, json={"error": "no model"}))
    with pytest.raises(LLMError, match="no model"):
        OllamaStructured("m", transport=err).chat_json("s", "u", {})


def test_ollama_embedder_batches() -> None:
    sizes = []

    def handler(req):
        n = len(json.loads(req.content)["input"])
        sizes.append(n)
        return httpx.Response(200, json={"embeddings": [[1.0]] * n})

    e = OllamaEmbedder(transport=httpx.MockTransport(handler))
    assert len(e.embed([str(i) for i in range(70)])) == 70
    assert sizes == [64, 6]
    assert e.embed([]) == []


# -- CLI end to end, with the LLM and embedder replaced ---------------------------------


def test_cli_extract_concepts_and_eval(tmp_path: Path, monkeypatch, capsys) -> None:
    doc = tmp_path / "trees.txt"
    doc.write_text("Binary search trees keep keys ordered.\n\nHeaps are not ordered.\n")
    out = tmp_path / "data"
    sid = ingest(doc, out=out).source.id
    reply = {
        "concepts": [_concept("binary search tree", ["BST"]), _concept("heap")],
        "relations": [
            {"subject": "BST", "predicate": "contrasts_with", "object": "heap", "confidence": 0.9}
        ],
        "facts": [],
    }

    class FakeLLM(ScriptedLLM):
        def __init__(self, model, *a, **k):
            super().__init__([reply] * 10)
            self.model = model

        def unload(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("domaingraph.llm.OllamaStructured", FakeLLM)
    monkeypatch.setattr("domaingraph.llm.OllamaEmbedder", lambda *a, **k: FakeEmbedder())
    assert main(["--out", str(out), "extract", "--model", "fake"]) == 0
    assert "2 concepts" in capsys.readouterr().out
    assert (out / "knowledge" / "fake.json").is_file()

    assert main(["--out", str(out), "concepts", "--model", "fake"]) == 0
    assert "binary search tree  (aka BST)" in capsys.readouterr().out

    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    g = _gold(source_id=sid)
    (gold_dir / "t.yaml").write_text(yaml.safe_dump(g))
    report = tmp_path / "r.md"
    rc = main(
        [
            "--out",
            str(out),
            "eval-extract",
            "--models",
            "fake",
            "--gold",
            str(gold_dir),
            "--report",
            str(report),
        ]
    )
    assert rc == 0
    text = report.read_text()
    # 2 of 2 extracted match; 2 of 3 gold found; the contrasts_with relation is right.
    assert "| fake | 100% / 67% / 80% |" in text
    assert "100% / 50%" in text  # relations P / R
