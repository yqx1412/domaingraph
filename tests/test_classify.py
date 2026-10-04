import json
from pathlib import Path

import pytest
import yaml

from domaingraph.classify import (
    DOMAINS,
    Example,
    TfidfLogReg,
    ZeroShotEmbed,
    ZeroShotLLM,
    clean_text,
    cluster_bootstrap,
    lecture_vote,
    load_corpus,
    macro_f1,
    short_text,
)
from domaingraph.llm import LLMError, StructuredReply

MANIFEST = Path(__file__).parents[1] / "benchmarks" / "classify" / "corpus.yaml"


def test_clean_text_strips_ocw_notice_and_course_numbers():
    raw = (
        "The following content is provided under a Creative Commons license. Your support "
        "will help MIT OpenCourseWare continue to offer high quality educational resources "
        "for free. To make a donation or view additional materials from hundreds of MIT "
        "courses, visit MIT OpenCourseWare at ocw.mit.edu. Today in 006 we do hashing,\n"
        "unlike 6.046 or 18.06."
    )
    assert clean_text(raw) == (
        "Today in this course we do hashing, unlike this course or this course."
    )


def test_short_text():
    assert short_text("a b c d e", 3) == "a b c"


def test_real_manifest_is_consistent():
    m = yaml.safe_load(MANIFEST.read_text("utf-8"))
    ids = [v for c in m["courses"] for v in c["lectures"] if v]
    assert len(ids) == len(set(ids))
    splits = m["splits"]
    assert not set(splits["train"]) & set(splits["test"])
    assert not set(splits["dev"]) & (set(splits["train"]) | set(splits["test"]))
    assert set(m.get("courses")[0]) >= {"course", "domain", "lectures"}
    assert {c["domain"] for c in m["courses"]} == set(DOMAINS)


def _source(root: Path, sid: str, vid: str, texts: list[str]) -> None:
    d = root / "sources" / sid
    d.mkdir(parents=True)
    (d / "source.json").write_text(
        json.dumps({"id": sid, "path": f"x/{vid}.m4a", "url": f"https://y/watch?v={vid}"})
    )
    (d / "chunks.jsonl").write_text(
        "\n".join(json.dumps({"index": i, "text": t}) for i, t in enumerate(texts))
    )


def test_load_corpus_joins_by_video_id_and_splits_by_lecture(tmp_path):
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "splits": {"train": [1], "dev": [2], "test": [3]},
                "courses": [
                    {
                        "course": "A",
                        "domain": "algorithms",
                        "lectures": ["aaaaaaaaaa1", None, "aaaaaaaaaa3"],
                    },
                    {
                        "course": "B",
                        "domain": "algorithms",
                        "split": "ooc",
                        "lectures": ["bbbbbbbbbb1"],
                    },
                ],
            }
        )
    )
    long = "one two three four five six"
    _source(tmp_path / "r1", "s1", "aaaaaaaaaa1", [long, "too short"])
    _source(tmp_path / "r2", "s3", "aaaaaaaaaa3", [long])
    _source(tmp_path / "r2", "s4", "bbbbbbbbbb1", [long])
    _source(tmp_path / "r2", "s5", "zzzzzzzzzzz", [long])  # not in the manifest: ignored
    ex = load_corpus(manifest, [tmp_path / "r1", tmp_path / "r2"])
    assert [(e.course, e.lecture, e.split) for e in ex] == [
        ("A", 1, "train"),
        ("A", 3, "test"),
        ("B", 1, "ooc"),
    ]


def test_load_corpus_reports_missing_lectures(tmp_path):
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "splits": {"train": [1]},
                "courses": [{"course": "A", "domain": "algorithms", "lectures": ["aaaaaaaaaa1"]}],
            }
        )
    )
    (tmp_path / "sources").mkdir()
    with pytest.raises(FileNotFoundError, match="not ingested"):
        load_corpus(manifest, [tmp_path])


def test_tfidf_learns_a_toy_split():
    x = ["matrix vector rank"] * 3 + ["raft leader replica"] * 3
    y = ["linear_algebra"] * 3 + ["distributed_systems"] * 3
    m = TfidfLogReg()
    m.fit(x, y)
    assert m.predict(["the rank of a matrix", "raft elects a leader"]) == [
        "linear_algebra",
        "distributed_systems",
    ]


class FakeEmbedder:
    model = "fake"

    def embed(self, texts):
        # one axis per domain, keyed on a marker word
        keys = ["sort", "matrix", "search", "proof", "replica"]
        return [[1.0 if k in t else 0.0 for k in keys] for t in texts]


def test_zeroshot_embed_picks_nearest_description():
    descs = {
        d: k for d, k in zip(DOMAINS, ["sort", "matrix", "search", "proof", "replica"], strict=True)
    }
    m = ZeroShotEmbed(FakeEmbedder(), descs)
    assert m.predict(["a replica", "a proof"]) == ["distributed_systems", "discrete_math"]


class FakeLLM:
    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.systems = []

    def chat_json(self, system, user, schema):
        self.systems.append(system)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return StructuredReply(data=r)


def test_zeroshot_llm_counts_invalid_replies():
    llm = FakeLLM([{"domain": "algorithms"}, {"domain": "cooking"}, LLMError("x"), {}])
    m = ZeroShotLLM(llm)
    assert m.predict(["a", "b", "c", "d"]) == ["algorithms"] + ["<invalid>"] * 3
    assert m.failures == 3
    assert all(f"- {d}:" in llm.systems[0] for d in DOMAINS)


def test_macro_f1_and_bootstrap():
    gold = ["a", "a", "b", "b"]
    assert macro_f1(gold, gold) == 1.0
    assert macro_f1(gold, ["a", "a", "a", "a"]) == pytest.approx((2 / 3 + 0) / 2)
    lo, hi = cluster_bootstrap(gold, ["a", "a", "a", "a"], ["g1", "g1", "g2", "g2"], n=500)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert cluster_bootstrap(gold, gold, ["g1", "g1", "g2", "g2"], n=50) == (1.0, 1.0)


def test_lecture_vote():
    ex = [
        Example("t", "a", "C", 1, "test"),
        Example("t", "a", "C", 1, "test"),
        Example("t", "a", "C", 1, "test"),
        Example("t", "b", "C", 2, "test"),
    ]
    assert lecture_vote(ex, ["a", "b", "a", "a"]) == 0.5
