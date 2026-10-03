"""D6 tests: the Obsidian exporter on hand-built graph data (no database)."""

from __future__ import annotations

import json

import yaml

from domaingraph.export import BEGIN, END, ExportData, VaultExporter, _split, safe_name

URL = "https://www.youtube.com/watch?v=abc"


def _data(**over) -> ExportData:
    d = {
        "concepts": [
            {"id": "radix-sort", "name": "radix sort", "aliases": ["LSD radix sort"],
             "type": "algorithm", "definition": "Sorts integers digit by digit.",
             "confidence": 0.9, "n_mentions": 5, "n_chunks": 3, "domains": ["algorithms"]},
            {"id": "counting-sort", "name": "counting sort", "aliases": [], "type": "algorithm",
             "definition": "Counts keys.", "confidence": 0.9, "n_mentions": 4, "n_chunks": 2,
             "domains": ["algorithms"]},
            {"id": "stability", "name": "stable sort", "aliases": [], "type": "property",
             "definition": "", "confidence": 0.8, "n_mentions": 1, "n_chunks": 1,
             "domains": ["algorithms"]},
            {"id": "o-n", "name": "O(n): linear?", "aliases": [], "type": "complexity",
             "definition": "", "confidence": 0.8, "n_mentions": 1, "n_chunks": 1,
             "domains": ["algorithms"]},
        ],
        "sources": [
            {"id": "s7", "title": "Lecture 7: Radix Sort", "kind": "audio", "url": URL,
             "duration": 3000.0, "pages": None, "domains": ["algorithms"]},
            {"id": "s9", "title": "Notes", "kind": "markdown", "url": None, "duration": None,
             "pages": None, "domains": ["algorithms"]},
        ],
        "relations": [
            {"subject": "radix-sort", "predicate": "uses", "object": "counting-sort", "n": 2},
            {"subject": "radix-sort", "predicate": "uses", "object": "stability", "n": 1},
            {"subject": "counting-sort", "predicate": "has_property", "object": "stability",
             "n": 1},
            {"subject": "radix-sort", "predicate": "has_property", "object": "o-n", "n": 1},
        ],
        "mentions": [
            {"concept": "radix-sort", "source": "s7", "chunk": 39, "start": 2727.0,
             "at": "45:27-46:50", "surfaces": ["radix sort"]},
            {"concept": "radix-sort", "source": "s7", "chunk": 38, "start": 2642.0,
             "at": "44:02-45:38", "surfaces": ["radix sort"]},
            {"concept": "counting-sort", "source": "s7", "chunk": 31, "start": 2202.0,
             "at": "36:42-38:11", "surfaces": []},
            {"concept": "counting-sort", "source": "s9", "chunk": 0, "start": None,
             "at": "Sorting > Counting", "surfaces": []},
            {"concept": "stability", "source": "s7", "chunk": 41, "start": 2900.0,
             "at": "48:20-49:30", "surfaces": []},
        ],
        "facts": [
            {"concept": "radix-sort", "statement": "Radix sort runs in linear time for small keys.",
             "source": "s7", "start": 3068.0, "at": "51:08-52:07"},
        ],
    }  # fmt: skip
    d.update(over)
    return ExportData(**d)


def _read(p):
    text = p.read_text(encoding="utf-8")
    return _split(text)[0], text


def test_concept_note_shape(tmp_path):
    st = VaultExporter(tmp_path).export(_data())
    assert (st.created, st.updated, st.unchanged) == (6, 0, 0)  # 4 concepts + 2 sources
    meta, text = _read(tmp_path / "Concepts" / "radix sort.md")
    assert meta["domaingraph_id"] == "radix-sort"
    assert meta["domain"] == "algorithms" and meta["type"] == "algorithm"
    assert meta["confidence"] == 0.9 and meta["aliases"] == ["LSD radix sort"]
    assert meta["tags"] == ["type/algorithm", "domain/algorithms"]
    assert meta["sources"] == ["Lecture 7: Radix Sort"]
    assert "- **Uses:** [[counting sort]], [[stable sort]]" in text  # most-mentioned first
    # Times in lecture order, each a link to that second of the video.
    assert (
        f"- [[Lecture 7 Radix Sort]]: [44:02-45:38]({URL}&t=2642s), [45:27-46:50]({URL}&t=2727s)"
    ) in text
    assert "Radix sort runs in linear time for small keys. ([[Lecture 7 Radix Sort]]" in text


def test_inverse_relations_and_sources_without_urls(tmp_path):
    VaultExporter(tmp_path).export(_data())
    _, text = _read(tmp_path / "Concepts" / "stable sort.md")
    assert "- **Used by:** [[radix sort]]" in text
    assert "- **Property of:** [[counting sort]]" in text
    _, cs = _read(tmp_path / "Concepts" / "counting sort.md")
    assert "- [[Notes]]: Sorting > Counting" in cs  # no URL, no seconds: plain locator


def test_file_names_are_safe_and_links_use_them(tmp_path):
    assert safe_name("a/b: c?") == "a b c"
    VaultExporter(tmp_path).export(_data())
    assert (tmp_path / "Concepts" / "O(n) linear.md").is_file()
    _, src = _read(tmp_path / "Sources" / "Lecture 7 Radix Sort.md")
    _, radix = _read(tmp_path / "Concepts" / "radix sort.md")
    assert "[[O(n) linear|O(n): linear?]]" in radix  # alias link where the name was changed
    assert src.index("[[counting sort]]") < src.index("[[radix sort]]")  # in lecture order
    assert f"[Watch]({URL})" in src


def test_rerun_is_idempotent_and_keeps_user_text(tmp_path):
    VaultExporter(tmp_path).export(_data())
    note = tmp_path / "Concepts" / "radix sort.md"
    text = note.read_text(encoding="utf-8")
    meta_end = text.index("\n---\n", 4) + 5
    edited = (
        text[:4]
        + "rating: 5\n"
        + text[4:meta_end]
        + "My own intro.\n\n"
        + text[meta_end:]
        + "\n## My notes\n\nRemember the spreadsheet trick.\n"
    )
    note.write_text(edited, encoding="utf-8")
    st = VaultExporter(tmp_path).export(_data())
    assert st.unchanged == 6 - 1 and st.updated == 1  # only the edited note is rewritten...
    again = note.read_text(encoding="utf-8")
    assert "My own intro." in again and "Remember the spreadsheet trick." in again
    assert yaml.safe_load(again.split("---\n")[1])["rating"] == 5
    st2 = VaultExporter(tmp_path).export(_data())
    assert st2.unchanged == 6  # ...and after that nothing changes at all
    assert again.count(BEGIN) == 1 and again.count(END) == 1


def test_graph_changes_update_in_place_even_after_rename(tmp_path):
    VaultExporter(tmp_path).export(_data())
    (tmp_path / "Concepts" / "radix sort.md").rename(tmp_path / "Concepts" / "Radix.md")
    d = _data()
    d.concepts[0]["definition"] = "Sorts integers one digit at a time."
    st = VaultExporter(tmp_path).export(d)
    assert not (tmp_path / "Concepts" / "radix sort.md").exists()
    assert "one digit at a time" in (tmp_path / "Concepts" / "Radix.md").read_text(encoding="utf-8")
    assert st.updated >= 1
    # Links elsewhere follow the note's new name.
    assert "[[Radix|radix sort]]" in (tmp_path / "Concepts" / "stable sort.md").read_text(
        encoding="utf-8"
    )


def test_removed_concepts_are_marked_not_deleted(tmp_path):
    VaultExporter(tmp_path).export(_data())
    d = _data()
    d.concepts = [c for c in d.concepts if c["id"] != "o-n"]
    st = VaultExporter(tmp_path).export(d)
    p = tmp_path / "Concepts" / "O(n) linear.md"
    assert st.removed == 1 and p.is_file()
    meta, _ = _read(p)
    assert meta["domaingraph_status"] == "removed"
    assert VaultExporter(tmp_path).export(d).removed == 0  # marked once
    st = VaultExporter(tmp_path, prune=True).export(d)
    assert st.deleted == 1 and not p.exists()


def test_prune_keeps_notes_with_user_text(tmp_path):
    VaultExporter(tmp_path).export(_data())
    p = tmp_path / "Concepts" / "O(n) linear.md"
    p.write_text(p.read_text(encoding="utf-8") + "\nMy thoughts.\n", encoding="utf-8")
    d = _data()
    d.concepts = [c for c in d.concepts if c["id"] != "o-n"]
    st = VaultExporter(tmp_path, prune=True).export(d)
    assert st.deleted == 0 and st.removed == 1 and p.is_file()


def test_min_chunks_filters_and_drops_dangling_links(tmp_path):
    st = VaultExporter(tmp_path, min_chunks=2).export(_data())
    assert st.skipped == 2
    assert not (tmp_path / "Concepts" / "stable sort.md").exists()
    _, text = _read(tmp_path / "Concepts" / "radix sort.md")
    assert "[[stable sort]]" not in text and "[[counting sort]]" in text


def test_graph_view_colours_written_once(tmp_path):
    VaultExporter(tmp_path).export(_data())
    cfg = tmp_path / ".obsidian" / "graph.json"
    groups = json.loads(cfg.read_text(encoding="utf-8"))["colorGroups"]
    assert {"query": "tag:#type/algorithm", "color": {"a": 1, "rgb": 0x4E79A7}} in groups
    cfg.write_text("{}", encoding="utf-8")  # the user's own settings win
    VaultExporter(tmp_path).export(_data())
    assert cfg.read_text(encoding="utf-8") == "{}"
