"""D2: extract concepts, relations and facts from chunks with a local LLM.

One structured-output call per chunk. The model's reply is constrained to the JSON schema of
:class:`ChunkExtraction` (Ollama ``format``), validated with Pydantic, cleaned, and stored
with a pointer back to the chunk and its time range / page / heading.

Output: ``data/extractions/<source id>/<model>.jsonl``, one :class:`ChunkResult` per line.
Re-running skips chunks already extracted with the same model and prompt version, so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from domaingraph.llm import LLMError, StructuredLLM
from domaingraph.models import Chunk, Source

PROMPT_VERSION = 2

# Caps per chunk, enforced by Ollama's grammar through ``maxItems``. Without them a model can
# list dozens of loosely related terms and run out of tokens mid-JSON (seen on qwen3:14b).
MAX_CONCEPTS, MAX_RELATIONS, MAX_FACTS = 12, 12, 8

ConceptType = Literal[
    "data_structure",
    "algorithm",
    "operation",
    "property",
    "complexity",
    "problem",
    "technique",
    "other",
]
Predicate = Literal["is_a", "part_of", "uses", "solves", "has_property", "contrasts_with"]
PREDICATES: tuple[str, ...] = Predicate.__args__  # type: ignore[attr-defined]
CONCEPT_TYPES: tuple[str, ...] = ConceptType.__args__  # type: ignore[attr-defined]


# --- what the model returns (this schema is sent to Ollama as ``format``) ----------------


class ExtractedConcept(BaseModel):
    name: str = Field(description="Canonical name, singular, lowercase unless a proper noun")
    aliases: list[str] = Field(default_factory=list, description="Other names used in the text")
    type: ConceptType
    definition: str = Field(description="One sentence, based on what the text says")
    confidence: float = Field(ge=0, le=1)


class ExtractedRelation(BaseModel):
    subject: str
    predicate: Predicate
    object: str
    confidence: float = Field(ge=0, le=1)


class ExtractedFact(BaseModel):
    statement: str = Field(description="A self-contained claim the text makes")
    concepts: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class ChunkExtraction(BaseModel):
    concepts: list[ExtractedConcept] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    facts: list[ExtractedFact] = Field(default_factory=list)


def response_schema() -> dict:
    """The JSON schema sent as Ollama ``format``: :class:`ChunkExtraction` plus item caps.
    The caps are not on the Pydantic model, so stored results with more items still load."""
    schema = ChunkExtraction.model_json_schema()
    for key, n in (("concepts", MAX_CONCEPTS), ("relations", MAX_RELATIONS), ("facts", MAX_FACTS)):
        schema["properties"][key]["maxItems"] = n
    return schema


# --- what we store ----------------------------------------------------------------------


class ChunkResult(BaseModel):
    chunk_id: str
    source_id: str
    index: int
    locator: str
    start: float | None = None
    end: float | None = None
    page_start: int | None = None
    model: str
    prompt_version: int = PROMPT_VERSION
    extraction: ChunkExtraction = Field(default_factory=ChunkExtraction)
    error: str | None = None  # set when the model's reply could not be used
    dropped: dict[str, int] = Field(default_factory=dict)  # what cleaning removed, by reason
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    @field_validator("dropped")
    @classmethod
    def _no_zero(cls, v: dict[str, int]) -> dict[str, int]:
        return {k: n for k, n in v.items() if n}


SYSTEM_PROMPT = """\
You build a study glossary and knowledge graph from lecture material.
From the passage, extract:

concepts: the specific technical ideas the passage teaches or substantially uses: named
  data structures, algorithms, operations, properties/invariants, complexity results,
  problems, techniques. At most 12, most important first. Only things a student would
  expect a glossary entry for. Skip: people, course logistics, example values, and
  generic words that are not a specific idea (node, pointer, value, key, tree, list,
  operation, data structure, time complexity, efficiency, constraint, n).
  Name each concept the way a textbook would: singular, lowercase unless a proper noun,
  no articles. Name operations by the operation itself ("insert", not "insert operation").
  Put abbreviations and other wordings used in the passage under aliases
  (e.g. name "binary search tree", aliases ["BST"]).
  definition: one sentence, from what the passage says.
relations: between two concepts you listed, using their exact names. Predicates:
  is_a (subtype or instance), part_of (component of), uses (relies on / built from),
  solves (algorithm or structure solves a problem), has_property (has a property,
  invariant or complexity), contrasts_with (the passage compares the two).
  Only relations the passage states or clearly shows; at most 12.
facts: up to 8 self-contained claims the passage makes about the concepts (e.g.
  "Insertion into a binary search tree takes O(h) time."), with the concept names they
  involve.

confidence: 0-1, how clearly the passage supports the item.
The passage is a speech transcript and may contain recognition errors; use the intended
technical term. If the passage teaches nothing technical, return empty lists."""


def build_prompt(source: Source, chunk: Chunk, known: Iterable[str] = ()) -> str:
    parts = [f"Source: {source.title} ({chunk.locator()})"]
    known = list(known)
    if known:
        # Earlier chunks' names, so the same idea gets the same name across the lecture.
        parts.append(
            "Concept names already used in this source (reuse when it is the same idea): "
            + ", ".join(known)
        )
    parts.append(f"Passage:\n{chunk.text}")
    return "\n\n".join(parts)


_WS = re.compile(r"\s+")
_PAREN = re.compile(r"^(?P<name>.+?)\s*\((?P<alias>[^()]+)\)\s*$")


def _clean_name(raw: str) -> tuple[str, list[str]]:
    """Trim and split "binary search tree (BST)" into a name and an alias."""
    s = _WS.sub(" ", raw).strip().strip(".,;:\"'`")
    m = _PAREN.match(s)
    if m:
        return m["name"].strip(), [m["alias"].strip()]
    return s, []


def clean(raw: ChunkExtraction) -> tuple[ChunkExtraction, dict[str, int]]:
    """Drop what can't be used: empty names, duplicates within the chunk, relations whose
    ends are not among the chunk's concepts, self-relations, empty facts."""
    dropped = {"empty": 0, "duplicate": 0, "dangling_relation": 0, "self_relation": 0}
    concepts: dict[str, ExtractedConcept] = {}
    lookup: dict[str, str] = {}  # any lowercased surface form -> concept name
    for c in raw.concepts:
        name, extra = _clean_name(c.name)
        if not name:
            dropped["empty"] += 1
            continue
        key = name.lower()
        if key in concepts:
            dropped["duplicate"] += 1
            continue
        aliases = []
        for a in [*extra, *c.aliases]:
            a, _ = _clean_name(a)
            if a and a.lower() != key and a.lower() not in (x.lower() for x in aliases):
                aliases.append(a)
        concepts[key] = c.model_copy(update={"name": name, "aliases": aliases})
        for form in [name, *aliases]:
            lookup.setdefault(form.lower(), name)

    relations, seen = [], set()
    for r in raw.relations:
        s = lookup.get(_clean_name(r.subject)[0].lower())
        o = lookup.get(_clean_name(r.object)[0].lower())
        if s is None or o is None:
            dropped["dangling_relation"] += 1
            continue
        if s == o:
            dropped["self_relation"] += 1
            continue
        k = (s, r.predicate, o)
        if k in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(k)
        relations.append(r.model_copy(update={"subject": s, "object": o}))

    facts = []
    for f in raw.facts:
        st = _WS.sub(" ", f.statement).strip()
        if not st:
            dropped["empty"] += 1
            continue
        names = [lookup[n.lower()] for n in f.concepts if n.lower() in lookup]
        facts.append(f.model_copy(update={"statement": st, "concepts": list(dict.fromkeys(names))}))

    return ChunkExtraction(
        concepts=list(concepts.values()), relations=relations, facts=facts
    ), dropped


def extract_chunk(
    llm: StructuredLLM, source: Source, chunk: Chunk, known: Iterable[str] = ()
) -> ChunkResult:
    result = ChunkResult(
        chunk_id=chunk.id,
        source_id=source.id,
        index=chunk.index,
        locator=chunk.locator(),
        start=chunk.start,
        end=chunk.end,
        page_start=chunk.page_start,
        model=llm.model,
    )
    schema = response_schema()
    try:
        reply = llm.chat_json(SYSTEM_PROMPT, build_prompt(source, chunk, known), schema)
    except LLMError as exc:
        result.error = str(exc)
        return result
    result.prompt_tokens = reply.prompt_tokens
    result.completion_tokens = reply.completion_tokens
    result.seconds = reply.seconds
    try:
        raw = ChunkExtraction.model_validate(reply.data)
    except ValidationError as exc:
        # The schema constrains decoding, so this is rare: e.g. confidence 1.5.
        result.error = (
            f"reply failed validation: {exc.error_count()} errors; {exc.errors()[0]['msg']}"
        )
        return result
    result.extraction, result.dropped = clean(raw)
    return result


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model)


def extraction_path(out: Path, source_id: str, model: str) -> Path:
    return out / "extractions" / source_id / f"{model_slug(model)}.jsonl"


def load_results(out: Path, source_id: str, model: str) -> list[ChunkResult]:
    p = extraction_path(out, source_id, model)
    if not p.is_file():
        return []
    rows = [
        ChunkResult.model_validate_json(x)
        for x in p.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    return sorted(rows, key=lambda r: r.index)


def extract_source(
    llm: StructuredLLM,
    source: Source,
    chunks: list[Chunk],
    *,
    out: Path,
    force: bool = False,
    known_names: bool = True,
    max_known: int = 60,
    on_chunk: Callable[[ChunkResult, bool], None] | None = None,
) -> list[ChunkResult]:
    """Extract every chunk in order, appending to the JSONL as it goes (so a crash loses at
    most one chunk). ``known_names`` feeds earlier concept names into later prompts."""
    path = extraction_path(out, source.id, llm.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, ChunkResult] = {}
    if not force:
        # Keep only usable results from this prompt version; failed chunks are retried.
        for r in load_results(out, source.id, llm.model):
            if r.prompt_version == PROMPT_VERSION and r.error is None:
                done[r.index] = r
    with path.open("w", encoding="utf-8") as f:
        for r in done.values():
            f.write(r.model_dump_json() + "\n")
        f.flush()
        known: dict[str, None] = {}
        for chunk in chunks:
            if chunk.index in done:
                r, reused = done[chunk.index], True
            else:
                names = list(known)[-max_known:] if known_names else []
                r, reused = extract_chunk(llm, source, chunk, names), False
                f.write(r.model_dump_json() + "\n")
                f.flush()
            for c in r.extraction.concepts:
                known[c.name] = None
            done[chunk.index] = r
            if on_chunk:
                on_chunk(r, reused)
    return sorted(done.values(), key=lambda r: r.index)
