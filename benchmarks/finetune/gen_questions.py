"""Generate questions for the training lectures (1-4, 8, 9) and the dev lecture (10).
Usage: python gen_questions.py [--limit N]   (N chunks per lecture, for a smoke test)"""

import sys
from pathlib import Path

from domaingraph.llm import OllamaStructured
from domaingraph.pipeline import list_sources, load_source
from domaingraph.questions import generate_questions

TEST = {"1c28a3b4720dda55", "9ef0cbfe913412e8", "9ec4d790bec6602e"}  # Lectures 5, 6, 7
DEV = {"f0e05624d768b1e8"}  # Lecture 10
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

sources = [s.id for s in list_sources(Path("data")) if s.id not in TEST]
assert len(sources) == 7 and not set(sources) & TEST, sources
llm = OllamaStructured("qwen3:8b", num_predict=1024, timeout=120)
for split, ids in (("dev", [s for s in sources if s in DEV]),
                   ("train", [s for s in sources if s not in DEV])):  # fmt: skip
    chunks = []
    for sid in ids:
        chunks += load_source(Path("data"), sid)[1][:limit]
    out = Path(f"data/finetune/questions_{split}.jsonl")
    qs = generate_questions(llm, chunks, out, log=lambda m: print(m, flush=True))
    print(f"{split}: {len(qs)} questions from {len(chunks)} chunks -> {out}", flush=True)
llm.unload()
