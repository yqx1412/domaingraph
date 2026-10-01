"""Transcript accuracy: word error rate (WER) against a reference text.

WER = (substitutions + deletions + insertions) / words in the reference, after both texts
are normalized: lower case, punctuation removed, numbers written as words ("7" -> "seven",
"150" -> "one hundred fifty"), so "Lecture 7" and "lecture seven" count as equal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_words(n: int) -> str:
    if n < 0:
        return "minus " + number_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f" {_ONES[ones]}" if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_ONES[hundreds]} hundred" + (f" {number_words(rest)}" if rest else "")
    if n < 1_000_000:
        thousands, rest = divmod(n, 1000)
        return f"{number_words(thousands)} thousand" + (f" {number_words(rest)}" if rest else "")
    return str(n)


def normalize(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    text = re.sub(r"\d+", lambda m: f" {number_words(int(m.group()))} ", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return [w.strip("'") for w in text.split() if w.strip("'")]


@dataclass
class WerResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int


def word_error_rate(reference: str, hypothesis: str) -> WerResult:
    ref, hyp = normalize(reference), normalize(hypothesis)
    # Levenshtein over words, keeping the operation counts of the best path.
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]  # (cost, sub, del, ins)
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur.append(prev[j - 1])
                continue
            sub, dele, ins = prev[j - 1], prev[j], cur[j - 1]
            best = min(
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
                (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
            )
            cur.append(best)
        prev = cur
    cost, s, d, n_ins = prev[-1]
    n = len(ref)
    return WerResult(cost / n if n else 0.0, s, d, n_ins, n)
