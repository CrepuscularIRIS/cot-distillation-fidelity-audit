"""Unified trace schema + problem-text normalization.

A single dataclass that every dataset adapter maps into, so downstream pairing,
judging, and aggregation never touch raw per-dataset field layouts.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

_WS = re.compile(r"\s+")


def normalize_problem(text: str) -> str:
    """Canonicalize a problem statement for cross-dataset matching.

    NFKC folds unicode hyphen/space variants (e.g. U+2011 -> '-'), then we
    lowercase and collapse whitespace. Conservative on purpose: we do NOT strip
    punctuation, to avoid collapsing distinct prompts.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.lower()
    t = _WS.sub(" ", t).strip()
    return t


def problem_hash(text: str) -> str:
    """Stable short key for exact (normalized) problem matching."""
    return hashlib.sha1(normalize_problem(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Trace:
    """One reasoning trace, normalized across datasets."""

    dataset: str
    teacher: str
    problem: str
    reasoning: str  # content of the <thinking> block (or model_thoughts)
    answer: str  # text after </thinking> (or model_response)
    domain: str = ""
    difficulty: str = ""
    uid: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def reasoning_chars(self) -> int:
        return len(self.reasoning)
