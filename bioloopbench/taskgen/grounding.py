"""Deterministic checks tying generated text to, or away from, the source."""

from __future__ import annotations

import re
import unicodedata

_PUNCT = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                        "—": "-", "−": "-", " ": " ", "μ": "u", "µ": "u"})


def normalise(text):
    text = unicodedata.normalize("NFKC", text).translate(_PUNCT).lower()
    return " ".join(text.split())


def quote_in_source(quote, source):
    """True if ``quote`` appears verbatim in ``source`` (case, spacing and
    typographic punctuation ignored). An ellipsis splits the quote into parts
    that must each appear, in order."""
    src = normalise(source)
    parts = [p.strip(" .") for p in re.split(r"\.\.\.|…|\[\.\.\.\]", normalise(quote))]
    parts = [p for p in parts if p]
    if not parts or sum(len(p) for p in parts) < 12:
        return False
    pos = 0
    for p in parts:
        i = src.find(p, pos)
        if i < 0:
            return False
        pos = i + len(p)
    return True


def _words(text):
    return re.findall(r"[a-z0-9]+", normalise(text))


def longest_shared_run(text, source):
    """Length in words of the longest word sequence shared by both texts."""
    a, b = _words(text), _words(source)
    if not a or not b:
        return 0
    index = {}
    for j, w in enumerate(b):
        index.setdefault(w, []).append(j)
    best, prev = 0, {}
    for w in a:
        cur = {}
        for j in index.get(w, ()):
            cur[j] = prev.get(j - 1, 0) + 1
            if cur[j] > best:
                best = cur[j]
        prev = cur
    return best
