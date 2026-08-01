"""Deterministic normalizer for elided cross-reference citations.

Statutes write reference lists elliptically: the governing word appears once and
is understood to distribute over the whole comma/or list —

    "a right of priority under section 119, 365(a), 365(b), 386(a), or 386(b)"

so only ``119`` keeps the word ``section``; ``365(a)`` and the rest are emitted
as bare fragments that no downstream resolver can key on.  Both legal-domain
pilots hit this exact failure (v1 §102(d)(2), v2 §878(a)/(b)).

This module recovers the elided governing word.  It is **fully dynamic**: it
does not know any section numbers, titles, or a fixed citation vocabulary
beyond the small closed set of English subdivision nouns the U.S. Code itself
uses as governing words.  The governing word for any bare reference is read out
of the evidence text that produced it, never from a lookup table.  A surface
that is not a bare reference sitting in a governed list is returned unchanged,
so ordinary (non-citation) nodes are never touched.

No ML, no network, no overlay writes.
"""

from __future__ import annotations

import re

# The closed set of English nouns the U.S. Code uses to introduce a numbered
# subdivision.  This is grammar, not domain knowledge: it is the same list a
# reader uses to expand "section 119, 365(a)" into two section citations.
_GOVERNING_WORDS = (
    "subparagraph",
    "subsection",
    "paragraph",
    "section",
    "clause",
    "chapter",
    "title",
    "subdivision",
    "subchapter",
    "part",
    "item",
    "rule",
)
_GOVERNING_ALT = "|".join(_GOVERNING_WORDS)

# A single reference token: a number optionally followed by parenthesized
# subdivisions, e.g. "119", "365(a)", "3056(a)(6)", "46501(2)".
_REF = r"\d+(?:\([A-Za-z0-9]+\))*"
_BARE_REF_RE = re.compile(rf"^{_REF}$")
_LEADING_GOVERNING_RE = re.compile(rf"^(?:{_GOVERNING_ALT})s?\b", re.IGNORECASE)

# A governed list: a governing word, then one or more reference tokens joined by
# commas / "or" / "and".  The governing word is captured so it can be
# distributed over every following reference.
_GOVERNED_LIST_RE = re.compile(
    rf"\b({_GOVERNING_ALT})s?\s+"
    rf"(?P<list>{_REF}(?:\s*(?:,|,?\s+or|,?\s+and)\s+{_REF})+)",
    re.IGNORECASE,
)
_REF_TOKEN_RE = re.compile(_REF)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def is_bare_reference(surface: str) -> bool:
    """True if ``surface`` is a numbered reference with no governing word."""
    return bool(_BARE_REF_RE.match(_normalise(surface)))


def governing_word_for(surface: str, evidence_sentence: str) -> str | None:
    """Return the governing word that the evidence applies to ``surface``.

    ``surface`` must be a bare reference (see :func:`is_bare_reference`).  The
    evidence is scanned for a governed list that literally contains it; the
    governing word is taken from that list.  Returns ``None`` when the surface
    is not a bare reference or does not occur in any governed list, so the
    caller can leave it untouched.
    """
    surface = _normalise(surface)
    if not is_bare_reference(surface):
        return None
    for match in _GOVERNED_LIST_RE.finditer(evidence_sentence or ""):
        refs = _REF_TOKEN_RE.findall(match.group("list"))
        if surface in refs:
            return match.group(1).lower()
    return None


def normalize_citation_surface(surface: str, evidence_sentence: str) -> tuple[str, bool]:
    """Recover an elided governing word for a bare reference surface.

    Returns ``(normalized_surface, changed)``.  ``changed`` is ``False`` and the
    surface is returned unmodified whenever it is already governed (starts with a
    governing word) or is not a bare reference found in a governed list — i.e.
    the normalizer is a no-op for everything that is not the elided-citation
    case it exists to fix.
    """
    clean = _normalise(surface)
    if not clean:
        return surface, False
    if _LEADING_GOVERNING_RE.match(clean):
        return clean, False
    governing = governing_word_for(clean, evidence_sentence)
    if governing is None:
        return surface, False
    return f"{governing} {clean}", True
