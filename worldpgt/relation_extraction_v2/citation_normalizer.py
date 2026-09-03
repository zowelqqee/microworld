"""Deterministic normalizer for elided cross-reference citations.

Statutes write reference lists elliptically: the governing word appears once and
is understood to distribute over the whole comma/or list —

    "a right of priority under section 119, 365(a), 365(b), 386(a), or 386(b)"

so only ``119`` keeps the word ``section``; ``365(a)`` and the rest are emitted
as bare fragments that no downstream resolver can key on.  Both legal-domain
pilots hit this exact failure (v1 §102(d)(2), v2 §878(a)/(b)).

This module recovers the elided governing word **without knowing what that word
is in advance**.  It carries no vocabulary of citation nouns ("section",
"article", "rule", ...): a governed list is recognized purely by its shape — one
plain word immediately followed by two or more numeric references joined by
comma/or/and — and whatever word actually precedes it in the evidence text is
what gets distributed. This is the same inference a reader performs, and it
generalizes to any citation convention (EU "Article 5, 6, or 7", a UK "Rule 12
and 13", …) without a line of code changing.

No ML, no network, no overlay writes.
"""

from __future__ import annotations

import re

# A single reference token: a number optionally followed by parenthesized
# subdivisions, e.g. "119", "365(a)", "3056(a)(6)", "46501(2)".
_REF = r"\d+(?:\([A-Za-z0-9]+\))*"
_BARE_REF_RE = re.compile(rf"^{_REF}$")

# A governed list: one plain word, then two or more reference tokens joined by
# commas / "or" / "and".  The word is captured so it can be distributed over
# every following reference — it is read from the text, not matched against a
# fixed vocabulary, which is what lets this recognize any citation convention.
_GOVERNED_LIST_RE = re.compile(
    rf"\b([A-Za-z]+)\s+"
    rf"(?P<list>{_REF}(?:\s*(?:,|,?\s+or|,?\s+and)\s+{_REF}){{1,}})"
)
_REF_TOKEN_RE = re.compile(_REF)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def is_bare_reference(surface: str) -> bool:
    """True if ``surface`` is a numbered reference with no governing word."""
    return bool(_BARE_REF_RE.match(_normalise(surface)))


def governing_word_for(surface: str, evidence_sentence: str) -> str | None:
    """Return the governing word that the evidence applies to ``surface``.

    ``surface`` must be a bare reference (see :func:`is_bare_reference`) —
    that requirement alone is what makes an already-governed surface such as
    "section 119" a no-op, since it is not a bare reference to begin with. The
    evidence is scanned for a governed list that literally contains the bare
    surface; the governing word is whatever word actually precedes that list
    in the text. Returns ``None`` when the surface is not a bare reference or
    does not occur in any governed list, so the caller can leave it untouched.
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
    surface is returned unmodified whenever it is not a bare reference (already
    governed, or not a citation at all) or is not found in any governed list in
    the evidence — i.e. the normalizer is a no-op for everything that is not
    the elided-citation case it exists to fix.
    """
    clean = _normalise(surface)
    if not clean:
        return surface, False
    governing = governing_word_for(clean, evidence_sentence)
    if governing is None:
        return surface, False
    return f"{governing} {clean}", True
