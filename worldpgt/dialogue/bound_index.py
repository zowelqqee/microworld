"""Per-request entity index decorator carrying resolved dialogue bindings.

The resolver decides what "it" denotes *before* parsing; this decorator is
only the transport of that decision into the untouched semantic parser: for
one request, ``find_in_text`` also yields the bound spans and ``resolve``
also knows the bound surfaces. When a question has no reference slots no
``BoundSurfaceIndex`` is ever constructed, so the single-turn path through
the parser is byte-identical to today's.

A bound span may carry several canonicals ("they" → two entities): one row
per canonical is emitted at the same span, in resolver (salience) order.
"""

from __future__ import annotations

import re

from worldpgt.dialogue.resolver import BoundSpan

_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip().lower())


class BoundSurfaceIndex:
    """Duck-typed stand-in for ``EntitySurfaceIndex`` (resolve / entity_type /
    find_in_text / has_surface) plus ``is_bound_span`` for the parser's
    partial-title guard."""

    def __init__(
        self,
        inner,
        bindings: tuple[BoundSpan, ...],
        entity_type_hints: dict[str, str] | None = None,
    ) -> None:
        self._inner = inner
        self._bindings = bindings
        self._type_hints = dict(entity_type_hints or {})
        self._by_surface: dict[str, str] = {}
        for span in bindings:
            surface = _norm(span.surface)
            if surface and span.canonicals:
                self._by_surface.setdefault(surface, span.canonicals[0])
        self._bound_spans = {(b.start, b.end) for b in bindings}

    # ── EntitySurfaceIndex interface ────────────────────────────────────────

    def resolve(self, surface: str):
        bound = self._by_surface.get(_norm(surface))
        if bound is not None:
            return bound
        return self._inner.resolve(surface)

    def entity_type(self, canonical: str):
        inner_type = self._inner.entity_type(canonical)
        if inner_type is not None:
            return inner_type
        return self._type_hints.get(canonical)

    def find_in_text(self, text: str) -> list[tuple[str, str, int, int]]:
        found = list(self._inner.find_in_text(text))
        for span in self._bindings:
            for canonical in span.canonicals:
                found.append((span.surface, canonical, span.start, span.end))
        found.sort(key=lambda row: (row[2], row[3]))
        return found

    def has_surface(self, text: str) -> bool:
        low = _norm(text)
        if any(surface in low for surface in self._by_surface):
            return True
        return self._inner.has_surface(text)

    def all_surfaces(self):
        return self._inner.all_surfaces()

    def known_surfaces_set(self):
        return self._inner.known_surfaces_set()

    # ── Parser guard ────────────────────────────────────────────────────────

    def is_bound_span(self, start: int, end: int) -> bool:
        return (start, end) in self._bound_spans
