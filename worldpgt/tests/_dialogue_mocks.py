"""Shared mock entity index / graph reader for dialogue-v2 tests.

Mimics the read interface of ``EntitySurfaceIndex`` over a tiny in-memory
world so dialogue tests are independent of overlay artifacts on disk.
"""

from __future__ import annotations

import re


class MockSurfaceIndex:
    """``entities`` maps canonical → (entity_type, [extra surfaces])."""

    def __init__(self, entities: dict[str, tuple[str | None, list[str]]]) -> None:
        self.entities = entities
        self._surface_to_canonical: dict[str, str] = {}
        for canonical, (_etype, surfaces) in entities.items():
            for surface in [canonical, *surfaces]:
                self._surface_to_canonical.setdefault(surface.lower(), canonical)
        ordered = sorted(self._surface_to_canonical, key=len, reverse=True)
        if ordered:
            self._combined_re = re.compile(
                r"(?<!\w)(?:" + "|".join(re.escape(s) for s in ordered) + r")(?!\w)",
                re.IGNORECASE,
            )
        else:
            self._combined_re = None

    def resolve(self, surface: str) -> str | None:
        normalized = re.sub(r"\s+", " ", (surface or "").strip().lower())
        return self._surface_to_canonical.get(normalized)

    def entity_type(self, canonical: str) -> str | None:
        entry = self.entities.get(canonical)
        return entry[0] if entry else None

    def find_in_text(self, text: str) -> list[tuple[str, str, int, int]]:
        if self._combined_re is None:
            return []
        out = []
        for m in self._combined_re.finditer(text):
            canonical = self._surface_to_canonical[m.group(0).lower()]
            out.append((m.group(0), canonical, m.start(), m.end()))
        return out

    def has_surface(self, text: str) -> bool:
        return bool(self.find_in_text(text))


class MockGraphReader:
    """``facts`` is a list of (subject, relation, object) triples."""

    def __init__(self, facts: list[tuple[str, str, str]]) -> None:
        self.facts = list(facts)

    def objects_for(self, subject: str, relation: str) -> tuple[str, ...]:
        return tuple(o for s, r, o in self.facts if s == subject and r == relation)

    def role_holders(self, anchor: str, relation: str) -> tuple[str, ...]:
        forward = self.objects_for(anchor, relation)
        reverse = tuple(s for s, r, o in self.facts if o == anchor and r == relation)
        return forward + reverse
