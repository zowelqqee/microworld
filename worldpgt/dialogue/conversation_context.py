"""Per-session dialogue context for interactive Microworld QA.

The context is intentionally in-memory only. It records what the controlled QA
surface has already said in one interactive session so later questions can
resolve explicit references without changing accepted memory or overlays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from worldpgt.entity_qa.types import SemanticQuery

_log = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    question: str
    semantic_query: SemanticQuery
    decision: str
    primary_entity: str | None
    mentioned_entities: list[str]
    relation_type: str | None
    # Type hints for entities that aren't in the static overlay's entity
    # index (e.g. a person named only in a live web-search answer), keyed by
    # canonical name. Session-only — never written to the overlay.
    entity_types: dict[str, str] = field(default_factory=dict)


@dataclass
class ConversationContext:
    turns: list[ConversationTurn] = field(default_factory=list)

    def last_entity(self) -> str | None:
        for turn in reversed(self.turns):
            if turn.primary_entity:
                return turn.primary_entity
        return None

    def last_mentioned_entities(self) -> list[str]:
        for turn in reversed(self.turns):
            if turn.mentioned_entities:
                return list(turn.mentioned_entities)
        return []

    def last_relation(self) -> str | None:
        for turn in reversed(self.turns):
            if turn.relation_type:
                return turn.relation_type
        return None

    def append_turn(self, turn: "ConversationTurn") -> None:
        _log.debug(
            "context.append_turn #%d decision=%s primary=%r mentioned=%r",
            len(self.turns), turn.decision, turn.primary_entity, turn.mentioned_entities,
        )
        self.turns.append(turn)

    def entities_in_window(self, n: int = 3) -> list[str]:
        # Audit turns don't count toward n — confirmed entities stay in window
        # across any number of intervening audit turns.
        seen: set[str] = set()
        per_turn: list[list[str]] = []
        confirmed_count = 0
        for turn in reversed(self.turns):
            if turn.decision == "audit":
                continue
            if confirmed_count >= n:
                break
            confirmed_count += 1
            turn_ents: list[str] = []
            for entity in [turn.primary_entity, *turn.mentioned_entities]:
                if not entity or entity in seen:
                    continue
                seen.add(entity)
                turn_ents.append(entity)
            per_turn.append(turn_ents)
        out: list[str] = []
        for turn_ents in reversed(per_turn):
            out.extend(turn_ents)
        _log.debug("entities_in_window(n=%d) → %r  (confirmed=%d, total_turns=%d)", n, out, confirmed_count, len(self.turns))
        return out

