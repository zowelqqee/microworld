"""Per-session dialogue context for interactive Microworld QA.

The context is intentionally in-memory only. It records what the controlled QA
surface has already said in one interactive session so later questions can
resolve explicit references without changing accepted memory or overlays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from worldpgt.entity_qa.types import SemanticQuery


@dataclass
class ConversationTurn:
    question: str
    semantic_query: SemanticQuery
    decision: str
    primary_entity: str | None
    mentioned_entities: list[str]
    relation_type: str | None


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

    def entities_in_window(self, n: int = 3) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for turn in self.turns[-n:]:
            for entity in [turn.primary_entity, *turn.mentioned_entities]:
                if not entity or entity in seen:
                    continue
                seen.add(entity)
                out.append(entity)
        return out

