"""Conservative rewrite layer for underspecified dialogue follow-ups.

The coreference resolver handles explicit references such as "it" and
"the company". This module handles a smaller conversational case: the user
omits the subject entirely ("who founded?") or names a bare entity after a
previous turn ("and Starlink?").
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from worldpgt.dialogue.conversation_context import ConversationContext
from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex


@dataclass(frozen=True)
class FollowupRewrite:
    original_question: str
    resolved_question: str
    answer_style: str = "normal"
    reason: str | None = None

    @property
    def changed(self) -> bool:
        return self.original_question != self.resolved_question


_LEADING_FOLLOWUP_RE = re.compile(
    r"^\s*(?:and|what about|а|а\s+что\s+про|а\s+про)\s+(.+?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_FOUNDING_ELLIPSIS = {
    "who founded",
    "who founded it",
    "who founded this",
    "who founded that",
    "who started it",
    "who started this",
    "кто основал",
    "а кто основал",
    "кто создал",
    "а кто создал",
}

_OWNERSHIP_ELLIPSIS = {
    "who owns it",
    "who owns this",
    "who owns that",
    "кто владеет",
    "а кто владеет",
}

_DEVELOPMENT_ELLIPSIS = {
    "what does it develop",
    "what does this develop",
    "what does that develop",
    "what do they develop",
    "что разрабатывает",
    "а что разрабатывает",
}


def rewrite_followup(
    question: str,
    context: ConversationContext,
    index: EntitySurfaceIndex,
) -> FollowupRewrite:
    """Rewrite only high-confidence dialogue ellipses.

    If no rewrite is needed, the original question is returned. ``answer_style``
    is set to ``followup`` for short contextual turns even when the rewrite is
    just a bare entity expansion.
    """

    if not context.turns:
        return FollowupRewrite(question, question)

    stripped = question.strip()
    normalized = _normalize_prompt(stripped)

    leading = _LEADING_FOLLOWUP_RE.match(stripped)
    if leading:
        tail = leading.group(1).strip()
        entity = _single_mentioned_entity(tail, index)
        if entity:
            return FollowupRewrite(
                question,
                f"Tell me about {entity}.",
                answer_style="followup",
                reason="bare_entity_followup",
            )

    last = _latest_question_subject(context) or context.last_entity()
    if not last:
        return FollowupRewrite(question, question)

    if normalized in _FOUNDING_ELLIPSIS:
        return FollowupRewrite(
            question,
            f"Who founded {last}?",
            answer_style="followup",
            reason="omitted_subject_founding",
        )

    if normalized in _OWNERSHIP_ELLIPSIS:
        return FollowupRewrite(
            question,
            f"Who owns {last}?",
            answer_style="followup",
            reason="omitted_subject_ownership",
        )

    if normalized in _DEVELOPMENT_ELLIPSIS:
        return FollowupRewrite(
            question,
            f"What does {last} develop?",
            answer_style="followup",
            reason="omitted_subject_development",
        )

    if len(stripped.split()) <= 4 and index.find_in_text(stripped):
        return FollowupRewrite(question, question, answer_style="followup", reason="short_followup")

    return FollowupRewrite(question, question)


def _single_mentioned_entity(text: str, index: EntitySurfaceIndex) -> str | None:
    matches = index.find_in_text(text)
    entities = []
    seen: set[str] = set()
    for _surface, canonical, _start, _end in matches:
        if canonical not in seen:
            seen.add(canonical)
            entities.append(canonical)
    return entities[0] if len(entities) == 1 else None


def _latest_question_subject(context: ConversationContext) -> str | None:
    for turn in reversed(context.turns):
        if turn.decision == "audit":
            continue
        entity = turn.semantic_query.entity_a
        if entity:
            return entity
    return None


def _normalize_prompt(text: str) -> str:
    normalized = re.sub(r"[?.!]+$", "", text.strip().lower())
    return re.sub(r"\s+", " ", normalized)
