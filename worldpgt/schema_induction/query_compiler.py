"""Compile a natural-language question into an auditable query plan.

The plan targets *generated relation families*, not hand-coded predicates. The
compiler:
- detects the entity mention,
- matches the question phrasing to a relation family (via generic surface cues),
- infers the target generic role from the interrogative.

Bilingual (English + Russian) cue tables. No LLM. No hidden completion — if no
family matches, the plan is an explicit ``audit``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from worldpgt.schema_induction.types import RelationFamily

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "").strip())


@dataclass(frozen=True)
class QueryPlan:
    operation: str               # find_role | open_synthesis | audit
    entity: str | None
    family_label: str | None
    target_role: str | None
    question: str
    language: str
    interrogative: str | None
    matched_cue: str | None
    reason: str | None = None


# Open-synthesis triggers (EN + RU).
_SYNTHESIS_CUES = (
    "tell me about", "what do you know about", "describe",
    "расскажи про", "расскажи о", "расскажи об", "что ты знаешь о",
)

# Interrogatives -> language + (optional role override).
_INTERROGATIVES: dict[str, tuple[str, str | None]] = {
    "why": ("en", "cause"),
    "where": ("en", "destination"),
    "when": ("en", "time"),
    "who": ("en", "agent"),
    "what": ("en", None),
    "how": ("en", None),
    "почему": ("ru", "cause"),
    "зачем": ("ru", "cause"),
    "куда": ("ru", "destination"),
    "где": ("ru", "place"),
    "когда": ("ru", "time"),
    "кто": ("ru", "agent"),
    "что": ("ru", None),
    "какие": ("ru", None),
    "как": ("ru", None),
}

# Relation cue -> (canonical family label, default target role). Generic verb
# surfaces in EN + RU; not a domain ontology.
_RELATION_CUES: tuple[tuple[str, str, str], ...] = (
    # (cue substring, family label, default role)
    ("require", "requires", "requirement"),
    ("requires", "requires", "requirement"),
    ("need", "requires", "requirement"),
    ("нужно", "requires", "requirement"),
    ("требует", "requires", "requirement"),
    ("требуется", "requires", "requirement"),
    ("необходимо", "requires", "requirement"),
    ("prohibit", "prohibits", "prohibition"),
    ("forbid", "prohibits", "prohibition"),
    ("ban", "prohibits", "prohibition"),
    ("запрещает", "prohibits", "prohibition"),
    ("запрещено", "prohibits", "prohibition"),
    ("запрет", "prohibits", "prohibition"),
    ("allow", "allows", "permission"),
    ("permit", "allows", "permission"),
    ("разрешает", "allows", "permission"),
    ("позволяет", "allows", "permission"),
    ("founded", "founded by", "agent"),
    ("основал", "founded by", "agent"),
    ("operated", "operated by", "agent"),
    ("управляет", "operated by", "agent"),
    ("migrate", "move/migrate", "destination"),
    ("move", "move/migrate", "destination"),
    ("мигрир", "move/migrate", "destination"),
    ("движ", "move/migrate", "destination"),
    ("перемещ", "move/migrate", "destination"),
)


def _detect_interrogative(low: str) -> tuple[str | None, str, str | None]:
    for word, (lang, role_override) in _INTERROGATIVES.items():
        if re.search(r"(^|\b)" + re.escape(word) + r"\b", low):
            return word, lang, role_override
    return None, "en", None


def _detect_entity(question: str, entity_surfaces: list[str]) -> str | None:
    low = question.lower()
    # Longest surface that appears in the question wins.
    best: str | None = None
    for surface in entity_surfaces:
        s = surface.lower().strip()
        if not s:
            continue
        if s in low and (best is None or len(s) > len(best.lower())):
            best = surface
    return best


def _detect_relation_cue(low: str) -> tuple[str, str, str] | None:
    for cue, label, role in _RELATION_CUES:
        if cue in low:
            return cue, label, role
    return None


def compile_query(
    question: str,
    families: list[RelationFamily],
    entity_surfaces: list[str],
) -> QueryPlan:
    """Compile a question into a query plan against generated families."""

    q = _norm(question)
    low = q.lower()

    # 1. Open synthesis?
    for cue in _SYNTHESIS_CUES:
        if cue in low:
            lang = "ru" if any(ord(c) > 127 for c in cue) else "en"
            entity = _detect_entity(q, entity_surfaces)
            # Entity may follow the cue directly.
            if entity is None:
                tail = low.split(cue, 1)[1].strip(" .?!")
                entity = tail or None
            return QueryPlan(
                operation="open_synthesis",
                entity=entity,
                family_label=None,
                target_role=None,
                question=q,
                language=lang,
                interrogative=None,
                matched_cue=cue,
            )

    interrogative, lang, role_override = _detect_interrogative(low)
    entity = _detect_entity(q, entity_surfaces)
    cue = _detect_relation_cue(low)

    if entity is None:
        return QueryPlan(
            operation="audit", entity=None, family_label=None, target_role=None,
            question=q, language=lang, interrogative=interrogative,
            matched_cue=cue[0] if cue else None,
            reason="no_entity_detected",
        )

    if cue is None:
        return QueryPlan(
            operation="audit", entity=entity, family_label=None, target_role=None,
            question=q, language=lang, interrogative=interrogative,
            matched_cue=None, reason="no_relation_cue_matched",
        )

    cue_text, family_label, default_role = cue
    target_role = role_override or default_role

    # Do any families with this label exist among the loaded schema?
    labelled = [f for f in families if f.canonical_label == family_label]
    if not labelled:
        return QueryPlan(
            operation="audit", entity=entity, family_label=family_label,
            target_role=target_role, question=q, language=lang,
            interrogative=interrogative, matched_cue=cue_text,
            reason="relation_family_not_found",
        )

    # If the inferred role is not part of ANY matching family, fall back to a
    # non-subject role that some family does carry, so we answer auditably
    # rather than guess.
    all_roles = {r for f in labelled for r in f.roles}
    if target_role not in all_roles:
        non_subject = [r for r in sorted(all_roles) if r != "subject"]
        target_role = non_subject[0] if non_subject else None

    return QueryPlan(
        operation="find_role",
        entity=entity,
        family_label=family_label,
        target_role=target_role,
        question=q,
        language=lang,
        interrogative=interrogative,
        matched_cue=cue_text,
    )
