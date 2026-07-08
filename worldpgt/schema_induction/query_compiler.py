"""Compile a natural-language question into an auditable query plan.

The plan targets *generated relation families*, not hand-coded predicates. The
compiler:
- detects the entity mention,
- matches the question phrasing to a relation family (via generic surface cues),
- infers the target generic role from the interrogative.

Bilingual (English + Russian) cue tables. No LLM. No hidden completion — if no
family matches, the plan is an explicit ``audit``.

Verb-lemma fallback (2026-07-07): ``_RELATION_CUES`` below is a small, hand
enumerated list (require/prohibit/allow/founded/operated/migrate) — verified
against a fresh test domain (Fields Medal, not in this repo) that families
raw_claim_extractor.py now discovers directly from arbitrary verbs ("win",
"die", "work", "study" -- see that module's own generic SVO fallback) are
invisible to this compiler if their verb isn't also hand-added here, even
though a family with that exact label already exists. ``_detect_verb_lemma_cue``
closes that gap: when no fixed cue matches, it lemmatizes the question's own
main verb via spaCy and looks for a family already labelled with that lemma —
no synonym dictionary entry needed per new verb, mirroring the extractor-side
fix. Falls back to no-op (behaves exactly as before) without spaCy installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from worldpgt.schema_induction.types import RelationFamily

_WS = re.compile(r"\s+")

# spaCy is optional -- see raw_claim_extractor.py's identical lazy-load
# pattern. Only used for the verb-lemma fallback below; everything else in
# this module is pure string matching and works without it.
_NLP = None
try:
    import spacy as _spacy_mod  # noqa: F401
    _SPACY_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without spaCy
    _SPACY_AVAILABLE = False


def _get_nlp():
    global _NLP
    if _NLP is None:
        if not _SPACY_AVAILABLE:
            return None
        try:
            _NLP = _spacy_mod.load("en_core_web_sm")
        except Exception:  # pragma: no cover - model missing
            return None
    return _NLP


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
    match_role: str = "subject"  # which frame role plan.entity must match


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


def _detect_verb_lemma_cue(
    question: str, families: list[RelationFamily]
) -> tuple[str, str, str] | None:
    """Fallback cue detection: match the question's own main verb (lemmatized)
    directly against an existing family's ``canonical_label``. Only fires when
    no fixed ``_RELATION_CUES`` entry matched -- see module docstring."""
    nlp = _get_nlp()
    if nlp is None or not families:
        return None
    labels = {f.canonical_label for f in families}
    doc = nlp(question)
    for token in doc:
        # Prefer POS=="VERB", but also accept the dependency ROOT regardless
        # of its POS tag: short inverted questions ("What did X win?") can
        # get their main verb mis-tagged as NOUN by the small English model,
        # while its ROOT status is still a reliable "this is the predicate"
        # signal.
        if token.pos_ != "VERB" and token.dep_ != "ROOT":
            continue
        lemma = token.lemma_.lower()
        if lemma in labels:
            # cue_text is the SURFACE form actually typed ("won"), not the
            # lemma ("win") -- irregular verbs don't contain their lemma as a
            # substring, which silently broke _is_reversed_direction's
            # position search (it found "decline" inside "declined" by
            # accident, but never found "win" inside "won").
            return token.text.lower(), lemma, "object"
    return None


def _is_reversed_direction(low: str, entity: str, cue_text: str) -> bool:
    """True when ``entity`` appears AFTER the relation cue in the question --
    a cheap, position-based signal that the entity is the grammatical OBJECT
    ("Who won ENTITY?") rather than the SUBJECT ("What did ENTITY win?").

    Generalizes the same idea entity_qa/semantic_question_parser.py applies
    only to "leader_of" via a hand-written regex (`_ACTIVE_LEADER_RE`) --
    here it applies to any relation, fixed-cue or verb-lemma-fallback alike,
    instead of needing a new one-off regex per predicate that turns out to
    have this same forward/reverse ambiguity.
    """
    entity_pos = low.find(entity.lower())
    cue_pos = low.find(cue_text.lower())
    if entity_pos < 0 or cue_pos < 0:
        return False
    return entity_pos > cue_pos


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
    fixed_cue = _detect_relation_cue(low)
    cue = fixed_cue or _detect_verb_lemma_cue(q, families)
    # Direction reversal (see _is_reversed_direction) is only meaningful for
    # the verb-lemma fallback: it's derived from spaCy's English parse, and
    # the fixed _RELATION_CUES table already covers its own directionality
    # correctly (including Russian cues, whose word order this English
    # position heuristic gets wrong -- e.g. "Куда мигрируют wildebeest?" has
    # the subject AFTER the verb, which isn't "reversed" in Russian at all).
    from_verb_lemma_fallback = fixed_cue is None and cue is not None

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
    match_role = "subject"

    # Reverse direction ("Who won ENTITY?" vs "What did ENTITY win?") -- see
    # _is_reversed_direction's docstring.
    if from_verb_lemma_fallback and _is_reversed_direction(low, entity, cue_text):
        match_role, target_role = default_role, "subject"

    # Do any families with this label exist among the loaded schema?
    labelled = [f for f in families if f.canonical_label == family_label]
    if not labelled:
        return QueryPlan(
            operation="audit", entity=entity, family_label=family_label,
            target_role=target_role, question=q, language=lang,
            interrogative=interrogative, matched_cue=cue_text,
            reason="relation_family_not_found", match_role=match_role,
        )

    # If the inferred role is not part of ANY matching family, fall back to a
    # role that some family does carry (other than what we're matching
    # against), so we answer auditably rather than guess.
    all_roles = {r for f in labelled for r in f.roles}
    if target_role not in all_roles:
        alt = [r for r in sorted(all_roles) if r != match_role]
        target_role = alt[0] if alt else None

    return QueryPlan(
        operation="find_role",
        entity=entity,
        family_label=family_label,
        target_role=target_role,
        question=q,
        language=lang,
        interrogative=interrogative,
        matched_cue=cue_text,
        match_role=match_role,
    )
