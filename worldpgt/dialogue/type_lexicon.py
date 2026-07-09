"""Speech-layer noun classification for the dialogue reference grammar.

Reference understanding — "which type does 'satellite' denote", "which role
does 'founder' denote" — is speech/understanding, not fact lookup, so it is
deliberately kept on the speech side of Microworld's fact/speech boundary
(the same separation the community-context layer already enforces: "may
shape how an answer is explained, may not make a factual claim true"). This
module therefore never reads a candidate entity's own truth-layer data
(overlay definitions, `EntitySurfaceIndex`) to interpret a *word*; it reads
:mod:`worldpgt.cognition.phrase_graph`, the already-trained speech artifact
that also drives rendering, via two lookups:

  * ``canonicalize_entity_type(noun)`` — the noun *is* one of the eleven
    canonical type names verbatim ("person", "vehicle", ...). This is a
    tautology over the graph's own closed type schema, not a vocabulary
    guess, so it is checked first and wins outright — it is what keeps a
    corpus artifact like "person" co-occurring with philosophy-adjacent
    definitions from ever outranking the literal word "person".
  * ``PhraseGraph.type_for_word(noun)`` — a reverse index learned during
    phrase_graph's normal training pass (same overlay walk that already
    builds rendering fragments), mapping a definition word to the type most
    often observed with it ("satellite" -> whatever Starlink-like entities
    are actually tagged, because "satellite" appears in their stored
    definitions). Improving coverage means retraining phrase_graph on a
    wider corpus, not editing this file.
  * ``PERSON_ROLE_TOKENS`` / ``relation_intent_from_text`` — role nouns
    ("founder", "ceo") are recognized via the same speech-layer token set
    that already drives He/They pronoun choice in phrase_graph, then
    resolved to a relation through the shared predicate keyword table
    (relation names are the graph's own schema, not invented vocabulary).

No embedding fallback: GloVe similarity over bare nouns produces real false
positives on ordinary abstract words ("the answer" -> develops 0.66, "the
goal" -> founded_by 0.74 — measured by hand before this module existed). A
noun neither check recognizes is not a recognized reference-grammar noun —
the caller must not form a slot for it, per the grammar's closed-form
"unrecognized -> no slot" rule.
"""

from __future__ import annotations

from worldpgt.cognition.phrase_graph import PERSON_ROLE_TOKENS, default_phrase_graph
from worldpgt.knowledge.entity_types import CanonicalEntityType, canonicalize_entity_type
from worldpgt.relation_extraction_v2.relation_policy import relation_intent_from_text

# Translation only (not classification): phrase_graph is trained from English
# overlay/community text. Mapping a foreign lexeme to its English gloss and
# running it through the same single lookup is a materially smaller and more
# honest piece of hardcoding than duplicating the understanding layer itself
# per language.
_RU_NOUN_GLOSS: dict[str, str] = {
    "компания": "company",
    "компании": "company",
    "организация": "organization",
    "организации": "organization",
    "ракета": "rocket",
    "человек": "person",
    "спутник": "satellite",
    "основатель": "founder",
    "создатель": "creator",
    "владелец": "owner",
    "руководитель": "leader",
}


def _singularize(noun: str) -> str | None:
    if noun.endswith("ies") and len(noun) > 4:
        return noun[:-3] + "y"
    if noun.endswith("s") and not noun.endswith("ss") and len(noun) > 3:
        return noun[:-1]
    return None


def _gloss(noun: str) -> str:
    return _RU_NOUN_GLOSS.get(noun, noun)


def _type_lookup(noun: str) -> CanonicalEntityType | None:
    exact = canonicalize_entity_type(noun)
    if exact is not None:
        return exact
    learned = default_phrase_graph().type_for_word(noun)
    return learned  # type: ignore[return-value]


def noun_to_type(noun: str) -> CanonicalEntityType | None:
    """Canonical type a bare noun denotes ("rocket" -> vehicle), or None."""

    normalized = _gloss(noun.strip().lower())
    if not normalized:
        return None
    found = _type_lookup(normalized)
    if found is not None:
        return found
    singular = _singularize(normalized)
    if singular and singular != normalized:
        return _type_lookup(singular)
    return None


def _role_relation(noun: str) -> str | None:
    return relation_intent_from_text(noun) or relation_intent_from_text(f"{noun} of")


def classify_reference_noun(noun: str) -> tuple[str, str] | None:
    """Classify a bare noun as either a role-relation or a type gate.

    Returns ``("role", relation_intent)`` or ``("type", canonical_type)``,
    or None if the noun matches neither. A non-person type wins outright —
    "the product"/"the company" are type references even though
    ``relation_intent_from_text("product of")`` happens to resolve
    (``product_of`` describes what line a product belongs to, not who holds
    a role; that overlap is a keyword-table coincidence, not role semantics).
    A noun with no type, or a person-shaped type, is checked for a role
    relation first — that covers agent nouns like "founder"/"ceo" (in
    ``PERSON_ROLE_TOKENS``) as well as untyped role nouns like
    "leader"/"head" that the type lookup doesn't recognize at all.
    """

    normalized = _gloss(noun.strip().lower())
    if not normalized:
        return None
    etype = noun_to_type(normalized)
    if etype is not None and etype != "person":
        return ("type", etype)
    if etype == "person" or normalized in PERSON_ROLE_TOKENS or etype is None:
        relation = _role_relation(normalized)
        if relation is not None:
            return ("role", relation)
    if etype == "person":
        return ("type", "person")
    return None
