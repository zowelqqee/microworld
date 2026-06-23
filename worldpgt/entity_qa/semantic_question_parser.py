"""Semantic SPO-style question parser for Entity QA.

This parser is intentionally deterministic. It resolves known entities through
the same ``EntitySurfaceIndex`` used by extraction, maps short relation phrases
through ``relation_policy``, and emits a small structured query that downstream
QA code can either use directly or fall back from.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Optional

from worldpgt.entity_qa.types import SemanticQuery
from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex
from worldpgt.relation_extraction_v2.relation_policy import relation_intent_from_text

_EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
_ACCEPTED_OVERLAY_PATH = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_PROMOTED_OVERLAY_PATH = (
    _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
)
_SNAPSHOT_OVERLAY_PATH = _EXPERIMENTS / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json"

_IS_A_RE = re.compile(r"^is\s+(.+?)\s+(?:a|an|the)\s+(.+?)[\?\.]?$", re.IGNORECASE)
_DEFINITION_RE = re.compile(
    r"^(?:who|what)\s+(?:is|are|was|were)\s+(.+?)[\?\.]?$|"
    r"^(?:define|tell\s+me\s+about)\s+(.+?)[\?\.]?$|"
    r"^what\s+(?:kind|type|sort)\s+of\s+.+?\s+is\s+(.+?)[\?\.]?$",
    re.IGNORECASE,
)
_PRODUCTS_MAKE_RE = re.compile(
    r"what\s+products?\s+does\s+.+?\s+(?:make|produce|manufacture|build)s?\b",
    re.IGNORECASE,
)
_PASSIVE_OPEN_RELATION_RE = re.compile(
    r"^what\s+(?:is|was)\s+(developed|produced|published)\s+by\s+.+?[\?\.]?$",
    re.IGNORECASE,
)
_DEFINE_KIND_PREFIX_RE = re.compile(r"^what\s+(?:kind|type|sort)\s+of\b", re.IGNORECASE)
_WHAT_IS_DEFINITION_PREFIX_RE = re.compile(r"^what\s+is\s+(?:a|an|the)\s+", re.IGNORECASE)
_OPEN_MANUFACTURE_RE = re.compile(r"^what\s+does\s+.+?\s+manufactures?\b", re.IGNORECASE)
_TITLE_TOKEN_RE = re.compile(r"[A-Z][A-Za-z0-9&.-]*")
_PATH_RE = re.compile(
    r"\b(?:how\s+(?:is|are|was|were)|what\s+path)\b.+\b(?:connected|related|linked|connects)\b",
    re.IGNORECASE,
)
_INTERSECTION_RE = re.compile(
    r"\b(?:have\s+in\s+common|in\s+common|share|common\s+between)\b",
    re.IGNORECASE,
)
# ── Open synthesis (layer 3) ──────────────────────────────────────────────
# Open-ended "tell me everything you know" style questions. These intentionally
# do NOT name a single relation_intent: the synthesis layer gathers *all* facts
# about the entity rather than answering one slot. "Tell me who X is" stays a
# definition (handled below), so the alternation here excludes "who".
_OPEN_QUERY_RE = re.compile(
    r"^(?:"
    r"tell\s+me\s+(?:about|everything\s+about|more\s+about|all\s+about)\s+(?P<a>.+?)|"
    r"what\s+(?:do|can)\s+you\s+(?:know|tell\s+me)\s+about\s+(?P<b>.+?)|"
    r"what\s+can\s+you\s+tell\s+me\s+about\s+(?P<c>.+?)|"
    r"(?:describe|summari[sz]e)\s+(?P<d>.+?)|"
    r"give\s+me\s+(?:an?\s+)?overview\s+of\s+(?P<e>.+?)|"
    r"what\s+does\s+(?P<f>.+?)\s+do|"
    r"how\s+(?:does|do)\s+(?P<g>.+?)\s+(?:work|works|operate|operates|function|functions)"
    r")[\?\.]?$",
    re.IGNORECASE,
)

_SUBJECT_WH_RE = re.compile(r"^(?:who|which|what)\b", re.IGNORECASE)
_PASSIVE_BY_RE = re.compile(r"\b(?:by|belong\s+to|belongs\s+to)\b", re.IGNORECASE)

_INVERSE_CANONICAL_RELATIONS = frozenset(
    {
        "developed_by",
        "created_by",
        "published_by",
        "product_of",
        "service_of",
        "platform_of",
        "subsidiary_of",
        "part_of",
        "leader_of",
    }
)
_ACTIVE_OWNER_RE = re.compile(r"\b(?:does|did)?\s*.+?\b(?:own|owns)\b", re.IGNORECASE)
_OWNER_ACTIVE_ENTITY_RE = re.compile(r"^\s*(?:which|what)\s+\w+.+\bdoes\s+.+?\s+own\b", re.IGNORECASE)
_PASSIVE_OWNED_BY_ENTITY_RE = re.compile(
    r"^\s*(?:which|what)\s+.+?\bowned\s+by\b",
    re.IGNORECASE,
)
_ACTIVE_LEADER_RE = re.compile(r"^(?:who|which\s+person)\s+(?:leads?|runs?|heads?)\b", re.IGNORECASE)
_ACTIVE_RELATION_SUBJECT_RE = re.compile(
    r"^(?:who|which\s+.+?)\s+"
    r"(?:develops?|builds?|makes?|produces?|manufactures?|publishes?)\b",
    re.IGNORECASE,
)

# ── Verb phrase extraction (for embedding fallback) ───────────────────────────

# Lemmas that carry no relation signal — filtered before embedding lookup.
_STOP_VERB_LEMMAS: frozenset[str] = frozenset({
    "be", "do", "have", "will", "would", "could", "should", "may",
    "might", "shall", "get", "go", "come", "say", "tell", "know",
    "what", "who", "which", "how", "where", "when", "whom", "whose",
})

# Matches hyphenated compounds like "co-established", "pre-funded", "re-launched"
_HYPHEN_COMPOUND_RE = re.compile(r'\b\w+(?:-\w+)+\b')

# Noun-forming suffixes — lemmas ending in these are nominal even when spaCy
# mislabels them as VERB (e.g. "leadership" tagged VB in predicative position).
_NOUN_SUFFIX_RE = re.compile(
    r'(?:ship|ment|tion|sion|ness|hood|ity|ism|ance|ence)$'
)

_NLP = None
_NLP_LOCK = None


def _get_spacy_nlp():
    """Lazy-load the shared spaCy model (en_core_web_sm)."""
    global _NLP, _NLP_LOCK
    if _NLP_LOCK is None:
        import threading
        _NLP_LOCK = threading.Lock()
    if _NLP is None:
        with _NLP_LOCK:
            if _NLP is None:
                try:
                    import spacy
                    _NLP = spacy.load("en_core_web_sm")
                except (ImportError, OSError):
                    _NLP = False  # sentinel: spaCy unavailable
    return _NLP if _NLP is not False else None


def extract_verb_phrases(text: str) -> list[str]:
    """Return candidate verb/phrase strings for embedding-based relation lookup.

    Uses two passes:
    1. Regex pre-pass: finds hyphenated compounds in the raw text (spaCy splits
       "co-established" into "co" + "-" + "established" as separate tokens, losing
       the compound). Adds each part and the full compound.
    2. spaCy dep-parse pass: ROOT verb + prt (phrasal particles), all VERB tokens,
       and the ROOT even when tagged NOUN (handles "What does SpaceX engineer?").

    Returns a deduplicated, lower-cased list, longest candidates first.
    """
    seen: set[str] = set()
    results: list[str] = []

    def _add(phrase: str) -> None:
        key = phrase.lower().strip()
        if key and key not in seen and key not in _STOP_VERB_LEMMAS:
            seen.add(key)
            results.append(key)

    # ── Pass 1: hyphenated compounds from raw text ────────────────────────────
    # spaCy splits "co-established" → ["co", "-", "established"], so the
    # hyphenated form is only recoverable from the original string.
    for m in _HYPHEN_COMPOUND_RE.finditer(text.lower()):
        compound = m.group(0)
        _add(compound)
        for part in compound.split("-"):
            _add(part)

    # ── Pass 2: spaCy dependency parse ───────────────────────────────────────
    nlp = _get_spacy_nlp()
    if nlp is None:
        results.sort(key=lambda s: -len(s))
        return results

    doc = nlp(text)

    def _process_tok(tok) -> None:
        lemma = tok.lemma_.lower()
        if lemma in _STOP_VERB_LEMMAS:
            return
        if _NOUN_SUFFIX_RE.search(lemma):
            return
        particles = [c.text.lower() for c in tok.children if c.dep_ == "prt"]
        if particles:
            _add(lemma + " " + particles[0])
        _add(lemma)
        surface = tok.text.lower()
        if "-" in surface:
            _add(surface)
            for part in surface.split("-"):
                _add(part)

    def _is_misparse_subject_as_verb(tok) -> bool:
        """Return True when spaCy mislabels a WH-question subject as a VERB ROOT."""
        if tok.dep_ != "ROOT" or tok.pos_ != "VERB":
            return False
        wh_lemmas = frozenset({"who", "which", "what", "whose", "whom"})
        has_wh = any(
            c.dep_ in ("nsubj", "det", "nsubjpass") and c.lemma_.lower() in wh_lemmas
            for c in tok.children
        )
        has_verb_dobj = any(c.dep_ == "dobj" and c.pos_ == "VERB" for c in tok.children)
        return has_wh and has_verb_dobj

    for tok in doc:
        if tok.dep_ == "ROOT":
            if tok.pos_ == "VERB" and not _is_misparse_subject_as_verb(tok):
                _process_tok(tok)
            elif tok.pos_ in ("NOUN", "PROPN") and any(
                c.dep_ == "aux" for c in tok.children
            ):
                _process_tok(tok)

    for tok in doc:
        if tok.pos_ == "VERB" and tok.dep_ != "ROOT":
            _process_tok(tok)

    results.sort(key=lambda s: -len(s))
    return results


# "Which [NOUN] that [ENTITY] [VERB1] [VERB2] [OBJECT]?"
# Captures: (entity_phrase, verb1, verb2, filter_object)
# verb1 maps via relation_intent_from_text to relation_intent (e.g. founded → founded_by)
# verb2 maps via relation_intent_from_text to filter_predicate  (e.g. develop → develops)
_FILTERED_LOOKUP_RE = re.compile(
    r"^which\s+\w+\s+that\s+(.+?)\s+"
    r"(found(?:ed)?|start(?:ed)?|launch(?:ed)?|establish(?:ed)?|creat(?:ed)?|kick(?:ed)?(?:\s+off)?)\s+"
    r"(develop(?:s)?|build(?:s)?|make(?:s)?|produce(?:s)?|manufactures?|publish(?:es)?|use(?:s)?)\s+"
    r"(.+?)[\?.]?$",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def default_surface_index() -> EntitySurfaceIndex:
    return EntitySurfaceIndex(
        accepted_overlay_path=_ACCEPTED_OVERLAY_PATH,
        promoted_overlay_path=_PROMOTED_OVERLAY_PATH,
        snapshot_overlay_path=_SNAPSHOT_OVERLAY_PATH,
    )


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip(" ?.\t\n\r"))


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _entity_mentions(
    question: str,
    index: EntitySurfaceIndex,
) -> list[tuple[str, str, int, int]]:
    mentions = index.find_in_text(question)
    out: list[tuple[str, str, int, int]] = []
    seen: set[str] = set()
    for surface, canonical, start, end in sorted(mentions, key=lambda row: row[2]):
        if _looks_like_partial_title_match(question, surface, start, end):
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append((surface, canonical, start, end))
    return out


def _looks_like_partial_title_match(text: str, surface: str, start: int, end: int) -> bool:
    before = text[:start].rstrip()
    after = text[end:].lstrip()
    prev = re.search(r"([A-Z][A-Za-z0-9&.-]*)\s*$", before)
    nxt = _TITLE_TOKEN_RE.match(after)
    return bool(prev or nxt)


def _definition_subject(match: re.Match[str]) -> str:
    for group in match.groups():
        if group:
            return re.sub(r"^(?:a|an|the)\s+", "", _clean(group), flags=re.IGNORECASE)
    return ""


def _exact_definition_entity(
    raw_subject: str,
    mentions: list[tuple[str, str, int, int]],
) -> str | None:
    raw_norm = _norm_text(raw_subject)
    for surface, canonical, _start, _end in mentions:
        if _norm_text(canonical) == raw_norm:
            return canonical
        if " " in raw_norm and _norm_text(surface) == raw_norm:
            return canonical
    return None


def _is_inverse_relation_question(question: str, relation: str) -> bool:
    q = question.lower()
    if relation == "owned_by":
        return bool(_OWNER_ACTIVE_ENTITY_RE.search(q) or _PASSIVE_OWNED_BY_ENTITY_RE.search(q))
    if relation == "leader_of" and _ACTIVE_LEADER_RE.search(q):
        return True
    if relation in {"develops", "produces", "publishes"} and _ACTIVE_RELATION_SUBJECT_RE.search(q):
        return True
    if relation in _INVERSE_CANONICAL_RELATIONS and _SUBJECT_WH_RE.search(q):
        return True
    if relation in {"developed_by", "created_by", "published_by"} and _PASSIVE_BY_RE.search(q):
        return True
    return False


def _unknown_position(question: str, relation: str | None) -> str:
    if _PATH_RE.search(question):
        return "path"
    if _INTERSECTION_RE.search(question):
        return "intersection"
    if relation and _is_inverse_relation_question(question, relation):
        return "subject"
    if relation:
        return "object"
    return "relation"


_EMBEDDING_SKIP_PREFIX_RE = re.compile(
    r"^(?:why|how|tell\s+me\s+why|explain\s+why)\b", re.IGNORECASE
)


def _relation_with_embedding_fallback(
    verb_phrases: list[str],
    confidence_out: list[float],
) -> Optional[str]:
    """Embedding-only relation lookup over pre-extracted verb phrases.

    Caller must already have checked ``relation_intent_from_text`` (exact match)
    and extracted verb phrases via ``extract_verb_phrases``.  This function only
    runs the embedding similarity step.

    Writes 0.8 into *confidence_out[0]* on a hit, 0.0 on a miss.
    """
    if not verb_phrases:
        confidence_out[0] = 0.0
        return None

    try:
        from worldpgt.knowledge.relation_embedding_index import get_default_index
    except ImportError:
        confidence_out[0] = 0.0
        return None

    index = get_default_index()
    intent, _sim = index.find_relation_intent(verb_phrases)
    if intent is not None:
        confidence_out[0] = 0.8
        return intent

    confidence_out[0] = 0.0
    return None


def parse_semantic_query(
    question: str,
    index: EntitySurfaceIndex | None = None,
) -> SemanticQuery:
    """Parse *question* into a structured semantic query."""

    q = _clean(question)
    surface_index = index or default_surface_index()
    mentions = _entity_mentions(q, surface_index)
    entities = [canonical for _surface, canonical, _start, _end in mentions]

    # Open synthesis: explicit "tell me about / what do you know about / how does
    # X work" phrasings. Resolve the entity loosely — synthesis tolerates an
    # unresolved subject and falls back to keyword overlap downstream.
    open_match = _OPEN_QUERY_RE.match(q)
    if open_match:
        raw_subject = _clean(next((g for g in open_match.groups() if g), ""))
        subject = surface_index.resolve(raw_subject)
        if subject is None:
            subject = entities[0] if entities else (raw_subject or None)
        return SemanticQuery(
            entity_a=subject,
            entity_b=None,
            relation_intent=None,
            unknown_position="relation",
            query_type="open_synthesis",
            confidence=0.85,
        )

    fl_match = _FILTERED_LOOKUP_RE.match(q)
    if fl_match:
        entity_raw = _clean(fl_match.group(1))
        verb1     = fl_match.group(2).lower().replace(" ", "_")
        verb2     = fl_match.group(3).lower()
        obj       = _clean(fl_match.group(4))
        entity_a  = surface_index.resolve(entity_raw) or entity_raw or (entities[0] if entities else None)
        rel_intent  = relation_intent_from_text(verb1) or relation_intent_from_text(fl_match.group(2))
        filt_pred   = relation_intent_from_text(verb2)
        return SemanticQuery(
            entity_a=entity_a,
            entity_b=None,
            relation_intent=rel_intent,
            unknown_position="object",
            query_type="filtered_lookup",
            confidence=0.9,
            filter_predicate=filt_pred,
            filter_object=obj,
        )

    # Fast path: exact keyword match — no spaCy, no embeddings.
    # Slow path: spaCy verb-phrase extraction + embedding similarity only when
    # exact match returns None.
    _relation_confidence: list[float] = [0.0]
    _embedding_matched = False
    relation = relation_intent_from_text(q)
    if relation is not None:
        _relation_confidence[0] = 1.0
    elif not _EMBEDDING_SKIP_PREFIX_RE.match(q.strip()):
        verb_phrases = extract_verb_phrases(q)
        relation = _relation_with_embedding_fallback(verb_phrases, _relation_confidence)
        _embedding_matched = _relation_confidence[0] == 0.8

    if relation == "develops" and _PRODUCTS_MAKE_RE.search(q):
        relation = "produces"
    if relation == "manufactures" and _OPEN_MANUFACTURE_RE.search(q):
        relation = "produces"

    is_a_match = _IS_A_RE.match(q)
    if is_a_match:
        left_subject = _clean(is_a_match.group(1))
        left_end = is_a_match.start(2)
        left_entities = [canonical for _surface, canonical, _start, end in mentions if end <= left_end]
        subject = left_entities[0] if left_entities else surface_index.resolve(left_subject)
        if subject is None and left_subject:
            subject = left_subject
        target = _clean(is_a_match.group(2))
        confidence = 0.95 if subject and target else 0.45
        return SemanticQuery(
            entity_a=subject,
            entity_b=target or None,
            relation_intent="is_a",
            unknown_position="relation",
            query_type="is_a",
            confidence=confidence,
        )

    passive_open = _PASSIVE_OPEN_RELATION_RE.match(q)
    if passive_open and entities:
        predicate = {
            "developed": "develops",
            "produced": "produces",
            "published": "publishes",
        }.get(passive_open.group(1).lower(), "develops")
        return SemanticQuery(
            entity_a=entities[0],
            entity_b=None,
            relation_intent=predicate,
            unknown_position="object",
            query_type="lookup",
            confidence=0.9,
        )

    if _INTERSECTION_RE.search(q):
        return SemanticQuery(
            entity_a=entities[0] if entities else None,
            entity_b=entities[1] if len(entities) > 1 else None,
            relation_intent=relation,
            unknown_position="intersection",
            query_type="comparative",
            confidence=0.95 if len(entities) >= 2 else 0.4,
        )

    if _PATH_RE.search(q):
        return SemanticQuery(
            entity_a=entities[0] if entities else None,
            entity_b=entities[1] if len(entities) > 1 else None,
            relation_intent=relation,
            unknown_position="path",
            query_type="lookup",
            confidence=0.95 if len(entities) >= 2 else 0.4,
        )

    definition_match = _DEFINITION_RE.match(q)
    if definition_match:
        raw_subject = _definition_subject(definition_match)
        exact_entity = _exact_definition_entity(raw_subject, mentions)
        if exact_entity and (
            relation is None
            or _DEFINE_KIND_PREFIX_RE.match(q)
            or _WHAT_IS_DEFINITION_PREFIX_RE.match(q)
        ):
            return SemanticQuery(
                entity_a=exact_entity,
                entity_b=None,
                relation_intent=None,
                unknown_position="relation",
                query_type="definition",
                confidence=0.9,
            )
        if relation is None:
            return SemanticQuery(
                entity_a=None,
                entity_b=None,
                relation_intent=None,
                unknown_position="relation",
                query_type="definition",
                confidence=0.1,
            )

    if relation and entities:
        unknown = _unknown_position(q, relation)
        base_conf = 0.8 if _embedding_matched else 0.9
        return SemanticQuery(
            entity_a=entities[0],
            entity_b=entities[1] if len(entities) > 1 else None,
            relation_intent=relation,
            unknown_position=unknown,  # type: ignore[arg-type]
            query_type="inverse" if unknown == "subject" else "lookup",
            confidence=base_conf,
        )

    if entities:
        return SemanticQuery(
            entity_a=entities[0],
            entity_b=entities[1] if len(entities) > 1 else None,
            relation_intent=None,
            unknown_position="relation",
            query_type="lookup",
            confidence=0.55,
        )

    return SemanticQuery(
        entity_a=None,
        entity_b=None,
        relation_intent=relation,
        unknown_position="relation",
        query_type="lookup",
        confidence=0.35 if relation else 0.1,
    )
