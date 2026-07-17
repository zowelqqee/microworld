"""Proposal-only, source-attributed acquisition from public open-web APIs.

This is deliberately not a general-purpose crawler.  A broad pump must not
turn arbitrary search snippets into memory: it uses a small, explicit source
allowlist, captures only metadata/abstracts, observes bounded request budgets,
and writes a separate proposal artifact.  Accepted and promoted memory are
never opened or mutated here.

Supported source families are deliberately complementary:

* OpenAlex: broad scholarly work metadata and reconstructed abstracts;
* Crossref: DOI catalogue metadata and publisher-provided abstracts;
* arXiv: openly available research abstracts, especially useful for STEM.

The output documents use the existing local-snapshot shape, so the established
extractor and precision firewalls can evaluate them without widening the
runtime's trusted-memory boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from worldpgt.knowledge_pump.extraction_yield_v2 import extract_yield_v2
from worldpgt.knowledge_pump.precision_firewall import apply_precision_firewall
from worldpgt.knowledge_pump.precision_firewall_v2 import apply_precision_firewall_v2
from worldpgt.relation_extraction_v2.sentence_splitter import extract_full_body, split_paragraphs, split_sentences
from worldpgt.wiki_snapshot_ingestion.types import ReadySnapshotDoc


@dataclass(frozen=True)
class OpenWebTopic:
    """A deliberately broad query, tagged for coverage accounting."""

    query: str
    bucket: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class OpenWebRecord:
    """A compact, attributable record suitable for proposal-only extraction."""

    source_id: str
    source_kind: str
    topic_bucket: str
    title: str
    source_url: str
    retrieved_at: str
    text: str
    license_note: str
    published_at: str = ""
    authors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Topics are intentionally concept-level rather than a list of fashionable
# entities.  One run can be bounded with --max-queries, while a full run is a
# cross-domain acquisition pass instead of another technology-only batch.
BROAD_OPEN_WEB_TOPICS: tuple[OpenWebTopic, ...] = (
    OpenWebTopic("world history", "history", ("openalex", "crossref")),
    OpenWebTopic("ancient civilizations", "history", ("openalex", "crossref")),
    OpenWebTopic("economic history", "history", ("openalex", "crossref")),
    OpenWebTopic("human geography", "geography", ("openalex", "crossref")),
    OpenWebTopic("urbanization and cities", "geography", ("openalex", "crossref")),
    OpenWebTopic("climate change", "earth_environment", ("openalex", "arxiv", "crossref")),
    OpenWebTopic("biodiversity conservation", "earth_environment", ("openalex", "crossref")),
    OpenWebTopic("geology of Earth", "earth_environment", ("openalex", "crossref")),
    OpenWebTopic("astronomy and cosmology", "physical_science", ("openalex", "arxiv", "crossref")),
    OpenWebTopic("quantum physics", "physical_science", ("openalex", "arxiv", "crossref")),
    OpenWebTopic("chemistry materials science", "physical_science", ("openalex", "arxiv", "crossref")),
    OpenWebTopic("molecular biology", "life_science", ("openalex", "crossref")),
    OpenWebTopic("evolutionary biology", "life_science", ("openalex", "crossref")),
    OpenWebTopic("public health epidemiology", "health", ("openalex", "crossref")),
    OpenWebTopic("mental health psychology", "health", ("openalex", "crossref")),
    OpenWebTopic("mathematics", "mathematics", ("openalex", "arxiv", "crossref")),
    OpenWebTopic("statistics and probability", "mathematics", ("openalex", "arxiv", "crossref")),
    OpenWebTopic("computer science", "computing", ("openalex", "arxiv", "crossref")),
    OpenWebTopic("artificial intelligence", "computing", ("openalex", "arxiv", "crossref")),
    OpenWebTopic("cybersecurity", "computing", ("openalex", "arxiv", "crossref")),
    OpenWebTopic("renewable energy", "engineering", ("openalex", "arxiv", "crossref")),
    OpenWebTopic("transportation engineering", "engineering", ("openalex", "crossref")),
    OpenWebTopic("agriculture and food systems", "engineering", ("openalex", "crossref")),
    OpenWebTopic("economics", "economics", ("openalex", "crossref")),
    OpenWebTopic("international trade", "economics", ("openalex", "crossref")),
    OpenWebTopic("political science", "society", ("openalex", "crossref")),
    OpenWebTopic("law and legal systems", "society", ("openalex", "crossref")),
    OpenWebTopic("education research", "society", ("openalex", "crossref")),
    OpenWebTopic("sociology", "society", ("openalex", "crossref")),
    OpenWebTopic("linguistics", "language", ("openalex", "crossref")),
    OpenWebTopic("language acquisition", "language", ("openalex", "crossref")),
    OpenWebTopic("philosophy", "humanities", ("openalex", "crossref")),
    OpenWebTopic("religious studies", "humanities", ("openalex", "crossref")),
    OpenWebTopic("literature", "arts_culture", ("openalex", "crossref")),
    OpenWebTopic("musicology", "arts_culture", ("openalex", "crossref")),
    OpenWebTopic("film studies", "arts_culture", ("openalex", "crossref")),
    OpenWebTopic("sports science", "sports", ("openalex", "crossref")),
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_LEAD_DEFINITION_RE = re.compile(
    r"^(?:the\s+)?(?P<subject>.+?)(?:\s*\([^)]{0,80}\))?\s+"
    r"(?:is|was|are|were)\s+(?:a\s+|an\s+|the\s+)?",
    re.IGNORECASE,
)
_UNSAFE_DEFINITION_TERMS = frozenset({
    "earliest", "essential", "driving", "best", "greatest", "largest", "first", "second", "third",
    "popular", "authoritative", "exhaustive", "comprehensive", "rigorous", "crucial", "worldwide",
    "leading", "major", "powerful", "recently", "right",
})
_SOURCE_SPECIFIC_RELATION_EXTRACTIONS = frozenset({
    "arxiv_explicit_relation_v1",
    "crossref_explicit_relation_v1",
    # DOI metadata is a separate, field-level Crossref extractor.  It remains
    # proposal-only and still traverses both generic precision firewalls.
    "crossref_doi_structured_metadata_v1",
    "openalex_api_topic_reference_structured_v1",
    "openalex_explicit_relation_v1",
    "wikidata_api_structured_property_v1",
})
_ABSTRACT_DEFINITION_HEADS = frozenset({
    "algorithm", "approach", "branch", "collection", "corpus", "database", "discipline", "field",
    "form", "framework", "language", "library", "method", "model", "platform", "process", "protocol",
    "program", "science", "standard", "study", "sub-area", "system", "technique", "technology",
    "theory", "type",
})
_ABSTRACT_DEFINITION_MODIFIERS = frozenset({
    "ab", "applied", "chemical", "computational", "digital", "formal", "formalised", "initio",
    "interdisciplinary", "large", "mathematical", "multidisciplinary", "natural", "open", "public",
    "quantum", "research", "scientific", "social", "structured",
})
_ABSTRACT_NONDEFINITION_PHRASES = (
    "can be studied", "can be used", "will reshape", "is intended to", "is designed to",
    "is important for", "is useful for", "is relevant for", "inspired by",
)
_DIRECT_ABSTRACT_DEFINITION_RE = re.compile(
    r"^(?:the\s+)?(?P<subject>[A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Za-z0-9'’.-]+){0,6})"
    r"(?:\s*\([^)]{1,80}\))?\s+(?:is|was|are|were)\s+(?:a\s+|an\s+|the\s+)(?P<definition>[^.!?:;]{8,220})[.!]$"
)
_ABSTRACT_SUBJECT_BLOCKLIST = frozenset({
    "the epilogue", "we", "this", "this chapter", "this paper", "the study", "he", "she", "it", "there",
})
_ABSTRACT_SUBJECT_PROSE_TOKENS = frozenset({
    "although", "argue", "argues", "because", "conclude", "concludes", "demonstrate", "demonstrates",
    "introduce", "introduced", "introduces", "paper", "present", "presents", "propose", "proposed",
    "proposes", "show", "shows", "study", "that", "we", "which", "while", "who",
})
_ABSTRACT_DEFINITION_CLAUSE_RE = re.compile(
    r"(?:\s+(?:that|which|who|where|while|although|but)\b|,\s+and\s+(?:is|was|are|were)\b)",
    re.IGNORECASE,
)
_EXPLORATORY_RELATION_PREDICATES = frozenset({
    "developed_by", "develops", "enables", "runs_on", "supports", "used_for", "uses", "works_by",
})
_EXPLORATORY_RELATION_SUBJECT_BLOCKLIST = frozenset({
    "a paper", "an article", "the article", "the authors", "the book", "the chapter", "the paper",
    "the report", "the study", "this article", "this book", "this chapter", "this paper", "this report",
    "this study", "we",
})

# This is deliberately an evidence-grounder, not a per-domain ontology or a
# list of paper-title rules.  Scholarly abstracts commonly state a relation in
# an authorial clause ("we introduce X ... that enables Y").  The generic
# relation extractor has no discourse model and may therefore substitute the
# record title for X.  We only retain a graph subject if the evidence itself
# names one: a distinctive token (acronym, identifier, hyphenated name) or a
# short sentence-leading concept phrase.  Everything else remains in the raw
# proposal lane with its provenance rather than becoming a queryable node.
_GROUNDING_DISTINCTIVE_SURFACE_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:[A-Z]{2,}[A-Za-z0-9_-]*|[A-Za-z][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*|[A-Z][A-Za-z]*-[A-Za-z0-9_-]+)(?![A-Za-z0-9_-])"
)
_GROUNDING_INITIAL_CONCEPT_RE = re.compile(
    r"^(?:the\s+)?(?P<surface>[A-Z][A-Za-z0-9_-]*(?:\s+[a-z][A-Za-z0-9_-]*){0,5})\s*$"
)
_GROUNDING_DESCRIPTOR_RE = re.compile(
    r",\s+(?:a|an|the)\s+(?P<descriptor>[^,.;]{3,180})",
    re.IGNORECASE,
)
_GROUNDING_DESCRIPTOR_CLAUSE_RE = re.compile(r"\s+(?:that|which|who)\b", re.IGNORECASE)
_QUALITY_DISCOURSE_OBJECT_STARTERS = frozenset({
    "and", "but", "for", "from", "in", "of", "on", "or", "that", "themselves",
    "them", "this", "to", "us", "with",
})
_GROUNDING_DISCOURSE_SUBJECTS = frozenset({
    "here", "it", "there", "these", "they", "this", "those", "we",
})
_PARENTHETICAL_ALIAS_RE = re.compile(
    r"(?P<left>[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){0,8})\s*"
    r"\((?P<right>[A-Za-z][A-Za-z0-9-]{1,20})\)"
)

# A scholarly title can be a perfectly useful graph subject, but people often
# omit a trailing method qualifier when asking about it.  These are deliberately
# *not* general title shortenings: only a long, named prefix before an explicit
# method phrase is eligible, and consolidation suppresses aliases shared by two
# different subjects.
_EXPERIMENTAL_TITLE_ALIAS_SPLIT_RE = re.compile(
    r"\s+(?:using|with|via|based\s+on)\s+", re.IGNORECASE,
)


def _compact(text: str) -> str:
    return _WS_RE.sub(" ", _HTML_TAG_RE.sub(" ", text or "")).strip()


def _experimental_title_alias_candidates(subject: str) -> list[str]:
    """Return conservative, human-typed aliases for a long paper title."""
    subject = _compact(subject)
    if len(subject) < 32 or len(subject.split()) < 5:
        return []
    prefix = _EXPERIMENTAL_TITLE_ALIAS_SPLIT_RE.split(subject, maxsplit=1)[0].strip(" :,-")
    if prefix == subject or len(prefix) < 24 or len(prefix.split()) < 5:
        return []
    return [prefix]


def _predicate_surface(predicate: str) -> str:
    """Recover the evidence wording from a typed predicate without a map."""
    return " ".join((predicate or "").replace("_", " ").split())


def _evidence_grounded_subject(edge: dict[str, Any]) -> tuple[str, str] | None:
    """Find the explicit subject surface immediately supporting one edge.

    The result is intentionally conservative.  A paper title is never used as
    a fallback subject: absence of an evidence-local name is an auditable gap,
    not a reason to manufacture a graph node.
    """
    evidence = _compact(str(edge.get("evidence_text") or edge.get("evidence_span") or ""))
    predicate_surface = _predicate_surface(str(edge.get("predicate") or ""))
    if not evidence or not predicate_surface:
        return None
    relation_start = evidence.casefold().find(predicate_surface.casefold())
    if relation_start <= 0:
        return None
    prefix = evidence[:relation_start]
    title_key = _norm(str(edge.get("source_record_title") or edge.get("source_page") or ""))

    def _is_discourse_subject(surface: str) -> bool:
        words = _norm(surface).split()
        return not words or words[0] in _GROUNDING_DISCOURSE_SUBJECTS

    # Distinctive names are portable across domains: project identifiers,
    # acronyms and hyphenated names need no hand-maintained vocabulary.
    distinctive = list(_GROUNDING_DISTINCTIVE_SURFACE_RE.finditer(prefix))
    if distinctive:
        surface = distinctive[-1].group(0).strip()
        if surface and _norm(surface) != title_key and not _is_discourse_subject(surface):
            return surface, "distinctive_evidence_surface"

    # For ordinary concepts such as "Graph neural networks use ...", the
    # subject is the compact noun phrase directly leading the predicate.  This
    # only accepts a whole uncluttered prefix, so authorial clauses and titles
    # embedded in prose cannot leak through.
    initial = _GROUNDING_INITIAL_CONCEPT_RE.fullmatch(prefix.strip(" ,:;"))
    if initial:
        surface = _compact(initial.group("surface"))
        if surface and _norm(surface) != title_key and not _is_discourse_subject(surface):
            return surface, "sentence_leading_concept"
    return None


def _evidence_descriptor(evidence: str, subject: str) -> str:
    """Return a directly appositive descriptor, if the evidence supplies one."""
    match = re.search(
        re.escape(subject) + _GROUNDING_DESCRIPTOR_RE.pattern,
        evidence,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    descriptor = _compact(match.group("descriptor"))
    clause = _GROUNDING_DESCRIPTOR_CLAUSE_RE.search(descriptor)
    if clause:
        descriptor = descriptor[:clause.start()].strip(" ,:;")
    return descriptor if 2 <= len(descriptor.split()) <= 16 else ""


def _relation_evidence_quality(edge: dict[str, Any]) -> dict[str, Any]:
    """Score local grammatical support without a topic or predicate allowlist.

    This asks only structural questions of the one evidence sentence: do the
    named subject and typed predicate occur in order, is the extracted object
    actually present after that predicate, and is it a content phrase rather
    than a dangling discourse fragment?  The score decides queryability in the
    experimental UI; lower-scoring rows remain inspectable review data.
    """
    evidence = _compact(str(edge.get("evidence_text") or edge.get("evidence_span") or ""))
    grounding = edge.get("evidence_grounding") if isinstance(edge.get("evidence_grounding"), dict) else {}
    subject = _compact(str(
        grounding.get("observed_subject_surface")
        or grounding.get("subject_surface")
        or edge.get("subject")
        or ""
    ))
    predicate_surface = _predicate_surface(str(edge.get("predicate") or ""))
    obj = _compact(str(edge.get("object") or ""))
    subject_start = evidence.casefold().rfind(subject.casefold())
    predicate_start = evidence.casefold().find(
        predicate_surface.casefold(),
        max(0, subject_start + len(subject)),
    )
    object_start = evidence.casefold().find(
        obj.casefold(),
        max(0, predicate_start + len(predicate_surface)),
    )
    score = 0
    signals: list[str] = []
    issues: list[str] = []
    if subject_start >= 0 and predicate_start >= subject_start + len(subject):
        score += 2
        signals.append("subject_precedes_predicate")
        between = evidence[subject_start + len(subject):predicate_start]
        if len(re.findall(r"[,:;]", between)) <= 1 and len(between.split()) <= 18:
            score += 1
            signals.append("compact_subject_predicate_span")
        else:
            issues.append("indirect_subject_predicate_span")
    else:
        issues.append("missing_direct_subject_predicate_order")
    if object_start >= 0:
        score += 2
        signals.append("object_occurs_after_predicate")
    else:
        issues.append("object_not_verbatim_after_predicate")
    object_words = obj.casefold().split()
    if object_words and object_words[0] in _QUALITY_DISCOURSE_OBJECT_STARTERS:
        score -= 2
        issues.append("object_starts_as_discourse_fragment")
    if len(object_words) > 28:
        score -= 1
        issues.append("object_phrase_too_long")
    if int(edge.get("supporting_source_count") or 0) > 1:
        score += 1
        signals.append("multi_source_support")
    return {
        "score": score,
        "queryable": score >= 4,
        "signals": signals,
        "issues": issues,
    }


def _alias_acronym_matches(long_text: str, short_text: str) -> bool:
    """Check an acronym relationship from spelling alone, not a term list."""
    short = re.sub(r"[^A-Za-z0-9]", "", short_text).casefold().removesuffix("s")
    if len(short) < 2:
        return False
    words = re.findall(r"[A-Za-z0-9]+", long_text)
    initials = "".join(word[0] for word in words if word)
    return initials.casefold() == short


def _evidence_alias_pairs(evidence: str) -> list[tuple[str, str]]:
    """Extract only self-explaining full-name/acronym pairs from evidence."""
    pairs: list[tuple[str, str]] = []
    for match in _PARENTHETICAL_ALIAS_RE.finditer(evidence):
        left = _compact(match.group("left"))
        right = _compact(match.group("right"))
        left_words = left.split()
        right_words = right.split()
        # A phrase before an acronym is the common form; find the shortest
        # suffix whose initials prove the relation, avoiding broad context
        # accidentally captured by the regex.
        for size in range(2, min(8, len(left_words)) + 1):
            candidate = " ".join(left_words[-size:])
            if _alias_acronym_matches(candidate, right):
                pairs.append((candidate, right))
                break
        # The reverse form ("LLM (Large Language Model)") is less common
        # but has equally explicit evidence.
        for size in range(2, min(8, len(right_words)) + 1):
            candidate = " ".join(right_words[:size])
            if _alias_acronym_matches(candidate, left):
                pairs.append((candidate, left))
                break
    return pairs


def _evidence_alias_index(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return canonical subject labels and aliases proved by the same evidence.

    Only alias pairs touching an already grounded subject participate.  Thus a
    parenthetical aside in an abstract cannot manufacture a new graph merge.
    """
    rows = list(rows)
    subject_keys = {_norm(str(row.get("subject") or "")) for row in rows}
    parent: dict[str, str] = {}
    display: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in rows:
        evidence = _compact(str(row.get("evidence_text") or row.get("evidence_span") or ""))
        for full_name, acronym in _evidence_alias_pairs(evidence):
            full_key, acronym_key = _norm(full_name), _norm(acronym)
            if not full_key or not acronym_key or not ({full_key, acronym_key} & subject_keys):
                continue
            display.setdefault(full_key, full_name)
            display.setdefault(acronym_key, acronym)
            union(full_key, acronym_key)

    groups: dict[str, set[str]] = {}
    for key in parent:
        groups.setdefault(find(key), set()).add(key)
    canonical_by_key: dict[str, str] = {}
    aliases_by_canonical: dict[str, list[str]] = {}
    for keys in groups.values():
        canonical_key = max(
            keys,
            key=lambda key: (len(display.get(key, key).split()), len(display.get(key, key)), display.get(key, key).casefold()),
        )
        canonical = display[canonical_key]
        aliases = sorted(
            (display[key] for key in keys if key != canonical_key),
            key=lambda value: (value.casefold(), value),
        )
        aliases_by_canonical[_norm(canonical)] = aliases
        for key in keys:
            canonical_by_key[key] = canonical
    return canonical_by_key, aliases_by_canonical


def build_evidence_grounded_experimental_graph(
    edges: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Project raw abstract edges onto entities explicitly named in evidence.

    The raw relation artifact remains untouched.  This adds a separate,
    proposal-only graph where every queryable subject has an evidence-local
    anchor and optional appositive description.  No accepted or promoted
    memory is involved.
    """
    grounded_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for edge in edges:
        grounded = _evidence_grounded_subject(edge)
        if grounded is None:
            rejected.append({
                "edge": edge,
                "reason": "no_explicit_evidence_local_subject",
            })
            continue
        subject, grounding_method = grounded
        obj = _compact(str(edge.get("object") or ""))
        predicate = _norm(str(edge.get("predicate") or ""))
        if not obj or not predicate:
            rejected.append({"edge": edge, "reason": "missing_relation_fields"})
            continue
        row = dict(edge)
        row.update({
            "subject": subject,
            "evidence_grounding": {
                "method": grounding_method,
                "subject_surface": subject,
                "source_record_title": str(edge.get("source_record_title") or edge.get("source_page") or ""),
            },
            "trust": "proposal_open_web_exploratory",
            "risk": "medium",
            "stability": "semi_stable",
            "requires_review": True,
            "experimental_query_only": True,
            "safe_for_general_runtime": False,
            "experimental_tier": "evidence_grounded_abstract_relation_v1",
        })
        grounded_rows.append(row)

    canonical_by_key, aliases_by_canonical = _evidence_alias_index(grounded_rows)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in grounded_rows:
        observed_subject = _compact(str(row.get("subject") or ""))
        canonical_subject = canonical_by_key.get(_norm(observed_subject), observed_subject)
        if canonical_subject != observed_subject:
            row["subject"] = canonical_subject
            row["evidence_grounding"]["canonical_subject"] = canonical_subject
            row["evidence_grounding"]["observed_subject_surface"] = observed_subject
        key = (
            _norm(str(row.get("subject") or "")),
            _norm(str(row.get("predicate") or "")),
            _norm(str(row.get("object") or "")),
        )
        grouped.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    query_relations: list[dict[str, Any]] = []
    review_relations: list[dict[str, Any]] = []
    descriptors: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for key in sorted(grouped):
        group = grouped[key]
        merged = dict(group[0])
        source_urls = sorted({str(item.get("source_url") or "") for item in group if item.get("source_url")})
        evidence = []
        for item in group:
            text = _compact(str(item.get("evidence_text") or ""))
            if text and text not in evidence:
                evidence.append(text)
        merged.update({
            "support_count": len(group),
            "supporting_source_count": len(source_urls),
            "supporting_sources": source_urls,
            "supporting_evidence": evidence[:3],
        })
        merged["evidence_quality"] = _relation_evidence_quality(merged)
        selected.append(merged)
        if merged["evidence_quality"]["queryable"]:
            query_relations.append(merged)
            for text in evidence:
                descriptor = _evidence_descriptor(text, str(merged["subject"]))
                if descriptor:
                    descriptors.setdefault(_norm(str(merged["subject"])), []).append((descriptor, merged))
        else:
            review_relations.append(merged)

    entities: list[dict[str, Any]] = []
    seen_subjects: set[str] = set()
    for edge in query_relations:
        subject = _compact(str(edge["subject"]))
        key = _norm(subject)
        if not subject or key in seen_subjects:
            continue
        seen_subjects.add(key)
        entities.append({
            "overlay_type": "overlay_entity",
            "label": subject,
            "aliases": aliases_by_canonical.get(key, []),
            "trust": "proposal_open_web_exploratory",
            "risk": "medium",
            "requires_review": True,
            "safe_for_general_runtime": False,
            "experimental_tier": "evidence_grounded_abstract_relation_v1",
        })

    definitions: list[dict[str, Any]] = []
    seen_definitions: set[tuple[str, str]] = set()
    for key, candidates in sorted(descriptors.items()):
        descriptor, edge = candidates[0]
        definition_key = (key, _norm(descriptor))
        if definition_key in seen_definitions:
            continue
        seen_definitions.add(definition_key)
        definitions.append({
            "overlay_type": "overlay_definition",
            "subject": str(edge["subject"]),
            "definition": descriptor,
            "predicate": "is_a",
            "source_page": str(edge.get("source_page") or ""),
            "source_record_title": str(edge.get("source_record_title") or edge.get("source_page") or ""),
            "source_url": str(edge.get("source_url") or ""),
            "source_kind": str(edge.get("source_kind") or ""),
            "evidence_text": str(edge.get("evidence_text") or ""),
            "evidence_span": str(edge.get("evidence_span") or edge.get("evidence_text") or ""),
            "trust": "proposal_open_web_exploratory",
            "risk": "medium",
            "stability": "semi_stable",
            "requires_review": True,
            "safe_for_general_runtime": False,
            "experimental_tier": "evidence_grounded_abstract_descriptor_v1",
        })
    return {
        "entities": entities,
        "definitions": definitions,
        "relations": selected,
        "query_relations": query_relations,
        "review_relations": review_relations,
        "rejected": rejected,
    }


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_filename(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return (stem[:90] or "untitled") + "-" + _sha256(text)[:12]


def _default_get_json(url: str, timeout_sec: float = 15.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "MicroworldOpenWebPump/1.0 (+proposal-only)"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # nosec B310: curated HTTPS endpoints
        payload = response.read().decode("utf-8", errors="replace")
    data = json.loads(payload)
    return data if isinstance(data, dict) else {}


def _default_get_text(url: str, timeout_sec: float = 15.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "MicroworldOpenWebPump/1.0 (+proposal-only)"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # nosec B310: curated HTTPS endpoint
        return response.read().decode("utf-8", errors="replace")


def _openalex_abstract(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, raw_positions in index.items():
        if not isinstance(word, str) or not isinstance(raw_positions, list):
            continue
        for position in raw_positions:
            if isinstance(position, int) and position >= 0:
                positions.append((position, word))
    return " ".join(word for _position, word in sorted(positions))


def _crossref_date(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    parts = value.get("date-parts")
    if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
        return ""
    first = parts[0]
    if not first:
        return ""
    return "-".join(str(part).zfill(2) for part in first[:3])


def parse_openalex(payload: dict[str, Any], topic: OpenWebTopic, retrieved_at: str) -> list[OpenWebRecord]:
    records: list[OpenWebRecord] = []
    for item in payload.get("results", []) if isinstance(payload.get("results"), list) else []:
        if not isinstance(item, dict):
            continue
        title = _compact(str(item.get("display_name") or ""))
        abstract = _compact(_openalex_abstract(item.get("abstract_inverted_index")))
        primary_location = item.get("primary_location") if isinstance(item.get("primary_location"), dict) else {}
        landing = str(primary_location.get("landing_page_url") or item.get("doi") or item.get("id") or "")
        if not title or not abstract or not landing:
            continue
        authors = tuple(
            _compact(str((row.get("author") or {}).get("display_name") or ""))
            for row in item.get("authorships", []) if isinstance(row, dict)
        )
        records.append(OpenWebRecord(
            source_id=str(item.get("id") or landing), source_kind="openalex", topic_bucket=topic.bucket,
            title=title, source_url=landing, retrieved_at=retrieved_at, text=abstract,
            license_note="OpenAlex work metadata and reconstructed abstract; proposal-only.",
            published_at=str(item.get("publication_year") or ""), authors=tuple(a for a in authors if a),
        ))
    return records


def parse_crossref(payload: dict[str, Any], topic: OpenWebTopic, retrieved_at: str) -> list[OpenWebRecord]:
    message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    records: list[OpenWebRecord] = []
    for item in message.get("items", []) if isinstance(message.get("items"), list) else []:
        if not isinstance(item, dict):
            continue
        title_values = item.get("title") if isinstance(item.get("title"), list) else []
        title = _compact(str(title_values[0] if title_values else ""))
        abstract = _compact(str(item.get("abstract") or ""))
        doi = _compact(str(item.get("DOI") or ""))
        url = _compact(str(item.get("URL") or (f"https://doi.org/{doi}" if doi else "")))
        if not title or not abstract or not url:
            continue
        authors = tuple(
            _compact(" ".join(str(row.get(key) or "") for key in ("given", "family")))
            for row in item.get("author", []) if isinstance(row, dict)
        )
        records.append(OpenWebRecord(
            source_id=doi or url, source_kind="crossref", topic_bucket=topic.bucket,
            title=title, source_url=url, retrieved_at=retrieved_at, text=abstract,
            license_note="Crossref metadata and publisher-provided abstract; proposal-only.",
            published_at=_crossref_date(item.get("published-print")) or _crossref_date(item.get("published-online")),
            authors=tuple(a for a in authors if a),
        ))
    return records


def parse_arxiv(xml_text: str, topic: OpenWebTopic, retrieved_at: str) -> list[OpenWebRecord]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    records: list[OpenWebRecord] = []
    for entry in root.findall("atom:entry", ns):
        title = _compact(entry.findtext("atom:title", default="", namespaces=ns))
        abstract = _compact(entry.findtext("atom:summary", default="", namespaces=ns))
        source_id = _compact(entry.findtext("atom:id", default="", namespaces=ns))
        published_at = _compact(entry.findtext("atom:published", default="", namespaces=ns))
        authors = tuple(_compact(author.findtext("atom:name", default="", namespaces=ns)) for author in entry.findall("atom:author", ns))
        if title and abstract and source_id:
            records.append(OpenWebRecord(
                source_id=source_id, source_kind="arxiv", topic_bucket=topic.bucket,
                title=title, source_url=source_id, retrieved_at=retrieved_at, text=abstract,
                license_note="arXiv record title and abstract; proposal-only.", published_at=published_at,
                authors=tuple(a for a in authors if a),
            ))
    return records


def build_query_plan(topics: Iterable[OpenWebTopic] = BROAD_OPEN_WEB_TOPICS) -> list[tuple[OpenWebTopic, str]]:
    """Return a deterministic, deduplicated cross-domain query plan."""
    seen: set[tuple[str, str]] = set()
    plan: list[tuple[OpenWebTopic, str]] = []
    for topic in topics:
        for source in topic.sources:
            key = (topic.query.casefold(), source)
            if source in {"openalex", "crossref", "arxiv"} and key not in seen:
                seen.add(key)
                plan.append((topic, source))
    return plan


def build_paged_query_plan(
    topics: Iterable[OpenWebTopic] = BROAD_OPEN_WEB_TOPICS,
    *,
    pages_per_query: int = 1,
    page_start: int = 0,
) -> list[tuple[OpenWebTopic, str, int]]:
    """Expand the broad source frontier into resumable, paged API requests.

    Page zero of every source query is scheduled before page one.  That keeps a
    partially completed multi-hour campaign broad across disciplines instead of
    exhausting one subject before touching the rest of the frontier.
    """
    if pages_per_query < 1:
        raise ValueError("pages_per_query must be at least 1")
    if page_start < 0:
        raise ValueError("page_start must be non-negative")
    base_plan = build_query_plan(topics)
    return [
        (topic, source, page_index)
        for page_index in range(page_start, page_start + pages_per_query)
        for topic, source in base_plan
    ]


def collect_records(
    *,
    topics: Iterable[OpenWebTopic] = BROAD_OPEN_WEB_TOPICS,
    start_query: int = 0,
    max_queries: int | None = None,
    records_per_query: int = 2,
    pages_per_query: int = 1,
    page_start: int = 0,
    allow_network: bool = False,
    skip_sources: Iterable[str] = (),
    request_delay_sec: float = 0.5,
    get_json: Callable[[str], dict[str, Any]] = _default_get_json,
    get_text: Callable[[str], str] = _default_get_text,
    sleep: Callable[[float], None] = time.sleep,
    retrieved_at: str | None = None,
) -> tuple[list[OpenWebRecord], dict[str, Any]]:
    """Collect bounded API records.  Without explicit network consent, plan only."""
    if records_per_query < 1:
        raise ValueError("records_per_query must be at least 1")
    if records_per_query > 200:
        raise ValueError("records_per_query must be at most 200 for the OpenAlex source")
    if pages_per_query < 1:
        raise ValueError("pages_per_query must be at least 1")
    if page_start < 0:
        raise ValueError("page_start must be non-negative")
    if records_per_query * (page_start + pages_per_query) > 10_000:
        raise ValueError("records_per_query * (page_start + pages_per_query) must not exceed 10,000 for Crossref offset paging")
    if start_query < 0:
        raise ValueError("start_query must be non-negative")
    if request_delay_sec < 0:
        raise ValueError("request_delay_sec must be non-negative")
    plan = build_paged_query_plan(topics, pages_per_query=pages_per_query, page_start=page_start)
    planned_total = len(plan)
    plan = plan[start_query:]
    if max_queries is not None:
        plan = plan[:max(0, max_queries)]
    skipped_sources = {source.strip().casefold() for source in skip_sources if source.strip()}
    if skipped_sources:
        plan = [
            (topic, source, page_index)
            for topic, source, page_index in plan
            if source.casefold() not in skipped_sources
        ]
    report: dict[str, Any] = {
        "network_calls": False,
        "planned_total": planned_total,
        "pages_per_query": pages_per_query,
        "page_start": page_start,
        "start_query": start_query,
        "skipped_sources": sorted(skipped_sources),
        "query_count": len(plan),
        "planned_by_source": dict(sorted(Counter(source for _topic, source, _page in plan).items())),
        "planned_by_bucket": dict(sorted(Counter(topic.bucket for topic, _source, _page in plan).items())),
        "errors": [],
    }
    if not allow_network:
        report["status"] = "planned_no_network"
        return [], report

    now = retrieved_at or _now_utc()
    records: list[OpenWebRecord] = []
    seen: set[tuple[str, str]] = set()
    rate_limited_sources: set[str] = set()
    for request_index, (topic, source, page_index) in enumerate(plan):
        if request_index:
            sleep(request_delay_sec)
        if source in rate_limited_sources:
            continue
        encoded = urllib.parse.quote(topic.query, safe="")
        try:
            if source == "openalex":
                url = (
                    f"https://api.openalex.org/works?search={encoded}&per-page={records_per_query}"
                    f"&page={page_index + 1}"
                )
                parsed = parse_openalex(get_json(url), topic, now)
            elif source == "crossref":
                url = (
                    f"https://api.crossref.org/works?query.bibliographic={encoded}&rows={records_per_query}"
                    f"&offset={page_index * records_per_query}"
                )
                parsed = parse_crossref(get_json(url), topic, now)
            else:
                url = (
                    f"https://export.arxiv.org/api/query?search_query=all:{encoded}"
                    f"&start={page_index * records_per_query}&max_results={records_per_query}"
                )
                parsed = parse_arxiv(get_text(url), topic, now)
            report["network_calls"] = True
        except Exception as exc:  # acquisition must be resumable despite one API outage
            error = str(exc)[:240]
            report["errors"].append({"source": source, "query": topic.query, "error": error})
            if "429" in error or "rate limit" in error.casefold():
                rate_limited_sources.add(source)
            continue
        for record in parsed:
            key = (record.source_kind, record.source_id.casefold())
            if key not in seen:
                seen.add(key)
                records.append(record)
    report.update({
        "status": "completed",
        "records_total": len(records),
        "records_by_source": dict(sorted(Counter(record.source_kind for record in records).items())),
        "records_by_bucket": dict(sorted(Counter(record.topic_bucket for record in records).items())),
        "rate_limited_sources": sorted(rate_limited_sources),
    })
    return records, report


def _normalized_doc(record: OpenWebRecord) -> str:
    authors = ", ".join(record.authors[:8])
    header = [
        f"# {record.title}", "", f"Source: {record.source_url}",
        f"Source type: {record.source_kind}", f"Topic bucket: {record.topic_bucket}",
        f"Retrieved at: {record.retrieved_at}", f"Published at: {record.published_at}",
        f"Authors: {authors}", f"License / use: {record.license_note}",
        "Status: LOCAL_OPEN_WEB_SNAPSHOT", "Safe for accepted memory: false",
        "Requires ingestion/quarantine/promotion/regression: true", "", record.text, "",
    ]
    return "\n".join(header)


def write_snapshot_artifacts(records: Iterable[OpenWebRecord], output_dir: str | Path) -> tuple[list[ReadySnapshotDoc], list[dict[str, Any]]]:
    """Persist records and return adapter docs; no trusted-memory paths are used."""
    root = Path(output_dir)
    raw_dir = root / "raw_records"
    doc_dir = root / "normalized_docs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)
    docs: list[ReadySnapshotDoc] = []
    manifest: list[dict[str, Any]] = []
    for record in records:
        raw_text = json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n"
        record_hash = _sha256(raw_text)
        stem = _safe_filename(f"{record.source_kind}-{record.source_id}-{record.title}")
        raw_path = raw_dir / f"{stem}.json"
        doc_path = doc_dir / f"{stem}.md"
        raw_path.write_text(raw_text, encoding="utf-8")
        doc_path.write_text(_normalized_doc(record), encoding="utf-8")
        row = {
            "title": record.title, "normalized_title": record.title, "source_url": record.source_url,
            "source_kind": record.source_kind, "topic_bucket": record.topic_bucket,
            "retrieved_at": record.retrieved_at, "published_at": record.published_at,
            "raw_text_sha256": record_hash, "raw_snapshot_path": str(raw_path),
            "normalized_doc_path": str(doc_path), "fetch_status": "success",
            "ready_for_self_ingestion": bool(record.text), "requires_quarantine": True,
            "safe_for_general_runtime": False,
        }
        manifest.append(row)
        docs.append(ReadySnapshotDoc(
            title=record.title, normalized_title=record.title, source_url=record.source_url,
            retrieved_at=record.retrieved_at, revision_id=None, raw_text_sha256=record_hash,
            normalized_doc_path=str(doc_path), manifest_row=row,
        ))
    (root / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return docs, manifest


def _with_record_provenance(items: Iterable[dict[str, Any]], docs: Iterable[ReadySnapshotDoc]) -> list[dict[str, Any]]:
    by_title = {doc.title: doc for doc in docs}
    out: list[dict[str, Any]] = []
    for item in items:
        enriched = dict(item)
        doc = by_title.get(str(item.get("source_record_title") or item.get("source_page") or ""))
        if doc is not None:
            row = doc.manifest_row
            enriched.update({
                "source_url": doc.source_url, "source_kind": row.get("source_kind", "open_web"),
                "topic_bucket": row.get("topic_bucket", ""), "source_retrieved_at": doc.retrieved_at,
                "pump_source_kind": "open_web_proposal", "trust": "proposal_open_web",
                "safe_for_general_runtime": False,
            })
        out.append(enriched)
    return out


def _norm(text: str) -> str:
    return " ".join((text or "").casefold().split())


def _clean_abstract_definition(text: str) -> str:
    cleaned = _compact(text).strip(" ,;:")
    clause = _ABSTRACT_DEFINITION_CLAUSE_RE.search(cleaned)
    if clause and clause.start() >= 24:
        cleaned = cleaned[:clause.start()].strip(" ,;:")
    return cleaned


def _clean_abstract_subject(text: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", _compact(text), flags=re.IGNORECASE).strip()


def _abstract_definition_head(text: str) -> str:
    words = _norm(text).split()
    while words and words[0] in _ABSTRACT_DEFINITION_MODIFIERS:
        words.pop(0)
    return words[0] if words else ""


def _is_term_shaped_abstract_subject(text: str) -> bool:
    """Reject sentence fragments while allowing ordinary lower-case term words.

    Abstracts often define terms such as ``Computer science`` or ``Peer
    review``.  The first implementation admitted only title-cased words, which
    threw those away.  It is still unsafe to treat an authorial clause such as
    ``The Epilogue concludes that ...`` as a term, so prose-bearing tokens are
    blocked explicitly.
    """
    words = _norm(text).split()
    return (
        bool(words)
        and len(words) <= 7
        and text[:1].isupper()
        and not set(words).intersection(_ABSTRACT_SUBJECT_PROSE_TOKENS)
    )


def extract_open_web_abstract_definitions(docs: Iterable[ReadySnapshotDoc]) -> list[dict[str, Any]]:
    """Extract only direct, title-cased copular definitions from abstracts.

    This is intentionally narrower than the normal snapshot extractor: source
    records describe papers, so their title is not assumed to be the thing
    being defined.  The claimed subject must instead appear verbatim at the
    start of a complete ``X is a/an/the Y`` sentence.  The original record
    title and URL stay on each candidate for review.
    """
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for doc in docs:
        raw = Path(doc.normalized_doc_path).read_text(encoding="utf-8")
        body = extract_full_body(raw)
        paragraphs = split_paragraphs(body)
        for paragraph in paragraphs[:3]:
            for sentence in split_sentences(paragraph)[:12]:
                evidence = _compact(sentence)
                match = _DIRECT_ABSTRACT_DEFINITION_RE.fullmatch(evidence)
                if not match:
                    continue
                subject = _clean_abstract_subject(match.group("subject"))
                definition = _clean_abstract_definition(match.group("definition"))
                if _norm(subject) in _ABSTRACT_SUBJECT_BLOCKLIST or not _is_term_shaped_abstract_subject(subject):
                    continue
                candidate_words = set(_norm(f"{subject} {definition}").split())
                if (
                    len(definition.split()) < 3
                    or candidate_words.intersection(_UNSAFE_DEFINITION_TERMS)
                    or any(mark in definition for mark in ('"', "“", "”", "'", "’"))
                    or " will " in f" {_norm(definition)} "
                    or _abstract_definition_head(definition) not in _ABSTRACT_DEFINITION_HEADS
                    or any(phrase in _norm(definition) for phrase in _ABSTRACT_NONDEFINITION_PHRASES)
                ):
                    continue
                key = (_norm(subject), _norm(definition))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    "overlay_type": "overlay_definition", "subject": subject,
                    "definition": definition, "predicate": "is_a", "source_page": subject,
                    "source_record_title": doc.title, "evidence_text": evidence, "evidence_span": evidence,
                    "trust": "proposal_open_web", "risk": "low", "stability": "stable",
                    "candidate_source": "open_web_abstract_definition_v2",
                    "open_web_extraction": "abstract_direct_definition_v2",
                    "safe_for_general_runtime": False,
                })
    return candidates


def build_exploratory_relation_overlay(items: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split direct abstract relations into raw and graph-ready experimental lanes.

    This lane deliberately has higher recall than the definition proposal
    overlay.  It still requires an explicit extracted predicate and sentence
    evidence, but it is never general-runtime memory: callers get both the
    complete raw relation set and a smaller typed relation graph for semantic
    exploration.  Source provenance and the experimental risk label remain on
    every edge.
    """
    raw: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        if item.get("overlay_type") != "overlay_relation":
            continue
        row = dict(item)
        raw.append(row)
        subject = _compact(str(row.get("subject") or ""))
        predicate = _norm(str(row.get("predicate") or ""))
        obj = _compact(str(row.get("object") or ""))
        if (
            predicate not in _EXPLORATORY_RELATION_PREDICATES
            or not subject
            or not obj
            or _norm(subject) in _EXPLORATORY_RELATION_SUBJECT_BLOCKLIST
            or not str(row.get("evidence_text") or "").strip()
        ):
            continue
        key = (_norm(subject), predicate, _norm(obj))
        if key in seen:
            continue
        seen.add(key)
        row.update({
            "trust": "proposal_open_web_exploratory",
            "risk": "high",
            "stability": "semi_stable",
            "requires_review": True,
            "safe_for_general_runtime": False,
            "experimental_tier": "typed_abstract_relation_v1",
        })
        selected.append(row)
    return {"raw": raw, "selected": selected}


def _open_web_source_gate(items: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Apply a source-specific precision boundary before generic firewalls.

    Research abstracts are evidence-rich but syntactically unlike encyclopedia
    leads.  The generic extractor can mistake an author's prose for a durable
    graph relation.  Until a dedicated abstract extractor exists, this lane
    admits only a direct subject-led definition. Everything else remains
    inspectable in quarantine instead of becoming a proposal fact.
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for item in items:
        if item.get("overlay_type") == "overlay_relation":
            extraction = str(item.get("open_web_extraction") or "")
            source_kind = str(item.get("source_kind") or "")
            if (
                extraction in _SOURCE_SPECIFIC_RELATION_EXTRACTIONS
                and extraction.startswith(f"{source_kind}_")
            ):
                accepted.append(item)
            else:
                quarantine.append({"item": item, "reason": "open_web_relation_requires_source_specific_extractor"})
            continue
        if item.get("overlay_type") != "overlay_definition":
            quarantine.append({"item": item, "reason": "open_web_relation_requires_source_specific_extractor"})
            continue
        subject = str(item.get("subject") or "").strip()
        source_page = str(item.get("source_page") or "").strip()
        evidence = _compact(str(item.get("evidence_text") or ""))
        definition = _compact(str(item.get("definition") or ""))
        match = _LEAD_DEFINITION_RE.match(evidence)
        source_specific = item.get("open_web_extraction") in {
            "abstract_direct_definition_v1", "abstract_direct_definition_v2",
        }
        if not subject or (not source_specific and _norm(subject) != _norm(source_page)):
            rejected.append({"item": item, "reason": "open_web_subject_not_source_title"})
            continue
        if not match or _norm(match.group("subject")) != _norm(subject):
            rejected.append({"item": item, "reason": "open_web_not_direct_title_led_definition"})
            continue
        words = set(_norm(definition).split())
        if words.intersection(_UNSAFE_DEFINITION_TERMS):
            rejected.append({"item": item, "reason": "open_web_evaluative_or_temporal_definition"})
            continue
        accepted.append(item)
    return {"accepted": accepted, "rejected": rejected, "quarantine": quarantine}


def build_proposal_overlay(
    docs: list[ReadySnapshotDoc],
    *,
    source_specific_candidates: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Extract and gate candidates, preserving rejected/quarantined evidence."""
    generic_candidates, extraction = extract_yield_v2(docs)
    abstract_candidates = extract_open_web_abstract_definitions(docs)
    candidates = _with_record_provenance(generic_candidates + abstract_candidates, docs)
    # These candidates already carry a source-specific parser provenance and
    # must retain it unchanged for the source gate below.  They still traverse
    # both precision firewalls with every other candidate.
    candidates.extend(dict(item) for item in source_specific_candidates)
    exploratory = build_exploratory_relation_overlay(candidates)
    source_gate = _open_web_source_gate(candidates)
    quality = apply_precision_firewall(source_gate["accepted"])
    precision_v2 = apply_precision_firewall_v2(quality["accepted"])
    accepted = precision_v2["accepted"]
    return {
        "proposal_only": True, "accepted_memory_modified": False, "promoted_overlay_modified": False,
        "runtime_behavior_modified": False, "safe_for_general_runtime": False,
        "candidates": candidates, "proposal_overlay": accepted,
        "exploratory_relation_candidates": exploratory["raw"],
        "exploratory_relation_overlay": exploratory["selected"],
        "rejected": source_gate["rejected"] + quality["rejected"] + precision_v2["rejected"],
        "quarantine": source_gate["quarantine"] + quality["quarantine"] + precision_v2["quarantine"],
        "extraction": {**extraction, "abstract_direct_definition_candidate_count": len(abstract_candidates)},
        "summary": {
            "docs_processed": len(docs), "candidate_count": len(candidates), "proposal_item_count": len(accepted),
            "rejected_count": len(source_gate["rejected"]) + len(quality["rejected"]) + len(precision_v2["rejected"]),
            "quarantine_count": len(source_gate["quarantine"]) + len(quality["quarantine"]) + len(precision_v2["quarantine"]),
            "exploratory_relation_candidate_count": len(exploratory["raw"]),
            "exploratory_relation_item_count": len(exploratory["selected"]),
            "exploratory_relation_items_by_predicate": dict(sorted(Counter(
                str(item.get("predicate") or "") for item in exploratory["selected"]
            ).items())),
            "source_kinds": dict(sorted(Counter(str(doc.manifest_row.get("source_kind", "")) for doc in docs).items())),
            "topic_buckets": dict(sorted(Counter(str(doc.manifest_row.get("topic_bucket", "")) for doc in docs).items())),
        },
    }


def run_open_web_pump(
    *,
    output_dir: str | Path,
    max_queries: int | None = None,
    start_query: int = 0,
    records_per_query: int = 2,
    pages_per_query: int = 1,
    page_start: int = 0,
    allow_network: bool = False,
    skip_sources: Iterable[str] = (),
    request_delay_sec: float = 0.5,
    topics: Iterable[OpenWebTopic] = BROAD_OPEN_WEB_TOPICS,
    get_json: Callable[[str], dict[str, Any]] = _default_get_json,
    get_text: Callable[[str], str] = _default_get_text,
) -> dict[str, Any]:
    """Run a bounded broad-acquisition pass and write only new proposal artifacts."""
    root = Path(output_dir)
    records, collection = collect_records(
        topics=topics, start_query=start_query, max_queries=max_queries, records_per_query=records_per_query,
        pages_per_query=pages_per_query, page_start=page_start,
        allow_network=allow_network, skip_sources=skip_sources, request_delay_sec=request_delay_sec,
        get_json=get_json, get_text=get_text,
    )
    result: dict[str, Any] = {
        "collection": collection, "proposal_only": True, "accepted_memory_modified": False,
        "promoted_overlay_modified": False, "runtime_behavior_modified": False,
    }
    if not allow_network:
        root.mkdir(parents=True, exist_ok=True)
        (root / "open_web_pump_plan.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result
    docs, manifest = write_snapshot_artifacts(records, root)
    proposal = build_proposal_overlay(docs)
    (root / "open_web_candidates.json").write_text(json.dumps(proposal["candidates"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "open_web_proposal_overlay.json").write_text(json.dumps(proposal["proposal_overlay"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "open_web_exploratory_relation_candidates.json").write_text(json.dumps(proposal["exploratory_relation_candidates"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "open_web_exploratory_relation_overlay.json").write_text(json.dumps(proposal["exploratory_relation_overlay"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "open_web_rejected.json").write_text(json.dumps(proposal["rejected"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "open_web_quarantine.json").write_text(json.dumps(proposal["quarantine"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result.update({"manifest_count": len(manifest), "proposal": proposal["summary"], "network_calls": collection["network_calls"]})
    (root / "open_web_pump_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def regate_existing_batch(output_dir: str | Path) -> dict[str, Any]:
    """Re-run deterministic gates over saved source docs without network access."""
    root = Path(output_dir)
    manifest_path = root / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs = [
        ReadySnapshotDoc(
            title=str(row["title"]), normalized_title=str(row.get("normalized_title") or row["title"]),
            source_url=str(row["source_url"]), retrieved_at=str(row["retrieved_at"]), revision_id=None,
            raw_text_sha256=str(row["raw_text_sha256"]), normalized_doc_path=str(row["normalized_doc_path"]),
            manifest_row=dict(row),
        )
        for row in manifest if isinstance(row, dict) and Path(str(row.get("normalized_doc_path") or "")).is_file()
    ]
    proposal = build_proposal_overlay(docs)
    (root / "open_web_candidates.json").write_text(json.dumps(proposal["candidates"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "open_web_proposal_overlay.json").write_text(json.dumps(proposal["proposal_overlay"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "open_web_exploratory_relation_candidates.json").write_text(json.dumps(proposal["exploratory_relation_candidates"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "open_web_exploratory_relation_overlay.json").write_text(json.dumps(proposal["exploratory_relation_overlay"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "open_web_rejected.json").write_text(json.dumps(proposal["rejected"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "open_web_quarantine.json").write_text(json.dumps(proposal["quarantine"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path = root / "open_web_pump_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    summary.update({"proposal": proposal["summary"], "regated_without_network": True, "accepted_memory_modified": False, "promoted_overlay_modified": False})
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def consolidate_regated_campaign(campaign_dir: str | Path) -> dict[str, Any]:
    """Merge already-gated segment overlays into one read-only campaign overlay."""
    root = Path(campaign_dir)
    segment_paths = sorted(root.glob("segment_*/open_web_proposal_overlay.json"))
    if not segment_paths:
        raise FileNotFoundError(f"no regated segment overlays found under: {root}")
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    source_counts: Counter[str] = Counter()
    for path in segment_paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        for item in rows if isinstance(rows, list) else []:
            if not isinstance(item, dict):
                continue
            key = (_norm(str(item.get("subject") or "")), _norm(str(item.get("definition") or "")))
            if not all(key) or key in seen:
                continue
            seen.add(key)
            items.append(dict(item))
            source_counts[str(item.get("source_kind") or "unknown")] += 1
    exploratory_paths = sorted(root.glob("segment_*/open_web_exploratory_relation_overlay.json"))
    exploratory_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for path in exploratory_paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        for item in rows if isinstance(rows, list) else []:
            if not isinstance(item, dict):
                continue
            key = (
                _norm(str(item.get("subject") or "")),
                _norm(str(item.get("predicate") or "")),
                _norm(str(item.get("object") or "")),
            )
            if all(key):
                exploratory_groups.setdefault(key, []).append(item)
    exploratory_items: list[dict[str, Any]] = []
    exploratory_predicates: Counter[str] = Counter()
    for key in sorted(exploratory_groups):
        group = exploratory_groups[key]
        merged = dict(group[0])
        source_urls = sorted({str(item.get("source_url") or "") for item in group if item.get("source_url")})
        evidence = []
        for item in group:
            text = _compact(str(item.get("evidence_text") or ""))
            if text and text not in evidence:
                evidence.append(text)
        merged.update({
            "support_count": len(group),
            "supporting_source_count": len(source_urls),
            "supporting_sources": source_urls,
            "supporting_evidence": evidence[:3],
        })
        exploratory_items.append(merged)
        exploratory_predicates[str(merged.get("predicate") or "")] += 1
    # Preserve the title-led graph for inspection, but build a separate main
    # experimental graph from subjects actually named in the abstract evidence.
    # This is the graph the UI should query; title-led edges remain in the raw
    # campaign artifact above for review and regression comparison.
    grounded_graph = build_evidence_grounded_experimental_graph(exploratory_items)

    experimental_subjects: list[tuple[str, str]] = []
    seen_experimental_entities: set[str] = set()
    for edge in exploratory_items:
        subject = _compact(str(edge.get("subject") or ""))
        key = _norm(subject)
        if not subject or key in seen_experimental_entities:
            continue
        seen_experimental_entities.add(key)
        experimental_subjects.append((key, subject))

    # Aliases must resolve to exactly one experimental entity.  This keeps a
    # convenient omitted-method suffix from silently changing the graph node.
    alias_owners: dict[str, set[str]] = {}
    alias_text: dict[str, str] = {}
    for key, subject in experimental_subjects:
        for alias in _experimental_title_alias_candidates(subject):
            alias_key = _norm(alias)
            alias_owners.setdefault(alias_key, set()).add(key)
            alias_text[alias_key] = alias

    experimental_entities: list[dict[str, Any]] = []
    for key, subject in experimental_subjects:
        aliases = [
            alias_text[alias_key]
            for alias_key, owners in alias_owners.items()
            if owners == {key}
        ]
        experimental_entities.append({
            "overlay_type": "overlay_entity",
            "label": subject,
            "aliases": aliases,
            "trust": "proposal_open_web_exploratory",
            "risk": "medium",
            "requires_review": True,
            "safe_for_general_runtime": False,
            "experimental_tier": "typed_abstract_relation_v1",
        })
    experimental_graph_edges = [
        {
            **edge,
            "trust": "proposal_open_web_exploratory",
            "risk": "medium",
            "experimental_query_only": True,
            "safe_for_general_runtime": False,
        }
        for edge in exploratory_items
    ]
    grounded_graph_items = [
        *grounded_graph["entities"],
        *grounded_graph["definitions"],
        *grounded_graph["query_relations"],
    ]
    summary = {
        "proposal_only": True,
        "accepted_memory_modified": False,
        "promoted_overlay_modified": False,
        "runtime_behavior_modified": False,
        "safe_for_general_runtime": False,
        "segment_count": len(segment_paths),
        "proposal_item_count": len(items),
        "proposal_items_by_source": dict(sorted(source_counts.items())),
        "exploratory_relation_item_count": len(exploratory_items),
        "experimental_graph_entity_count": len(experimental_entities),
        "experimental_graph_item_count": len(experimental_entities) + len(experimental_graph_edges),
        "evidence_grounded_relation_item_count": len(grounded_graph["relations"]),
        "evidence_grounded_queryable_relation_item_count": len(grounded_graph["query_relations"]),
        "evidence_grounded_review_relation_item_count": len(grounded_graph["review_relations"]),
        "evidence_grounded_entity_count": len(grounded_graph["entities"]),
        "evidence_grounded_definition_count": len(grounded_graph["definitions"]),
        "evidence_grounded_graph_item_count": len(grounded_graph_items),
        "evidence_grounding_rejected_count": len(grounded_graph["rejected"]),
        "exploratory_relation_items_by_predicate": dict(sorted(exploratory_predicates.items())),
        "segment_paths": [str(path) for path in segment_paths],
        "exploratory_relation_segment_paths": [str(path) for path in exploratory_paths],
    }
    (root / "open_web_campaign_proposal_overlay.json").write_text(
        json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (root / "open_web_campaign_exploratory_relation_overlay.json").write_text(
        json.dumps(exploratory_items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (root / "open_web_campaign_exploratory_graph_overlay.json").write_text(
        json.dumps([*experimental_entities, *experimental_graph_edges], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "open_web_campaign_evidence_grounded_graph_overlay.json").write_text(
        json.dumps(grounded_graph_items, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "open_web_campaign_evidence_grounding_rejected.json").write_text(
        json.dumps(grounded_graph["rejected"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "open_web_campaign_evidence_grounded_review.json").write_text(
        json.dumps(grounded_graph["review_relations"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "open_web_campaign_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
