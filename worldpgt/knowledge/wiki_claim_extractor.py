"""Deterministic claim extraction for Wikipedia Ingestion v2.

Extracts, from a curated page record:
* one ``definition_claim`` (first-sentence ``is_a`` definition)
* zero or more ``relation_claim`` items (safe deterministic relations)
* zero or more ``context_link`` items (one weak edge per outgoing link)
* zero or more ``source_qualified_fact`` items (volatile, time-sensitive)

All rules are explicit regex/keyword rules. Volatile / source-dependent
claims are routed to ``source_qualified_fact`` and never become stable
relation claims.
"""

from __future__ import annotations

import re

from worldpgt.knowledge.wiki_claim_normalizer import (
    classify_risk,
    classify_stability,
    is_volatile_text,
)
from worldpgt.knowledge.wiki_ingestion_v2_types import (
    ContextLink,
    DefinitionClaim,
    RelationClaim,
    SourceQualifiedFact,
    WikiPageRecord,
)

# Sentence splitter on terminal punctuation followed by whitespace.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Copula + article for definition extraction.
_COPULA_RE = re.compile(
    r"\b(?:is|are|was|were)\s+(?:a|an|the)\s+(.+)", re.IGNORECASE
)

# Markers at which a definition object is truncated to stay a clean noun phrase.
_DEF_TRUNCATE_MARKERS: tuple[str, ...] = (
    " known for",
    " known as",
    " that ",
    " which ",
    " based in",
    " founded ",
    ", ",
)

# Leadership relation: "known for his leadership of Tesla and SpaceX".
_LEADERSHIP_RE = re.compile(
    r"((?:known for\s+(?:his|her|its|their)\s+)?leadership of\s+(.+?))(?:\.|$)",
    re.IGNORECASE,
)

# Generic known_for (non-leadership).
_KNOWN_FOR_RE = re.compile(
    r"known for\s+(?:his|her|its|their)?\s*(.+?)(?:\.|$)", re.IGNORECASE
)

# Production / development / publishing verbs in lead text.
_VERB_RELATIONS: tuple[tuple[str, str], ...] = (
    (r"\bmanufactures\b", "produces"),
    (r"\bproduces\b", "produces"),
    (r"\bdevelops\b", "develops"),
    (r"\bpublishes\b", "publishes"),
)

# Splits an object list on "and" / commas / semicolons.
_OBJ_LIST_SPLIT = re.compile(r"\s+and\s+|,\s*|;\s*", re.IGNORECASE)

# Volatile estimation patterns -> source_qualified_fact.
_NET_WORTH_RE = re.compile(r"net[\s-]?worth", re.IGNORECASE)
_AMOUNT_RE = re.compile(
    r"(US\$\s?[\d.,]+\s*(?:trillion|billion|million)?)", re.IGNORECASE
)
_WEALTHIEST_RE = re.compile(r"\b(wealthiest|richest)\b", re.IGNORECASE)
_RANKED_RE = re.compile(r"\branked\s+(?:number\s*)?(\d+)\b", re.IGNORECASE)

# Recognised source organisations for source-qualified facts.
_KNOWN_SOURCES: tuple[str, ...] = ("Forbes", "Bloomberg", "Fortune")


class WikiClaimExtractor:
    """Builds definition / relation / context / source-qualified candidates."""

    def extract_definition(self, page: WikiPageRecord) -> DefinitionClaim | None:
        sentences = self._sentences(page.lead_paragraph)
        if not sentences:
            return None
        first = sentences[0]
        match = _COPULA_RE.search(first)
        if not match:
            return None
        obj = self._truncate_definition(match.group(1))
        if not obj:
            return None
        # Definitions are inherently stable/low regardless of surface tokens.
        return DefinitionClaim(
            subject=page.title,
            predicate="is_a",
            object=obj,
            source_page=page.title,
            evidence_text=first.strip(),
            stability="stable",
            risk="low",
            status="candidate",
        )

    def extract_relations(self, page: WikiPageRecord) -> list[RelationClaim]:
        relations: list[RelationClaim] = []
        for sentence in self._sentences(page.lead_paragraph):
            # Volatile estimation sentences are handled as source-qualified
            # facts only — never as stable relation claims.
            if self._is_source_qualified_sentence(sentence):
                continue
            relations.extend(self._relations_from_sentence(page, sentence))
        relations.extend(self._relations_from_infobox(page))
        return relations

    def extract_context_links(self, page: WikiPageRecord) -> list[ContextLink]:
        links: list[ContextLink] = []
        for link in page.links:
            links.append(
                ContextLink(
                    source_page=page.title,
                    surface=link.surface,
                    target=link.target,
                    relation="mentioned_with",
                    strength="weak",
                    status="candidate",
                )
            )
        return links

    def extract_source_qualified_facts(
        self, page: WikiPageRecord
    ) -> list[SourceQualifiedFact]:
        facts: list[SourceQualifiedFact] = []
        as_of = self._as_of(page)
        for sentence in self._sentences(page.lead_paragraph):
            if not self._is_source_qualified_sentence(sentence):
                continue
            fact = self._sqf_from_sentence(page, sentence, as_of)
            if fact is not None:
                facts.append(fact)
        return facts

    # ------------------------------------------------------------------
    # Relation helpers
    # ------------------------------------------------------------------

    def _relations_from_sentence(
        self, page: WikiPageRecord, sentence: str
    ) -> list[RelationClaim]:
        out: list[RelationClaim] = []
        lead_match = _LEADERSHIP_RE.search(sentence)
        if lead_match:
            evidence = lead_match.group(1).strip()
            for obj in self._split_objects(lead_match.group(2)):
                out.append(
                    self._relation(page, "leader_of", obj, evidence)
                )
            # Leadership consumed this sentence's "known for".
            return out

        kf_match = _KNOWN_FOR_RE.search(sentence)
        if kf_match:
            evidence = kf_match.group(0).strip()
            for obj in self._split_objects(kf_match.group(1)):
                out.append(self._relation(page, "known_for", obj, evidence))

        for pattern, predicate in _VERB_RELATIONS:
            m = re.search(pattern + r"\s+(.+?)(?:\.|$)", sentence, re.IGNORECASE)
            if m:
                evidence = m.group(0).strip()
                for obj in self._split_objects(m.group(1)):
                    out.append(self._relation(page, predicate, obj, evidence))
        return out

    def _relations_from_infobox(
        self, page: WikiPageRecord
    ) -> list[RelationClaim]:
        out: list[RelationClaim] = []
        lowered = {k.lower(): v for k, v in page.infobox.items()}

        if "known_for" in lowered:
            evidence = f"known_for: {lowered['known_for']}"
            for obj in self._split_objects(lowered["known_for"]):
                out.append(self._relation(page, "known_for", obj, evidence))

        for key in ("products", "produces"):
            if key in lowered:
                evidence = f"{key}: {lowered[key]}"
                for obj in self._split_objects(lowered[key]):
                    out.append(self._relation(page, "produces", obj, evidence))

        for key in ("founder", "founders", "founded_by"):
            if key in lowered:
                for founder in self._split_objects(lowered[key]):
                    evidence = f"{founder} founded {page.title}"
                    out.append(
                        RelationClaim(
                            subject=founder,
                            predicate="founded",
                            object=page.title,
                            source_page=page.title,
                            evidence_text=evidence,
                            stability="semi_stable",
                            risk="medium",
                            status="candidate",
                        )
                    )
        return out

    def _relation(
        self, page: WikiPageRecord, predicate: str, obj: str, evidence: str
    ) -> RelationClaim:
        stability = classify_stability(evidence, predicate)
        risk = classify_risk(stability, predicate)
        return RelationClaim(
            subject=page.title,
            predicate=predicate,  # type: ignore[arg-type]
            object=obj,
            source_page=page.title,
            evidence_text=evidence,
            stability=stability,
            risk=risk,
            status="candidate",
        )

    # ------------------------------------------------------------------
    # Source-qualified-fact helpers
    # ------------------------------------------------------------------

    def _is_source_qualified_sentence(self, sentence: str) -> bool:
        # Strict, shared detector: a concrete current/source-dependent measurement.
        return is_volatile_text(sentence)

    def _sqf_from_sentence(
        self, page: WikiPageRecord, sentence: str, as_of: str
    ) -> SourceQualifiedFact | None:
        source_name = self._source_name(sentence)
        lowered = sentence.lower()

        if _NET_WORTH_RE.search(sentence) and (
            "estimat" in lowered or _AMOUNT_RE.search(sentence)
        ):
            amount = _AMOUNT_RE.search(sentence)
            obj = (
                amount.group(1).strip()
                if amount
                else "net worth estimate (amount not specified)"
            )
            return SourceQualifiedFact(
                subject=page.title,
                predicate="estimated_net_worth",
                object=obj,
                source_name=source_name,
                as_of=as_of,
                claim_type="time_sensitive_estimate",
                source_page=page.title,
                evidence_text=sentence.strip(),
                stability="volatile",
                risk="high",
                status="candidate",
                requires_recheck=True,
            )

        if _WEALTHIEST_RE.search(sentence):
            return SourceQualifiedFact(
                subject=page.title,
                predicate="wealth_rank",
                object=sentence.strip(),
                source_name=source_name,
                as_of=as_of,
                claim_type="time_sensitive_ranking",
                source_page=page.title,
                evidence_text=sentence.strip(),
                stability="volatile",
                risk="high",
                status="candidate",
                requires_recheck=True,
            )

        rank = _RANKED_RE.search(sentence)
        if rank:
            return SourceQualifiedFact(
                subject=page.title,
                predicate="ranking",
                object=f"rank {rank.group(1)}",
                source_name=source_name,
                as_of=as_of,
                claim_type="time_sensitive_ranking",
                source_page=page.title,
                evidence_text=sentence.strip(),
                stability="volatile",
                risk="high",
                status="candidate",
                requires_recheck=True,
            )

        # Other recognised current-measurement patterns (office, market cap,
        # price, population). Detected by the shared strict detector; captured
        # explicitly as source-qualified so nothing volatile is silently lost.
        if is_volatile_text(sentence):
            predicate, claim_type = self._classify_volatile_predicate(lowered)
            return SourceQualifiedFact(
                subject=page.title,
                predicate=predicate,
                object=sentence.strip(),
                source_name=source_name,
                as_of=as_of,
                claim_type=claim_type,
                source_page=page.title,
                evidence_text=sentence.strip(),
                stability="volatile",
                risk="high",
                status="candidate",
                requires_recheck=True,
            )
        return None

    def _classify_volatile_predicate(self, lowered: str) -> tuple[str, str]:
        if "current" in lowered and any(
            o in lowered for o in ("ceo", "chief executive", "president", "office")
        ):
            return "current_office_holder", "time_sensitive_office"
        if "market cap" in lowered:
            return "market_capitalization", "time_sensitive_estimate"
        if "price" in lowered:
            return "current_price", "time_sensitive_estimate"
        if "population" in lowered:
            return "current_population", "time_sensitive_estimate"
        return "current_estimate", "time_sensitive_estimate"

    def _source_name(self, sentence: str) -> str:
        for src in _KNOWN_SOURCES:
            if src.lower() in sentence.lower():
                return src
        return "unknown_source"

    def _as_of(self, page: WikiPageRecord) -> str:
        retrieved = page.source.retrieved_at or ""
        m = re.match(r"(\d{4})-(\d{2})", retrieved)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
        return retrieved

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _sentences(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]

    def _truncate_definition(self, raw: str) -> str:
        obj = raw.strip()
        lowered = obj.lower()
        cut = len(obj)
        for marker in _DEF_TRUNCATE_MARKERS:
            idx = lowered.find(marker)
            if idx != -1:
                cut = min(cut, idx)
        obj = obj[:cut].strip()
        return obj.rstrip(".").strip()

    def _split_objects(self, raw: str) -> list[str]:
        out: list[str] = []
        for piece in _OBJ_LIST_SPLIT.split(raw or ""):
            cleaned = piece.strip().rstrip(".").strip()
            # Drop leading articles for cleaner object surfaces.
            cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
            if cleaned:
                out.append(cleaned)
        return out
