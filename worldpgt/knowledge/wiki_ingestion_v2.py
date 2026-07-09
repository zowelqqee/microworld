"""Wikipedia/Wikidata Ingestion v2 orchestrator (read-only / dry-run).

Converts curated Wikipedia-style page records into structured *candidate*
memory items, runs a deterministic review gate, and builds a summary.

Nothing here is trusted runtime memory. ``safe_for_runtime_memory`` is always
False. No module writes to sense_memory.py, accepted memory artifacts, or any
generation behavior, and no network access is performed.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from worldpgt.knowledge.wiki_claim_extractor import WikiClaimExtractor
from worldpgt.knowledge.wiki_claim_normalizer import is_volatile_text
from worldpgt.knowledge.wiki_entity_extractor import WikiEntityExtractor
from worldpgt.knowledge.wiki_ingestion_v2_types import (
    ContextLink,
    DefinitionClaim,
    EntityCard,
    RelationClaim,
    ReviewIssue,
    SourceQualifiedFact,
    WikiPageRecord,
)
from worldpgt.knowledge.wiki_page_reader import WikiPageReader


@dataclass
class IngestionResult:
    """All candidate items plus review + summary for one ingestion run."""

    candidates: list = field(default_factory=list)
    review_issues: list[ReviewIssue] = field(default_factory=list)
    duplicates_removed: int = 0
    summary: dict = field(default_factory=dict)


def as_dict(obj: object) -> object:
    """Recursively convert dataclasses/containers into JSON-safe structures."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: as_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [as_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: as_dict(v) for k, v in obj.items()}
    return obj


class WikiIngestionV2:
    """Deterministic read-only ingestion pipeline (candidate generation only)."""

    def __init__(self) -> None:
        self._entity_extractor = WikiEntityExtractor()
        self._claim_extractor = WikiClaimExtractor()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> list[WikiPageRecord]:
        return WikiPageReader().load(path)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_page(self, page: WikiPageRecord) -> list:
        """Return all candidate items for one page (order is deterministic)."""
        items: list = [self._entity_extractor.extract(page)]
        definition = self._claim_extractor.extract_definition(page)
        if definition is not None:
            items.append(definition)
        items.extend(self._claim_extractor.extract_relations(page))
        items.extend(self._claim_extractor.extract_context_links(page))
        items.extend(self._claim_extractor.extract_source_qualified_facts(page))
        return items

    def run(self, pages: list[WikiPageRecord]) -> IngestionResult:
        raw: list = []
        for page in pages:
            raw.extend(self.extract_page(page))

        deduped, dup_issues = deduplicate(raw)
        issues = review_candidates(deduped) + dup_issues
        summary = build_summary(pages, deduped, issues)
        return IngestionResult(
            candidates=deduped,
            review_issues=issues,
            duplicates_removed=len(raw) - len(deduped),
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _candidate_key(item: object) -> str:
    return json.dumps(as_dict(item), sort_keys=True, ensure_ascii=False)


def deduplicate(candidates: list) -> tuple[list, list[ReviewIssue]]:
    """Remove exact-duplicate candidates; report each removal as a warning."""
    seen: set[str] = set()
    out: list = []
    issues: list[ReviewIssue] = []
    for idx, item in enumerate(candidates):
        key = _candidate_key(item)
        if key in seen:
            issues.append(
                ReviewIssue(
                    severity="warning",
                    code="duplicate_candidate",
                    message="exact duplicate candidate removed",
                    item_type=getattr(item, "item_type", "unknown"),
                    item_index=idx,
                )
            )
            continue
        seen.add(key)
        out.append(item)
    return out, issues


# ---------------------------------------------------------------------------
# Review gate
# ---------------------------------------------------------------------------


def review_candidates(candidates: list) -> list[ReviewIssue]:
    """Deterministically validate candidates. Never weakens; only flags.

    Checks: missing source, missing evidence for relation claims, volatile
    claim not marked volatile, source-qualified fact without ``as_of``,
    context_link marked strong/trusted, relation claim with empty
    subject/object, entity card with no label, and duplicate candidates.
    """
    issues: list[ReviewIssue] = []
    seen_keys: set[str] = set()

    for idx, item in enumerate(candidates):
        item_type = getattr(item, "item_type", "unknown")

        # Duplicate detection (report-only here; pipeline also dedupes).
        key = _candidate_key(item)
        if key in seen_keys:
            issues.append(
                ReviewIssue(
                    severity="warning",
                    code="duplicate_candidate",
                    message="duplicate candidate detected",
                    item_type=item_type,
                    item_index=idx,
                )
            )
        seen_keys.add(key)

        if isinstance(item, ContextLink):
            issues.extend(_review_context_link(item, idx))
            continue

        # Source presence (all non-context items carry source_page).
        if not getattr(item, "source_page", ""):
            issues.append(
                ReviewIssue(
                    severity="error",
                    code="missing_source",
                    message="candidate is missing source_page",
                    item_type=item_type,
                    item_index=idx,
                )
            )

        if isinstance(item, EntityCard) and not item.label:
            issues.append(
                ReviewIssue(
                    severity="error",
                    code="entity_card_no_label",
                    message="entity card has no label",
                    item_type=item_type,
                    item_index=idx,
                )
            )

        if isinstance(item, (RelationClaim,)):
            issues.extend(_review_relation_claim(item, idx))

        if isinstance(item, SourceQualifiedFact):
            issues.extend(_review_source_qualified_fact(item, idx))

        if isinstance(item, DefinitionClaim):
            if not item.subject or not item.object:
                issues.append(
                    ReviewIssue(
                        severity="error",
                        code="definition_empty_field",
                        message="definition claim has empty subject/object",
                        item_type=item_type,
                        item_index=idx,
                    )
                )

    return issues


def _review_context_link(item: ContextLink, idx: int) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    if not item.source_page:
        issues.append(
            ReviewIssue(
                severity="error",
                code="missing_source",
                message="context_link is missing source_page",
                item_type="context_link",
                item_index=idx,
            )
        )
    if item.strength != "weak":
        issues.append(
            ReviewIssue(
                severity="error",
                code="context_link_not_weak",
                message="context_link must be weak; hyperlink is not a trusted fact",
                item_type="context_link",
                item_index=idx,
            )
        )
    if item.status != "candidate":
        issues.append(
            ReviewIssue(
                severity="error",
                code="context_link_trusted",
                message="context_link must remain a candidate, never trusted",
                item_type="context_link",
                item_index=idx,
            )
        )
    if str(getattr(item, "relation", "")) in {"is_a", "leader_of", "founded"}:
        issues.append(
            ReviewIssue(
                severity="error",
                code="context_link_strong_relation",
                message="context_link relation must stay weak/associative",
                item_type="context_link",
                item_index=idx,
            )
        )
    return issues


def _review_relation_claim(item: RelationClaim, idx: int) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    if not item.subject or not item.object:
        issues.append(
            ReviewIssue(
                severity="error",
                code="relation_empty_field",
                message="relation claim has empty subject/object",
                item_type="relation_claim",
                item_index=idx,
            )
        )
    if not item.evidence_text:
        issues.append(
            ReviewIssue(
                severity="error",
                code="relation_missing_evidence",
                message="relation claim has no evidence_text",
                item_type="relation_claim",
                item_index=idx,
            )
        )
    # Volatile content must not be presented as a stable relation claim.
    if is_volatile_text(item.evidence_text + " " + item.object) and item.stability != "volatile":
        issues.append(
            ReviewIssue(
                severity="error",
                code="volatile_not_marked",
                message="volatile content in a non-volatile relation claim",
                item_type="relation_claim",
                item_index=idx,
            )
        )
    return issues


def _review_source_qualified_fact(
    item: SourceQualifiedFact, idx: int
) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    if not item.as_of:
        issues.append(
            ReviewIssue(
                severity="error",
                code="sqf_missing_as_of",
                message="source_qualified_fact is missing as_of",
                item_type="source_qualified_fact",
                item_index=idx,
            )
        )
    if item.stability != "volatile":
        issues.append(
            ReviewIssue(
                severity="error",
                code="sqf_not_volatile",
                message="source_qualified_fact must be marked volatile",
                item_type="source_qualified_fact",
                item_index=idx,
            )
        )
    if not item.source_name:
        issues.append(
            ReviewIssue(
                severity="warning",
                code="sqf_missing_source_name",
                message="source_qualified_fact has no source_name",
                item_type="source_qualified_fact",
                item_index=idx,
            )
        )
    return issues


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def build_summary(
    pages: list[WikiPageRecord], candidates: list, issues: list[ReviewIssue]
) -> dict:
    by_item_type: Counter[str] = Counter()
    by_stability: Counter[str] = Counter()
    by_risk: Counter[str] = Counter()

    for item in candidates:
        by_item_type[getattr(item, "item_type", "unknown")] += 1
        stability = getattr(item, "stability", None)
        if stability is not None:
            by_stability[stability] += 1
        risk = getattr(item, "risk", None)
        if risk is not None:
            by_risk[risk] += 1

    errors = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")

    return {
        "pages_total": len(pages),
        "candidates_total": len(candidates),
        "by_item_type": dict(sorted(by_item_type.items())),
        "by_stability": dict(sorted(by_stability.items())),
        "by_risk": dict(sorted(by_risk.items())),
        "review_errors_count": errors,
        "review_warnings_count": warnings,
        "volatile_claims_count": by_stability.get("volatile", 0),
        "context_links_count": by_item_type.get("context_link", 0),
        "relation_claims_count": by_item_type.get("relation_claim", 0),
        "entity_cards_count": by_item_type.get("entity_card", 0),
        "definition_claims_count": by_item_type.get("definition_claim", 0),
        "source_qualified_facts_count": by_item_type.get("source_qualified_fact", 0),
        "safe_for_runtime_memory": False,
    }
