"""Deterministic entity-card extraction for Wikipedia Ingestion v2.

Produces exactly one :class:`EntityCard` per page. Candidate only.
"""

from __future__ import annotations

import re

from worldpgt.knowledge.wiki_claim_normalizer import (
    classify_risk,
    infer_entity_type,
    normalize_entity_id,
)
from worldpgt.knowledge.wiki_ingestion_v2_types import EntityCard, WikiPageRecord

# Matches a leading "Full Proper Name ..." prefix used as the canonical alias.
_FULL_NAME_RE = re.compile(r"^([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){1,3})\b")


class WikiEntityExtractor:
    """Builds one candidate entity card per curated page."""

    def extract(self, page: WikiPageRecord) -> EntityCard:
        entity_id = normalize_entity_id(page.wikidata_id, page.title)
        entity_type = infer_entity_type(page)
        aliases = self._extract_aliases(page)
        return EntityCard(
            entity_id=entity_id,
            label=page.title,
            aliases=aliases,
            entity_type=entity_type,
            source_page=page.title,
            risk=classify_risk("stable", None),
            status="candidate",
        )

    def _extract_aliases(self, page: WikiPageRecord) -> list[str]:
        """Derive deterministic aliases without losing original text.

        Aliases come from: the full proper name opening the lead paragraph, and
        the last token of the label (e.g. surname) for person-like pages.
        """
        aliases: list[str] = []
        label = page.title.strip()

        match = _FULL_NAME_RE.match(page.lead_paragraph.strip())
        if match:
            full_name = match.group(1).strip()
            if full_name and full_name != label:
                aliases.append(full_name)

        # Single-word last token (surname / short form) if multi-word label.
        parts = label.split()
        if len(parts) > 1:
            last = parts[-1]
            if last not in aliases and last != label:
                aliases.append(last)

        # Deduplicate while preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for a in aliases:
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out
