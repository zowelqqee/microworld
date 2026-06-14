"""Wiki page candidate builder for Wikipedia Self-Ingestion v1.

Converts normalized local documents into the existing curated wiki page shape
(matching ``wiki_pages_curated_v2.json``) when they follow the simple local
convention::

    TITLE: Falcon 9
    TYPE: product
    LEAD: Falcon 9 is a reusable launch vehicle developed by SpaceX.
    INFOBOX:
    developed_by: SpaceX
    category: reusable launch vehicle
    LINKS:
    SpaceX
    reusable rocket

Documents that do not parse into a structured page are returned as
``RawClaim`` objects for the conflict detector. The structured pages are fed to
the existing (unchanged) ingestion v2 pipeline.
"""

from __future__ import annotations

from worldpgt.self_ingestion.types import NormalizedDocument, RawClaim

_RETRIEVED_AT = "2026-06-14"


def _parse_structured(doc: NormalizedDocument) -> dict | None:
    """Parse the TITLE/TYPE/LEAD/INFOBOX/LINKS convention into a page dict."""
    text = doc.raw_text or ""
    if "title:" not in text.lower():
        return None

    title = type_hint = lead = ""
    infobox: dict[str, str] = {}
    links: list[dict] = []
    section = None

    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("TITLE:"):
            title = stripped.split(":", 1)[1].strip(); section = None
        elif upper.startswith("TYPE:"):
            type_hint = stripped.split(":", 1)[1].strip().lower(); section = None
        elif upper.startswith("LEAD:"):
            lead = stripped.split(":", 1)[1].strip(); section = None
        elif upper.startswith("INFOBOX:"):
            section = "infobox"
        elif upper.startswith("LINKS:"):
            section = "links"
        elif not stripped:
            continue
        elif section == "infobox" and ":" in stripped:
            k, v = stripped.split(":", 1)
            infobox[k.strip()] = v.strip()
        elif section == "links":
            links.append({"surface": stripped, "target": stripped})

    if not title or not lead:
        return None

    return {
        "page_id": f"wiki:{title.replace(' ', '_')}",
        "title": title,
        "wikidata_id": None,
        "entity_type_hint": type_hint or None,
        "lead_paragraph": lead,
        "infobox": infobox,
        "links": links,
        "categories": [],
        "source": {
            "source_id": f"self_ingestion_v1:{doc.source_id}",
            "source_type": "curated_wiki_snippet",
            "source_url": None,
            "retrieved_at": _RETRIEVED_AT,
        },
    }


def build_pages(documents: list[NormalizedDocument]) -> tuple[list[tuple[str, dict]], list[RawClaim]]:
    """Return (structured pages tagged by source_id, raw claims)."""
    pages: list[tuple[str, dict]] = []
    raw_claims: list[RawClaim] = []
    for doc in documents:
        if doc.read_errors:
            raw_claims.append(RawClaim(doc.source_id, doc.trust_tier,
                                       doc.raw_text.strip() or "(unreadable)"))
            continue
        page = _parse_structured(doc)
        if page is not None:
            pages.append((doc.source_id, page))
        else:
            text = (doc.raw_text or "").strip()
            if text:
                raw_claims.append(RawClaim(doc.source_id, doc.trust_tier, text))
    return pages, raw_claims
