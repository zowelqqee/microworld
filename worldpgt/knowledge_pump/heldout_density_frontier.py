"""Build a target-first acquisition frontier for a leakage-free held-out pool.

This module deliberately selects existing graph subjects with one disjoint
predicate group.  It never promotes facts and never reads benchmark question
or answer text: only opaque relation/evidence identifiers are used to exclude
the main split.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from worldpgt.benchmarks.open_book_qa.dataset import _compact, _norm, _valid_relation, relation_id
from worldpgt.benchmarks.open_book_qa.heldout_v1 import heldout_pool_diagnostics


def build_density_frontier(relations: Iterable[dict], main_ids: set[str]) -> tuple[list[dict], dict]:
    """Return deterministic subjects that need an independent second relation.

    A target is retained only when its complete currently clean group is
    disjoint from the main split.  The frontier carries no benchmark question,
    expected answer, or selected target edge.
    """

    relations = list(relations)
    clean_groups, by_subject, pool = heldout_pool_diagnostics(relations, main_ids)
    predicates_touched_by_main: dict[str, set[str]] = {}
    for relation in relations:
        if _valid_relation(relation) is not None or relation_id(relation) not in main_ids:
            continue
        subject = _norm(str(relation["subject"]))
        predicates_touched_by_main.setdefault(subject, set()).add(str(relation["predicate"]))
    rows: list[dict] = []
    for subject_key, predicates in sorted(by_subject.items()):
        if len(predicates) != 1:
            continue
        predicate, group = next(iter(predicates.items()))
        exemplar = group[0]
        rows.append({
            "subject": _compact(exemplar["subject"]),
            "surface_subject": _compact(exemplar["subject"]),
            "normalized_subject": subject_key,
            "canonical_entity": None,
            "canonical_resolution_status": "unresolved_not_fetched",
            "existing_clean_predicates": [predicate],
            "predicates_touched_by_main": sorted(predicates_touched_by_main.get(subject_key, set())),
            "required_new_predicate": True,
            "existing_clean_relation_count": len(group),
            "existing_source_ids": sorted({
                _compact(value)
                for item in group
                for value in [item.get("source_url"), item.get("source_page")]
                if _compact(value)
            }),
            "acquisition_goal": "find a new independently evidenced predicate group",
            "selection_basis": "zero_overlap_subject_with_exactly_one_clean_predicate_group",
        })
    summary = {
        "frontier_version": "heldout_density_frontier_v1",
        "target_subject_count": len(rows),
        "target_density_before_acquisition": {"predicate_groups_per_subject": {"1": len(rows)}},
        "selection_reads": "relation_ids/evidence_ids and graph relation metadata only",
        "selection_excludes": "benchmark questions, expected answers, templates, and failure artifacts",
        "source_predicate_distribution": dict(sorted(
            Counter(row["existing_clean_predicates"][0] for row in rows).items()
        )),
        "pool": pool,
    }
    return rows, summary


def attach_wikipedia_resolution_layer(frontier: Iterable[dict], manifest: Iterable[dict]) -> list[dict]:
    """Attach optional canonical pages without replacing evidence-local terms.

    Each returned row retains ``surface_subject`` even when MediaWiki resolves
    a redirect or disambiguates the page title.  A missing page is a useful
    result: it remains available to non-Wikipedia acquisition lanes.
    """

    by_surface = {_norm(str(row.get("title") or "")): row for row in manifest}
    resolved: list[dict] = []
    for target in frontier:
        row = dict(target)
        surface = _compact(str(row.get("surface_subject") or row.get("subject") or ""))
        fetched = by_surface.get(_norm(surface))
        canonical = _compact(str((fetched or {}).get("normalized_title") or ""))
        success = (fetched or {}).get("fetch_status") == "success" and bool(canonical)
        row.update({
            "surface_subject": surface,
            "canonical_entity": canonical if success else None,
            "canonical_source_url": _compact(str((fetched or {}).get("source_url") or "")) if success else None,
            "canonical_resolution_status": (
                "resolved_wikipedia_title" if success else "unresolved_wikipedia_title"
            ),
            "surface_retained": True,
            "canonical_is_additive": True,
        })
        resolved.append(row)
    return resolved


def attach_wikidata_exact_resolution(
    frontier: Iterable[dict],
    search_results: dict[str, Iterable[dict]],
    *,
    entities: dict[str, dict] | None = None,
    enable_alias_disambiguation: bool = False,
) -> list[dict]:
    """Attach only one unambiguous exact Wikidata item to each surface subject.

    This is deliberately a resolution layer, not a filter: every input target
    is returned.  Canonical entities and unresolved evidence-local fragments
    therefore remain separate, measurable cohorts instead of the latter
    crowding out the former through an arbitrary alphabetical frontier.
    """

    if enable_alias_disambiguation:
        if entities is None:
            raise ValueError("entities are required when alias/disambiguation is enabled")
        return attach_wikidata_alias_disambiguated_resolution(
            frontier, search_results, entities=entities,
        )

    resolved: list[dict] = []
    for target in frontier:
        row = dict(target)
        surface = _compact(str(row.get("surface_subject") or row.get("subject") or ""))
        exact = {
            (str(hit.get("id") or ""), _compact(str(hit.get("label") or "")))
            for hit in search_results.get(_norm(surface), ())
            if str(hit.get("id") or "").startswith("Q")
            and _norm(str(hit.get("label") or "")) == _norm(surface)
        }
        if len(exact) == 1:
            qid, label = next(iter(exact))
            row.update({
                "canonical_entity": label,
                "canonical_qid": qid,
                "canonical_resolution_status": "resolved_wikidata_exact",
                "canonical_is_additive": True,
                "surface_retained": True,
            })
        else:
            row.update({
                "canonical_entity": None,
                "canonical_qid": None,
                "canonical_resolution_status": (
                    "ambiguous_wikidata_exact" if len(exact) > 1 else "unresolved_wikidata_exact"
                ),
                "canonical_is_additive": True,
                "surface_retained": True,
            })
        resolved.append(row)
    return resolved


# These are metadata/publication-like instances, not subjects that should win
# a concept-vs-work label collision.  The set is deliberately about Wikidata
# classes, never particular surface forms or acronyms.
_WIKIDATA_NON_CONTENT_P31 = frozenset({
    "Q571",       # book
    "Q7725634",   # literary work
    "Q13442814",  # scholarly article
    "Q47461344",  # written work
    "Q5633421",   # academic journal
    "Q4167410",   # Wikimedia disambiguation page
    "Q27949697",  # Wikidata reason for deprecation
})


def attach_wikidata_alias_disambiguated_resolution(
    frontier: Iterable[dict], search_results: dict[str, Iterable[dict]], *, entities: dict[str, dict],
) -> list[dict]:
    """Conservative additive resolver: unique exact alias or one content label.

    This does not replace :func:`attach_wikidata_exact_resolution`; callers
    explicitly opt in after fetching candidate P31 claims.
    """
    resolved = []
    for target in frontier:
        row = dict(target)
        surface = _compact(str(row.get("surface_subject") or row.get("subject") or ""))
        hits = [h for h in search_results.get(_norm(surface), ()) if str(h.get("id") or "").startswith("Q")]
        def english_or_multilingual(hit: dict) -> bool:
            language = str(((hit.get("display") or {}).get("label") or {}).get("language") or "")
            return language in {"en", "mul"}
        exact = [h for h in hits if english_or_multilingual(h) and _norm(h.get("label") or "") == _norm(surface)]
        aliases = [h for h in hits if english_or_multilingual(h) and _norm((h.get("match") or {}).get("text") or "") == _norm(surface) and (h.get("match") or {}).get("type") == "alias"]
        def content_candidate(hit: dict) -> bool:
            claims = (entities.get(str(hit.get("id") or ""), {}).get("claims") or {}).get("P31", [])
            values = {str((((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {}).get("id") or "") for claim in claims}
            return not bool(values & _WIKIDATA_NON_CONTENT_P31)
        selected, method = None, None
        if len(exact) > 1:
            candidates = [hit for hit in exact if content_candidate(hit)]
            if len(candidates) == 1:
                selected, method = candidates[0], "exact_label_disambiguated_by_p31"
        elif not exact and len(aliases) == 1 and content_candidate(aliases[0]):
            selected, method = aliases[0], "unique_wikidata_alias"
        if selected:
            row.update({"canonical_entity": _compact(str(selected.get("label") or "")), "canonical_qid": str(selected["id"]), "canonical_resolution_status": "resolved_wikidata_alias_or_disambiguated", "canonical_resolution_method": method, "canonical_is_additive": True, "surface_retained": True})
        else:
            row.update({"canonical_entity": None, "canonical_qid": None, "canonical_resolution_status": "unresolved_after_alias_disambiguation", "canonical_resolution_method": None, "canonical_is_additive": True, "surface_retained": True})
        resolved.append(row)
    return resolved


def require_wikipedia_anchor(
    resolved: Iterable[dict],
    entities: dict[str, dict],
) -> list[dict]:
    """Require an independent English Wikipedia anchor before auto-proposing.

    An exact Wikidata label alone is insufficient for short labels and
    acronyms: it can name a different entity with the same surface.  Rows that
    lack an anchor remain in the manifest with their exact QID for later
    review, rather than being silently removed from the acquisition cohort.
    """

    anchored: list[dict] = []
    for original in resolved:
        row = dict(original)
        qid = str(row.get("canonical_qid") or "")
        enwiki = (entities.get(qid, {}).get("sitelinks") or {}).get("enwiki") or {}
        title = _compact(str(enwiki.get("title") or ""))
        if row.get("canonical_resolution_status") == "resolved_wikidata_exact" and title:
            row.update({
                "canonical_resolution_status": "resolved_wikidata_exact_enwiki",
                "canonical_wikipedia_title": title,
                "canonical_source_url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            })
        elif qid:
            row.update({
                "canonical_resolution_status": "exact_wikidata_without_enwiki_anchor",
                "canonical_wikipedia_title": None,
                "canonical_source_url": None,
            })
        anchored.append(row)
    return anchored
