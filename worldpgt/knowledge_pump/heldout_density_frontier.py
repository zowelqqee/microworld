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
