"""One-shot held-out builder for paraphrase and multi-evidence QA.

The main dataset is read only to obtain an opaque set of relation/evidence IDs.
Question selection never reads its questions, answers, templates, or failures.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable

from .dataset import _case, _compact, _evidence, _norm, _source_ids, _valid_relation, load_experimental_relations, read_jsonl, relation_id


_PARAPHRASE_FORMS = {
    "uses": ("Which resources does {subject} draw upon?", "What does {subject} depend on?"),
    "enables": ("Which possibilities are opened by {subject}?", "In what way does {subject} allow outcomes to occur?"),
    "supports": ("What role does {subject} play in sustaining?", "What is reinforced by {subject}?"),
    "runs_on": ("Which execution environment hosts {subject}?",),
    "used_for": ("For what application is {subject} employed?",),
    "works_by": ("By what process does {subject} operate?",),
    "developed_by": ("By whom was {subject} engineered?",),
}

_EXPLICIT_CLAUSES = {
    "uses": "which resources does {subject} draw upon",
    "enables": "which outcomes can {subject} bring about",
    "supports": "what does {subject} reinforce",
    "runs_on": "which execution environment hosts {subject}",
    "used_for": "for what application is {subject} employed",
    "works_by": "by what process does {subject} operate",
    "developed_by": "by whom was {subject} engineered",
}


def _case_for_groups(groups: list[list[dict]], *, question: str, category: str, multi_kind: str | None = None) -> dict:
    selected = [item for group in groups for item in group]
    primary = selected[0]
    return _case(
        primary,
        question=question,
        category=category,
        relations=selected,
        predicates=[_compact(item["predicate"]) for item in selected],
        multi_kind=multi_kind,
    )


def _main_relation_ids(main_dataset_path: str | Path) -> set[str]:
    """Read only opaque IDs from the main dataset; never inspect question text."""

    return {
        value
        for row in read_jsonl(main_dataset_path)
        for value in [*(row.get("relation_ids") or ()), *(row.get("evidence_ids") or ())]
        if isinstance(value, str)
    }


def heldout_pool_diagnostics(relations: Iterable[dict], main_ids: set[str]) -> tuple[dict[tuple[str, str], list[dict]], dict[str, dict[str, list[dict]]], dict]:
    """Return the disjoint pool and its predicate-density distribution.

    Density is measured after the main-v2 ID exclusion, because this is the
    actual pool from which a leakage-free held-out set can be drawn.
    """

    valid = [row for row in relations if _valid_relation(row) is None]
    all_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in valid:
        all_groups[(_norm(row["subject"]), str(row["predicate"]))].append(row)
    # A group is clean only if *every* valid relation for subject+predicate is
    # unseen.  This makes its "all valid objects" expectation disjoint too.
    clean_groups = {
        key: sorted(rows, key=relation_id)
        for key, rows in all_groups.items()
        if not {relation_id(row) for row in rows} & main_ids
    }
    by_subject: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for (subject, predicate), rows in clean_groups.items():
        by_subject[subject][predicate] = rows
    density = Counter(len(groups) for groups in by_subject.values())
    return clean_groups, by_subject, {
        "relation_density_distribution": {
            "predicate_groups_per_subject": dict(sorted(density.items())),
            "subjects_with_1_predicate_group": density[1],
            "subjects_with_2_predicate_groups": density[2],
            "subjects_with_3_or_more_predicate_groups": sum(
                count for degree, count in density.items() if degree >= 3
            ),
        },
        "clean_subject_predicate_groups": len(clean_groups),
        "clean_subjects": len(by_subject),
    }


def build_heldout_cases(relations: Iterable[dict], main_ids: set[str]) -> tuple[list[dict], dict]:
    """Construct a disjoint 20+10+10 held-out set or fail before any run."""

    clean_groups, by_subject, pool_summary = heldout_pool_diagnostics(relations, main_ids)

    used_ids: set[str] = set()
    cases: list[dict] = []

    # The implicit stratum is most constrained: exactly two clean predicate
    # groups prevents an evaluator from hiding an arbitrary third relation.
    # Each selected group is included in full, so multiple valid objects are
    # never reduced to a hidden arbitrary target.
    implicit = []
    for subject, groups in sorted(by_subject.items()):
        selected = [row for rows in groups.values() for row in rows]
        if len(groups) != 2 or not 2 <= len(selected) <= 4:
            continue
        label = _compact(selected[0]["subject"])
        implicit.append(_case_for_groups(
            [groups[predicate] for predicate in sorted(groups)],
            question=f"Tell me two key relations about {label}.",
            category="multi_evidence_implicit",
            multi_kind="implicit_two_relation_synthesis",
        ))
    implicit = implicit[:10]
    if len(implicit) < 10:
        raise RuntimeError(f"held-out implicit multi-evidence unavailable: need 10, found {len(implicit)}")
    cases.extend(implicit)
    used_ids.update(value for case in implicit for value in case["relation_ids"])

    explicit = []
    for subject, groups in sorted(by_subject.items()):
        available = [predicate for predicate in sorted(groups) if predicate in _EXPLICIT_CLAUSES and not {relation_id(row) for row in groups[predicate]} & used_ids]
        if len(available) < 2:
            continue
        left, right = available[:2]
        selected_groups = [groups[left], groups[right]]
        if sum(len(group) for group in selected_groups) > 4:
            continue
        label = _compact(selected_groups[0][0]["subject"])
        left_clause = _EXPLICIT_CLAUSES[left].format(subject=label)
        right_clause = _EXPLICIT_CLAUSES[right].format(subject=label)
        explicit.append(_case_for_groups(
            selected_groups,
            question=left_clause[:1].upper() + left_clause[1:] + ", and " + right_clause + "?",
            category="multi_evidence_explicit",
            multi_kind="specified_distinct_predicates_heldout",
        ))
    explicit = explicit[:10]
    if len(explicit) < 10:
        raise RuntimeError(f"held-out explicit multi-evidence unavailable: need 10, found {len(explicit)}")
    cases.extend(explicit)
    used_ids.update(value for case in explicit for value in case["relation_ids"])

    paraphrase = []
    form_index = Counter()
    for (subject, predicate), rows in sorted(clean_groups.items()):
        ids = {relation_id(row) for row in rows}
        if ids & used_ids or predicate not in _PARAPHRASE_FORMS:
            continue
        label = _compact(rows[0]["subject"])
        forms = _PARAPHRASE_FORMS[predicate]
        question = forms[form_index[predicate] % len(forms)].format(subject=label)
        form_index[predicate] += 1
        paraphrase.append(_case_for_groups([rows], question=question, category="paraphrase"))
    paraphrase = paraphrase[:20]
    if len(paraphrase) < 20:
        raise RuntimeError(f"held-out paraphrase unavailable: need 20, found {len(paraphrase)}")
    cases.extend(paraphrase)

    heldout_ids = {value for case in cases for value in [*case["relation_ids"], *case["evidence_ids"]]}
    overlap = heldout_ids & main_ids
    if overlap:
        raise AssertionError("held-out overlap detected")
    summary = {
        "version": "heldout_v1",
        "total_cases": len(cases),
        "cases_per_category": dict(sorted(Counter(case["category"] for case in cases).items())),
        "main_relation_or_evidence_id_count": len(main_ids),
        "heldout_relation_or_evidence_id_count": len(heldout_ids),
        "zero_overlap_relation_ids": len(overlap) == 0,
        "zero_overlap_evidence_ids": len(overlap) == 0,
        "overlap_count": len(overlap),
        "expected_objects_are_complete_per_selected_subject_predicate_group": True,
        **pool_summary,
    }
    return cases, summary


def write_heldout_dataset(output_dir: str | Path, *, main_dataset_path: str | Path) -> dict:
    """Write the held-out split; this is the sole generation point before run."""

    main_ids = _main_relation_ids(main_dataset_path)
    cases, summary = build_heldout_cases(load_experimental_relations(), main_ids)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("dataset.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases), encoding="utf-8"
    )
    root.joinpath("dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
