"""Independent paraphrase held-out set: non-standard grammatical forms.

Purpose: measure the semantic predicate fallback on question shapes that were
NOT consulted while developing it.  Every template below was written once for
this builder and never used in the main dataset builder, heldout_v1/v2/v3
templates, or the fallback's example-phrase table.  The grammatical families
are deliberately awkward: fronted-agent passives, nominalisations ("the
founders of", "the maker of"), and questions with no predicate verb at all
("What are the products of X?").

Evidence graph: the frozen heldout_v3 relation snapshot, reused unchanged so
results are directly comparable to heldout_v3 runs.  Where possible, cases
draw on relations whose IDs do not appear in any heldout_v3 case.

Negative cases ask a paraphrased question about a predicate the subject does
not have (and whose wording is absent from the evidence text) — the direct
guardrail measurement for false confident matches introduced by the fallback.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
import shutil

from worldpgt.benchmarks.open_book_qa.dataset import (
    _case,
    _evidence,
    _norm,
    _valid_relation,
    read_jsonl,
    relation_id,
)

_ROOT = Path("artifacts/open_book_qa")
_SOURCE = _ROOT / "heldout_v3"
_OUTPUT = _ROOT / "independent_paraphrase_v1"

# One or two novel grammatical forms per predicate present in the frozen graph.
# form kinds: fronted-agent passive, nominalisation, verbless.
_INDEPENDENT_FORMS: dict[str, tuple[str, ...]] = {
    "developed_by": (
        "By which company was {subject} developed?",
        "Who is behind the development of {subject}?",
    ),
    "founded_by": (
        "Who were the founders of {subject}?",
        "By whom was {subject} set up?",
    ),
    "headquartered_in": (
        "Where does {subject} keep its head office?",
        "In which city are the headquarters of {subject}?",
    ),
    "owned_by": (
        "Who is the owner of {subject}?",
        "By which organisation is {subject} controlled?",
    ),
    "product_of": (
        "Which firm built {subject}?",
        "Who is the maker of {subject}?",
    ),
    "produces": ("What are the products of {subject}?",),
    "used_for": (
        "For which purposes is {subject} intended?",
        "What is the intended use of {subject}?",
    ),
    "created_by": ("Whose creation is {subject}?",),
    "parent_company_of": ("Which subsidiaries sit under {subject}?",),
    "runs_on": ("On top of what does {subject} run?",),
}

# Negative probes: predicate asked, but absent for the subject.  Kept to the
# same novel wording so the guardrail measures the new fallback path.
_NEGATIVE_FORMS: dict[str, str] = {
    "developed_by": "By which company was {subject} developed?",
    "founded_by": "Who were the founders of {subject}?",
    "headquartered_in": "Where does {subject} keep its head office?",
    "owned_by": "Who is the owner of {subject}?",
}

_ANSWER_CASES = 16
_NEGATIVE_CASES = 4


def build() -> dict:
    rng = random.Random(20260716)
    relations = [
        row for row in json.load((_SOURCE / "frozen_relations.json").open())
        if _valid_relation(row) is None
    ]
    v3_used_ids = {
        rid
        for case in read_jsonl(_SOURCE / "dataset.jsonl")
        for rid in case["relation_ids"]
    }

    by_predicate: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(relations, key=relation_id):
        by_predicate[str(row["predicate"])].append(row)
    # Prefer relations heldout_v3 never asked about.
    for rows in by_predicate.values():
        rows.sort(key=lambda row: (relation_id(row) in v3_used_ids, relation_id(row)))

    cases: list[dict] = []
    form_cursor: dict[str, int] = defaultdict(int)
    predicate_cycle = [p for p in _INDEPENDENT_FORMS if by_predicate.get(p)]
    cursor: dict[str, int] = defaultdict(int)
    while len(cases) < _ANSWER_CASES and predicate_cycle:
        for predicate in list(predicate_cycle):
            rows = by_predicate[predicate]
            if cursor[predicate] >= len(rows):
                predicate_cycle.remove(predicate)
                continue
            row = rows[cursor[predicate]]
            cursor[predicate] += 1
            forms = _INDEPENDENT_FORMS[predicate]
            template = forms[form_cursor[predicate] % len(forms)]
            form_cursor[predicate] += 1
            question = template.format(subject=row["subject"])
            # A repeated question (same subject, same predicate, different
            # object) measures nothing new and double-counts fan-out effects.
            if any(case["question"] == question for case in cases):
                continue
            cases.append(_case(
                row,
                question=question,
                category="paraphrase",
            ))
            if len(cases) == _ANSWER_CASES:
                break

    # Negatives: subject lacks the asked predicate, and the predicate's plain
    # wording does not occur in the evidence span.
    subjects: dict[str, list[dict]] = defaultdict(list)
    for row in relations:
        subjects[_norm(row["subject"])].append(row)
    negatives: list[dict] = []
    used_subjects = {case["expected_subject"] for case in cases}
    for subject_rows in (subjects[key] for key in sorted(subjects)):
        primary = subject_rows[0]
        if primary["subject"] in used_subjects:
            continue
        present = {str(row["predicate"]) for row in subject_rows}
        evidence = " ".join(_evidence(row) for row in subject_rows)
        for predicate, template in _NEGATIVE_FORMS.items():
            if predicate in present:
                continue
            if predicate.replace("_", " ") in _norm(evidence):
                continue
            negatives.append(_case(
                primary,
                question=template.format(subject=primary["subject"]),
                category="negative",
                expected_decision="unknown",
                predicates=[predicate],
            ))
            break
        if len(negatives) == _NEGATIVE_CASES:
            break

    cases = [*cases, *negatives]
    rng.shuffle(cases)

    _OUTPUT.mkdir(parents=True, exist_ok=True)
    (_OUTPUT / "dataset.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    shutil.copy(_SOURCE / "frozen_relations.json", _OUTPUT / "frozen_relations.json")
    summary = {
        "total_cases": len(cases),
        "answer_cases": _ANSWER_CASES,
        "negative_cases": len(negatives),
        "evidence_graph": "heldout_v3 frozen_relations.json (unchanged)",
        "independence": "templates written once for this builder; never used in dataset/heldout builders or the fallback example-phrase table",
    }
    (_OUTPUT / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
