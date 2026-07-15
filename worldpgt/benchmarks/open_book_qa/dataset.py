"""Build deterministic, evidence-scoped cases from the live proposal overlay."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import random
import re
import subprocess
from typing import Iterable

from worldpgt.api import server

_DEICTIC = re.compile(
    r"^(?:(?:our|my|we|us|i|this|these|those|that)\b|it$|they$|them$|the$)",
    re.I,
)
_SHORT_ALIAS = re.compile(r"^[A-Z0-9]{1,3}$")
_QUESTIONS = {
    "uses": "What does {subject} use?",
    "enables": "What does {subject} enable?",
    "supports": "What does {subject} support?",
    "runs_on": "What does {subject} run on?",
    "used_for": "What is {subject} used for?",
    "works_by": "How does {subject} work?",
    "developed_by": "Who developed {subject}?",
}
_PARAPHRASES = {
    "uses": ("What technology does {subject} rely on?", "What does {subject} employ?", "What is used by {subject}?"),
    "enables": ("What does {subject} make possible?", "What capability does {subject} provide?"),
    "supports": ("What does {subject} help support?", "What outcome is supported by {subject}?"),
    "runs_on": ("What platform does {subject} operate on?",),
    "used_for": ("What purpose does {subject} serve?",),
    "works_by": ("What mechanism does {subject} use?",),
    "developed_by": ("Who created {subject}?",),
}
_NEGATIVE_PREDICATES = ("founded_by", "runs_on", "developed_by", "uses", "enables", "supports")
_MULTI_CLAUSES = {
    "uses": "what does {subject} use",
    "enables": "what does {subject} enable",
    "supports": "what does {subject} support",
    "runs_on": "what does {subject} run on",
    "used_for": "what is {subject} used for",
    "works_by": "how does {subject} work",
    "developed_by": "who developed {subject}",
}


def _compact(value: object) -> str:
    return " ".join(str(value or "").split())


def _norm(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _stable_id(relation: dict) -> str:
    payload = "\x1f".join(_compact(relation.get(key)) for key in ("subject", "predicate", "object", "evidence_text", "source_url"))
    return "obqa-" + sha256(payload.encode("utf-8")).hexdigest()[:16]


def relation_id(relation: dict) -> str:
    return "edge:" + "|".join(_norm(relation.get(key)) for key in ("subject", "predicate", "object"))


def _source_ids(relation: dict) -> list[str]:
    values = [relation.get("source_url"), relation.get("source_page"), *(relation.get("supporting_sources") or [])]
    return sorted({_compact(value) for value in values if _compact(value)})


def _evidence(relation: dict) -> str:
    return _compact(relation.get("evidence_text") or relation.get("evidence_span"))


def load_experimental_relations(overlay: str = "pump-dry-run+experimental-web-graph") -> list[dict]:
    """Use the same composed-overlay loader and serving filter as the API."""
    if overlay != "pump-dry-run+experimental-web-graph":
        raise ValueError("only the reproducible composed serving overlay is supported")
    paths = server._available_experimental_web_graph_paths()
    rows = server._merge_experimental_graph_items(
        item for path in paths for item in server._load_overlay_items(str(path))
    )
    return [
        row for row in rows
        if row.get("overlay_type") == "overlay_relation"
        and str(row.get("experimental_tier") or "").startswith("evidence_grounded_")
    ]


def _valid_relation(relation: dict) -> str | None:
    subject, predicate, obj, evidence = (_compact(relation.get(k)) for k in ("subject", "predicate", "object", "evidence_text"))
    if not subject or not predicate or not obj:
        return "malformed_relation"
    if _DEICTIC.match(subject) or _DEICTIC.match(obj):
        return "deictic_node"
    if _SHORT_ALIAS.match(subject):
        return "ambiguous_short_alias"
    if not evidence:
        return "empty_evidence"
    if not _source_ids(relation):
        return "missing_source_id"
    if _norm(obj) not in _norm(evidence):
        return "object_not_in_evidence"
    if predicate not in _QUESTIONS:
        return "unsupported_predicate"
    return None


def _case(relation: dict, *, question: str, category: str, expected_decision: str = "answer", contexts: list[str] | None = None, relations: list[dict] | None = None, predicates: list[str] | None = None, multi_kind: str | None = None) -> dict:
    selected = relations or [relation]
    identity = question + "\x1f" + "\x1f".join(relation_id(item) for item in selected)
    return {
        "id": _stable_id(relation) + "-" + category + "-" + sha256(identity.encode()).hexdigest()[:8],
        "question": question,
        "contexts": contexts if contexts is not None else [_evidence(item) for item in selected],
        "expected_subject": _compact(relation.get("subject")),
        "expected_predicate": predicates or [_compact(item.get("predicate")) for item in selected],
        "expected_objects": [_compact(item.get("object")) for item in selected] if expected_decision == "answer" else [],
        "expected_decision": expected_decision,
        "relation_ids": [relation_id(item) for item in selected],
        "evidence_ids": [relation_id(item) for item in selected],
        "source_ids": sorted({source for item in selected for source in _source_ids(item)}),
        "category": category,
        **({"multi_kind": multi_kind} if multi_kind else {}),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_dataset(overlay: str = "pump-dry-run+experimental-web-graph", *, seed: int = 42) -> tuple[list[dict], list[dict], dict]:
    """Return 250 fixed-seed cases plus every excluded real relation and reason."""
    rng = random.Random(seed)
    candidates, rejected = [], []
    for relation in load_experimental_relations(overlay):
        reason = _valid_relation(relation)
        if reason:
            rejected.append({"relation": relation, "reason": reason})
        else:
            candidates.append(relation)
    rng.shuffle(candidates)
    if len(candidates) < 150:
        raise RuntimeError(f"need at least 150 valid relations, found {len(candidates)}")

    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_subject[_norm(row["subject"])].append(row)
    multi = []
    for rows in by_subject.values():
        # A generic "What is known about X?" cannot identify which two of a
        # high-degree node's facts the evaluator expects. Pick one stable edge
        # per predicate and ask explicitly for a *pair of distinct predicates*.
        # The question therefore identifies its expected evidence bundle;
        # neither the planner nor Qwen receives a hidden target pair.
        by_predicate: dict[str, dict] = {}
        for row in sorted(rows, key=relation_id):
            predicate = str(row["predicate"])
            if predicate in _MULTI_CLAUSES:
                by_predicate.setdefault(predicate, row)
        for left, right in combinations(sorted(by_predicate), 2):
            selected = [by_predicate[left], by_predicate[right]]
            primary = selected[0]
            first_clause = _MULTI_CLAUSES[left].format(subject=primary["subject"])
            question = (
                first_clause[:1].upper() + first_clause[1:]
                + ", and "
                + _MULTI_CLAUSES[right].format(subject=primary["subject"])
                + "?"
            )
            multi.append(_case(
                primary, question=question, category="multi_evidence",
                relations=selected, multi_kind="specified_distinct_predicates",
            ))
    rng.shuffle(multi)
    # The overlay presently supports only this many non-ambiguous predicate
    # pairs. Keeping the smaller stratum is more honest than manufacturing
    # repeated generic questions with hidden expected edges.
    multi = multi[:50]
    if not multi:
        raise RuntimeError("need at least one specified multi-evidence case")

    direct_count = 250 - 50 - 50 - len(multi)
    direct = [_case(
        row, question=_QUESTIONS[row["predicate"]].format(subject=row["subject"]), category="direct",
    ) for row in candidates[:direct_count]]
    paraphrase_source = candidates[direct_count:]
    paraphrase = []
    for index, row in enumerate(paraphrase_source):
        templates = _PARAPHRASES.get(row["predicate"], ())
        if templates:
            paraphrase.append(_case(row, question=templates[index % len(templates)].format(subject=row["subject"]), category="paraphrase"))
        if len(paraphrase) == 50:
            break
    if len(paraphrase) < 50:
        raise RuntimeError("not enough supported-predicate relations for paraphrases")

    negative = []
    for row in candidates:
        evidence = _evidence(row)
        predicate = next((value for value in _NEGATIVE_PREDICATES if value != row["predicate"] and value.replace("_", " ") not in _norm(evidence)), None)
        if predicate:
            question = _QUESTIONS.get(predicate, f"What does {{subject}} {predicate.replace('_', ' ')}?").format(subject=row["subject"])
            negative.append(_case(row, question=question, category="negative", expected_decision="unknown", predicates=[predicate]))
        if len(negative) == 50:
            break
    if len(negative) < 50:
        raise RuntimeError("not enough safe negative contexts")
    cases = [*direct, *paraphrase, *negative, *multi]
    rng.shuffle(cases)
    fingerprint = sha256("\n".join(sorted(relation_id(row) + "\x1f" + _evidence(row) for row in candidates)).encode()).hexdigest()
    summary = {
        "total_cases": len(cases), "cases_per_category": dict(Counter(case["category"] for case in cases)),
        "predicate_distribution": dict(Counter(predicate for case in cases for predicate in case["expected_predicate"])),
        "unique_subjects": len({case["expected_subject"] for case in cases}),
        "unique_evidence_spans": len({context for case in cases for context in case["contexts"]}),
        "multi_evidence_question_contract": "two explicit, distinct relation predicates per question",
        "excluded_counts_by_reason": dict(Counter(item["reason"] for item in rejected)),
        "overlay": overlay, "overlay_fingerprint": fingerprint, "build_commit": _git_commit(),
        "generated_timestamp": datetime.now(timezone.utc).isoformat(), "random_seed": seed,
    }
    return cases, rejected, summary


def write_dataset(output: str | Path, **kwargs: object) -> dict:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    cases, rejected, summary = build_dataset(**kwargs)
    for name, rows in (("dataset.jsonl", cases), ("rejected_candidates.jsonl", rejected)):
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (output / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
