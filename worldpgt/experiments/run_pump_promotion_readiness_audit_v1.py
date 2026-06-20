"""Pump Promotion Readiness Audit v1.

Read-only audit over the precision-cleaned Knowledge Pump v1 facts. The audit
does not promote, ingest, auto-apply, or modify accepted memory/overlays. It
writes only under ``knowledge_pump_v1/promotion_readiness_audit_v1`` by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.knowledge_pump.frontier_title_extractor import (
    _PROSE_TERMINAL_WORDS,
    _has_prose_period_break,
)
from worldpgt.knowledge.temporal_classification import (
    classify_temporal_class,
)

_EXPERIMENTS = Path(__file__).resolve().parent
_PUMP_DIR = _EXPERIMENTS / "knowledge_pump_v1"
_OUT_DIR = _PUMP_DIR / "promotion_readiness_audit_v1"

_PRECISION_FACTS = _PUMP_DIR / "pump_precision_answerable_delta.json"
_QA_SUMMARY = _PUMP_DIR / "pump_fact_qa_v1" / "pump_fact_qa_summary.json"
_QA_OUTPUTS = _PUMP_DIR / "pump_fact_qa_v1" / "pump_fact_qa_outputs.json"
_PUMP_SUMMARY = _PUMP_DIR / "pump_summary.json"

_ACCEPTED = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_PROMOTED = _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
_SNAPSHOT_DRY_RUN = _EXPERIMENTS / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json"
_TRUSTED = _EXPERIMENTS / "accepted_knowledge_memory_v1.json"
_SENSE_MEMORY = _ROOT / "worldpgt" / "continuation" / "sense_memory.py"

CLASS_PROMOTION = "promotion_candidate"
CLASS_PROPOSAL = "proposal_only"
CLASS_REVIEW = "needs_review"
CLASS_REJECT = "reject_recommendation"

ANSWERABLE_TYPES = {"overlay_relation", "overlay_definition"}
SAFE_RELATION_PREDICATES = {
    "is_a",
    "founded_by",
    "founded",
    "publishes",
    "service_of",
    "subsidiary_of",
}
HIGH_REVIEW_PREDICATES = {
    "owned_by",
    "develops",
    "developed_by",
    "headquartered_in",
    "leader_of",
}
TRUNCATION_TERMINALS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "in",
    "of",
    "or",
    "the",
    "to",
    "with",
}
CURRENT_TERMS = {
    "current",
    "currently",
    "today",
    "latest",
    "ongoing",
    "forecasted",
    "planned",
    "proposed",
    "under development",
}
TOO_BROAD_DEFINITIONS = {
    "company",
    "corporation",
    "french company",
    "american company",
    "publicly traded company",
    "private corporation",
    "businessman",
    "engineer",
    "entrepreneur",
    "manufacturer",
    "service",
}


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING"


def _protected_hashes(
    *,
    experiments_dir: Path = _EXPERIMENTS,
    root: Path = _ROOT,
) -> dict[str, str]:
    return {
        "trusted_memory": _sha(experiments_dir / "accepted_knowledge_memory_v1.json"),
        "accepted_overlay": _sha(experiments_dir / "accepted_wiki_memory_overlay_v1.json"),
        "promoted_overlay": _sha(
            experiments_dir / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
        ),
        "snapshot_dry_run_overlay": _sha(
            experiments_dir / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json"
        ),
        "sense_memory": _sha(root / "worldpgt" / "continuation" / "sense_memory.py"),
    }


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _fact_signature(item: dict[str, Any]) -> tuple[str, str, str, str] | None:
    otype = item.get("overlay_type")
    if otype == "overlay_relation":
        return (
            "relation",
            _norm(item.get("subject")),
            _norm(item.get("predicate")),
            _norm(item.get("object")),
        )
    if otype == "overlay_definition":
        return (
            "definition",
            _norm(item.get("subject")),
            "is_a",
            _norm(item.get("definition")),
        )
    return None


def _duplicate_index(paths: dict[str, Path]) -> dict[tuple[str, str, str, str], list[str]]:
    index: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for name, path in paths.items():
        rows = _read_json(path, [])
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, dict):
                continue
            sig = _fact_signature(item)
            if sig is not None:
                index[sig].append(name)
    return index


def _is_answerable(item: dict[str, Any]) -> bool:
    return item.get("overlay_type") in ANSWERABLE_TYPES


def _is_weak_context(item: dict[str, Any]) -> bool:
    return item.get("overlay_type") == "overlay_context_link" or item.get("trust") == "weak_context_only"


def _fact_value(item: dict[str, Any]) -> str:
    if item.get("overlay_type") == "overlay_definition":
        return str(item.get("definition", ""))
    return str(item.get("object", ""))


def _has_current_signal(item: dict[str, Any]) -> bool:
    stability = _norm(item.get("stability"))
    if stability in {"current", "volatile", "live"}:
        return True
    text = _norm(" ".join([_fact_value(item), str(item.get("evidence_text", ""))]))
    for term in CURRENT_TERMS:
        if " " in term:
            if term in text:
                return True
            continue
        if re.search(rf"\b{re.escape(term)}\b", text):
            return True
    return False


def _temporal_class(item: dict[str, Any]) -> str | None:
    explicit = item.get("temporal_class")
    if explicit:
        return str(explicit)
    return classify_temporal_class(
        str(item.get("predicate") or ("is_a" if item.get("overlay_type") == "overlay_definition" else "")),
        str(item.get("stability") or ""),
        overlay_type=str(item.get("overlay_type") or ""),
        claim_type=str(item.get("claim_type") or ""),
    )


def _subject_problem(subject: str) -> str | None:
    subject = subject.strip()
    if not subject:
        return "missing_subject"
    if len(subject) < 2:
        return "subject_too_short"
    if len(subject.split()) > 8:
        return "subject_too_long"
    if _has_prose_period_break(subject):
        return "subject_sentence_fragment"
    if subject.split() and subject.split()[-1] in _PROSE_TERMINAL_WORDS:
        return "subject_sentence_fragment"
    if subject in {"Many companies", "In", "During", "The"}:
        return "subject_not_canonical"
    return None


def _looks_truncated(value: str) -> bool:
    value = value.strip()
    if not value:
        return True
    words = value.split()
    if words[-1].casefold().rstrip(".,;:") in TRUNCATION_TERMINALS:
        return True
    if value.endswith(("-", " -", ",", ";", ":")):
        return True
    if re.search(r"\b[A-Z]\.$", value):
        return True
    return False


def _definition_reason(item: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    subject = str(item.get("subject", "")).strip()
    definition = str(item.get("definition", "")).strip()
    subject_issue = _subject_problem(subject)
    if subject_issue:
        reasons.append(subject_issue)
    if _looks_truncated(definition):
        reasons.append("definition_truncated_or_incomplete")
    if len(definition.split()) < 2:
        reasons.append("definition_too_short")
    if _norm(definition) in TOO_BROAD_DEFINITIONS:
        reasons.append("definition_too_broad")
    if re.fullmatch(r"(american|british|french|german|canadian|south african|private|public)\s+\w+", _norm(definition)):
        reasons.append("definition_broad_class")
    if item.get("source_page") and _norm(item.get("source_page")) != _norm(subject):
        reasons.append("source_page_subject_mismatch")
    temporal_class = _temporal_class(item)
    if temporal_class is None:
        reasons.append("temporal_class_requires_review")
    if temporal_class == "snapshot" and not item.get("as_of"):
        reasons.append("snapshot_requires_as_of")
    if temporal_class == "aggregate" and not item.get("as_of"):
        reasons.append("aggregate_requires_as_of")
    if _has_current_signal(item):
        reasons.append("current_or_volatile")

    if (
        "current_or_volatile" in reasons
        or "definition_truncated_or_incomplete" in reasons
        or "snapshot_requires_as_of" in reasons
    ):
        return CLASS_REJECT, reasons
    if "aggregate_requires_as_of" in reasons:
        return CLASS_REJECT, reasons
    if subject_issue or "definition_too_short" in reasons:
        return CLASS_REJECT, reasons
    if "definition_too_broad" in reasons or "definition_broad_class" in reasons:
        return CLASS_REVIEW, reasons
    if "source_page_subject_mismatch" in reasons:
        return CLASS_REVIEW, reasons
    return CLASS_PROMOTION, ["stable_complete_definition"]


def _object_problem(obj: str) -> str | None:
    obj = obj.strip()
    if not obj:
        return "missing_object"
    if _has_prose_period_break(obj):
        return "object_sentence_fragment"
    if _looks_truncated(obj):
        return "object_truncated_or_incomplete"
    if len(obj.split()) > 8:
        return "object_too_complex"
    return None


def _relation_reason(item: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    subject = str(item.get("subject", "")).strip()
    predicate = str(item.get("predicate", "")).strip()
    obj = str(item.get("object", "")).strip()
    subject_issue = _subject_problem(subject)
    object_issue = _object_problem(obj)
    if subject_issue:
        reasons.append(subject_issue)
    if object_issue:
        reasons.append(object_issue)
    temporal_class = _temporal_class(item)
    if temporal_class is None:
        reasons.append("temporal_class_requires_review")
    if temporal_class == "snapshot" and not item.get("as_of"):
        reasons.append("snapshot_requires_as_of")
    if temporal_class == "aggregate" and not item.get("as_of"):
        reasons.append("aggregate_requires_as_of")
    if predicate in HIGH_REVIEW_PREDICATES and not (
        temporal_class == "snapshot" and item.get("as_of")
    ):
        reasons.append("predicate_high_review")
    if predicate == "owned_by" and not (temporal_class == "snapshot" and item.get("as_of")):
        reasons.append("ownership_semantic_ambiguity")
    if predicate in {"develops", "developed_by"} and len(obj.split()) > 3:
        reasons.append("complex_development_object")
    if predicate == "headquartered_in":
        reasons.append("headquarters_currentness_risk")
    if predicate == "leader_of" and not (temporal_class == "snapshot" and item.get("as_of")):
        reasons.append("leader_currentness_risk")
    if predicate not in SAFE_RELATION_PREDICATES and predicate not in HIGH_REVIEW_PREDICATES:
        reasons.append("predicate_not_allowlisted")
    if str(item.get("risk", "")).casefold() == "high":
        reasons.append("high_risk_fact")
    if _has_current_signal(item):
        reasons.append("current_or_volatile")

    if (
        "current_or_volatile" in reasons
        or "snapshot_requires_as_of" in reasons
        or "aggregate_requires_as_of" in reasons
        or subject_issue
        or object_issue
        or "high_risk_fact" in reasons
    ):
        return CLASS_REJECT, reasons
    if temporal_class == "snapshot" and item.get("as_of"):
        return CLASS_PROMOTION, ["snapshot_as_of_recheck_required"]
    if predicate in HIGH_REVIEW_PREDICATES:
        return CLASS_REVIEW, reasons
    if predicate in SAFE_RELATION_PREDICATES:
        return CLASS_PROMOTION, ["stable_trustworthy_predicate"]
    return CLASS_PROPOSAL, reasons or ["predicate_not_promoted_by_policy"]


def _qa_positive_keys(qa_outputs: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for row in qa_outputs:
        if not isinstance(row, dict):
            continue
        prompt = row.get("prompt") or {}
        if prompt.get("category") != "positive":
            continue
        if row.get("classification") not in {"ok", "planner_gap", "source_or_volatility_sensitive"}:
            continue
        keys.add((
            _norm(prompt.get("subject")),
            _norm(prompt.get("predicate") or "is_a"),
            _norm(prompt.get("obj")),
        ))
    return keys


def _qa_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (_norm(item.get("subject")), _norm(item.get("predicate") or "is_a"), _norm(_fact_value(item)))


def _classify_item(
    item: dict[str, Any],
    *,
    duplicate_locations: list[str],
    qa_covered: bool,
    qa_current: bool,
) -> dict[str, Any]:
    if item.get("overlay_type") == "overlay_definition":
        classification, reasons = _definition_reason(item)
        fact_kind = "definition"
        value = str(item.get("definition", ""))
    else:
        classification, reasons = _relation_reason(item)
        fact_kind = "relation"
        value = str(item.get("object", ""))

    if duplicate_locations and classification == CLASS_PROMOTION:
        classification = CLASS_PROPOSAL
        reasons = ["duplicate_existing_memory"] + reasons
    if not qa_covered and classification == CLASS_PROMOTION:
        classification = CLASS_REVIEW
        reasons = ["missing_positive_qa_coverage"] + reasons
    if not qa_current and classification == CLASS_PROMOTION:
        classification = CLASS_REVIEW
        reasons = ["qa_not_current"] + reasons

    return {
        "classification": classification,
        "reasons": reasons,
        "primary_reason": reasons[0] if reasons else "",
        "fact_kind": fact_kind,
        "overlay_type": item.get("overlay_type", ""),
        "subject": item.get("subject", ""),
        "predicate": item.get("predicate", "is_a"),
        "object": value,
        "source_page": item.get("source_page", ""),
        "risk": item.get("risk", ""),
        "stability": item.get("stability", ""),
        "temporal_class": _temporal_class(item) or "",
        "as_of": item.get("as_of", ""),
        "trust": item.get("trust", ""),
        "evidence_text": item.get("evidence_text", ""),
        "qa_covered": qa_covered,
        "qa_current": qa_current,
        "duplicate_of": duplicate_locations,
        "duplicate_of_text": "|".join(duplicate_locations),
    }


def _by_predicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("predicate", ""))].append(row)
    out: list[dict[str, Any]] = []
    for predicate, items in grouped.items():
        counts = Counter(row["classification"] for row in items)
        out.append({
            "predicate": predicate,
            "total": len(items),
            "promotion_candidate": counts[CLASS_PROMOTION],
            "proposal_only": counts[CLASS_PROPOSAL],
            "needs_review": counts[CLASS_REVIEW],
            "reject_recommendation": counts[CLASS_REJECT],
            "candidate_rate": round(counts[CLASS_PROMOTION] / len(items), 4),
            "top_reasons": dict(Counter(r["primary_reason"] for r in items if r["classification"] != CLASS_PROMOTION).most_common(8)),
        })
    return sorted(out, key=lambda r: (-r["promotion_candidate"], r["predicate"]))


def _by_source_page(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source_page", "") or "unknown")].append(row)
    out: list[dict[str, Any]] = []
    for page, items in grouped.items():
        counts = Counter(row["classification"] for row in items)
        out.append({
            "source_page": page,
            "total": len(items),
            "promotion_candidate": counts[CLASS_PROMOTION],
            "proposal_only": counts[CLASS_PROPOSAL],
            "needs_review": counts[CLASS_REVIEW],
            "reject_recommendation": counts[CLASS_REJECT],
            "candidate_rate": round(counts[CLASS_PROMOTION] / len(items), 4),
            "top_reasons": dict(Counter(r["primary_reason"] for r in items if r["classification"] != CLASS_PROMOTION).most_common(8)),
        })
    return sorted(out, key=lambda r: (-r["promotion_candidate"], -r["candidate_rate"], r["source_page"]))


def _summary_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["classification"] for row in rows)
    return {
        "promotion_candidate_count": counts[CLASS_PROMOTION],
        "proposal_only_count": counts[CLASS_PROPOSAL],
        "needs_review_count": counts[CLASS_REVIEW],
        "reject_recommendation_count": counts[CLASS_REJECT],
    }


def run(
    *,
    pump_dir: Path = _PUMP_DIR,
    out_dir: Path = _OUT_DIR,
    experiments_dir: Path = _EXPERIMENTS,
    root: Path = _ROOT,
) -> dict[str, Any]:
    before = _protected_hashes(experiments_dir=experiments_dir, root=root)

    precision_path = pump_dir / "pump_precision_answerable_delta.json"
    qa_summary_path = pump_dir / "pump_fact_qa_v1" / "pump_fact_qa_summary.json"
    qa_outputs_path = pump_dir / "pump_fact_qa_v1" / "pump_fact_qa_outputs.json"
    pump_summary_path = pump_dir / "pump_summary.json"

    raw_items = _read_json(precision_path, [])
    if not isinstance(raw_items, list):
        raw_items = []
    answerable = [item for item in raw_items if isinstance(item, dict) and _is_answerable(item)]
    non_answerable_excluded = [
        item for item in raw_items
        if isinstance(item, dict) and not _is_answerable(item)
    ]
    weak_context_excluded = [
        item for item in raw_items
        if isinstance(item, dict) and _is_weak_context(item)
    ]

    qa_summary = _read_json(qa_summary_path, {})
    qa_outputs = _read_json(qa_outputs_path, [])
    pump_summary = _read_json(pump_summary_path, {})
    if not isinstance(qa_outputs, list):
        qa_outputs = []
    relation_count = sum(1 for item in answerable if item.get("overlay_type") == "overlay_relation")
    definition_count = sum(1 for item in answerable if item.get("overlay_type") == "overlay_definition")
    fact_count_matches_qa = qa_summary.get("pump_fact_qa_fact_count") == len(answerable)
    relation_count_matches_qa = qa_summary.get("pump_fact_qa_relation_fact_count") == relation_count
    definition_count_matches_qa = qa_summary.get("pump_fact_qa_definition_fact_count") == definition_count
    qa_status = pump_summary.get("pump_fact_qa_status")
    summary_says_current = qa_status == "current_from_qa_artifact"
    qa_is_current = bool(
        fact_count_matches_qa
        and relation_count_matches_qa
        and definition_count_matches_qa
        and qa_summary.get("pump_fact_qa_all_critical_passed") is True
        and qa_summary.get("pump_fact_qa_positive_wrong_count", qa_summary.get("wrong_answer_count", 0)) == 0
        and qa_summary.get("pump_fact_qa_positive_unsupported_count", qa_summary.get("unsupported_answer_count", 0)) == 0
        and qa_summary.get("pump_fact_qa_adversarial_fail_count", qa_summary.get("adversarial_fail_count", 0)) == 0
        and qa_summary.get("pump_fact_qa_current_safety_fail_count", qa_summary.get("current_safety_fail_count", 0)) == 0
        and summary_says_current
    )

    dupes = _duplicate_index({
        "accepted_overlay": experiments_dir / "accepted_wiki_memory_overlay_v1.json",
        "promoted_overlay": experiments_dir / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json",
        "snapshot_dry_run_overlay": experiments_dir / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json",
    })
    qa_keys = _qa_positive_keys(qa_outputs)

    classified: list[dict[str, Any]] = []
    for idx, item in enumerate(answerable):
        sig = _fact_signature(item)
        duplicate_locations = dupes.get(sig, []) if sig is not None else []
        row = _classify_item(
            item,
            duplicate_locations=duplicate_locations,
            qa_covered=_qa_key(item) in qa_keys,
            qa_current=qa_is_current,
        )
        row["fact_index"] = idx
        classified.append(row)

    counts = _summary_counts(classified)
    by_predicate = _by_predicate(classified)
    by_source_page = _by_source_page(classified)
    not_promoted_reasons = Counter(
        row["primary_reason"] for row in classified if row["classification"] != CLASS_PROMOTION
    )
    candidates = [row for row in classified if row["classification"] == CLASS_PROMOTION]
    needs_review = [row for row in classified if row["classification"] == CLASS_REVIEW]
    reject_recommendations = [row for row in classified if row["classification"] == CLASS_REJECT]
    proposal_only = [row for row in classified if row["classification"] == CLASS_PROPOSAL]
    suspicious = needs_review + reject_recommendations
    duplicates = [row for row in classified if row["duplicate_of"]]
    promotion_ready = bool(
        qa_is_current
        and len(classified) == qa_summary.get("pump_fact_qa_fact_count")
        and counts["promotion_candidate_count"] > 0
        and all(row["qa_covered"] for row in candidates)
    )

    after = _protected_hashes(experiments_dir=experiments_dir, root=root)
    confirmations = {
        "network_calls": False,
        "trusted_memory_modified": before["trusted_memory"] != after["trusted_memory"],
        "accepted_overlay_modified": before["accepted_overlay"] != after["accepted_overlay"],
        "promoted_overlay_modified": before["promoted_overlay"] != after["promoted_overlay"],
        "snapshot_dry_run_overlay_modified": before["snapshot_dry_run_overlay"] != after["snapshot_dry_run_overlay"],
        "sense_memory_modified": before["sense_memory"] != after["sense_memory"],
        "auto_ingest": False,
        "auto_promote": False,
        "auto_apply": False,
        "weak_context_excluded": True,
        "entity_cards_excluded": True,
        "neural_gpt_training_embedding_code_added": False,
        "nanogpt_touched": False,
    }

    summary = {
        "audit_name": "pump_promotion_readiness_audit_v1",
        "promotion_ready": promotion_ready,
        "total_world_model_items": len(raw_items),
        "total_answerable_facts": len(classified),
        "relations_count": relation_count,
        "definitions_count": definition_count,
        **counts,
        "qa_is_current": qa_is_current,
        "fact_count_matches_qa_fact_count": fact_count_matches_qa,
        "relation_count_matches_qa_relation_count": relation_count_matches_qa,
        "definition_count_matches_qa_definition_count": definition_count_matches_qa,
        "summary_pump_fact_qa_status": qa_status,
        "summary_says_pump_fact_qa_status_current_from_qa_artifact": summary_says_current,
        "all_promotion_candidates_covered_by_qa": all(row["qa_covered"] for row in candidates),
        "duplicate_candidate_count": sum(1 for row in candidates if row["duplicate_of"]),
        "duplicate_fact_count": len(duplicates),
        "entity_card_excluded_count": sum(1 for item in non_answerable_excluded if item.get("overlay_type") == "overlay_entity"),
        "weak_context_excluded_count": len(weak_context_excluded),
        "top_reasons_for_not_promoting": dict(not_promoted_reasons.most_common(12)),
        "most_trustworthy_predicates": [
            row for row in by_predicate
            if row["promotion_candidate"] > 0
        ][:10],
        "cleanest_source_pages": [
            row for row in by_source_page
            if row["promotion_candidate"] > 0
        ][:15],
        "source_artifacts": {
            "precision_answerable_delta": str(precision_path),
            "pump_dry_run_overlay": str(pump_dir / "pump_dry_run_overlay.json"),
            "pump_fact_qa_summary": str(qa_summary_path),
            "precision_firewall_v2": str(pump_dir / "precision_firewall_v2"),
            "precision_cleanup_v2_1": str(pump_dir / "precision_cleanup_v2_1"),
        },
        "confirmations": confirmations,
    }

    report = {
        "summary": summary,
        "counts_by_predicate": by_predicate,
        "counts_by_source_page": by_source_page,
        "top_reasons_for_not_promoting": summary["top_reasons_for_not_promoting"],
        "sample_promotion_candidates": candidates[:20],
        "sample_needs_review": needs_review[:20],
        "sample_reject_recommendations": reject_recommendations[:20],
        "proposal_only": proposal_only[:50],
        "duplicates": duplicates,
        "suspicious_facts": suspicious,
        "qa_summary": qa_summary,
    }

    fields = [
        "fact_index",
        "classification",
        "primary_reason",
        "fact_kind",
        "subject",
        "predicate",
        "object",
        "source_page",
        "risk",
        "stability",
        "temporal_class",
        "as_of",
        "qa_covered",
        "qa_current",
        "duplicate_of_text",
    ]
    _write_json(out_dir / "promotion_readiness_summary.json", summary)
    _write_json(out_dir / "promotion_readiness_report.json", report)
    _write_json(out_dir / "promotion_readiness_candidates.json", candidates)
    _write_csv(out_dir / "promotion_readiness_candidates.csv", candidates, fields)
    _write_json(out_dir / "promotion_readiness_needs_review.json", needs_review)
    _write_json(out_dir / "promotion_readiness_reject_recommendations.json", reject_recommendations)
    _write_json(out_dir / "promotion_readiness_by_predicate.json", by_predicate)
    _write_json(out_dir / "promotion_readiness_by_source_page.json", by_source_page)
    _write_json(out_dir / "promotion_readiness_suspicious_facts.json", suspicious)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Pump Promotion Readiness Audit v1")
    parser.add_argument("--pump-dir", default=str(_PUMP_DIR))
    parser.add_argument("--out-dir", default=str(_OUT_DIR))
    args = parser.parse_args(argv)

    summary = run(pump_dir=Path(args.pump_dir), out_dir=Path(args.out_dir))
    print("Pump Promotion Readiness Audit v1")
    for key in (
        "total_answerable_facts",
        "relations_count",
        "definitions_count",
        "promotion_candidate_count",
        "proposal_only_count",
        "needs_review_count",
        "reject_recommendation_count",
        "qa_is_current",
        "fact_count_matches_qa_fact_count",
        "summary_pump_fact_qa_status",
        "all_promotion_candidates_covered_by_qa",
        "duplicate_fact_count",
        "promotion_ready",
    ):
        print(f"  {key}: {summary.get(key)}")
    print("  top_reasons_for_not_promoting:")
    for reason, count in summary.get("top_reasons_for_not_promoting", {}).items():
        print(f"    {reason}: {count}")
    print(f"  artifacts: {Path(args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
