"""Executes a QueryPlan step-by-step against the overlay provider.

Safety: reuses validate_hop_safety from path_validator — volatile, high-risk,
and current-sensitive relations are excluded from Find/Traverse results with a
trace note. An empty result after safety filtering causes an audit PlanResult.
"""

from __future__ import annotations

import re
from typing import Any

from worldpgt.knowledge.ontology_traversal import find_is_a_path
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider
from worldpgt.multihop_qa.path_validator import validate_hop_safety
from worldpgt.multihop_qa.types import HopEdge
from worldpgt.query_engine.primitives import (
    Aggregate,
    Classify,
    Compare,
    Count,
    Filter,
    Find,
    Traverse,
)
from worldpgt.query_engine.types import PlanResult, QueryPlan

_AUDIT_EMPTY = "no matching facts found in the overlay after safety filtering"
_AUDIT_NO_STEPS = "query plan has no executable steps"


def execute(
    plan: QueryPlan,
    provider: WikiMemoryOverlayProvider,
    ontology_items: list[dict] | None = None,
) -> PlanResult:
    """Execute *plan* against *provider* and return a structured PlanResult."""

    if not plan.steps:
        return PlanResult(
            success=False, data=None, trace=["no steps"],
            decision="audit", audit_reason=_AUDIT_NO_STEPS,
        )

    results: dict[str, Any] = {}
    trace: list[str] = []

    for step in plan.steps:
        if isinstance(step, Find):
            data = _exec_find(step, provider, trace)
        elif isinstance(step, Filter):
            data = _exec_filter(step, results, provider, trace)
        elif isinstance(step, Count):
            data = _exec_count(step, results, trace)
        elif isinstance(step, Compare):
            data = _exec_compare(step, results, trace)
        elif isinstance(step, Traverse):
            data = _exec_traverse(step, provider, trace)
        elif isinstance(step, Classify):
            data = _exec_classify(step, provider, ontology_items or [], trace)
        elif isinstance(step, Aggregate):
            data = _exec_aggregate(step, results, trace)
        else:
            trace.append(f"unknown step type: {type(step)}")
            continue

        results[step.result_key] = data

        # Early-exit if a Find/Filter returns empty — downstream Count/Compare
        # would be meaningless; audit immediately.
        if isinstance(step, (Find, Filter)):
            if isinstance(data, list) and len(data) == 0:
                return PlanResult(
                    success=False, data=data, trace=trace,
                    decision="audit", audit_reason=_AUDIT_EMPTY,
                )

    final_data = results.get(plan.steps[-1].result_key)
    answer_text = _render(plan, results, final_data)
    support = _infer_support_kind(results)

    return PlanResult(
        success=True,
        data=final_data,
        trace=trace,
        decision="answer",
        audit_reason=None,
        answer_text=answer_text,
        support_kind=support,
    )


# ── Step executors ─────────────────────────────────────────────────────────

def _exec_find(
    step: Find,
    provider: WikiMemoryOverlayProvider,
    trace: list[str],
) -> list[dict]:
    if step.target == "object":
        if step.subject and step.predicate:
            candidates = provider.get_relations(step.subject, step.predicate)
        elif step.subject:
            candidates = provider.get_relations(step.subject)
        else:
            candidates = provider.all_relations()
    else:
        # Inverse: find subjects where (?, predicate, known_value)
        pred = step.predicate
        known = _norm(step.known_value or "")
        candidates = [
            r for r in provider.all_relations()
            if r.get("predicate") == pred
            and _norm(r.get("object", "")) == known
        ]

    safe: list[dict] = []
    for r in candidates:
        hop = _to_hop_edge(r)
        valid, reason = validate_hop_safety(hop)
        if valid:
            safe.append(r)
        else:
            trace.append(
                f"safety_excluded: {r.get('subject')}|{r.get('predicate')}|{r.get('object')}: {reason}"
            )

    trace.append(
        f"Find({step.subject!r}, {step.predicate!r}, target={step.target!r}) → {len(safe)} safe results"
    )
    return safe


def _exec_filter(
    step: Filter,
    results: dict[str, Any],
    provider: WikiMemoryOverlayProvider,
    trace: list[str],
) -> list[dict]:
    source: list[dict] = results.get(step.source_key, [])
    cond = step.condition
    out: list[dict] = []

    for r in source:
        # Metadata filters
        if "stability" in cond and r.get("stability") != cond["stability"]:
            continue
        if "temporal_class" in cond and r.get("temporal_class") != cond["temporal_class"]:
            continue

        # Secondary relation filter: check that r's object entity has a
        # relation with the given predicate (and, if specified, a specific
        # object value) in the overlay.
        sec_objs: list[str] = []
        if "predicate" in cond:
            obj_entity = r.get("object", "")
            sec_pred = cond["predicate"]
            sec_relations = provider.get_relations(obj_entity, sec_pred)
            # Safety-filter the secondary relations too
            sec_safe = [s for s in sec_relations if validate_hop_safety(_to_hop_edge(s))[0]]
            # If a specific object value is required, narrow further
            if "object" in cond:
                required = _norm(str(cond["object"]))
                sec_safe = [s for s in sec_safe if _norm(s.get("object", "")) == required]
            if not sec_safe:
                continue
            sec_objs = [str(s.get("object", "")) for s in sec_safe if s.get("object")]

        if sec_objs:
            r = dict(r)
            r["_sec_objects"] = sec_objs
        out.append(r)

    trace.append(
        f"Filter({cond}) on {len(source)} items → {len(out)} passed"
    )
    return out


def _exec_count(
    step: Count,
    results: dict[str, Any],
    trace: list[str],
) -> int:
    source = results.get(step.source_key, [])
    n = len(source) if isinstance(source, list) else int(source)
    trace.append(f"Count({step.source_key}) → {n}")
    return n


def _exec_compare(
    step: Compare,
    results: dict[str, Any],
    trace: list[str],
) -> dict:
    a: list[dict] = results.get(step.key_a, [])
    b: list[dict] = results.get(step.key_b, [])

    def _key(r: dict) -> tuple[str, str]:
        return (str(r.get("predicate", "")), str(r.get("object", "")))

    if step.mode == "intersection":
        b_keys = {_key(r) for r in b}
        common = [r for r in a if _key(r) in b_keys]
        trace.append(f"Compare(intersection) {step.label_a} ∩ {step.label_b} → {len(common)} items")
        return {
            "mode": "intersection",
            "items": common,
            "label_a": step.label_a,
            "label_b": step.label_b,
        }

    if step.mode == "difference":
        b_keys = {_key(r) for r in b}
        diff = [r for r in a if _key(r) not in b_keys]
        trace.append(f"Compare(difference) → {len(diff)} unique to {step.label_a}")
        return {
            "mode": "difference",
            "items": diff,
            "label_a": step.label_a,
            "label_b": step.label_b,
        }

    # size_compare
    ca, cb = len(a), len(b)
    larger = "a" if ca > cb else ("b" if cb > ca else "equal")
    trace.append(
        f"Compare(size_compare) {step.label_a}={ca} vs {step.label_b}={cb} → {larger}"
    )
    return {
        "mode": "size_compare",
        "count_a": ca,
        "count_b": cb,
        "label_a": step.label_a,
        "label_b": step.label_b,
        "larger": larger,
        "items_a": a,
        "items_b": b,
    }


def _exec_traverse(
    step: Traverse,
    provider: WikiMemoryOverlayProvider,
    trace: list[str],
) -> list[dict]:
    pred1, pred2 = step.path
    hop1_relations = provider.get_relations(step.entity, pred1)
    hop1_safe = [r for r in hop1_relations if validate_hop_safety(_to_hop_edge(r))[0]]

    results: list[dict] = []
    for r1 in hop1_safe:
        via = r1.get("object", "")
        hop2_relations = provider.get_relations(via, pred2)
        hop2_safe = [r for r in hop2_relations if validate_hop_safety(_to_hop_edge(r))[0]]
        for r2 in hop2_safe:
            results.append({"via": via, "endpoint": r2.get("object", ""), "hop1": r1, "hop2": r2})

    trace.append(
        f"Traverse({step.entity!r}, {step.path}) → {len(results)} 2-hop paths"
    )
    return results


def _exec_classify(
    step: Classify,
    provider: WikiMemoryOverlayProvider,
    ontology_items: list[dict],
    trace: list[str],
) -> dict:
    all_items = [*provider.all_definitions(), *provider.all_relations(), *ontology_items]
    path = find_is_a_path(step.entity, step.target_class, all_items)
    is_member = path is not None
    trace.append(
        f"Classify({step.entity!r}, {step.target_class!r}) → is_member={is_member}"
    )
    return {
        "entity": step.entity,
        "target_class": step.target_class,
        "is_member": is_member,
        "path": path or [],
    }


def _exec_aggregate(
    step: Aggregate,
    results: dict[str, Any],
    trace: list[str],
) -> Any:
    source = results.get(step.source_key, [])
    if not isinstance(source, list):
        source = [source]

    if step.operation == "count":
        result: Any = len(source)
    elif step.operation == "list":
        result = [r.get("object", r) if isinstance(r, dict) else r for r in source]
    elif step.operation == "earliest":
        result = source[0] if source else None
    elif step.operation == "latest":
        result = source[-1] if source else None
    else:
        result = source

    trace.append(f"Aggregate({step.operation}) on {len(source)} items → {result!r}")
    return result


# ── Rendering ──────────────────────────────────────────────────────────────

def _render(plan: QueryPlan, results: dict[str, Any], final_data: Any) -> str:
    hint = plan.question_type_hint

    if hint == "count":
        find_step = next((s for s in plan.steps if isinstance(s, Find)), None)
        count_step = next((s for s in plan.steps if isinstance(s, Count)), None)
        n = final_data if isinstance(final_data, int) else 0
        if find_step is None:
            return f"Count: {n}."
        subject = find_step.subject or find_step.known_value or "?"
        pred = find_step.predicate or "?"
        # Pull the underlying items to list them alongside the count
        source_items: list[dict] = []
        if count_step is not None:
            src = results.get(count_step.source_key, [])
            if isinstance(src, list):
                source_items = src
        objs = list(dict.fromkeys(
            str(r.get("object", "")) for r in source_items if r.get("object")
        ))
        if pred == "founded":
            noun = "company" if n == 1 else "companies"
            if objs:
                return f"{subject} founded {n} {noun}: {_join_list(objs)}."
            return f"{subject} founded {n} {noun}."
        suffix = "" if n == 1 else "s"
        if objs:
            return f"{subject} has {n} {pred} result{suffix}: {_join_list(objs)}."
        return f"{subject} has {n} {pred} result{suffix}."

    if hint == "size_compare":
        d = final_data if isinstance(final_data, dict) else {}
        la = d.get("label_a", "A")
        lb = d.get("label_b", "B")
        ca = d.get("count_a", 0)
        cb = d.get("count_b", 0)
        find_step = next((s for s in plan.steps if isinstance(s, Find)), None)
        pred = (find_step.predicate if find_step else None) or "relations"
        larger = d.get("larger", "equal")
        if larger == "a":
            return f"{la} has more {pred} relations ({ca}) than {lb} ({cb})."
        if larger == "b":
            return f"{lb} has more {pred} relations ({cb}) than {la} ({ca})."
        return f"{la} and {lb} each have {ca} {pred} relation(s)."

    if hint == "intersection":
        d = final_data if isinstance(final_data, dict) else {}
        items: list[dict] = d.get("items", [])
        la = d.get("label_a", "A")
        lb = d.get("label_b", "B")
        if not items:
            return f"{la} and {lb} share no common relations in the overlay."
        grouped: dict[str, list[str]] = {}
        for r in items:
            grouped.setdefault(str(r.get("predicate", "?")), []).append(str(r.get("object", "?")))
        parts = [f"both {pred} {', '.join(objs)}" for pred, objs in grouped.items()]
        return f"{la} and {lb}: {'; '.join(parts)}."

    if hint == "filtered_lookup":
        items = final_data if isinstance(final_data, list) else []
        if not items:
            return ""
        find_step = next((s for s in plan.steps if isinstance(s, Find)), None)
        filter_step = next((s for s in plan.steps if isinstance(s, Filter)), None)
        subject = (find_step.subject if find_step else None) or "?"
        pred = (find_step.predicate if find_step else None) or "?"
        sec_pred = (filter_step.condition.get("predicate") if filter_step else None) or "?"
        primary_objs = list(dict.fromkeys(str(r.get("object", "?")) for r in items))
        all_sec_objs: list[str] = []
        for r in items:
            for x in r.get("_sec_objects", []):
                if x and x not in all_sec_objs:
                    all_sec_objs.append(x)
        obj_str = _join_list(primary_objs)
        if all_sec_objs:
            return f"{subject} {pred} {obj_str}, which {sec_pred} {_join_list(all_sec_objs)}."
        return f"{subject} {pred} {obj_str}, which also {sec_pred}."

    if hint == "is_a":
        d = final_data if isinstance(final_data, dict) else {}
        entity = d.get("entity", "?")
        cls = d.get("target_class", "?")
        if d.get("is_member"):
            return f"Yes, {entity} is a {cls}."
        return f"{entity} is not confirmed as a {cls} in the overlay."

    if hint in ("lookup", "inverse_lookup"):
        items = final_data if isinstance(final_data, list) else []
        if not items:
            return ""
        # Aggregate "list" produces plain strings rather than relation dicts.
        if items and isinstance(items[0], str):
            return ", ".join(items) + "."
        if hint == "lookup":
            subj = str(items[0].get("subject", "?"))
            grouped: dict[str, list[str]] = {}
            for r in items:
                grouped.setdefault(str(r.get("predicate", "?")), []).append(str(r.get("object", "?")))
            parts = [f"{subj} {pred} {', '.join(objs)}" for pred, objs in grouped.items()]
            return ". ".join(parts) + "."
        else:
            obj = str(items[0].get("object", "?"))
            pred = str(items[0].get("predicate", "?"))
            subjs = list(dict.fromkeys(str(r.get("subject", "?")) for r in items))
            return f"{', '.join(subjs)} {pred} {obj}."

    return str(final_data) if final_data is not None else ""


# ── Helpers ────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[''`]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _to_hop_edge(r: dict) -> HopEdge:
    return HopEdge(
        subject=str(r.get("subject", "")),
        predicate=str(r.get("predicate", "")),
        object=str(r.get("object", "")),
        overlay_type=str(r.get("overlay_type", "overlay_relation")),
        trust=str(r.get("trust", "")),
        stability=str(r.get("stability", "")),
        risk=str(r.get("risk", "")),
        source_page=str(r.get("source_page", "")),
        temporal_class=str(r.get("temporal_class", "")),
        as_of=str(r.get("as_of", "")),
    )


def _join_list(items: list[str]) -> str:
    items = list(dict.fromkeys(items))
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _infer_support_kind(results: dict[str, Any]) -> str:
    """Infer support_kind from the stability of underlying relation items."""
    for data in results.values():
        if isinstance(data, list):
            for r in data:
                if isinstance(r, dict) and r.get("stability") == "semi_stable":
                    return "semi_stable_relation"
        if isinstance(data, dict):
            for r in data.get("items", []) + data.get("items_a", []) + data.get("items_b", []):
                if isinstance(r, dict) and r.get("stability") == "semi_stable":
                    return "semi_stable_relation"
    return "stable_relation"
