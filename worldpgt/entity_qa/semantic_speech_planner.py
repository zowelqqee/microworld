"""Semantic speech planning for open entity synthesis.

This module turns a :class:`SynthesisAnswer` into a small deterministic speech
plan before rendering. It does not add facts; it only chooses which supported
facts to foreground, how to group them, and how to phrase known gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

from worldpgt.entity_qa.types import SynthesisAnswer


_GENERIC_TERMS = frozenset({
    "aircraft",
    "battery energy storage",
    "battery storage",
    "electric car",
    "electric cars",
    "launch vehicle",
    "launch vehicles",
    "report",
    "reports",
    "rocket",
    "rockets",
    "satellite constellation",
    "satellite internet",
    "spacecraft",
    "vehicle",
    "vehicles",
})

_NON_CONTENT_TYPES = frozenset({"", "other", "unknown"})

_DISPLAY_TERMS = {
    "battery storage": "battery energy storage",
    "electric car": "electric cars",
    "launch vehicle": "launch vehicles",
}


@dataclass
class SpeechPlan:
    subject: str
    style: str = "overview"
    answer_style: str = "normal"
    seed: str = ""
    reference: str = ""
    intro: str = ""
    activity: list[str] = field(default_factory=list)
    origin: list[str] = field(default_factory=list)
    ownership: list[str] = field(default_factory=list)
    classification: list[str] = field(default_factory=list)
    recognition: list[str] = field(default_factory=list)
    mechanism: list[str] = field(default_factory=list)
    purpose: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)
    snapshots: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


def build_speech_plan(
    result: SynthesisAnswer,
    question: str = "",
    *,
    answer_style: str = "normal",
) -> SpeechPlan:
    subject = result.subject or ""
    plan = SpeechPlan(
        subject=subject,
        style=_question_style(question),
        answer_style=answer_style,
        seed=_speech_seed(result, question),
        reference=_followup_reference(subject, result.entity_type, result.definition),
    )

    if result.definition:
        plan.intro = _definition_sentence(subject, result.definition)
    elif result.entity_type and result.entity_type not in _NON_CONTENT_TYPES:
        plan.intro = f"{subject} is {_article_phrase(result.entity_type)}."

    for group in result.groups:
        if group.tier == "SNAPSHOT":
            plan.snapshots.append(_snapshot_sentence(subject, group))
            continue
        _add_verified_group(plan, group)

    plan.gaps.extend(result.unknown_notes)
    return plan


def render_speech_plan(plan: SpeechPlan) -> str:
    from worldpgt.entity_qa.symbolic_text_generator import generate_text

    return generate_text(plan)


def _add_verified_group(plan: SpeechPlan, group) -> None:
    pred = str(group.predicate or "")
    objects = [str(obj) for obj in group.objects if str(obj)]
    if not objects:
        return

    if group.kind == "inverse_relation":
        if pred == "founded":
            plan.origin.extend(objects)
        elif pred == "owned_by":
            plan.ownership.append(f"owns {_join_list(objects)}")
        else:
            plan.other.append(_inverse_relation_phrase(pred, objects))
        return

    if pred == "is_a":
        plan.classification.extend(_lowercase_first(obj) for obj in objects)
    elif pred in {"develops", "produces", "operates", "publishes"}:
        plan.activity.append(_activity_phrase(pred, objects))
    elif pred in {"provides", "enables"}:
        plan.purpose.append(_activity_phrase(pred, objects))
    elif pred in {"uses", "works_by"}:
        plan.mechanism.append(_mechanism_phrase(pred, objects))
    elif pred == "used_for":
        plan.purpose.append(f"is used for {_compact_objects(objects)}")
    elif pred == "founded_by":
        plan.origin.extend(objects)
    elif pred == "founded":
        plan.origin.append(f"founded {_compact_objects(objects)}")
    elif pred == "owned_by":
        plan.ownership.append(f"is owned by {_join_list(objects)}")
    elif pred == "known_for":
        plan.recognition.extend(objects)
    else:
        plan.other.append(f"is linked to {_join_list(objects)} via {pred}")


def _known_sentence(plan: SpeechPlan) -> str:
    buckets: dict[str, list[str]] = {
        "classification": [],
        "activity": [],
        "origin": [],
        "ownership": [],
        "recognition": [],
        "mechanism": [],
        "purpose": [],
        "other": [],
    }
    if plan.classification:
        classes = _compact_objects(plan.classification)
        buckets["classification"].append(f"is classified as {classes}")
    if plan.activity:
        buckets["activity"].extend(plan.activity)
    if plan.origin:
        founder_names = [x for x in plan.origin if not x.startswith("founded ")]
        founded_objects = [x for x in plan.origin if x.startswith("founded ")]
        if founder_names:
            buckets["origin"].append(f"was founded by {_join_list(founder_names)}")
        buckets["origin"].extend(founded_objects)
    if plan.ownership:
        buckets["ownership"].extend(plan.ownership)
    if plan.recognition:
        buckets["recognition"].append(f"is known for {_join_list(plan.recognition)}")
    if plan.mechanism:
        buckets["mechanism"].extend(plan.mechanism)
    if plan.purpose:
        buckets["purpose"].extend(plan.purpose)
    if plan.other:
        buckets["other"].extend(plan.other)

    clauses = _ordered_clauses(plan, buckets)
    if plan.answer_style == "brief":
        clauses = [_brief_clause(clauses[0])] if clauses else []
    elif plan.answer_style == "followup":
        clauses = clauses[:3]
    elif plan.answer_style == "important":
        clauses = clauses[:2]
    elif plan.answer_style == "simple":
        clauses = _simple_clauses(clauses)
    if not clauses:
        return ""
    return f"{plan.reference} {_join_clauses(clauses)}."


def _activity_phrase(predicate: str, objects: list[str]) -> str:
    phrase = {
        "develops": "develops",
        "produces": "produces",
        "operates": "operates",
        "provides": "provides",
        "publishes": "publishes",
    }.get(predicate, predicate)
    return f"{phrase} {_compact_objects(objects)}"


def _mechanism_phrase(predicate: str, objects: list[str]) -> str:
    if predicate == "works_by":
        return f"works by {_compact_objects(objects)}"
    return f"uses {_compact_objects(objects)}"


def _inverse_relation_phrase(predicate: str, objects: list[str]) -> str:
    phrase = {
        "develops": "is developed by",
        "produces": "is produced by",
        "publishes": "is published by",
        "leader_of": "is led by",
    }.get(predicate)
    if phrase:
        return f"{phrase} {_join_list(objects)}"
    return f"is linked to {_join_list(objects)} via {predicate}"


def _compact_objects(objects: list[str], *, max_named: int = 4) -> str:
    items = _dedupe_objects([obj for obj in objects if obj])
    generic: list[str] = []
    named: list[str] = []
    for item in items:
        key = item.lower()
        if key in _GENERIC_TERMS:
            generic.append(_DISPLAY_TERMS.get(key, _lowercase_first(item)))
        else:
            named.append(item)

    named_sample = named[:max_named]
    remaining = len(named) - len(named_sample)
    if generic and named_sample:
        suffix_items = list(named_sample)
        if remaining > 0:
            suffix_items.append(
                f"{remaining} other named item" + ("" if remaining == 1 else "s")
            )
        return f"{_join_list(generic)}, including {_join_list(suffix_items)}"

    out = generic + named_sample
    if remaining > 0:
        out.append(f"{remaining} other named item" + ("" if remaining == 1 else "s"))
    return _join_list(out)


def _snapshot_sentence(subject: str, group) -> str:
    obj = group.objects[0] if group.objects else ""
    pred_phrase = str(group.predicate or "").replace("_", " ")
    source = group.source_name or "an unknown source"
    as_of = group.as_of or "an unknown date"
    return (
        f"According to {source} (as of {as_of}), {subject}'s {pred_phrase} is {obj} "
        f"— a volatile, source-qualified estimate that should be rechecked."
    )


def _definition_sentence(subject: str, def_text: str) -> str:
    article = _article_for(def_text)
    return f"{subject} is {article} {def_text}."


def _article_phrase(text: str) -> str:
    text = str(text).strip()
    if not text:
        return text
    if text.lower().startswith(("a ", "an ", "the ")):
        return text
    return f"{_article_for(text)} {text}"


def _article_for(text: str) -> str:
    text = str(text or "").strip()
    return "an" if text[:1].lower() in "aeiou" else "a"


def _join_clauses(clauses: list[str]) -> str:
    clauses = _dedupe(clauses)
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    return ", ".join(clauses[:-1]) + f", and {clauses[-1]}"


def _ordered_clauses(plan: SpeechPlan, buckets: dict[str, list[str]]) -> list[str]:
    orders = {
        "how": [
            (4, ("mechanism", "purpose", "classification", "ownership", "activity", "origin", "recognition", "other")),
            (2, ("purpose", "mechanism", "ownership", "classification", "activity", "origin", "recognition", "other")),
        ],
        "knowledge": [
            (3, ("activity", "purpose", "origin", "classification", "ownership", "mechanism", "recognition", "other")),
            (2, ("classification", "activity", "purpose", "origin", "ownership", "mechanism", "recognition", "other")),
            (2, ("origin", "activity", "purpose", "recognition", "ownership", "classification", "mechanism", "other")),
        ],
        "overview": [
            (3, ("activity", "purpose", "origin", "ownership", "classification", "mechanism", "recognition", "other")),
            (2, ("origin", "activity", "purpose", "ownership", "recognition", "classification", "mechanism", "other")),
            (2, ("classification", "activity", "purpose", "origin", "ownership", "mechanism", "recognition", "other")),
        ],
    }
    order = _weighted_choice(
        plan.seed,
        f"order:{plan.style}",
        orders.get(plan.style, orders["knowledge"]),
    )
    clauses: list[str] = []
    for name in order:
        clauses.extend(buckets.get(name, []))
    return clauses


def _simple_clauses(clauses: list[str]) -> list[str]:
    out: list[str] = []
    for clause in clauses:
        simplified = (
            clause
            .replace("is classified as", "is a kind of")
            .replace("is owned by", "belongs to")
            .replace("works by", "works by using")
        )
        out.append(simplified)
        if len(out) == 3:
            break
    return out


def _brief_clause(clause: str) -> str:
    if ", including " in clause:
        return clause.split(", including ", 1)[0]
    return clause


def _weighted_choice(seed: str, node: str, weighted_options):
    total = sum(weight for weight, _value in weighted_options)
    if total <= 0:
        return weighted_options[0][1]
    value = int(sha256(f"{seed}:{node}".encode("utf-8")).hexdigest()[:12], 16)
    pick = value % total
    acc = 0
    for weight, option in weighted_options:
        acc += weight
        if pick < acc:
            return option
    return weighted_options[-1][1]


def _speech_seed(result: SynthesisAnswer, question: str) -> str:
    parts = [question or "", result.subject or "", result.definition or ""]
    for group in result.groups:
        parts.append(str(group.kind))
        parts.append(str(group.predicate))
        parts.extend(str(obj) for obj in group.objects)
    parts.extend(result.unknown_notes)
    return "|".join(parts)


def _followup_reference(subject: str, entity_type: str | None, definition: str | None) -> str:
    text = f"{entity_type or ''} {definition or ''}".lower()
    if "businessman" in text or "person" in text or "entrepreneur" in text:
        parts = [p for p in subject.split() if p]
        return parts[-1] if len(parts) > 1 else subject
    return "It"


def _join_list(items: list[str]) -> str:
    items = _dedupe(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _dedupe_objects(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = _DISPLAY_TERMS.get(item.lower(), item).lower()
        display = _DISPLAY_TERMS.get(item.lower(), item)
        if key and key not in seen:
            seen.add(key)
            out.append(display)
    return out


def _lowercase_first(s: str) -> str:
    if not s:
        return s
    return s[0].lower() + s[1:]


def _question_style(question: str) -> str:
    q = (question or "").strip().lower()
    if q.startswith("how "):
        return "how"
    if "what do you know" in q or "what can you tell" in q:
        return "knowledge"
    return "overview"
