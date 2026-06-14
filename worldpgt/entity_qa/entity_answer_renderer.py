"""Entity answer renderer v1.

Converts EntityQAPlan into short, honest, source-aware answer strings.
Deterministic rule-based only. No ML. No network.
"""

from __future__ import annotations

import re

from worldpgt.entity_qa.types import EntityQAPlan

_AUDIT_PREFIX = "I cannot answer from the wiki overlay because "

_VERB_OBJECT_PHRASE: dict[str, str] = {
    "produces": "produces",
    "develops": "develops",
    "publishes": "publishes",
    "known_for": "is known for",
    "leader_of": "leads",
    "founded": "was founded by",
    "related_to": "is related to",
    "part_of": "is part of",
}


def render(plan: EntityQAPlan) -> str:
    if plan.decision == "audit":
        return _render_audit(plan)

    t = plan.render_template
    args = plan.render_args

    if t == "define_entity":
        return _render_define(args)
    if t == "relation_lookup":
        return _render_relation(args)
    if t == "link_explanation":
        return _render_link_explanation(args)
    if t == "source_fact_lookup":
        return _render_source_fact(args)
    if t == "stability_check":
        return _render_stability_check(args)
    if t == "link_policy":
        return _render_link_policy(args)

    return _render_audit(plan)


# ------------------------------------------------------------------
# Template renderers
# ------------------------------------------------------------------


def _render_audit(plan: EntityQAPlan) -> str:
    reason = plan.render_args.get("reason", plan.audit_reason or "unsupported query")
    return _AUDIT_PREFIX + _lowercase_first(str(reason))


def _render_define(args: dict) -> str:
    entity = args.get("entity")
    definition = args.get("definition")
    relations: list[dict] = args.get("relations", [])
    subject: str = args.get("subject", "")

    parts: list[str] = []

    if definition:
        def_text = definition["definition"]
        entity_label = definition["subject"]
        # Check for article already in definition
        if def_text and def_text[0].lower() in "aeiou":
            article = "an"
        else:
            article = "a"
        parts.append(f"{entity_label} is {article} {def_text}.")
    elif entity:
        label = entity["label"]
        etype = entity.get("entity_type", "entity")
        parts.append(f"{label} is a {etype}.")

    if entity and entity.get("entity_type"):
        pass  # already covered above

    if relations:
        # Relations here all have the described entity as their *subject*, so
        # directional predicates must be expressed from the subject's perspective.
        pred_groups: dict[str, list[str]] = {}
        for r in relations:
            pred_groups.setdefault(r["predicate"], []).append(r["object"])

        # Copular clauses read after "is" (no leading subject); verb clauses
        # carry their own verb and must NOT be prefixed with "is".
        copular_clauses: list[str] = []
        verb_clauses: list[str] = []
        for pred, objects in pred_groups.items():
            obj_list = _join_list(objects)
            if pred == "leader_of":
                copular_clauses.append(f"linked to {obj_list} through leadership")
            elif pred == "known_for":
                copular_clauses.append(f"known for {obj_list}")
            elif pred == "founded":
                # Subject founded the objects — "is the founder of X", not "founded by X".
                copular_clauses.append(f"listed as the founder of {obj_list}")
            elif pred == "produces":
                verb_clauses.append(f"produces {obj_list}")
            elif pred == "develops":
                verb_clauses.append(f"develops {obj_list}")
            elif pred == "publishes":
                verb_clauses.append(f"publishes {obj_list}")
            else:
                copular_clauses.append(f"linked to {obj_list} via {pred}")

        label = entity["label"] if entity else subject
        if copular_clauses and verb_clauses:
            parts.append(
                f"In the overlay, {label} is {_join_clauses(copular_clauses)}, "
                f"and it {_join_clauses(verb_clauses)}."
            )
        elif copular_clauses:
            parts.append(f"In the overlay, {label} is {_join_clauses(copular_clauses)}.")
        elif verb_clauses:
            parts.append(f"In the overlay, {label} {_join_clauses(verb_clauses)}.")

    return " ".join(parts).strip()


def _render_relation(args: dict) -> str:
    subject: str = args.get("subject", "")
    predicate: str = args.get("predicate") or ""
    relations: list[dict] = args.get("relations", [])
    founder_lookup: bool = args.get("founder_lookup", False)

    if not relations:
        return f"No relation data found for {subject} in the overlay."

    # "Who founded X?" — relations have X as object, founders as subject.
    if founder_lookup:
        founders = [r["subject"] for r in relations if r.get("subject")]
        if founders:
            founder_list = _join_list(founders)
            return f"{subject} was founded by {founder_list}."

    # Group by predicate, collecting objects.
    pred_groups: dict[str, list[str]] = {}
    for r in relations:
        pred_groups.setdefault(r["predicate"], []).append(r["object"])

    sentences: list[str] = []
    for pred, objects in pred_groups.items():
        obj_list = _join_list(objects)
        if pred == "leader_of":
            sentences.append(f"{subject} is linked to {obj_list} through leadership.")
        elif pred == "known_for":
            sentences.append(f"{subject} is known for {obj_list}.")
        elif pred == "produces":
            sentences.append(f"{subject} produces {obj_list}.")
        elif pred == "develops":
            sentences.append(f"{subject} develops {obj_list}.")
        elif pred == "publishes":
            sentences.append(f"{subject} publishes {obj_list}.")
        elif pred == "founded":
            # Subject is the founder here — never invert to "founded by".
            sentences.append(f"{subject} founded {obj_list}.")
        else:
            verb = _VERB_OBJECT_PHRASE.get(pred, f"is linked to via {pred}")
            sentences.append(f"{subject} {verb} {obj_list}.")

    return " ".join(sentences)


def _render_link_explanation(args: dict) -> str:
    subject: str = args.get("subject", "")
    secondary: str = args.get("secondary", "")
    links: list[dict] = args.get("links", [])

    if not links:
        return (
            f"{secondary} does not appear as a context link on the {subject} page "
            "in this overlay."
        )

    source_pages = list({lnk["source_page"] for lnk in links})
    page_str = _join_list(source_pages)

    return (
        f"{secondary} is linked from the {page_str} page as a weak contextual mention. "
        f"It is not treated as a stable factual relation by this overlay."
    )


def _render_source_fact(args: dict) -> str:
    facts: list[dict] = args.get("facts", [])
    if not facts:
        return "No source-qualified fact found in the overlay for this query."

    parts: list[str] = []
    for f in facts:
        source = f.get("source_name", "an unknown source")
        as_of = f.get("as_of", "an unknown date")
        predicate = f.get("predicate", "")
        obj = f.get("object", "")
        subject = f.get("subject", "")

        pred_phrase = predicate.replace("_", " ")
        parts.append(
            f"According to {source} as of {as_of}, "
            f"{subject}'s {pred_phrase} is {obj}. "
            f"This is a volatile source-qualified estimate and should be rechecked."
        )

    return " ".join(parts)


def _render_stability_check(args: dict) -> str:
    subject: str = args.get("subject", "")
    predicate_hint: str = args.get("predicate_hint", "")
    facts: list[dict] = args.get("facts", [])

    if predicate_hint == "stability_check":
        if facts:
            f = facts[0]
            source = f.get("source_name", "a source")
            as_of = f.get("as_of", "unknown date")
            return (
                f"No, {subject}'s net worth is not a stable fact. "
                f"It is classified as volatile and source-qualified "
                f"(source: {source}, as of {as_of}). "
                f"It requires rechecking."
            )
        return (
            f"No, {subject}'s net worth is not a stable fact in this overlay. "
            f"It is classified as volatile and source-qualified. It requires rechecking."
        )

    if predicate_hint == "recheck_reason":
        if facts:
            f = facts[0]
            source = f.get("source_name", "a source")
            as_of = f.get("as_of", "unknown date")
            return (
                f"{subject}'s net worth should be rechecked because it is a "
                f"time-sensitive estimate from {source} (as of {as_of}). "
                f"Wealth figures change frequently and are source-dependent."
            )
        return (
            f"{subject}'s net worth should be rechecked because it is a "
            f"time-sensitive, source-qualified estimate. "
            f"Wealth figures change frequently and are not stable facts."
        )

    if predicate_hint == "source_qualified_confirm":
        if facts:
            f = facts[0]
            source = f.get("source_name", "a source")
            as_of = f.get("as_of", "unknown date")
            return (
                f"Yes. In this overlay, the {source} estimate of {subject}'s net worth "
                f"is a source-qualified, volatile estimate (source: {source}, as of {as_of}) "
                f"that requires rechecking. It is not a stable or current fact."
            )
        return (
            f"Yes. {subject}'s net worth in this overlay is a source-qualified, volatile "
            f"estimate that requires rechecking. It is not a stable or current fact."
        )

    return "No source-qualified stability information found in the overlay."


def _render_link_policy(args: dict) -> str:
    return (
        "No. In this overlay, a wiki link is stored as a weak context link "
        "(weak_context_only trust) and is not treated as a stable factual relation. "
        "A weak link does not establish that a claim is true."
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _join_clauses(clauses: list[str]) -> str:
    """Join predicate clauses with commas and a final 'and'."""
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return f"{clauses[0]} and {clauses[1]}"
    return ", ".join(clauses[:-1]) + f", and {clauses[-1]}"


def _join_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _lowercase_first(s: str) -> str:
    if not s:
        return s
    return s[0].lower() + s[1:]
