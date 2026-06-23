"""Entity answer renderer v1.

Converts EntityQAPlan into short, honest, source-aware answer strings.
Deterministic rule-based only. No ML. No network.
"""

from __future__ import annotations

import re

from worldpgt.entity_qa.types import EntityQAPlan
from worldpgt.knowledge.temporal_classification import temporal_caveat, weakest_temporal_class
from worldpgt.relation_extraction_v2.relation_policy import should_hedge_render

_ORDINAL_START_RE = re.compile(
    r"^\s*(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|"
    r"eighteenth|nineteenth|twentieth|\d+(?:st|nd|rd|th))\s+",
    re.IGNORECASE,
)
_SUPERLATIVE_START_RE = re.compile(
    r"^\s*(?:largest|smallest|biggest|richest|poorest|fastest|slowest|"
    r"highest|lowest|most\s+\w+|best|worst|oldest|newest)\b",
    re.IGNORECASE,
)


def _should_use_the(def_text: str) -> bool:
    return bool(_ORDINAL_START_RE.match(def_text) or _SUPERLATIVE_START_RE.match(def_text))

_VERB_OBJECT_PHRASE: dict[str, str] = {
    "produces": "produces",
    "develops": "develops",
    "publishes": "publishes",
    "known_for": "is known for",
    "leader_of": "leads",
    "founded": "was founded by",
    "founded_by": "was founded by",
    "owned_by": "is owned by",
    "related_to": "is related to",
    "part_of": "is part of",
    "alias": "is also known as",
    "based_at": "is based at",
    "ceased_operations": "ceased operations in",
    "construction_started": "construction began in",
    "filed_for_bankruptcy": "filed for",
    "first_released": "was first released in",
    "funded_by": "was funded by",
    "has_facility": "has",
    "hosted_flight_to": "flew missions to",
    "introduced": "was introduced in",
    "located_in": "is located in",
    "marketed_as": "is marketed as",
    "merged_with": "merged with",
    "offers": "is offered on",
    "runs_on": "runs on",
    "supports": "supports",
    "variant_of": "is a variant of",
}


def render(plan: EntityQAPlan) -> str:
    if plan.decision == "audit":
        return _render_audit(plan)
    if plan.decision == "no":
        return _render_negated_is_a(plan.render_args)

    t = plan.render_template
    args = plan.render_args

    if t == "open_synthesis":
        return _render_open_synthesis(args)
    if t == "define_entity":
        return _render_define(args)
    if t == "relation_lookup":
        return _render_relation(args)
    if t == "inverse_relation_lookup":
        return _render_inverse_relation(args)
    if t == "comparative_intersection":
        return _render_comparative_intersection(args)
    if t == "ontology_is_a":
        return _render_ontology_is_a(args)
    if t == "negated_is_a":
        return _render_negated_is_a(args)
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
    reason = str(plan.render_args.get("reason", plan.audit_reason or ""))
    r_lower = reason.lower()
    if "no relevant information" in r_lower:
        return "I don't have verified information about this."
    if "no stable definition" in r_lower:
        return "I don't have a definition for this in my knowledge base."
    if reason:
        clean = (
            reason
            .replace("wiki overlay", "my knowledge base")
            .replace(" in the overlay", "")
            .replace(" from the overlay", "")
            .strip()
            .rstrip(".")
        )
        return f"I don't have verified information about this. {clean}."
    return "I don't have verified information about this."


def _render_negated_is_a(args: dict) -> str:
    subject = str(args.get("subject") or "")
    target = str(args.get("target") or "")
    actual_class = str(args.get("actual_class") or "")
    support_kind = str(args.get("support_kind") or "")

    if support_kind == "entity_type_mismatch":
        return (
            f"No. {subject} is {_article_phrase(actual_class)}, "
            f"not {_article_phrase(target)}. The question's premise is incompatible "
            f"with {subject}'s known type."
        )

    return (
        f"No. {subject} is known to be {_article_phrase(actual_class)}, "
        f"which contradicts {_article_phrase(target)}."
    )


def _definition_sentence(subject: str, def_text: str) -> str:
    if def_text and _should_use_the(def_text):
        article = "the"
    elif def_text and def_text[0].lower() in "aeiou":
        article = "an"
    else:
        article = "a"
    return f"{subject} is {article} {def_text}."


# Forward predicate -> "{subject} <clause> {objects}." (subject is the actor).
_FWD_SYNTH_VERB: dict[str, str] = {
    "develops": "develops",
    "produces": "produces",
    "publishes": "publishes",
    "known_for": "is known for",
    "leader_of": "leads",
    "founded": "founded",
    "owned_by": "is owned by",
    "operates": "operates",
    "provides": "provides",
}
# Inverse predicate -> "{subject} <clause> {others}." (subject is the target).
_INV_SYNTH_VERB: dict[str, str] = {
    "founded": "was founded by",
    "leader_of": "is led by",
    "owned_by": "owns",
    "develops": "is developed by",
    "produces": "is produced by",
    "publishes": "is published by",
}


def _synthesis_clause(subject: str, group) -> str:
    objs = _join_list(list(group.objects))
    pred = group.predicate
    if group.kind == "inverse_relation":
        verb = _INV_SYNTH_VERB.get(pred)
        if verb:
            return f"{subject} {verb} {objs}."
        return f"{subject} is linked to {objs} via {pred}."
    # forward relation
    if pred == "is_a":
        return f"{subject} is {_article_phrase(objs)}."
    verb = _FWD_SYNTH_VERB.get(pred)
    if verb:
        return f"{subject} {verb} {objs}."
    return f"{subject} is linked to {objs} via {pred}."


def _snapshot_clause(subject: str, group) -> str:
    obj = group.objects[0] if group.objects else ""
    pred_phrase = group.predicate.replace("_", " ")
    source = group.source_name or "an unknown source"
    as_of = group.as_of or "an unknown date"
    return (
        f"According to {source} (as of {as_of}), {subject}'s {pred_phrase} is {obj} "
        f"— a volatile, source-qualified estimate that should be rechecked."
    )


def _synthesis_footer(verified: int, snapshot: int, unknown_notes: list[str]) -> str:
    segments: list[str] = []
    if verified:
        noun = "fact" if verified == 1 else "facts"
        segments.append(f"VERIFIED — {verified} {noun} from knowledge base")
    if snapshot:
        noun = "fact" if snapshot == 1 else "facts"
        segments.append(f"SNAPSHOT — {snapshot} dated {noun}")
    if unknown_notes:
        segments.append(f"UNKNOWN — {_join_list(list(unknown_notes))}")
    if not segments:
        segments.append("UNKNOWN — no verified facts found")
    return "[" + " | ".join(segments) + "]"


def _render_open_synthesis(args: dict) -> str:
    result = args.get("synthesis")
    if result is None or not getattr(result, "matched", False):
        return "I don't have verified information about this."

    subject = result.subject or ""
    lines: list[str] = []

    if result.match_kind == "keyword":
        lines.append(
            f"I don't have an entity named exactly that, but here is the closest "
            f"match I know — {subject}:"
        )

    if result.definition:
        lines.append(_definition_sentence(subject, result.definition))
    elif result.entity_type:
        lines.append(f"{subject} is {_article_phrase(result.entity_type)}.")

    snapshot_lines: list[str] = []
    for group in result.groups:
        if group.tier == "SNAPSHOT":
            snapshot_lines.append(_snapshot_clause(subject, group))
        else:
            lines.append(_synthesis_clause(subject, group))
    lines.extend(snapshot_lines)

    for note in result.unknown_notes:
        lines.append(f"I don't have verified information about {note}.")

    footer = _synthesis_footer(
        result.verified_count, result.snapshot_count, result.unknown_notes
    )
    return " ".join(lines) + "\n\n" + footer


def _render_define(args: dict) -> str:
    entity = args.get("entity")
    definition = args.get("definition")
    relations: list[dict] = args.get("relations", [])
    subject: str = args.get("subject", "")

    parts: list[str] = []

    if definition:
        def_text = definition["definition"]
        entity_label = definition["subject"]
        if def_text and _should_use_the(def_text):
            article = "the"
        elif def_text and def_text[0].lower() in "aeiou":
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
            if should_hedge_render(pred):
                copular_clauses.append(f"linked to {obj_list} through leadership")
            elif pred == "known_for":
                copular_clauses.append(f"known for {obj_list}")
            elif pred == "founded":
                # Subject founded the objects — "is the founder of X", not "founded by X".
                copular_clauses.append(f"listed as the founder of {obj_list}")
            elif pred == "founded_by":
                copular_clauses.append(f"founded by {obj_list}")
            elif pred == "owned_by":
                copular_clauses.append(f"owned by {obj_list}")
            elif pred == "produces":
                verb_clauses.append(f"produces {obj_list}")
            elif pred == "develops":
                verb_clauses.append(f"develops {obj_list}")
            elif pred == "publishes":
                verb_clauses.append(f"publishes {obj_list}")
            elif pred == "is_a":
                copular_clauses.append(_article_phrase(obj_list))
            elif pred in _VERB_OBJECT_PHRASE:
                phrase = _VERB_OBJECT_PHRASE[pred]
                if phrase.startswith("is "):
                    copular_clauses.append(f"{phrase.removeprefix('is ')} {obj_list}")
                else:
                    verb_clauses.append(f"{phrase} {obj_list}")
            else:
                copular_clauses.append(f"linked to {obj_list} via {pred}")

        label = entity["label"] if entity else subject
        if copular_clauses and verb_clauses:
            parts.append(
                f"In the overlay, {label} is {_join_clauses(copular_clauses)}, "
                f"and it {_join_clauses(verb_clauses)}."
            )
        elif copular_clauses:
            prefix = "In the overlay, " if parts else ""
            parts.append(f"{prefix}{label} is {_join_clauses(copular_clauses)}.")
        elif verb_clauses:
            prefix = "In the overlay, " if parts else ""
            parts.append(f"{prefix}{label} {_join_clauses(verb_clauses)}.")

    caveat = _relations_temporal_caveat(relations)
    if caveat:
        parts.append(caveat)
    return " ".join(parts).strip()


def _render_relation(args: dict) -> str:
    subject: str = args.get("subject", "")
    predicate: str = args.get("predicate") or ""
    relations: list[dict] = args.get("relations", [])
    founder_lookup: bool = args.get("founder_lookup", False)

    if not relations:
        return f"No relation data found for {subject}."

    # "Who founded X?" — relations have X as object, founders as subject.
    if founder_lookup:
        founders = [
            r["object"] if r.get("predicate") == "founded_by" else r["subject"]
            for r in relations
            if r.get("subject") and r.get("object")
        ]
        if founders:
            founder_list = _join_list(founders)
            caveat = _relations_temporal_caveat(relations)
            suffix = f" {caveat}" if caveat else ""
            return f"{subject} was founded by {founder_list}.{suffix}"

    # Group by predicate, collecting objects.
    pred_groups: dict[str, list[str]] = {}
    for r in relations:
        pred_groups.setdefault(r["predicate"], []).append(r["object"])

    sentences: list[str] = []
    for pred, objects in pred_groups.items():
        obj_list = _join_list(objects)
        if should_hedge_render(pred):
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
        elif pred == "founded_by":
            sentences.append(f"{subject} was founded by {obj_list}.")
        elif pred == "owned_by":
            sentences.append(f"{subject} is owned by {obj_list}.")
        elif pred == "is_a":
            sentences.append(f"{subject} is {_article_phrase(obj_list)}.")
        else:
            verb = _VERB_OBJECT_PHRASE.get(pred, f"is linked to via {pred}")
            sentences.append(f"{subject} {verb} {obj_list}.")

    caveat = _relations_temporal_caveat(relations)
    if caveat:
        sentences.append(caveat)
    return " ".join(sentences)


def _render_inverse_relation(args: dict) -> str:
    known_object = args.get("known_object")
    predicate = args.get("predicate")
    relations: list[dict] = args.get("relations", [])
    subjects = _join_list([r.get("subject", "") for r in relations if r.get("subject")])
    caveat = _sentence_suffix(_relations_temporal_caveat(relations))

    if not subjects:
        return f"No {predicate} relation found for {known_object}."

    if predicate == "owned_by":
        return f"{known_object} owns {subjects}.{caveat}"
    if predicate == "leader_of":
        return f"{subjects} leads {known_object}.{caveat}"
    if predicate == "develops":
        return f"{subjects} develops {known_object}.{caveat}"
    if predicate == "produces":
        return f"{subjects} produces {known_object}.{caveat}"
    if predicate == "publishes":
        return f"{subjects} publishes {known_object}.{caveat}"
    if predicate == "developed_by":
        return f"{subjects} was developed by {known_object}.{caveat}"
    if predicate == "created_by":
        return f"{subjects} was created by {known_object}.{caveat}"
    if predicate == "published_by":
        return f"{subjects} was published by {known_object}.{caveat}"
    if predicate == "subsidiary_of":
        return f"{subjects} is a subsidiary of {known_object}.{caveat}"

    return f"{subjects} has relation {predicate} to {known_object}.{caveat}"


def _render_comparative_intersection(args: dict) -> str:
    entity_a = str(args.get("entity_a") or "")
    entity_b = str(args.get("entity_b") or "")
    common_pairs: list[dict] = args.get("common_pairs", [])
    common_classes: list[dict] = args.get("common_classes", [])

    clauses: list[str] = []
    classes = [str(item.get("class") or "") for item in common_classes if item.get("class")]
    if classes:
        clauses.append(f"both are {_join_list([_article_phrase(c) for c in classes])}")

    grouped: dict[str, list[str]] = {}
    for item in common_pairs:
        predicate = str(item.get("predicate") or "")
        obj = str(item.get("object") or "")
        if predicate and obj:
            grouped.setdefault(predicate, []).append(obj)

    for predicate, objects in grouped.items():
        obj_list = _join_list(objects)
        if predicate == "develops":
            clauses.append(f"both develop {obj_list}")
        elif predicate == "produces":
            clauses.append(f"both produce {obj_list}")
        elif predicate == "publishes":
            clauses.append(f"both publish {obj_list}")
        elif predicate == "known_for":
            clauses.append(f"both are known for {obj_list}")
        elif predicate == "owned_by":
            clauses.append(f"both are owned by {obj_list}")
        elif predicate == "founded_by":
            clauses.append(f"both were founded by {obj_list}")
        else:
            phrase = _VERB_OBJECT_PHRASE.get(predicate)
            if phrase:
                clauses.append(f"both {phrase} {obj_list}")
            else:
                clauses.append(f"both have {predicate} {obj_list}")

    if not clauses:
        return f"No common facts found for {entity_a} and {entity_b}."

    return f"{entity_a} and {entity_b} have these facts in common: {_join_clauses(clauses)}."


def _render_ontology_is_a(args: dict) -> str:
    subject: str = args.get("subject", "")
    path: list = args.get("path", [])
    if not path:
        return f"No explicit is_a path found for {subject}."

    clauses: list[str] = []
    for idx, edge in enumerate(path):
        obj = _article_phrase(edge.object)
        if idx == 0:
            clauses.append(f"{edge.subject} is {obj}")
        else:
            clauses.append(f"which is {obj}")
    return "Yes. " + ", ".join(clauses) + "."


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
        return "No source-qualified fact found for this query."

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
            f"This is a snapshot, volatile source-qualified estimate and should be rechecked."
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

    return "No source-qualified stability information found."


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
    items = list(dict.fromkeys(items))  # deduplicate, preserve order
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _relations_temporal_caveat(relations: list[dict]) -> str:
    classes = [str(r.get("temporal_class", "")) for r in relations if r.get("temporal_class")]
    if not classes:
        return ""
    temporal_class = weakest_temporal_class(classes)
    return temporal_caveat(temporal_class, [str(r.get("as_of", "")) for r in relations])


def _sentence_suffix(text: str) -> str:
    return f" {text}" if text else ""


def _article_phrase(text: str) -> str:
    text = str(text).strip()
    if not text:
        return text
    if text.lower().startswith(("a ", "an ", "the ")):
        return text
    article = "an" if text[0].lower() in "aeiou" else "a"
    return f"{article} {text}"


def _lowercase_first(s: str) -> str:
    if not s:
        return s
    return s[0].lower() + s[1:]
