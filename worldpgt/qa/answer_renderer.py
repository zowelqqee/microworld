"""Short answer renderer for AnswerPlanner v1.

Renders AnswerPlan objects into short, evidence-grounded answer strings.

Rule-based and deterministic only.  No ML libraries.
No learned model parameters.  No language-model inference.  No back-prop.
"""

from __future__ import annotations

from worldpgt.qa.answer_planner import _SENSE_BASE
from worldpgt.qa.types import AnswerPlan

_SENSE_LABEL: dict[str, str] = {
    "financial_institution": "financial bank",
    "river_edge": "river bank",
    "animal": "animal",
    "sports_equipment": "sports bat",
    "closure_stamp": "closure seal",
    "bird": "crane bird",
    "machine": "construction crane",
    "stone": "rock stone",
    "music": "rock music",
    "season": "spring season",
    "coil": "spring coil",
}

_AUDIT_MESSAGES: dict[str, str] = {
    "underconstrained_question": (
        "I cannot answer safely because the question is underconstrained."
    ),
    "missing_term_or_sense": (
        "I cannot answer safely because the term or sense could not be identified."
    ),
    "missing_cue_or_sense": (
        "I cannot answer safely because the cue or sense could not be identified."
    ),
    "cue_not_found": (
        "I cannot answer safely because the cue is not found in the known sense memory."
    ),
    "missing_context_or_term": (
        "I cannot answer safely because no context or term was identified."
    ),
    "no_cue_evidence_in_context": (
        "I cannot answer safely because the context contains no known sense cues."
    ),
    "score_below_threshold": (
        "I cannot answer safely because the evidence score is below the required threshold."
    ),
    "margin_below_threshold": (
        "I cannot answer safely because the margin between senses is below the required threshold."
    ),
    "conflict_detected": (
        "I cannot answer safely because the context contains conflicting sense cues."
    ),
    "missing_senses_for_distinction": (
        "I cannot answer safely because one or both senses could not be identified."
    ),
    "insufficient_evidence_for_distinction": (
        "I cannot answer safely because one or both senses lack sufficient evidence."
    ),
    "sense_not_in_memory": (
        "I cannot answer safely because this sense is not in the known memory."
    ),
    "unsupported_intent": (
        "I cannot answer safely because this question type is not supported."
    ),
}


def _join_list(items: list[str], connector: str = "and") -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {connector} " + items[-1]


def render(plan: AnswerPlan) -> str:
    """Return a short answer string from the plan."""
    if plan.decision == "audit":
        reason = plan.audit_reason or "unknown"
        return _AUDIT_MESSAGES.get(
            reason,
            f"I cannot answer safely because: {reason}.",
        )

    tmpl = plan.render_template
    args = plan.render_args

    if tmpl == "define_sense":
        term = args["term"]
        sense_id = args["sense_id"]
        base_desc = args["base_desc"]
        cues = args.get("cues", [])
        actions = args.get("typical_actions", [])
        locations = args.get("typical_locations", [])

        parts = [f"A {_SENSE_LABEL.get(sense_id, sense_id)} is {base_desc}."]
        if cues:
            parts.append(f"It is associated with {_join_list(cues)}.")
        if actions:
            parts.append(f"Typical actions include {_join_list(actions)}.")
        if locations:
            parts.append(f"It is commonly found near {_join_list(locations)}.")
        return " ".join(parts)

    if tmpl == "explain_cue":
        cue = args["cue"]
        term = args["term"]
        sense_id = args["sense_id"]
        related = args.get("related_cues", [])
        actions = args.get("typical_actions", [])

        label = _SENSE_LABEL.get(sense_id, sense_id)
        parts = [
            f"The cue '{cue}' supports {term} as {label} because it appears"
            f" in contexts involving {term} as {label}."
        ]
        if related:
            parts.append(f"Related cues include {_join_list(related)}.")
        if actions:
            parts.append(f"Typical actions include {_join_list(actions)}.")
        return " ".join(parts)

    if tmpl == "classify_context":
        term = args["term"]
        sense_id = args["sense_id"]
        matched = args.get("matched_cues", [])

        label = _SENSE_LABEL.get(sense_id, sense_id)
        parts = [f"Here, {term} means {label}"]
        if matched:
            parts[-1] += f" because the context includes {_join_list(matched)}."
        else:
            parts[-1] += "."
        return " ".join(parts)

    if tmpl == "distinguish_senses":
        term = args["term"]
        sense_a = args["sense_a"]
        sense_b = args["sense_b"]
        sa = args["summary_a"]
        sb = args["summary_b"]

        label_a = _SENSE_LABEL.get(sense_a, sense_a)
        label_b = _SENSE_LABEL.get(sense_b, sense_b)

        desc_a = _SENSE_BASE.get((term, sense_a), "")
        desc_b = _SENSE_BASE.get((term, sense_b), "")

        parts: list[str] = []
        if desc_a:
            parts.append(f"A {label_a} is {desc_a}")
            if sa["cues"]:
                parts[-1] += f", associated with {_join_list(sa['cues'][:3])}."
            else:
                parts[-1] += "."
        if desc_b:
            parts.append(f"A {label_b} is {desc_b}")
            if sb["cues"]:
                parts[-1] += f", associated with {_join_list(sb['cues'][:3])}."
            else:
                parts[-1] += "."
        return " ".join(parts) if parts else (
            f"A {label_a} differs from a {label_b} in its typical context and cues."
        )

    return "I cannot answer safely."
