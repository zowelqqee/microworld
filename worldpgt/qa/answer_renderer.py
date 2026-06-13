"""Short answer renderer for AnswerPlanner v1.

Renders AnswerPlan objects into short, evidence-grounded answer strings.

Rule-based and deterministic only.  No ML libraries.
No learned model parameters.  No language-model inference.  No back-prop.
"""

from __future__ import annotations

import re

from worldpgt.qa.answer_planner import _SENSE_BASE
from worldpgt.qa.types import AnswerPlan

# Sentence-subject display labels for define_sense and distinguish_senses.
# Value includes the article where appropriate; mass nouns and uncountable terms
# appear without an article ("rock music", "spring").
# Keyed by (term, sense_id) so homographs get distinct natural labels.
_DISPLAY_LABEL: dict[tuple[str, str], str] = {
    ("bank", "financial_institution"): "a financial bank",
    ("bank", "river_edge"): "a river bank",
    ("bat", "animal"): "a bat",
    ("bat", "sports_equipment"): "a baseball bat",
    ("seal", "animal"): "a seal",
    ("seal", "closure_stamp"): "a wax seal",
    ("crane", "bird"): "a crane bird",
    ("crane", "machine"): "a construction crane",
    ("rock", "stone"): "a rock",
    ("rock", "music"): "rock music",
    ("spring", "season"): "spring",
    ("spring", "coil"): "a spring coil",
}

# Inline labels for explain_cue and classify_context.
# Values preserve keyword tokens required by qa_prompts_v1.csv expected checks.
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

# Natural prepositional phrases for typical_location values.
_LOCATION_PREP: dict[str, str] = {
    "branch": "at a bank branch",
    "document": "on documents",
    "device": "inside devices",
    "river": "along rivers",
    "cave": "in caves",
    "attic": "in attics",
    "construction site": "at construction sites",
    "coast": "along coasts",
    "shore": "along shores",
    "plate": "at home plate",
    "stage": "on stage",
    "cliff": "near cliffs",
    "mountain": "near mountains",
    "wetland": "in wetlands",
    "lake": "near lakes",
}

# Strips a trailing ", associated with ..." clause from a base description so
# the renderer can append its own cue sentence without duplication.
_TRAILING_ASSOC_RE = re.compile(r",\s*associated with\s+[^.]*$")


def _strip_trailing_associated(desc: str) -> str:
    """Remove trailing ', associated with ...' clause from a base description."""
    return _TRAILING_ASSOC_RE.sub("", desc).rstrip()


def _start_sentence(label: str) -> str:
    """Capitalize the first character of a display label to open a sentence."""
    return label[0].upper() + label[1:] if label else label


def _loc_phrase(loc: str) -> str:
    """Return a natural prepositional phrase for a location value."""
    return _LOCATION_PREP.get(loc, f"near {loc}")


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

        display = _start_sentence(
            _DISPLAY_LABEL.get((term, sense_id), f"a {sense_id}")
        )
        clean_desc = _strip_trailing_associated(base_desc)
        parts = [f"{display} is {clean_desc}."]
        if cues:
            parts.append(f"It is associated with {_join_list(cues)}.")
        if actions:
            parts.append(f"Typical actions include {_join_list(actions)}.")
        if locations:
            loc_phrases = [_loc_phrase(loc) for loc in locations]
            parts.append(f"It is commonly found {_join_list(loc_phrases)}.")
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

        display_a = _start_sentence(
            _DISPLAY_LABEL.get((term, sense_a), f"a {sense_a}")
        )
        display_b = _start_sentence(
            _DISPLAY_LABEL.get((term, sense_b), f"a {sense_b}")
        )
        label_b_raw = _DISPLAY_LABEL.get((term, sense_b), f"a {sense_b}")

        desc_a = _strip_trailing_associated(_SENSE_BASE.get((term, sense_a), ""))
        desc_b = _strip_trailing_associated(_SENSE_BASE.get((term, sense_b), ""))

        parts: list[str] = []
        if desc_a:
            parts.append(f"{display_a} is {desc_a}")
            if sa["cues"]:
                parts[-1] += f", associated with {_join_list(sa['cues'][:3])}."
            else:
                parts[-1] += "."
        if desc_b:
            parts.append(f"{display_b} is {desc_b}")
            if sb["cues"]:
                parts[-1] += f", associated with {_join_list(sb['cues'][:3])}."
            else:
                parts[-1] += "."
        return " ".join(parts) if parts else (
            f"{display_a} differs from {label_b_raw} in its typical context and cues."
        )

    return "I cannot answer safely."
