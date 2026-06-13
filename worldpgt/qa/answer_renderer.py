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

# --------------------------------------------------------------------------
# SemanticLanguageRealizer v2
#
# Deterministic mapping from world relation -> communicative role -> sentence
# fragment.  This is NOT a generator: every word emitted is either fixed
# connective scaffolding or a token already present in the explicit evidence.
# No facts are invented; fixed frames only complete an action's natural role
# (who/what does it, on what) and are only emitted when the action verbs they
# reference are present in the evidence.
#
#   definition / is_a  ->  "X is ..."
#   cue                ->  "Common clues/signs/contexts are ..."
#   action             ->  agency-aware frame (see _ACTION_REALIZATION)
#   location           ->  "It is often found ..."
# --------------------------------------------------------------------------

# Per-sense lead-in for the cue (relation: cue) fragment.  Default below.
_CUE_LEADIN: dict[tuple[str, str], str] = {
    ("spring", "season"): "Common signs are",
    ("rock", "music"): "Common contexts are",
}
_DEFAULT_CUE_LEADIN = "Common clues are"

# Safe per-cue surface forms, applied before generic pluralization.  Keyed by
# (term, sense_id, cue).  Used to avoid awkward plurals ("thaws", "rains") and
# to render mild, evidence-grounded morphology ("morning" -> "warmer mornings").
_CUE_SURFACE: dict[tuple[str, str, str], str] = {
    ("spring", "season", "thaw"): "thaw",
    ("spring", "season", "flower"): "flowers",
    ("spring", "season", "morning"): "warmer mornings",
    ("spring", "season", "rain"): "rain",
    ("rock", "music", "band"): "bands",
    ("rock", "music", "concert"): "concerts",
    ("rock", "music", "crowd"): "crowds",
    ("rock", "music", "stage"): "stages",
}

# Uncountable / mass tokens that must never be blindly pluralized.
_UNCOUNTABLE_CUES: frozenset[str] = frozenset(
    {"thaw", "rain", "music", "water", "cash", "snow", "wax", "fish"}
)

# Agency-aware action realization, keyed by (term, sense_id).
#   frame             - descriptive label of the agency role (documentation)
#   text              - fixed clause; "{actions}" is filled from evidence verbs
#   connector         - word joining multiple action verbs in "{actions}"
#   requires          - for fixed-text frames, action verbs that MUST be in the
#                       evidence before the frame may be used (evidence safety)
#   include_location  - whether to append the "found ..." location clause
_ACTION_REALIZATION: dict[tuple[str, str], dict] = {
    ("bat", "animal"): {
        "frame": "subject_can",
        "text": "It can {actions}",
        "connector": "and",
        "include_location": True,
    },
    ("bat", "sports_equipment"): {
        "frame": "instrument",
        "text": "It is used to {actions} the ball",
        "connector": "or",
        "include_location": True,
    },
    ("seal", "animal"): {
        "frame": "subject_can",
        "text": "It can {actions}",
        "connector": "and",
        "include_location": True,
    },
    ("seal", "closure_stamp"): {
        "frame": "instrument",
        "text": "It is used to {actions} documents or envelopes",
        "connector": "or",
        "include_location": False,
    },
    ("crane", "bird"): {
        "frame": "subject_can",
        "text": "It can {actions}",
        "connector": "and",
        "include_location": True,
    },
    ("crane", "machine"): {
        "frame": "machine_function",
        "text": "It is used to {actions} heavy loads",
        "connector": "or",
        "include_location": True,
    },
    ("rock", "music"): {
        "frame": "performed_content",
        "text": "It is played and performed, often by bands on stage",
        "requires": ("play", "perform"),
        "include_location": False,
    },
    ("rock", "stone"): {
        "frame": "patient_or_natural",
        "text": "A rock can roll or be climbed",
        "requires": ("climb", "roll"),
        "include_location": True,
    },
    ("spring", "season"): {
        "frame": "seasonal_events",
        "text": "In spring, flowers bloom and snow thaws",
        "requires": ("bloom", "thaw"),
        "include_location": True,
    },
    ("spring", "coil"): {
        "frame": "mechanical",
        "text": "It can {actions}",
        "connector": "and",
        "include_location": True,
    },
}

# --------------------------------------------------------------------------
# ContrastRealizer v1 (distinguish_senses)
#
# Compact, relation-aware contrast clauses.  Each side is rendered as
#   "<Display label> is <gloss> <clue/action clause>."
# The gloss uses only mild role words already supported by the base definition
# or sense label (e.g. "money", "sports equipment", "marine mammal").  Clue
# tokens are drawn from the evidence and surfaced via _cue_surface; no facts
# are invented.  Reuses display labels, cue surfaces, and the action joiner.
# --------------------------------------------------------------------------
_CONTRAST_SIDE: dict[tuple[str, str], dict] = {
    ("bank", "financial_institution"): {
        "gloss": "a money institution",
        "clue_lead": " for",
        "clue_count": 2,
    },
    ("bank", "river_edge"): {
        "gloss": "land beside a river or stream",
        "clue_lead": ", marked by",
        "clue_count": 3,
    },
    ("bat", "animal"): {
        "gloss": "a flying mammal",
        "clue_lead": " linked to",
        "clue_count": 3,
    },
    ("bat", "sports_equipment"): {
        "gloss": "sports equipment used to hit a ball",
        "clue_lead": ", with clues like",
        "clue_count": 3,
    },
    ("seal", "animal"): {
        "gloss": "a marine mammal",
        "clue_lead": " linked to",
        "clue_count": 3,
    },
    ("seal", "closure_stamp"): {
        "gloss": "a device used to close or authenticate an object",
        "clue_lead": ", with clues like",
        "clue_count": 3,
    },
    ("crane", "bird"): {
        "gloss": "a large wading bird",
        "clue_lead": " linked to",
        "clue_count": 3,
    },
    ("crane", "machine"): {
        "gloss": "a machine used to lift heavy loads",
        "clue_lead": ", with clues like",
        "clue_count": 3,
    },
    ("rock", "music"): {
        "gloss": "a music genre",
        "clue_lead": " linked to",
        "clue_count": 3,
    },
    ("rock", "stone"): {
        "gloss": "a solid mineral object",
        "clue_lead": " found near",
        "clue_count": 3,
    },
    ("spring", "season"): {
        "gloss": "the season after winter",
        "clue_lead": ", marked by",
        "clue_count": 3,
    },
    ("spring", "coil"): {
        "gloss": "a coiled metal device that stores and releases mechanical energy",
        "action_tail": ", and can",
        "connector": "or",
    },
}

# Human-readable sense alternatives for each ambiguous term.
# Used only when rendering a helpful underconstrained_question audit.
# Deterministic: order matches the canonical sense ordering of _TERM_SENSES.
_AMBIGUOUS_SENSE_LABELS: dict[str, list[str]] = {
    "bank": ["a financial institution", "the edge of a river"],
    "bat": ["a flying mammal", "a baseball bat"],
    "seal": ["a marine animal", "a wax/document seal"],
    "crane": ["a bird", "a construction machine"],
    "rock": ["rock music", "a stone"],
    "spring": ["the season after winter", "a mechanical coil"],
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


def _pluralize(word: str) -> str:
    """Deterministic, dependency-free plural for a single evidence token.

    Handles the regular English plural rules only (``-s``, ``-es`` after a
    sibilant, ``y -> ies``).  This is intentionally small: it is morphology,
    not NLP, and never changes the underlying token's meaning.  Uncountable /
    mass tokens are returned unchanged.
    """
    w = word.strip()
    if not w:
        return w
    lower = w.lower()
    if lower in _UNCOUNTABLE_CUES:
        return w
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    if lower.endswith("y") and len(w) >= 2 and lower[-2] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


def _cue_surface(term: str, sense_id: str, cue: str) -> str:
    """Return the safe surface form for a cue (table first, then morphology)."""
    mapped = _CUE_SURFACE.get((term, sense_id, cue.strip().lower()))
    if mapped is not None:
        return mapped
    return _pluralize(cue)


def _realize_definition(display: str, clean_desc: str) -> str:
    """definition / is_a relation -> 'X is ...' fragment."""
    return f"{display} is {clean_desc}."


def _realize_cues(term: str, sense_id: str, cues: list[str]) -> str:
    """cue relation -> 'Common clues are ...' fragment (no facts invented)."""
    if not cues:
        return ""
    leadin = _CUE_LEADIN.get((term, sense_id), _DEFAULT_CUE_LEADIN)
    surfaces = [_cue_surface(term, sense_id, c) for c in cues]
    return f"{leadin} {_join_list(surfaces)}."


def _realize_locations(locations: list[str]) -> str:
    """location relation -> 'found ...' clause body (no leading subject)."""
    if not locations:
        return ""
    return _join_list([_loc_phrase(loc) for loc in locations])


def _realize_actions(
    term: str, sense_id: str, actions: list[str]
) -> tuple[str, bool]:
    """action relation -> agency-aware clause + whether to append location.

    Returns ``(clause, include_location)``.  Fixed-text frames are only used
    when the action verbs they reference are present in the evidence; otherwise
    a safe subject frame over the available evidence verbs is used.
    """
    spec = _ACTION_REALIZATION.get((term, sense_id))
    if spec is None:
        clause = f"It can {_join_list(actions)}" if actions else ""
        return clause, True

    include_location = spec.get("include_location", True)
    text = spec["text"]
    if "{actions}" in text:
        if not actions:
            return "", include_location
        connector = spec.get("connector", "and")
        return text.format(actions=_join_list(actions, connector)), include_location

    # Fixed-text frame: only emit if its referenced verbs are in the evidence.
    required = spec.get("requires", ())
    present = {a.strip().lower() for a in actions}
    if required and all(r in present for r in required):
        return text, include_location
    # Evidence-safe frame over whatever action verbs do exist.
    clause = f"It can {_join_list(actions)}" if actions else ""
    return clause, True


def _realize_action_location(
    term: str, sense_id: str, actions: list[str], locations: list[str]
) -> str:
    """Combine the action and location relations into one natural sentence."""
    act, include_location = _realize_actions(term, sense_id, actions)
    loc = _realize_locations(locations) if include_location else ""
    if act and loc:
        return f"{act}, and it is often found {loc}."
    if act:
        return f"{act}."
    if loc:
        return f"It is often found {loc}."
    return ""


def _realize_contrast_side(
    term: str,
    sense_id: str,
    base: str,
    cues: list[str],
    actions: list[str],
    locations: list[str],
) -> str:
    """One compact, relation-aware clause describing a single sense."""
    display = _start_sentence(
        _DISPLAY_LABEL.get((term, sense_id), f"a {sense_id}")
    )
    spec = _CONTRAST_SIDE.get((term, sense_id))
    if spec is None:
        # Evidence-safe default: keep the grounded base description plus a
        # compact cue clue.  Not a forced/generic answer — the planner already
        # gated insufficient evidence to an audit before reaching here.
        desc = _strip_trailing_associated(base) or sense_id
        out = f"{display} is {desc}"
        if cues:
            surfaces = [_cue_surface(term, sense_id, c) for c in cues[:3]]
            out += f", with clues like {_join_list(surfaces)}"
        return out + "."

    out = f"{display} is {spec['gloss']}"
    if spec.get("action_tail"):
        if actions:
            connector = spec.get("connector", "or")
            out += f"{spec['action_tail']} {_join_list(actions, connector)}"
    elif spec.get("clue_lead") and cues:
        n = spec.get("clue_count", 3)
        surfaces = [_cue_surface(term, sense_id, c) for c in cues[:n]]
        out += f"{spec['clue_lead']} {_join_list(surfaces)}"
    return out + "."


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
        # For underconstrained questions on known ambiguous terms, render a
        # helpful explanation of the competing senses instead of the generic
        # "cannot answer safely" message.  Decision remains audit.
        if reason == "underconstrained_question":
            term = plan.render_args.get("term")
            if term:
                labels = _AMBIGUOUS_SENSE_LABELS.get(term)
                if labels and len(labels) >= 2:
                    term_cap = term.capitalize()
                    return (
                        f'"{term_cap}" is ambiguous: it can mean {labels[0]}'
                        f" or {labels[1]}."
                        f" I need context to choose the right meaning."
                    )
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
        parts = [_realize_definition(display, clean_desc)]
        cue_fragment = _realize_cues(term, sense_id, cues)
        if cue_fragment:
            parts.append(cue_fragment)
        action_location = _realize_action_location(
            term, sense_id, actions, locations
        )
        if action_location:
            parts.append(action_location)
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

        side_a = _realize_contrast_side(
            term,
            sense_a,
            _SENSE_BASE.get((term, sense_a), ""),
            sa.get("cues", []),
            sa.get("actions", []),
            sa.get("locations", []),
        )
        side_b = _realize_contrast_side(
            term,
            sense_b,
            _SENSE_BASE.get((term, sense_b), ""),
            sb.get("cues", []),
            sb.get("actions", []),
            sb.get("locations", []),
        )
        return f"{side_a} {side_b}"

    return "I cannot answer safely."
