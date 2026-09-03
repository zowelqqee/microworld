"""Deterministic shape recognition for statutory questions.

The analyzer keys on *interrogative shape* only — the grammatical frame of the
question — never on legal vocabulary. It contains no list of statutes, offences,
doctrines, or legal terms, so it behaves identically on a chapter it has never
seen. What the question is *about* is left entirely to content-based retrieval
in the planner.

Shapes recognized:
  definition      "what is X", "what does X mean", "how is X defined"
  penalty         "what is the penalty for X", "how is X punished"
  cross_reference "which section does X cite", "what does X refer to"
  scope           "what does X apply to", "who does X govern"
  conditional     "is X ...", "can X ...", "does X ...", "when is X ..."
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AnalyzedLegalQuestion:
    question: str
    shape: str          # one of the shapes above, or "unknown"
    focus: str          # the content phrase the question is about
    expects_yes_no: bool


# Interrogative frames. Each is a *grammatical* pattern; the capture group is
# the content the question is about, which the analyzer never interprets.
_SHAPES: tuple[tuple[str, str], ...] = (
    ("penalty", r"^(?:what\s+(?:is|are)\s+the\s+)?penal(?:ty|ties)\s+(?:for|under)\s+(?P<f>.+)$"),
    ("penalty", r"^what\s+(?:is|are)\s+the\s+penal(?:ty|ties)\s+(?P<f>.+)$"),
    ("penalty", r"^what\s+penalty\s+(?:applies\s+to|attaches\s+to|for)\s+(?P<f>.+)$"),
    ("penalty", r"^how\s+is\s+(?P<f>.+?)\s+punish(?:ed|able)\??$"),
    ("cross_reference", r"^which\s+(?:section|subsection|paragraph|provision)s?\s+(?P<f>.+)$"),
    ("cross_reference", r"^what\s+(?:section|subsection|paragraph|provision)s?\s+(?P<f>.+)$"),
    ("scope", r"^(?:what|who|whom)\s+does\s+(?P<f>.+?)\s+(?:apply\s+to|govern|cover)\??$"),
    ("scope", r"^what\s+is\s+the\s+scope\s+of\s+(?P<f>.+)$"),
    ("definition", r"^what\s+does\s+(?P<f>.+?)\s+(?:mean|include)\??$"),
    ("definition", r"^how\s+is\s+(?P<f>.+?)\s+defined\??$"),
    ("definition", r"^what\s+(?:is|are)\s+(?:a|an|the)?\s*(?P<f>.+)$"),
)

# Yes/no and rule-application frames — these ask whether a rule *holds*, which
# is the shape whose answer is worthless without its conditions and exceptions.
_CONDITIONAL_LEAD = re.compile(
    r"^(?:is|are|was|were|does|do|did|can|could|may|might|must|shall|will|would)\b",
    re.IGNORECASE,
)
_CONDITIONAL_WH = re.compile(
    r"^(?:when|under\s+what|in\s+what\s+(?:case|circumstance)s?)\b", re.IGNORECASE
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).rstrip("?").strip()


def analyze(question: str) -> AnalyzedLegalQuestion:
    """Parse one statutory question into a shape plus its content focus."""

    q = _clean(question)
    if not q:
        return AnalyzedLegalQuestion(question, "unknown", "", False)
    low = q.lower()

    # A yes/no or "when/under what" frame is a rule-application question, and it
    # takes precedence: its answer is unsafe without guards, so it must never be
    # mistaken for a bare lookup.
    if _CONDITIONAL_LEAD.match(low) or _CONDITIONAL_WH.match(low):
        focus = re.sub(
            r"^(?:is|are|was|were|does|do|did|can|could|may|might|must|shall|will|would|"
            r"when|under\s+what|in\s+what\s+(?:case|circumstance)s?)\s+",
            "", low,
        )
        return AnalyzedLegalQuestion(
            question, "conditional", _clean(focus), bool(_CONDITIONAL_LEAD.match(low))
        )

    for shape, pattern in _SHAPES:
        m = re.match(pattern, low, re.IGNORECASE)
        if m:
            return AnalyzedLegalQuestion(question, shape, _clean(m.group("f")), False)

    return AnalyzedLegalQuestion(question, "unknown", q, False)
