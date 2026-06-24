"""Small deterministic answer-style normalizer for Assistant Surface.

This does not add facts or change planner behavior. It only strips explicit
style instructions from the user's question and returns a rendering style hint.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class AnswerStyleResolution:
    original_question: str
    question: str
    answer_style: str = "normal"

    @property
    def changed(self) -> bool:
        return self.original_question != self.question or self.answer_style != "normal"


_STYLE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("brief", r"(?:briefly|short(?:ly)?|in\s+short|коротко|кратко|в\s+двух\s+словах)"),
    ("detailed", r"(?:in\s+detail|more\s+detail|подробнее|подробно|развернуто|развёрнуто)"),
    ("simple", r"(?:simply|in\s+simple\s+terms|explain\s+like\s+i'?m\s+five|простыми\s+словами|как\s+реб[её]нку)"),
    ("important", r"(?:what'?s\s+important|most\s+important|самое\s+важное|главное)"),
)

_PREFIX_RE = re.compile(
    r"^\s*(?P<style>{styles})\s*(?:[:,\-—]\s*)?(?P<rest>.+?)\s*$".format(
        styles="|".join(f"(?:{pattern})" for _style, pattern in _STYLE_PREFIXES)
    ),
    re.IGNORECASE,
)

_RU_ABOUT_RE = re.compile(r"^(?:про|о|об)\s+(.+)$", re.IGNORECASE)
_EN_ABOUT_RE = re.compile(r"^about\s+(.+)$", re.IGNORECASE)


def resolve_answer_style(question: str) -> AnswerStyleResolution:
    text = (question or "").strip()
    if not text:
        return AnswerStyleResolution(question, question)

    match = _PREFIX_RE.match(text)
    if not match:
        return AnswerStyleResolution(question, question)

    style_text = match.group("style")
    rest = match.group("rest").strip()
    style = _style_for_prefix(style_text)
    cleaned = _normalize_rest(rest)
    return AnswerStyleResolution(question, cleaned, style)


def _style_for_prefix(style_text: str) -> str:
    for style, pattern in _STYLE_PREFIXES:
        if re.fullmatch(pattern, style_text.strip(), re.IGNORECASE):
            return style
    return "normal"


def _normalize_rest(rest: str) -> str:
    if _looks_like_question(rest):
        return rest

    about = _RU_ABOUT_RE.match(rest) or _EN_ABOUT_RE.match(rest)
    if about:
        subject = about.group(1).strip().rstrip("?.!")
        return f"Tell me about {subject}."

    if len(rest.split()) <= 5:
        subject = rest.strip().rstrip("?.!")
        return f"Tell me about {subject}."

    return rest


def _looks_like_question(text: str) -> bool:
    return bool(
        re.match(
            r"^\s*(?:who|what|where|when|why|how|is|are|was|were|does|do|did|tell|describe|summari[sz]e)\b",
            text,
            re.IGNORECASE,
        )
    )
