import re
from typing import Optional
from .events import Event

# Maps each known verb form to a canonical relation family.
# Extend this dict to support more languages or verb forms.
VERB_TO_RELATION: dict[str, str] = {
    # produces_result
    "дала": "produces_result",
    "дал": "produces_result",
    "дает": "produces_result",
    "даёт": "produces_result",
    "снесла": "produces_result",
    "произвел": "produces_result",
    "произвёл": "produces_result",
    "производит": "produces_result",
    "написал": "produces_result",
    "получила": "produces_result",
    "получил": "produces_result",
    # causes_effect
    "вызвал": "causes_effect",
    # contains
    "содержит": "contains",
    # grows_into
    "вырастает": "grows_into",
    # requires
    "требует": "requires",
    # prevents
    "уничтожает": "prevents",
    "мешает": "prevents",
    # reduces
    "уменьшает": "reduces",
    "сокращает": "reduces",
    # enables
    "помогает": "enables",
    "усиливает": "enables",
}

# Sorted longest-first so greedy matching picks the right verb
_VERBS_PATTERN = "|".join(
    re.escape(v) for v in sorted(VERB_TO_RELATION, key=len, reverse=True)
)
# Optional Russian preposition between verb and object (e.g. "вырастает в яблоню")
_PREP = r"(?:(?:в|на|из|к|у|со?|до|за|при|под|над|об?|по|про|через)\s+)?"
_EVENT_RE = re.compile(
    rf"^(?P<subject>\S+)\s+(?P<verb>{_VERBS_PATTERN})\s+{_PREP}(?P<object>\S+)$",
    re.UNICODE,
)


class TinyParser:
    """Rule-based parser for simple Russian SVO sentences."""

    def parse(self, text: str) -> Optional[Event]:
        cleaned = text.strip().rstrip(".")
        m = _EVENT_RE.match(cleaned)
        if not m:
            return None
        return Event(
            subject=m.group("subject"),
            verb=m.group("verb"),
            object=m.group("object"),
            raw_text=text,
        )

    def relation_type(self, verb: str) -> Optional[str]:
        return VERB_TO_RELATION.get(verb)
