"""Deterministic answer-sentence extraction from a retrieved article/snippet.

Given a natural-language question and a block of text (a Wikipedia extract or a
search snippet), pick the single sentence most likely to contain the specific
fact asked. Rule-based and stdlib-only, matching the project's no-ML ethos:

- map the question's relation trigger words to a set of answer keywords
  (e.g. "married/marry/wife" -> {married, wife, husband, spouse});
- score each sentence by relation-keyword hits (strong) plus overlap with the
  question's content tokens (weak), with a small earlier-sentence bonus;
- return the best sentence when it clears a threshold, else None so the caller
  can fall back to the intro.

For a bare "who/what is X" definition question (no relation cue), the first
substantive sentence — the intro definition — is returned.
"""

from __future__ import annotations

import re

# Relation trigger words in the question -> answer keywords to seek in text.
# Order matters only for readability; all matching groups contribute.
_RELATION_CUES: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (frozenset({"president", "minister", "mayor", "governor", "officeholder", "incumbent"}),
     frozenset({"officeholder", "incumbent", "serving", "current"})),
    # Answer side deliberately omits "wife"/"husband": they collide with title
    # phrases like "The Good Wife". "married"/"spouse" are far more reliable.
    (frozenset({"married", "marry", "wife", "husband", "spouse", "wed", "wedding"}),
     frozenset({"married", "spouse", "wed"})),
    (frozenset({"born", "birthplace", "birth", "raised", "from"}),
     frozenset({"born", "raised"})),
    (frozenset({"die", "died", "death", "killed", "dead"}),
     frozenset({"died", "death", "killed", "assassinated", "dead"})),
    (frozenset({"discover", "discovered", "invent", "invented", "discovery"}),
     frozenset({"discovered", "invented", "developed", "discovery", "known"})),
    (frozenset({"school", "study", "studied", "educated", "education", "college",
                "university", "graduate", "graduated", "attend", "attended"}),
     frozenset({"educated", "graduated", "attended", "university", "college",
                "school", "studied", "alma"})),
    (frozenset({"play", "plays", "played", "position", "team"}),
     frozenset({"played", "plays", "position", "quarterback", "guard", "forward",
                "signed"})),
    (frozenset({"currency"}), frozenset({"currency"})),
    (frozenset({"language", "speak", "spoken", "languages"}),
     frozenset({"language", "spoken", "languages"})),
    (frozenset({"capital"}), frozenset({"capital"})),
    (frozenset({"wrote", "write", "written", "book", "books", "author", "novel"}),
     frozenset({"wrote", "author", "book", "novel", "published", "writer"})),
    (frozenset({"founded", "founder", "found", "created", "create"}),
     frozenset({"founded", "co-founded", "founder", "established", "created"})),
    (frozenset({"government"}),
     frozenset({"republic", "monarchy", "democracy", "parliamentary", "federal",
                "government"})),
    # "How does X work?" / "What is the mechanism of X?" — favor sentences
    # that describe the actual operating mechanism over the generic intro
    # sentence (which is what a no-cue definitional match would return).
    (frozenset({"work", "works", "working", "function", "functions", "functioning",
                "mechanism", "operate", "operates", "operation"}),
     frozenset({"via", "through", "consists", "consist", "consisting", "comprises",
                "comprising", "comprised", "works", "operates", "operation", "system",
                "process", "technology", "designed", "allows", "enabling", "enable",
                "communicate", "communicates", "connecting", "connects", "connected",
                "network", "satellites", "orbit", "ground", "terminal", "terminals",
                "signal", "signals", "transmit", "transmits", "infrastructure",
                "constellation", "powered", "generates", "converts"})),
)

_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "and", "or", "to", "for", "with",
    "from", "is", "are", "was", "were", "did", "does", "do", "who", "what",
    "when", "where", "which", "whom", "whose", "why", "how", "his", "her",
    "he", "she", "it", "they", "that", "this", "as", "by", "be", "been",
    "have", "has", "had", "will", "would", "before", "after", "into",
})

_WORD_RE = re.compile(r"[a-z0-9]+")
# Split on sentence boundaries and newlines (Wikipedia extracts put section
# headers on their own lines).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|\n+")
# A sentence LED by a subject pronoun refers to the article's main subject, so
# it counts as naming the subject even without a proper noun. Possessives
# (His/Her/Their) are excluded: "His teammate Ryan Hite ..." is about Ryan, not
# the subject.
_SUBJECT_PRONOUN_RE = re.compile(r"^(?:He|She|It|They)\b")
# Wikipedia extract section headers ("== History ==", "=== Background ===").
# These mark where the lead paragraph ends — everything before the first one
# is reliably about the article's subject by encyclopedia convention, no
# matter how many sentences that takes (a well-developed topic like a country
# can have a 10+ sentence lead; a short biography's lead may be 3 sentences).
_SECTION_HEADER_RE = re.compile(r"^==+\s*[^=\n]+?\s*==+\s*$", re.MULTILINE)


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _content_tokens(question: str) -> set[str]:
    return {t for t in _tokens(question) if t not in _STOPWORDS and len(t) > 2}


def _relation_keywords(question: str) -> frozenset[str]:
    qtokens = set(_tokens(question))
    keywords: set[str] = set()
    for triggers, answer_kws in _RELATION_CUES:
        if qtokens & triggers:
            keywords |= answer_kws
    return frozenset(keywords)


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(text or "")
    return [p.strip() for p in parts if p and len(p.strip()) > 15]


_DEFAULT_LEAD_SENTENCE_COUNT = 6


def _lead_sentence_count(text: str) -> int:
    """How many sentences make up the lead paragraph(s), by real structure.

    Finds the first section header and counts sentences before it, instead
    of assuming a fixed number — a long, well-developed lead (a country, a
    company) can run to 10+ sentences and is still entirely on-subject; a
    short one (a minor topic) might be 2-3. When no header is found (a short
    snippet with no visible structure, e.g. from DuckDuckGo, or a synthetic
    text in a test), falls back to the previous fixed guess rather than
    trusting the entire text — an unstructured blob gives no real evidence
    that everything in it is still about the same subject.
    """

    header = _SECTION_HEADER_RE.search(text or "")
    if header is None:
        return _DEFAULT_LEAD_SENTENCE_COUNT
    return len(_split_sentences(text[: header.start()]))


def extract_answer(
    question: str,
    text: str,
    *,
    subject: str | None = None,
    max_len: int = 320,
) -> str | None:
    """Return the sentence in ``text`` best answering ``question``, or None.

    When the question carries a recognizable relation cue, a sentence must hit
    at least one relation keyword to qualify. For a plain definition question
    (no cue), the first substantive sentence (the intro) is returned.

    ``subject`` (the resolved article title) lets sentences that name the
    subject — or lead with a pronoun referring to it — count as on-subject, so
    "She was born in Bridgetown" is not mistaken for an off-topic sentence.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return None

    content = _content_tokens(question)
    relation_kws = _relation_keywords(question)

    if not relation_kws:
        # Definitional "who/what is X" — the intro sentence is the answer.
        return _trim(sentences[0], max_len)

    # Tokens that actually name the subject: question content tokens (minus the
    # relation triggers) plus the resolved title's tokens.
    entity_tokens = {t for t in content if t not in relation_kws}
    if subject:
        entity_tokens |= {t for t in _tokens(subject) if len(t) > 2}

    lead_sentence_count = _lead_sentence_count(text)
    best_sentence: str | None = None
    best_score = 0.0
    for idx, sentence in enumerate(sentences):
        stoks = set(_tokens(sentence))
        relation_hits = len(stoks & relation_kws)
        if relation_hits == 0:
            continue
        entity_hits = len(stoks & entity_tokens)
        subject_ref = entity_hits > 0 or bool(_SUBJECT_PRONOUN_RE.match(sentence))
        # A relation keyword deep in the article with NO tie to the subject is
        # almost always about someone/something else ("The Good Wife", a
        # teammate's college). Require the sentence to name the subject, lead
        # with a subject pronoun, or sit within the article's actual lead
        # section (see ``_lead_sentence_count`` — not a fixed sentence count).
        if not subject_ref and idx >= lead_sentence_count:
            continue
        # A mild intro nudge, kept small so a strongly-relational sentence
        # deeper in the article (e.g. birthplace vs birth-date) can still win.
        early_bonus = max(0.0, 2.0 - idx * 0.5)
        score = entity_hits * 5 + relation_hits * 3 + (2 if subject_ref else 0) + early_bonus
        if score > best_score:
            best_score = score
            best_sentence = sentence

    if best_sentence is None:
        return None
    return _trim(best_sentence, max_len)


def _trim(sentence: str, max_len: int) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    if len(sentence) <= max_len:
        return sentence
    return sentence[:max_len].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
