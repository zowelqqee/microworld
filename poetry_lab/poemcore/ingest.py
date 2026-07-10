"""Corpus ingestion pipeline — poetry replaces the wiki/Reddit knowledge source.

This is the module the experiment swaps out. In production, ingestion reads
Wikipedia/Reddit and emits fact overlays (subject–predicate–object rows) into
JSON artifacts. Here it reads the poetry corpus and emits the *same shape* of
artifacts — a concept graph, a phrase model, rhyme groups, and per-author style
profiles — into ``artifacts/``. Every downstream layer reads only these JSON
files, so the layer boundary is preserved exactly: reasoning and language never
touch the raw corpus.

What is emitted is statistical structure (co-occurrence, transitions, rhyme
clusters, frequency), never memorised whole lines. The novelty guard
(``novelty.py``) later enforces that generation cannot echo a corpus 4-gram.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from poemcore.concept_graph import ConceptGraph
from poemcore.entity_types import (
    LOCATIVE_PREPOSITIONS,
    SPEECH_VERB_STEMS,
    EntityEvidence,
    build_entity_types,
)
from poemcore.morphology import GenderEvidence, build_gender_map, is_finite_verb, is_infinitive, past_gender
from poemcore.phrase_model import PhraseModel
from poemcore.text import STOPWORDS, content_words, line_syllables, rhyme_key, sentences, words

# Any preposition, not just locative ones: a token following any of them is a
# noun phrase head, which is exactly the signal morphology needs to reject
# nouns that look like masculine past verbs ("стол", "угол", "пол").
_MORPH_PREPOSITIONS = frozenset({
    "в", "во", "на", "за", "под", "из", "к", "ко", "по", "у", "с", "со",
    "от", "до", "над", "при", "про", "о", "об", "обо", "для", "без", "через",
})
# A token must follow a preposition at least this often before we call it a
# noun; a single stray hit should not silence a real verb.
_NOUN_LIKE_MIN = 2

_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = _ROOT / "corpus"
ARTIFACT_DIR = _ROOT / "artifacts"

# Adjective surface endings — a light morphological heuristic (no POS model),
# used only to record noun<-epithet pairs. Mis-tags are harmless: an epithet
# the generator never looks up is simply inert, same tolerance the production
# phrase graph has for an unused learned fragment.
_ADJ_END = re.compile(r"(ый|ий|ой|ая|яя|ое|ее|ые|ие|ым|им|ом|ей|ою|ую|юю)$")

# case-preserving word tokenizer, for proper-noun detection during ingest
_CASE_WORD_RE = re.compile(r"[А-ЯЁа-яё]+(?:-[А-ЯЁа-яё]+)?")


def _looks_adjective(word: str) -> bool:
    if bool(_ADJ_END.search(word)) and len(word) >= 5:
        return True
    # Short-form adjectives have no long adjective ending ("невидима",
    # "любима"), but they are still not noun heads or predicates.  This is
    # an ending family learned by morphology, not a word allow/deny list.
    return len(word) >= 6 and word.endswith(("има", "ыма", "ема", "ома", "ена", "ана", "ита", "ута", "ята"))


def _nominative_attribute_agrees(adjective: str, noun: str) -> bool:
    """Keep only observed adjective+noun pairs that can form a subject phrase."""

    if noun.endswith(("а", "я", "ь")):
        return adjective.endswith(("ая", "яя"))
    if noun.endswith(("о", "е")):
        return adjective.endswith(("ое", "ее"))
    return adjective.endswith(("ый", "ий", "ой"))


def _past_predicate_agrees(subject: str, predicate: str) -> bool:
    """A conservative agreement check for a noun-led descriptive fact."""

    gender = past_gender(predicate)
    if not gender:
        return False
    if subject.endswith(("а", "я", "ь")):
        return gender == "f"
    if subject.endswith(("о", "е")):
        return gender == "n"
    return gender == "m"


def _attribute_agrees_subject(attribute: str, subject: str) -> bool:
    """Reject comparative/adverbial tails that cannot modify the subject."""

    if subject.endswith(("а", "я", "ь")):
        return attribute.endswith(("ая", "яя", "ой", "ей", "ою", "ею"))
    if subject.endswith(("о", "е")):
        return attribute.endswith(("ое", "ее", "ым", "им"))
    return attribute.endswith(("ый", "ий", "ой", "ым", "им", "ом", "ем"))


# Clause boundaries inside one prose sentence. ``words`` drops punctuation, so
# any adjacency-based fact scanned over a whole sentence would fabricate a
# relation across a comma ("за город, в рощу" -> a false link on "город").
# The hyphen is deliberately absent: it joins tokens ("кое-как"), it does not
# separate clauses.
_CLAUSE_BREAK_RE = re.compile(r"[,;:()«»„“\"—–…]")


# Fallback when a corpus file has no "# author:" header (e.g. a raw collected
# edition dropped in directly) — keeps the style-profile key consistent with
# the Cyrillic name the engine's author-alias table expects.
_STEM_AUTHORS = {
    "pushkin": "Пушкин", "lermontov": "Лермонтов", "tyutchev": "Тютчев",
    "fet": "Фет", "blok": "Блок", "akhmatova": "Ахматова",
}


def _author_of(path: Path, lines: list[str]) -> str:
    for line in lines[:3]:
        m = re.match(r"#\s*author:\s*(.+)", line.strip())
        if m:
            return m.group(1).strip()
    return _STEM_AUTHORS.get(path.stem.lower(), path.stem)


# Structural noise found in raw scanned/collected editions: ALL-CAPS titles,
# standalone page numbers or years, roman-numeral section markers, and
# asterisk dividers. None of these are verse lines, so learning transitions or
# rhymes from them would pollute the phrase model with title fragments and
# page numbers rather than poetic language.
_NOISE_RE = re.compile(
    r"^(?:"
    r"[0-9]+"                      # page number / bare year
    r"|[IVXLCDM]+"                  # roman-numeral section marker
    r"|[*\-\s]+"                    # divider (***, ---, ...)
    r"|[А-ЯЁ0-9 .,:;!?\"'\-]+"       # ALL-CAPS title line
    r")$"
)


_DIVIDER_RE = re.compile(r"^[*\-\s]+$")


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _DIVIDER_RE.match(stripped):
        return True
    if not _NOISE_RE.match(stripped):
        return False
    # an all-caps/digit match must actually contain a letter or digit (not just
    # punctuation) and must contain no lowercase Cyrillic letter, else it is a
    # real verse line that happens to satisfy the branch trivially
    return any(ch.isalpha() or ch.isdigit() for ch in stripped) and not any(
        "а" <= ch <= "я" or ch == "ё" for ch in stripped
    )


def _stanzas(body_lines: list[str]) -> list[list[str]]:
    stanzas, current = [], []
    for line in body_lines:
        if line.strip():
            current.append(line.strip())
        elif current:
            stanzas.append(current)
            current = []
    if current:
        stanzas.append(current)
    return stanzas


def build_artifacts() -> dict:
    phrase = PhraseModel()
    graph = ConceptGraph()
    rhyme_groups: dict[str, Counter] = defaultdict(Counter)
    style: dict[str, dict] = {}
    node_freq: Counter = Counter()
    # Proper-noun detection (no NER model): a token seen Capitalized *mid-line*
    # is a name (Астрахань, Стамати, Татищев); at line start capitalization is
    # ambiguous, so those occurrences are ignored. A token is kept as a proper
    # name only if its mid-line-capitalized count dominates its lowercase count,
    # which filters ordinary words that merely start a sentence mid-line.
    cap_mid: Counter = Counter()
    lower_any: Counter = Counter()

    corpus_files = sorted(CORPUS_DIR.glob("*.txt"))
    total_lines = 0

    for path in corpus_files:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
        author = _author_of(path, raw_lines)
        body = [
            ln for ln in raw_lines
            if not ln.strip().startswith("#") and not _is_noise_line(ln)
        ]

        opener_counter: Counter = Counter()
        syllable_hist: Counter = Counter()
        vocab: Counter = Counter()

        for line in body:
            if not line.strip():
                continue
            total_lines += 1
            phrase.learn_line(line)
            toks = words(line)
            if toks:
                opener_counter[toks[0]] += 1
                syllable_hist[line_syllables(line)] += 1
                rw = toks[-1]
                rhyme_groups[rhyme_key(rw)][rw] += 1
            # case-preserving pass for proper-noun detection
            case_toks = _CASE_WORD_RE.findall(line)
            for pos, ct in enumerate(case_toks):
                low = ct.lower()
                if pos > 0 and ct[:1].isupper():
                    cap_mid[low] += 1
                elif ct[:1].islower():
                    lower_any[low] += 1
            cwords = content_words(line)
            for w in cwords:
                node_freq[w] += 1
                vocab[w] += 1
            # co-occurrence edges within a line (imagery that travels together)
            for i, a in enumerate(cwords):
                for b in cwords[i + 1 : i + 5]:
                    graph.add_edge(a, b, 1.0)
            # epithet capture: adjective immediately before a content word
            for a, b in zip(toks, toks[1:]):
                if _looks_adjective(a) and b in cwords and not _looks_adjective(b):
                    graph.epithet[b][a] = graph.epithet[b].get(a, 0) + 1

        # Broader stanza-level association (looser thematic co-occurrence).
        # Bounded to a sliding window rather than full pairwise: a raw
        # collected edition may lack blank lines for pages at a time (long
        # narrative poems like "Демон"/"Мцыри"), so an unbounded stanza can
        # hold hundreds of unique words — full pairwise there is O(n^2) and
        # exploded to millions of edges on the full-works corpus. The window
        # keeps "loose thematic co-occurrence" for the common short stanza
        # (window far exceeds a normal stanza's word count, so behaviour is
        # unchanged there) while capping the pathological long ones to O(n).
        _STANZA_WINDOW = 20
        for stanza in _stanzas(body):
            stanza_words = []
            for line in stanza:
                stanza_words.extend(content_words(line))
            uniq = list(dict.fromkeys(stanza_words))
            for i, a in enumerate(uniq):
                for b in uniq[i + 1 : i + 1 + _STANZA_WINDOW]:
                    graph.add_edge(a, b, 0.25)

        style[author] = {
            "avg_syllables": _weighted_avg(syllable_hist),
            "common_syllables": [s for s, _ in syllable_hist.most_common(3)],
            "top_openers": [w for w, _ in opener_counter.most_common(12)],
            "signature_vocab": _signature(vocab, node_freq),
            "line_count": sum(syllable_hist.values()),
        }

    graph.weight = {w: float(c) for w, c in node_freq.items()}

    proper_names = sorted(
        w for w, mid in cap_mid.items()
        if mid >= 2 and mid > lower_any.get(w, 0)
    )

    artifact = {
        "meta": {
            "authors": sorted(style),
            "total_lines": total_lines,
            "concept_nodes": len(graph.weight),
            "rhyme_groups": len(rhyme_groups),
        },
        "phrase_model": phrase.to_dict(),
        "concept_graph": graph.to_dict(),
        "rhyme_groups": {k: dict(c) for k, c in rhyme_groups.items() if sum(c.values()) >= 2},
        "style_profiles": style,
        "proper_names": proper_names,
    }
    return artifact


def _weighted_avg(hist: Counter) -> float:
    total = sum(hist.values())
    if not total:
        return 0.0
    return round(sum(s * n for s, n in hist.items()) / total, 2)


def _signature(vocab: Counter, global_freq: Counter, k: int = 25) -> list[str]:
    """Author-characteristic words: high local frequency relative to global
    frequency (a crude tf/df). This is what 'in the style of X' biases toward."""

    scored = []
    for w, local in vocab.items():
        if local < 1 or len(w) < 3:
            continue
        score = local / (1 + global_freq.get(w, 0) - local + 1)
        scored.append((score * local, w))
    scored.sort(reverse=True)
    return [w for _s, w in scored[:k]]


def write_artifacts(target: Path | None = None) -> Path:
    artifact = build_artifacts()
    ARTIFACT_DIR.mkdir(exist_ok=True)
    out = target or (ARTIFACT_DIR / "poetry_model.json")
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# -- narrative corpus -------------------------------------------------------

_NARRATIVE_SOURCE = CORPUS_DIR / "мастер и маргарита.txt"
_DIALOGUE_RE = re.compile(r"^\s*[–—]")
_NARRATIVE_VERB = re.compile(
    r"(?:ться|тся|[аяеои]л(?:а|о|и)?|лся|лась|лось|лись|"
    r"ают|яют|уют|ат|ят|ает|яет|ует|ит|ёт|ет|"
    r"аешь|яешь|уешь|ишь|ешь|аете|яете|уете|ите|"
    r"аю|яю|ую|[аяеёиоуыэюя]ть)$"
)
_NARRATIVE_OBJECT_NOISE = frozenset({
    "этого", "него", "того", "том", "очень", "ним", "даже", "совершенно", "ничего",
    "здесь", "сейчас", "вдруг", "почему", "который", "какой", "тогда", "потом", "ну", "перед",
})
_DESCRIPTION_ENTITY_NOISE = frozenset({
    "него", "нее", "них", "ним", "ними", "ему", "ей", "ими", "себя",
    "нему", "ней", "нею", "нем", "нём", "том", "этом", "себе", "собой",
    "мне", "нам", "вам", "тебе", "тех", "той",
    "каждый", "каждая", "каждое", "всякий", "всякая", "всякое",
    "этот", "эта", "это", "тот", "та", "то", "такой", "такая", "такое",
})


def _narrative_paths(source: Path | None = None) -> tuple[Path, ...]:
    """Resolve one file or a directory of uploaded prose files."""
    if source is None:
        if _NARRATIVE_SOURCE.exists():
            return (_NARRATIVE_SOURCE,)
        # The original single-work fixture may be replaced by uploaded works.
        # Select prose-like files automatically; verse remains available via
        # an explicit ``--source`` directory when a caller wants it.
        candidates = tuple(sorted(CORPUS_DIR.glob("*.txt")))
        prose = []
        for path in candidates:
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            average = sum(len(words(line)) for line in lines) / max(1, len(lines))
            if average >= 8.0:
                prose.append(path)
        return tuple(prose or candidates)
    if source.is_dir():
        paths = tuple(sorted(source.glob("*.txt")))
        return paths or (source,)
    return (source,)


def build_narrative_artifacts(source: Path | None = None) -> dict:
    """Build the same core artifacts from prose sentences, not verse lines.

    The graph, phrase model, proper-name detector, and novelty support remain
    the same objects.  Only corpus segmentation and prose-oriented metadata
    change, which makes the transfer boundary explicit and inspectable.
    """

    paths = _narrative_paths(source)
    raw = "\n\n".join(path.read_text(encoding="utf-8") for path in paths)
    prose_sentences = sentences(raw)
    phrase = PhraseModel()
    graph = ConceptGraph()
    node_freq: Counter = Counter()
    cap_any: Counter = Counter()
    cap_mid: Counter = Counter()
    lower_any: Counter = Counter()
    location_freq: Counter = Counter()
    object_freq: Counter = Counter()
    # Corpus evidence for the ported entity-type classifier (see entity_types.py).
    evidence = EntityEvidence()
    # Corpus evidence for the morphology layer (see morphology.py):
    #  * gender_evidence — which past-tense gender each name governs;
    #  * after_preposition — tokens that follow в/на/под/..., i.e. nouns, which
    #    is what separates "стол" from the identically-ending verb "сказал".
    gender_evidence = GenderEvidence()
    after_preposition: Counter = Counter()
    object_evidence: Counter = Counter()
    standalone_evidence: Counter = Counter()
    dative_evidence: Counter = Counter()
    transitive_evidence: Counter = Counter()
    terminal_evidence: Counter = Counter()
    frame_counts: Counter = Counter()
    frame_subjects: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    descriptive_subject_evidence: Counter = Counter()
    prose_tokens: list[list[str]] = []
    sentence_lengths: list[int] = []
    dialogue_count = description_count = narration_count = 0
    opener_counter: Counter = Counter()

    for sentence in prose_sentences:
        phrase.learn_line(sentence)
        toks = words(sentence)
        if not toks:
            continue
        prose_tokens.append(toks)
        if is_finite_verb(toks[-1]):
            terminal_evidence[toks[-1]] += 1
        opener_counter[toks[0]] += 1
        sentence_lengths.append(len(toks))
        dialogue = bool(_DIALOGUE_RE.match(sentence))
        dialogue_count += int(dialogue)
        adjective_count = sum(_looks_adjective(token) for token in toks)
        if not dialogue and adjective_count >= max(1, len(toks) // 6):
            description_count += 1
        elif not dialogue:
            narration_count += 1
        cwords = content_words(sentence)
        for word in cwords:
            node_freq[word] += 1
            if (
                len(word) >= 3 and word not in STOPWORDS and word not in _NARRATIVE_OBJECT_NOISE
                and not _looks_adjective(word) and not _NARRATIVE_VERB.search(word)
            ):
                object_freq[word] += 1
        for i, left in enumerate(cwords):
            for right in cwords[i + 1 : i + 5]:
                graph.add_edge(left, right, 1.0)
        for left, right in zip(toks, toks[1:]):
            if _looks_adjective(left) and right in cwords and not _looks_adjective(right):
                graph.epithet[right][left] = graph.epithet[right].get(left, 0) + 1

        case_tokens = _CASE_WORD_RE.findall(sentence)
        for pos, token in enumerate(case_tokens):
            lower = token.lower()
            if token[:1].isupper():
                cap_any[lower] += 1
                if pos:
                    cap_mid[lower] += 1
                previous = case_tokens[pos - 1].lower() if pos else ""
                following = case_tokens[pos + 1].lower() if pos + 1 < len(case_tokens) else ""
                if previous in LOCATIVE_PREPOSITIONS:
                    location_freq[lower] += 1
                    evidence.locative[lower] += 1
                # Agent evidence: the name drives a finite verb ("Маргарита
                # летела") or follows a speech verb in the inversion Russian
                # prose uses constantly ("сказал Воланд"). Patient evidence:
                # the name follows a non-speech verb, i.e. it is acted upon.
                elif following and _NARRATIVE_VERB.search(following):
                    evidence.agent[lower] += 1
                elif previous.startswith(SPEECH_VERB_STEMS):
                    evidence.agent[lower] += 1
                elif previous and _NARRATIVE_VERB.search(previous):
                    evidence.patient[lower] += 1
                # Gender agreement evidence: the past-tense verb this name
                # governs marks the name's own gender on its surface, in either
                # word order ("Маргарита сказала" / "сказала Маргарита").
                for neighbour in (following, previous):
                    gender = past_gender(neighbour)
                    if gender:
                        gender_evidence.observe(lower, gender)
                        break
            else:
                lower_any[lower] += 1

        # A token right after a preposition is a noun, never a finite verb.
        # This is what lets morphology tell "под стол" from "он сказал".
        for previous, token in zip(toks, toks[1:]):
            if previous in _MORPH_PREPOSITIONS:
                after_preposition[token] += 1
        # A repeated token after a corpus-observed finite predicate is also
        # noun evidence. This supplements the preposition signal for heads
        # such as ``бал`` that may occur without a preposition.
        for _subject, predicate, object_ in zip(toks, toks[1:], toks[2:]):
            if (
                _subject not in STOPWORDS
                and _subject not in _DESCRIPTION_ENTITY_NOISE
                and not is_finite_verb(_subject)
                and not _looks_adjective(_subject)
                and is_finite_verb(predicate)
                and past_gender(predicate) is not None
                and _past_predicate_agrees(_subject, predicate)
                and object_ not in STOPWORDS
                and not is_finite_verb(object_)
                and not is_infinitive(object_)
                and (
                    (_looks_adjective(object_) and _attribute_agrees_subject(object_, _subject))
                    or (predicate.endswith(("ся", "сь")) and object_.endswith("о"))
                )
            ):
                descriptive_subject_evidence[_subject] += 1
            if (
                is_finite_verb(predicate)
                and object_ not in STOPWORDS
                and len(object_) >= 2
                and not _looks_adjective(object_)
                and not is_finite_verb(object_)
            ):
                object_evidence[object_] += 1
                transitive_evidence[predicate] += 1
                if _subject not in STOPWORDS and not _looks_adjective(_subject):
                    frame = (predicate, object_)
                    frame_counts[frame] += 1
                    frame_subjects[frame].add(_subject)
            if (
                not is_finite_verb(predicate)
                and is_finite_verb(object_)
                and (predicate.endswith(("о", "е")) or predicate in STOPWORDS)
            ):
                standalone_evidence[object_] += 1
            if is_finite_verb(predicate) and object_.endswith(("у", "ю", "ому", "ему", "ам", "ям")):
                dative_evidence[predicate] += 1

    # Paragraph-level association is the prose analogue of stanza association:
    # a bounded window lets characters, locations, and objects stay connected
    # through nearby sentences without creating a document-wide hub.
    for paragraph in re.split(r"\n\s*\n", raw):
        paragraph_words = list(dict.fromkeys(content_words(paragraph)))
        for i, left in enumerate(paragraph_words):
            for right in paragraph_words[i + 1 : i + 21]:
                graph.add_edge(left, right, 0.25)

    graph.weight = {word: float(count) for word, count in node_freq.items()}
    characters = [
        word for word, count in cap_mid.most_common()
        if len(word) >= 3 and word not in STOPWORDS and count >= 2
        and count > lower_any.get(word, 0) and word not in location_freq
    ][:30]
    character_stems = {word[:4] for word in characters}
    locations = [
        word for word, count in location_freq.most_common()
        if len(word) >= 3 and count >= 2 and word[:4] not in character_stems
    ][:20]
    recurring_objects = [word for word, _count in object_freq.most_common(30) if word not in characters][:20]
    # Type every recognised name from the accumulated corpus evidence. This is
    # the QA knowledge layer's classifier, advisory only — an "unknown" type
    # leaves the planner exactly as it was.
    entity_types = build_entity_types(list(dict.fromkeys((*characters, *locations))), evidence)
    # Morphology layer: gender for every name plus every frequent content word,
    # and the noun evidence that keeps "стол" from being read as a past verb.
    gender_map = build_gender_map(
        list(dict.fromkeys((*characters, *locations, *recurring_objects))), gender_evidence
    )
    prep_heads = {
        word for word, count in after_preposition.items() if count >= _NOUN_LIKE_MIN
    }
    # Case forms carry the same noun head ("на балу", "с балом" -> "бал").
    # Project that repeated prepositional evidence back onto an observed
    # nominative surface instead of hand-listing ambiguous words.
    case_endings = ("иями", "ами", "ями", "ого", "ему", "ому", "ами", "ями", "ом", "ем", "ам", "ям", "ою", "ею", "у", "ю", "а", "я", "е", "и", "ы")
    prep_stems = {
        token[:-len(ending)]
        for token in prep_heads
        for ending in case_endings
        if token.endswith(ending) and len(token) - len(ending) >= 3
    } | prep_heads
    stem_heads = {
        candidate for candidate in node_freq
        if any(candidate == stem or candidate.startswith(stem) for stem in prep_stems)
        and not is_finite_verb(candidate)
    }
    raw_noun_like = stem_heads | {
        word for word, count in object_evidence.items() if count >= _NOUN_LIKE_MIN
    } | set(descriptive_subject_evidence)
    # A repeated adverb such as "тоже" can appear in the same local slot as
    # an object. Keep -о/-е words only when the preposition evidence observed
    # that exact surface as a noun head (e.g. "на солнце", "в пальто").
    noun_like = sorted(
        word for word in raw_noun_like
        if not word.endswith(("о", "е")) or word in prep_heads
    )
    noun_set = set(noun_like)
    proper_set = set(characters) | set(locations)
    description_relation_counts: Counter = Counter()

    def _description_subject_ok(subject: str, *, allow_neuter: bool = False) -> bool:
        """A nominative-looking noun head that may anchor a descriptive fact.

        ``allow_neuter`` admits -о/-е surfaces ("солнце", "окно") for the kinds
        whose predicate-agreement check ("село" is neuter) separates a neuter
        nominative from an oblique case such as "в комнате".
        """
        oblique = ("у", "ю", "и", "ы", "ом", "ем", "ами", "ями") if allow_neuter else (
            "у", "ю", "е", "и", "ы", "ом", "ем", "ами", "ями",
        )
        return (
            subject in noun_set
            and subject not in proper_set
            and subject not in _DESCRIPTION_ENTITY_NOISE
            and not is_finite_verb(subject)
            and not _NARRATIVE_VERB.search(subject)
            and not _looks_adjective(subject)
            and not subject.endswith(oblique)
        )

    def _link_object_ok(other: str) -> bool:
        """A noun head that can stand as the far end of an observed relation."""
        return (
            other not in STOPWORDS
            and other not in proper_set
            and other not in _DESCRIPTION_ENTITY_NOISE
            and len(other) >= 3
            and not is_finite_verb(other)
            and not is_infinitive(other)
            and not _looks_adjective(other)
            # Instrumental-plural adjectives and possessives ("со своими")
            # end in -ими/-ыми; noun heads there end in -ами/-ями instead.
            and not other.endswith(("ими", "ыми"))
        )

    def _place_predicate_ok(subject: str, predicate: str) -> bool:
        """An agreeing past predicate that forms a complete clause on its own.

        A transitive predicate is rejected: its direct object was elided by
        the local scan, and "человек погружал в сосуд" is not a clause.
        """
        return (
            is_finite_verb(predicate)
            and past_gender(predicate) is not None
            and _past_predicate_agrees(subject, predicate)
            and (
                predicate.endswith(("ся", "сь"))
                or transitive_evidence.get(predicate, 0) < 2
            )
        )

    # This is the narrow factual layer used by topic descriptions.  A relation
    # is admitted only when its subject, predicate, and complement occurred in
    # one corpus sentence; it deliberately does not inherit paragraph-level or
    # two-hop graph associations.  Thus a street can keep "тянулась далеко",
    # while a nearby doctor cannot become a property of that street.
    for toks in prose_tokens:
        for subject, predicate, detail in zip(toks, toks[1:], toks[2:]):
            if (
                _description_subject_ok(subject)
                and is_finite_verb(predicate)
                and past_gender(predicate) is not None
                and _past_predicate_agrees(subject, predicate)
                and detail not in STOPWORDS
                and detail not in proper_set
                and detail not in _DESCRIPTION_ENTITY_NOISE
                and detail not in noun_set
                and not is_finite_verb(detail)
                and not is_infinitive(detail)
                and len(detail) >= 3
                and (
                    (_looks_adjective(detail) and _attribute_agrees_subject(detail, subject))
                    or (predicate.endswith(("ся", "сь")) and detail.endswith("о"))
                )
            ):
                description_relation_counts[(subject, predicate, detail, "property")] += 1
        # Inverted temporal/state clauses start with the predicate ("наступил
        # вечер").  They can be placed in a location by the reasoning layer,
        # but must still be a directly observed corpus relation.
        for index, (predicate, subject) in enumerate(zip(toks, toks[1:])):
            if (
                index == 0
                and _description_subject_ok(subject)
                and is_finite_verb(predicate)
                and past_gender(predicate) is not None
                and not predicate.endswith(("ся", "сь"))
            ):
                description_relation_counts[(subject, predicate, "", "predicate_before_subject")] += 1
        for adjective, noun in zip(toks, toks[1:]):
            if (
                adjective in STOPWORDS or noun not in noun_set or noun in proper_set
                or noun in _DESCRIPTION_ENTITY_NOISE or _looks_adjective(noun)
                or is_finite_verb(noun) or _NARRATIVE_VERB.search(noun)
                or noun.endswith(("у", "ю", "е", "и", "ы", "ом", "ем", "ами", "ями"))
            ):
                continue
            if (
                _looks_adjective(adjective) and adjective not in noun_set
                and _nominative_attribute_agrees(adjective, noun)
            ):
                description_relation_counts[(noun, "", adjective, "epithet")] += 1
    # Object-to-object links and placed events are scanned per clause, not per
    # sentence, and the subject may not itself follow a preposition — both
    # guards stop an oblique run such as "за город в рощу" from fabricating a
    # fact about "город".
    for sentence_text in prose_sentences:
        for chunk in _CLAUSE_BREAK_RE.split(sentence_text):
            toks = words(chunk)
            # A noun head joined to another noun head by one observed
            # preposition ("дверь в коридор", "комната в трактире").
            for position, (subject, prep, other) in enumerate(zip(toks, toks[1:], toks[2:])):
                if (
                    _description_subject_ok(subject)
                    and (position == 0 or toks[position - 1] not in _MORPH_PREPOSITIONS)
                    and prep in _MORPH_PREPOSITIONS
                    and _link_object_ok(other)
                ):
                    description_relation_counts[(subject, prep, other, "object_link")] += 1
            # An event anchored to a place ("солнце село за строенье"): the
            # past predicate must agree with the subject, which also lets a
            # neuter -о/-е nominative in without admitting oblique cases.
            for position, (subject, predicate, prep, other) in enumerate(
                zip(toks, toks[1:], toks[2:], toks[3:])
            ):
                if (
                    _description_subject_ok(subject, allow_neuter=True)
                    and (position == 0 or toks[position - 1] not in _MORPH_PREPOSITIONS)
                    and _place_predicate_ok(subject, predicate)
                    and prep in _MORPH_PREPOSITIONS
                    and _link_object_ok(other)
                ):
                    description_relation_counts[
                        (subject, predicate, f"{prep} {other}", "event_place")
                    ] += 1
            # The same relation in the word order Russian prefers for scene
            # description ("за заводами кончался город"): place first, then
            # the agreeing predicate, then the nominative subject.
            for prep, other, predicate, subject in zip(toks, toks[1:], toks[2:], toks[3:]):
                if (
                    prep in _MORPH_PREPOSITIONS
                    and _link_object_ok(other)
                    and _place_predicate_ok(subject, predicate)
                    and _description_subject_ok(subject, allow_neuter=True)
                ):
                    description_relation_counts[
                        (subject, predicate, f"{prep} {other}", "event_place_inverted")
                    ] += 1
    description_relations: dict[str, list[dict]] = defaultdict(list)
    for (subject, predicate, detail, kind), weight in description_relation_counts.items():
        description_relations[subject].append({
            "predicate": predicate, "detail": detail, "kind": kind, "weight": weight,
        })
    for rows in description_relations.values():
        rows.sort(key=lambda item: (-item["weight"], item["kind"], item["predicate"], item["detail"]))
    action_sets = {
        word for word, count in standalone_evidence.items() if count >= 2
    } | {
        word for word, count in terminal_evidence.items() if count >= 2
    }
    # Prefer overt oblique-case endings for subject substitution. Bare
    # nominative-looking objects ("женщина", "глаза") often came from an
    # inverted clause and produced malformed universal frames.
    safe_object_endings = ("у", "ю", "ом", "ем", "ой", "ою", "ью", "ами", "ями")
    universal_speech_frames = [
        {"action": action, "object": object_, "weight": count, "subject_count": len(frame_subjects[(action, object_)])}
        for (action, object_), count in frame_counts.items()
        if count >= 2 and len(frame_subjects[(action, object_)]) >= 2
        and object_ not in set(characters) | set(locations)
        and object_ not in cap_mid
        and object_ not in STOPWORDS and object_ not in _NARRATIVE_OBJECT_NOISE
        and not _looks_adjective(object_)
        and not is_finite_verb(object_)
        and object_.endswith(safe_object_endings)
        and action not in STOPWORDS and action not in noun_like
        and action not in cap_mid
        and not _looks_adjective(action)
        and not action.endswith(("о", "е", "ся", "сь"))
        and past_gender(action) is not None
        and (
            action in action_sets
            or transitive_evidence.get(action, 0) >= 2
        )
    ]
    universal_speech_frames.sort(key=lambda item: (-item["weight"], item["action"], item["object"]))
    total = len(prose_sentences)
    lengths = sorted(sentence_lengths)
    avg_words = round(sum(sentence_lengths) / total, 2) if total else 0.0
    median_words = lengths[total // 2] if total else 0
    meta = {
        "mode": "narrative",
        "source": paths[0].name if len(paths) == 1 else "merged_narrative_corpus",
        "sources": [path.name for path in paths],
        "total_sentences": total,
        "concept_nodes": len(graph.weight),
        "sentence_length": {"average_words": avg_words, "median_words": median_words},
        "dialogue": {"sentences": dialogue_count, "ratio": round(dialogue_count / total, 3) if total else 0.0},
        "discourse_modes": {
            "narration_sentences": narration_count,
            "description_sentences": description_count,
            "dialogue_sentences": dialogue_count,
        },
        "recurring_characters": characters,
        "recurring_locations": locations,
        "recurring_objects": recurring_objects,
        "entity_types": entity_types,
        "noun_like_sources": {
            "after_preposition": dict(after_preposition),
            "after_predicate": dict(object_evidence),
            "case_stem_projection": sorted(stem_heads),
        },
        "verb_lexicon": {
            "standalone_actions": sorted(word for word, count in standalone_evidence.items() if count >= 2),
            "dative_actions": sorted(word for word, count in dative_evidence.items() if count >= 2),
            "transitive_actions": sorted(word for word, count in transitive_evidence.items() if count >= 2),
            "terminal_actions": sorted(word for word, count in terminal_evidence.items() if count >= 2),
        },
        "universal_speech_frames": universal_speech_frames,
        "description_relations": dict(description_relations),
    }
    return {
        "meta": meta,
        "phrase_model": phrase.to_dict(),
        "concept_graph": graph.to_dict(),
        "rhyme_groups": {},
        "style_profiles": {"prose": {"top_openers": [word for word, _count in opener_counter.most_common(20)]}},
        "proper_names": list(dict.fromkeys((*characters, *locations))),
        "entity_types": entity_types,
        "gender_map": gender_map,
        "noun_like": noun_like,
        "verb_lexicon": {
            "standalone_actions": sorted(word for word, count in standalone_evidence.items() if count >= 2),
            "dative_actions": sorted(word for word, count in dative_evidence.items() if count >= 2),
            "transitive_actions": sorted(word for word, count in transitive_evidence.items() if count >= 2),
            "terminal_actions": sorted(word for word, count in terminal_evidence.items() if count >= 2),
        },
        "universal_speech_frames": universal_speech_frames,
        "description_relations": dict(description_relations),
    }


def write_narrative_artifacts(target: Path | None = None, source: Path | None = None) -> Path:
    artifact = build_narrative_artifacts(source)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    out = target or (ARTIFACT_DIR / "narrative_model.json")
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_artifacts(source: Path | None = None) -> dict:
    src = source or (ARTIFACT_DIR / "poetry_model.json")
    return json.loads(src.read_text(encoding="utf-8"))


if __name__ == "__main__":
    path = write_artifacts()
    data = load_artifacts()
    meta = data["meta"]
    print(f"Wrote {path}")
    print(
        f"authors={meta['authors']} lines={meta['total_lines']} "
        f"concepts={meta['concept_nodes']} rhyme_groups={meta['rhyme_groups']}"
    )
