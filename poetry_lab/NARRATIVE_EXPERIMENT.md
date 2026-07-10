# Narrative Transfer Experiment

This experiment reuses the existing offline MicroWorld graph, deterministic
trigram phrase model, discourse salience, candidate ranking, and novelty gate.
It changes only the corpus segmentation, planning vocabulary, and realization
surface.

## Corpus

`corpus/мастер и маргарита.txt` is ingested alone into
`artifacts/narrative_model.json`; the verse files remain untouched as the
previous experiment's control corpus.

Rebuilt-artifact statistics (full novel, both parts):

- 9,409 sentences; average 11.97 words, median 10 words.
- Dialogue: 2,596 sentences (27.6%).
- Narration: 4,073 sentences; description: 2,740 sentences.
- The metadata retains recurring character, location, and object candidates
  inferred from capitalized mid-sentence tokens, location prepositions, and
  frequent content words.
- Derived layers: 50 typed entities (26 person / 19 place / 3 object /
  2 unknown), a 62-word gender map, and 1,510 `noun_like` tokens.

An earlier run ingested a truncated file that ended at the close of Part One,
which is why questions about Margarita or the Ball at Satan's could not be
grounded at all — that content was simply absent, not mis-parsed.

## Surface Substitution

| Poetry assumption | Narrative equivalent |
| --- | --- |
| line | sentence |
| stanza | paragraph / scene |
| `PoemGoal` | `SceneGoal` |
| stanza plan | `ScenePlan` |
| line decision | `SentencePlan` |
| meter and rhyme gates | target sentence length and entity/topic constraints |

The narrative pipeline is:

```text
prompt -> SceneGoal -> ScenePlan -> SentencePlan -> sentence realization -> paragraph
```

`NarrativeGenerator` still creates eight deterministic candidates. It ranks
them with the transferred discourse salience plus planned subject/action/object
coverage and penalizes unplanned named entities. The trigram traversal refuses
any next token that would close a corpus 4-gram, so the novelty gate is active
during realization as well as in final reporting.

## Evaluation

Run:

```bash
python3 cli.py ingest-narrative
python3 eval/eval_narrative.py
```

The evaluator runs the six narrative prompts from the experiment brief and
reports structural proxies: learned-trigram coherence, planned-entity
consistency, pronoun-set consistency, graph-field topic drift, novelty, mean
generation time, and artifact size. These are not literary-quality metrics.

## Result

Replacing the corpus did not require a new core architecture. The graph,
trigram language layer, state tracking, ranking, and novelty gate remain in
place. Changes were confined to corpus ingestion, scene/sentence planning, and
paragraph realization. The remaining failure mode is visible short or awkward
sentences when novelty blocks a copied continuation; that is a limitation of
the fixed trigram surface layer, not evidence that the core architecture had to
change.

## Question-prompt fix (`plan_scene`, `poemcore/narrative.py`)

Testing on direct questions ("Кто такой Воланд?", "Что случилось с
Берлиозом?") found `SceneGoal.topic/speaker/location` collapsing onto the
*interrogative word itself* — "такой", "случилось", "почему", "куда" — instead
of the entity the question asked about, because `content_words()` does not
strip Russian question pronouns/adverbs (only "кто"/"что"/"где" are in the
shared `STOPWORDS`; their neighbours are not) and `seeds[0]` (first content
word in reading order) was taken as the topic unconditionally.

Three fixes in `plan_scene`, none touching the shared `STOPWORDS` set (which
also gates concept-graph training on the novel's own sentences, where these
same words are ordinary content):

1. **`_INTERROGATIVE_MARKERS`** — a local filter (почему, зачем, куда, откуда,
   каков-/чей-/какой- forms, такой-forms, and question-frame verbs
   случилось/произошло/происходит/делает) stripped from prompt seeds before
   topic selection, with the unfiltered list kept as a last resort for a
   purely interrogative prompt ("Кто это?").
2. **Named-entity-first seed ordering** — any recognized character/location
   is promoted ahead of a generic word regardless of sentence position, so
   "Почему Понтий Пилат наказал Иешуа?" grounds on Пилат, not on "почему".
3. **Declension-tolerant, case-normalizing entity matching**
   (`_canonical_entity`, stem-prefix match — the same principle
   `_corpus_form` already uses) — "Берлиозом" (instrumental) resolves to
   whichever form ("берлиоз") the ingest lexicon actually recorded. Needed
   because `proper_names` only contains the individual surface forms whose
   *own* mid-sentence-capitalized count cleared the ingest threshold, and
   Russian spreads one name across up to six case forms. Without this, the
   scene correctly grounded on the entity but then tried to use the
   instrumental form as a sentence subject, which reads as broken agreement
   ("Берлиоза заседании приступили").

Before / after on the same prompts:

| Prompt | Before | After |
|---|---|---|
| Кто такой Воланд? | `topic=такой speaker=такой location=такой` | `topic=воланд` — *"Воланд может запорошить... Профессор пугливо оглянулся, а он заранее понимаете."* |
| Что случилось с Берлиозом? | `topic=случилось` | `topic=берлиоз` — *"Берлиоз и возразил... Берлиоз упал навзничь, а он даже изменился."* (grammatically stable subject throughout; "упал навзничь" also happens to match his canonical death scene) |
| Почему Понтий Пилат наказал Иешуа? | `topic=почему` | `topic=понтия`, rendered subject "Пилат" throughout |
| Что делает Иван Бездомный? | `topic=делает` | `topic=иван` |

Regression checks: a declarative control prompt ("Воланд появился на
Патриарших прудах") was unaffected (`topic=воланд location=патриарших`,
unchanged before/after); generation stayed deterministic; the poetry engine
(a separate module, untouched) was unaffected.

## Knowledge layer: advisory entity types (`poemcore/entity_types.py`)

Ported from `worldpgt/knowledge/entity_type_classifier.py`. Production types an
entity from its Wikipedia *definition* text; we have no definitions, only the
novel, so the classifier's **shape** is kept (rules in priority order,
first-match-wins, non-committal default) and its **input** is swapped for
evidence `ingest` already accumulates while reading prose:

```text
production: definition text  -> keyword rules -> canonical type
here:       corpus evidence  -> ratio rules   -> canonical type
```

Evidence: `locative` (token follows в/на/из/к/по/у/под), `agent` (token precedes
a finite verb, or follows a speech verb in inversion — "сказал Воланд"),
`patient`. A type is asserted only above 3 observations with 60% dominance;
otherwise `unknown`.

**Advisory, not a gate — the one place the QA analogy is deliberately broken.**
QA uses the type to decide whether it may answer at all (unsupported → audit).
Here it only picks a *role*: a `place` never enters the agent slot, a `person`
may. An `unknown` type reproduces the previous behaviour exactly. Nothing is
ever refused.

Result on the full novel: 26 `person`, 19 `place`, 3 `object`, 2 `unknown`. The
two entities that broke generation typed correctly — `патриарших → place`,
`бегемот → person`.

| Prompt | Before | After |
|---|---|---|
| …о Патриарших прудах | `topic=speaker=location=патриарших` (a pond as agent) | `location=патриарших`, `speaker=маргарита` |
| …о коте Бегемоте | `topic=коте` (common noun beat the name) | `topic=бегемот` |
| …о Понтии Пилате | `topic=понтия`, broken clauses | `topic=пилат`, stable throughout |

Two supporting fixes fell out of this: entity resolution now runs *before* the
`word in graph.weight` filter (a rare inflected name like "Бегемоте" is absent
from the graph even though its canonical "бегемот" is present, so filtering
first silently discarded the entity the prompt was about), and a compound name
("Понтий Пилат", "Иван Бездомный") resolves to its more corpus-salient half.

## Morphology layer (`poemcore/morphology.py`)

The entity-type transfer fixed *role* selection but could not touch *agreement*:
"Маргарита погубил одного" picks the right subject and a real corpus verb, but
the verb is masculine and the subject is not.

Russian makes this tractable without a dictionary or a tagger, because **the
past tense marks gender on its own surface** (`-л` / `-ла` / `-ло` / `-ли`). The
only unknown is the *subject's* gender, and that is learned exactly the way
`entity_types.py` learns person-vs-place: by counting what the corpus put next
to each name. A name that repeatedly governs `-ла` verbs is feminine. An ending
heuristic is a fallback only where the corpus stayed silent; undecidable words
are absent from the map and constrain nothing.

**The hard part was not gender — it was verb-hood.** Surface endings cannot
separate a masculine past verb from a noun: `сказал`/`стол` both end in vowel+л,
`видел`/`мел` both in `-ел`. Two candidate discriminators were tested:

1. *"Does a plural past form exist?"* — rejected. It silently fails on hapaxes:
   `погубил` occurs once and has no `погубили`, i.e. it kills precisely the verb
   this layer exists to catch.
2. *"Does the token follow a preposition?"* — adopted. A noun does, a finite verb
   does not. In this corpus `стол`/`угол`/`пол` follow в/на/под 7/7/29 times;
   `сказал`/`погубил`/`вступил` follow one zero times. Collected at ingest as
   `noun_like`.

Applied in two places, both scoring rather than hard rejection where an
agreeing candidate may not exist: an agreement **gate** in
`_choose_reasoned_fragment` (a disagreeing fragment never becomes a candidate),
and **penalties** in `NarrativeGenerator._select` for residual disagreement, for
a clause with no finite verb ("Пилат и всеобщий"), and for an infinitive where a
predicate belongs ("Маргарита шелестеть листами").

Measured over the five 10-sentence prompts (50 sentences), scored both times
against the true gender map:

| | morphology OFF | morphology ON |
|---|---|---|
| gender-agreement errors | **9** | **0** |
| sentences with no finite verb | 0 | 1 |

Same paragraph, before and after: `Маргарита погубил одного` / `Маргарита опять
вступил` / `Маргарита шелестеть листами` became `Маргарита вздохнула стала` /
`Маргарита уже соскучилась` / `Маргарита сразу узнала`.

### Known defect, harmless by construction

`москве → n` — the learned gender is wrong (Москва is feminine); the
prepositional surface form misled the ending fallback. It is inert because
gender is only consulted for the *subject*, and the entity-type layer already
guarantees a place never becomes one. Worth naming rather than hiding: the two
layers cover each other's errors here by accident, not by design.

### What the earlier question-prompt fix did *not* solve — three distinct, honest limits

Three of the eight originally-tested questions stayed broken after the fix,
each for a different, non-code reason:

- **Corpus completeness, not a bug.** "Кто такая Маргарита?" and "Что
  произошло на балу у сатаны?" cannot be fixed by any parsing change: the
  ingested corpus file ends exactly at the last line of Part One ("настает
  пора переходить нам ко второй части этого правдивого повествования. За
  мной, читатель!"). Margarita and the Ball at Satan's are Part Two content —
  "маргарит" (any case) has **zero** matches in the corpus file. No code
  change grounds a question about text that was never ingested.
- **An ingest-heuristic edge case.** "Куда исчез кот Бегемот?" still
  resolves `topic=исчез`. "Бегемот" appears capitalized 3 times in the
  corpus, but the character-detection heuristic in `ingest.py` only counts a
  capitalized token toward the name lexicon when it appears *mid-sentence*
  (`pos > 0`) — sentence-initial capitalization is deliberately excluded
  there to avoid false positives from ordinary words that merely open a
  sentence. If all 3 occurrences happen to be sentence-initial in this
  corpus, the character never clears the threshold. A real limitation of a
  deliberately conservative heuristic, not the question-parsing bug this
  round fixed.
- **Compound-name granularity.** "Понтий" and "Пилат" are separate lexicon
  entries (no multi-word name grouping exists), so which one wins as `topic`
  depends on reading order and which surface form the ingest threshold
  happened to keep. Cosmetic — the rendered text stayed grounded on "Пилат"
  regardless — but worth naming as a scope boundary, not a fixed bug.
