# poetry_lab — architecture transfer experiment

**Research question.** Can the same core architecture that powers MicroWorld's
factual QA also produce coherent *creative* writing after changing **only** the
knowledge source?

This is a research experiment, not a product. It began fully separated from the
production runtime and lives in this folder — but the transfer later became
**bidirectional**: the description-mode fact-bundling worked out here (see
[Reverse transfer](#reverse-transfer-fact-bundling-back-into-production-qa))
flowed *back* into `worldpgt/`, so the claim "nothing under `worldpgt/` was
modified" no longer holds. The production changes are listed in that section.

> Note: `artifacts/*.json` (the generated concept/phrase models) are git-ignored.
> Regenerate them with `python cli.py ingest` and `python cli.py ingest-narrative`
> before running anything below. The full mixed-corpus narrative model is
> ~230 MB on disk (~1 GB resident); the on-device build uses a slim copy
> (`python cli.py slim-narrative` → ~24 MB / ~380 MB resident, same output).

## Narrative transfer

The prose surface began as a single-corpus experiment (`Мастер и Маргарита`)
and has since grown into the **Creative mode that ships in the iOS app**
([`../ios_demo/`](../ios_demo/)), running fully offline on an iPhone 11. Two
things changed along the way:

1. **Language became configurable.** The primitives were Cyrillic-only
   (tokenizer, verb/adjective detection, case morphology). They now handle
   English alongside Russian, so the same machine ingests an English
   public-domain corpus (Shakespeare + eight Victorian/adventure classics —
   Conan Doyle, Austen, Wilde, Verne, Stevenson, Wells) and generates from it.
   Where Russian keys off case endings, English keys off word order and small
   closed-class lists — the mechanism is unchanged, only its inputs are
   re-domained. See `poemcore/text.py`, `poemcore/morphology.py`,
   `poemcore/entity_types.py`.

2. **Description/scene generation became a real three-layer pipeline** — the QA
   architecture, not a flat template. See [The three-layer creative
   generator](#the-three-layer-creative-generator) below.

```bash
# English multi-work corpus (one .txt per work under corpus/english/):
python3 cli.py ingest-narrative --source corpus/english/
python3 cli.py narrate "Write a scene about a storm" --sentences 6
python3 cli.py narrate "Describe the sea" --trace
python3 cli.py slim-narrative           # memory-lean copy for the phone build
```

See [`NARRATIVE_EXPERIMENT.md`](NARRATIVE_EXPERIMENT.md) for the original corpus
analysis and evaluation definitions.

## The three-layer creative generator

The description/scene path mirrors QA's `open_synthesis` pipeline — the same
knowledge → reasoning → speech separation — instead of the old single-relation
template that produced monotone "N adjectives + noun" output. All three layers
live in `poemcore/narrative.py`:

| Layer | QA analogue | Here | Role |
|---|---|---|---|
| **1 — knowledge** | `entity_answer_planner` / `synthesis_engine` | `_gather_topic_knowledge` → `TopicKnowledge` | Read every observed relation about the topic, bucketed by role: epithets, properties, prepositional links, fronted predicates, place-events, and real subject→verb→object **actions** pulled from the phrase graph. No wording yet. |
| **2 — reasoning / discourse** | `semantic_speech_planner` (bucket + seeded clause-order) | `_plan_topic_discourse` | Pick one **rhetorical schema** per sentence from a weighted pool (seeded, so each sentence differs), fill its clauses from the buckets, and track what's been used so a several-sentence paragraph is *multiple facets*, not one clause restated. |
| **3 — speech** | `phrase_graph.generate` (per-node seeded pick) | `_render_description_sentence` / `_grow_clause_bridge` | Realize each clause's tokens, letting the learned phrase graph supply connective tissue ("sea *of* glory", "love *that* burns") where the corpus actually has it — never a fabricated transition. |

Three things make it read as language rather than a fill-in-the-blank:

- **Collocations.** The 2nd/3rd epithet is weighted not just by topic-fit but by
  how often it is actually observed *next to* an already-chosen adjective in the
  corpus (`adjective_collocations`, extracted at ingest). "true" pulls
  "fair/false/good/kind" (attested pairings), not just any frequent adjective.
- **Universal scene vs. description — one machine, re-weighted.** "Describe X"
  and "Write a scene/story about X" run the *same* planner; a `scene_bias` flag
  only tilts the shared schema pool toward event/action shapes (something
  happens) vs. static description. No separate templated path. This is why
  "Write a scene about a murder" now narrates ("Murder lurked. Murder
  committed.") instead of falling back to "Murder paused."
- **Creative licence ("allowed to lie").** Unlike QA, which may only state
  grounded facts, this layer may borrow an epithet from a related field concept
  or reach for a figurative connector — combinations the corpus never literally
  showed. Hard-safety screening still runs first (a private/current-sensitive
  ask audits even under a creative framing), preserved on the app path in
  `../ios_demo/.../mw_ios.py`.

## The claim being tested

MicroWorld's runtime contract is:

```text
Text → Semantic Structures → Reasoning → Speech Plan → Language Renderer → Answer
```

The hypothesis: that pipeline is *source-agnostic*. If you keep the reasoning
mechanism (a typed graph with spreading activation) and the language mechanism
(a learned frequency phrase graph traversed with a seeded deterministic pick),
and you swap the ingested knowledge from Wikipedia/Reddit facts to a poetry
corpus, the same machine should produce verse instead of answers.

The one deliberate change beyond the source: the **support gate is inverted**
(see below), because "refuse anything not explicitly supported" is the single
most QA-specific assumption and is the opposite of what creative recombination
needs.

## What was reused vs. replaced

Full module classification is in [`ANALYSIS.md`](ANALYSIS.md). Summary:

| Production layer | Status | poetry_lab module |
|---|---|---|
| Spreading activation over a typed graph (`cognition/semantic_thought_graph.py::_activate`) | **ported verbatim** | `poemcore/concept_graph.py::activate` |
| Seeded deterministic weighted pick (`cognition/phrase_graph.py::_seeded_weighted_pick`) | **ported verbatim** | `poemcore/phrase_model.py::seeded_weighted_pick` |
| Frequency phrase graph + traversal (`cognition/phrase_graph.py`) | reused as mechanism | `poemcore/phrase_model.py`, `poemcore/generator.py` |
| Multi-word **fragment context** (QA stitched learned clauses, not single words) | reused as mechanism (order-2 phrase model) | `poemcore/phrase_model.py` (`forward2`/`backward2`) |
| Explicit typed reasoning artifact (`ReasoningTrace`) | reused as pattern | `poemcore/planner.py::PoemPlan` |
| JSON artifacts as the layer boundary | reused | `artifacts/poetry_model.json` |
| Knowledge ingestion (wiki/Reddit → fact overlays) | **replaced** | `poemcore/ingest.py` (poetry → structures) |
| Support gate "reject unsupported" (`cognition/support_guard.py`) | **inverted** | `poemcore/novelty.py` (reject *memorised*) |
| Question parsing / routing / QA validators / dialogue | **removed** | — (replaced by small intent reader in `engine.py`) |

### The inverted gate

```text
QA gate      : allow output only if every claim is grounded in accepted memory.
poetry gate  : allow output only if it does NOT reproduce a corpus 4-gram.
```

Same architectural slot (a gate between reasoning and final output), opposite
polarity. Creative combination of learned images is allowed; reciting the
training text is blocked. The system must **recombine, not recite**.

## Layout

```
poetry_lab/
  corpus/            Russian verse (Пушкин, Лермонтов, Тютчев, Фет, Блок, Ахматова) + prose (Булгаков, Чехов)
    english/         English narrative corpus (Shakespeare + 8 classics) — the Creative-mode source
  poemcore/
    text.py          syllable/rhyme/token primitives (Russian + English)
    morphology.py    verb / adjective / gender detection (Russian case endings + English word-order rules)
    entity_types.py  person / place / object classifier (advisory)
    ingest.py        corpus → artifacts + slim_narrative_artifact (the swapped knowledge pipeline)
    concept_graph.py reasoning: spreading activation (ported)
    phrase_model.py  language: frequency graph + seeded pick (ported)
    narrative.py     the three-layer description/scene generator (knowledge → discourse → speech)
    planner.py       verse: poetic move selection → PoemPlan
    reasoning.py     verse: poem goal → stanza plan → line decisions
    generator.py     verse: render plan into metered, rhymed lines
    novelty.py       inverted support gate
    engine.py        orchestration
  cli.py             command-line interface (ingest / ingest-narrative / slim-narrative / write / narrate)
  eval/              evaluation scripts + run_all.py
  artifacts/         generated JSON model (the layer boundary; git-ignored)
```

No third-party dependencies (stdlib only) — see `requirements.txt`.

## Usage

```bash
cd poetry_lab
python cli.py ingest                       # build artifacts/poetry_model.json
python cli.py write "write a poem about autumn"
python cli.py write "write in the style of Pushkin" --stanzas 3 --trace
python cli.py write "write a poem about space using classical Russian imagery"
python cli.py continue "Мороз и солнце; день чудесный!"
python eval/run_all.py                     # all metrics
```

`--trace` prints the reasoning artifact (activated concepts + per-line plan),
so you can see the layer boundary the same way production exposes a
`ReasoningTrace`.

### Explicit reasoning transfer

The current experiment adds a narrow planning stage above candidate generation:

```text
Prompt → PoemGoal → StanzaPlan → LineDecision → LineIntent → trigram realization
```

`poemcore/reasoning.py` chooses a theme-grounded goal, a purpose and anchor for
each stanza, then a subject/action/object decision for every line. The decision
is passed to the existing intent-seeded traversal and candidate scorer; it does
not replace the trigram graph or relax meter, rhyme, novelty, or discourse
gates. `--trace` and `--json` expose every object. `eval/eval_reasoning.py`
compares generic line intents against explicit decisions on the fixed battery.

## Example output

`write a poem about space using classical Russian imagery` seeds
`звезда · звёзды · небо · лазурь · луна`; activation spreads to neighbouring
images from the corpus and the generator assembles ABAB stanzas around them —
word sequences that never appear verbatim in the corpus (novelty gate passes),
built entirely from learned transitions.

## Results (10-prompt battery, `eval/run_all.py`)

Three corpus configurations were run, each a deliberately unbalanced scaling
test: the original hand-picked sample (6 authors, ~60–90 lines each, 371 lines
total), then a full collected edition of Lermontov dropped in on top of the
small samples (9568 lines total), then full collected editions of Pushkin and
Blok added as well (43973 lines total — Пушкин 23840, Блок 10303, Лермонтов
9234, vs. Ахматова/Фет/Тютчев still at 56–59 each).

| Metric | Small, balanced (371 lines) | + full Lermontov (9568 lines) | + full Pushkin & Blok (43973 lines) | Reading |
|---|---|---|---|---|
| **Novelty** (lines with no corpus 4-gram) | 0.93 | 1.00 | **1.00** | Stable at the ceiling once the corpus is large enough — more text only ever adds transition options, never forces a full memorised 4-gram back out. |
| Bigram reuse | ~1.0 (by construction) | ~1.0 (by construction) | ~1.0 (by construction) | Every step is a learned transition regardless of scale; novelty lives at the 3–4-gram level. |
| **Meter** (within ±1 syllable of target) | 89%, dev 0.72 | 84%, dev 0.65 | **78%, dev 0.79** | Degrading gradually — a far bigger, more varied transition table gives the backward walk more plausible-but-off-target paths to wander down before hitting the syllable count. |
| **Rhyme** (planned ABAB pairs share a rhyme key) | 0.85 | 0.83 | **0.80** | Same direction as meter — more rhyme groups (249 → 2163 → 5535) means more *options*, which dilutes the frequency signal the seeded pick leans on. |
| **Coherence** (on-theme content-word share, 2-hop) | 0.25 | 0.64 | **0.67** | Keeps improving — a denser concept graph (974 → 12024 → 37481 nodes) keeps giving spreading activation more to work with. The most reliably scale-positive metric of the five. |
| **Style separation** (own vs. other signature overlap) | +0.02 (weak) | −0.01 (collapsed) | **−0.01 (still collapsed, different author)** | Still broken by imbalance, see below — but *which* author reads as distinctive shifted. |

### The fragment-context fix: order-1 → order-2 phrase model

The three runs above all used an **order-1 (bigram)** language layer — each
word chosen from only its immediate predecessor. That is why the generated
lines held meter and rhyme but read as word salad at the phrase level: a bigram
walk has no memory of the word before last, so it cannot keep a three-word
grammatical span intact. Production QA never had this problem because its
`cognition/phrase_graph.py` never generated word-by-word — it stitched whole
multi-word *fragments* ("was founded by {object_list}"), carrying local grammar
in the learned span itself.

Porting that idea meant raising the phrase model to **order-2 (trigram)**: every
word is now chosen conditioned on the *two* words already placed, falling back
to bigram only when that exact two-word context was never seen. Novelty still
lives at the 4-gram gate, so lines recombine real 3-word fragments without
reciting a 4-word one. Measured on the full 43973-line corpus:

| Metric | order-1 (bigram) | order-2 (trigram) | Effect |
|---|---|---|---|
| **Local grammaticality** (generated 3-word windows attested in corpus) | **0.19** | **0.79** | **The fix.** 4x more of each line is a real grammatical span → phrases stop being word salad. |
| Novelty (no corpus 4-gram) | 1.00 | 0.99 | Order-2 makes an accidental real 4-gram likelier, so the per-line re-roll budget was raised 4→8 to keep the novelty gate satisfied; net drop is negligible (0.99). |
| Meter (within ±1 syllable) | 78% | 80% | Small gain — trigram context favours more natural line-length phrasing. |
| Rhyme | 0.80 | 0.83 | Small gain, same reason. |
| Thematic coherence (graph on-theme share) | 0.67 | 0.60 | Small *drop* — see below. |

**Local grammaticality (0.19 → 0.79) is the headline.** Before/after on the same
prompt: `да здравствует великий глюк` / `собаки нет счастье на бледный цвет`
(bigram) became `И будто путник запоздалый / Очарованье доцвело` and `Гневной
совести смиритель` / `Его убийца хладнокровно` (trigram).

The one metric that *dropped* is the interesting part: **grammatical coherence
and thematic coherence measure different things and trade off slightly.** Order-2
sometimes spends a word on a grammatical connective ("будто", "иль") to keep the
span real, instead of on an on-theme image the concept graph activated — so the
share of on-theme content words dips even as the lines read far better. Two
distinct notions of "coherent," pulling in slightly different directions, now
each have their own metric (`eval_coherence.py` reports both). This is the same
kind of reasoning-vs-language tension the scaling test surfaced, made sharper.

### The discourse-state transfer: from local grammar to cross-line coherence

Order-2 fixed grammar *inside* a line. It did nothing for coherence *across*
lines — each line was still grown independently and the generator kept the first
valid one, so a poem's images did not develop from line to line. In QA that
cross-sentence coherence came from a different mechanism entirely: an explicit
discourse state (`dialogue/state.py::EntityActivation`) plus a **salience score**
(`dialogue/salience.py`) that ranks candidates against it with an integer
breakdown, so successive sentences stay about the same, still-active subject.

That mechanism ports directly, because the generator already produces a *pool*
of candidate lines (its 8-attempt retry loop) and was discarding all but the
first valid one. The transfer (`poemcore/discourse.py`, ~110 lines) is: keep a
per-poem `DiscourseState` of active images (decaying by recency, counted for
repetition), and change the generator's accept step from "first valid" to
"score every valid candidate by how well it continues the active images, keep
the argmax, feed its images back into the state." The chosen line's concepts
raise the salience of their graph neighbours for the *next* line — spreading
activation now runs across the whole poem, not just inside one planning step.
Design rationale and the full mechanism ranking are in
[`DESIGN_next_experiment.md`](DESIGN_next_experiment.md).

Measured with a new metric, **inter-line continuity** (`eval_continuity.py`) —
the average concept-graph link share between adjacent lines' content words —
run with the salience selection off vs. on (everything else identical):

| Metric | ranking OFF (first valid) | ranking ON (discourse salience) | Effect |
|---|---|---|---|
| **Inter-line continuity** | **0.13** | **0.23** | **The target.** +75% — adjacent lines now share/develop images instead of each re-seeding. |
| Thematic coherence (on-theme share) | 0.60 | 0.68 | Rose too — selecting for continuity also keeps lines nearer the theme cluster. |
| Meter (within ±1 syllable) | 80% | **91%** | *Rose*, see below. |
| Rhyme | 0.83 | 0.80 | Flat within noise. |
| Novelty | 0.99 | 1.00 | Flat. |
| Local grammaticality | 0.79 | 0.76 | Flat within noise — order-2 still owns within-line grammar. |

**An honest correction mid-experiment.** The design doc claimed meter "cannot
regress" because it is gated before ranking. The first implementation proved
that wrong: meter *dropped* to 74%. The gate was only a *lower* syllable bound,
and salience correlates with content-word count, so the ranker systematically
preferred longer, overshooting lines. Two small fixes closed it: a **two-sided**
meter window in the candidate gate (so every ranked line is on-meter both ways)
and **normalizing salience per content word** (so it rewards continuity
*density*, not line length). The two-sided window also rejects overshoot for
*non*-ranked reasons, which is why meter came out *above* the pre-transfer
baseline (80% → 91%). The lesson is exactly the kind a real transfer surfaces:
a ported mechanism interacts with the host system's other constraints in ways
the source never had to consider.

This is the **third** QA mechanism transferred (after spreading activation and
fragment context), and the first that improves coherence *above* the single
line. Same pattern each time: find the production mechanism that already solved
the analogous problem, port its shape unchanged, re-domain its inputs.

### The line-intent layer: giving each line a subject and an action

Discourse state kept images *active*, but no line had an explicit *intent* — it
was still a rhyme-anchored walk that happened to be scored for continuity. The
lines read as fluent fragments, not as clauses that assert something. The QA
analogue is `entity_qa/semantic_speech_planner.py::SpeechPlan`: before rendering,
QA committed to *what each sentence is about* (a subject and a role-bucketed
predicate). Porting that idea gives `poemcore/line_plan.py`: a `PoemIntent`
(theme, speaker, setting, mood, core images) and a per-line `LineIntent`
(subject, action, object, modifier, mood, relation-to-previous), built from the
same structural plan the generator already uses.

Per the constraints of this experiment, the generator was **not** rewritten and
the n-gram order, corpus, and traversal are unchanged. The intent enters as one
additional scoring term in the existing candidate selection: on top of
meter/rhyme/novelty (gates) and discourse salience (continuity), each candidate
gets a `line_plan_score` that rewards a detectable subject+action, rewards
realizing the planned concepts, and **penalizes** two things the earlier output
did badly — random proper names (a corpus-derived name lexicon built at ingest,
no NER model) and abrupt entity jumps (content words with no graph link to the
active image field). Subject/action/proper-noun detection is light Russian
morphology (verb endings, a closed pronoun set), not a parser.

A/B on the full battery — discourse ranking alone vs. discourse ranking **plus**
the line-intent score, everything else identical:

| Metric | line-intent OFF | line-intent ON | Effect |
|---|---|---|---|
| **Subject/action presence** (lines with both a subject and a verb) | **0.45** | **0.79** | **The target.** +76% — most lines now read as clauses that assert something, not image fragments. |
| Proper-name rate / line | 0.013 | 0.013 | Held near zero (the penalty keeps random names out; a few rare inflected names still slip the lexicon). |
| Meter (within ±1 syllable) | 91% | **95%** | Rose — action-bearing lines the intent favours also sit cleaner on the meter grid. |
| Rhyme | 0.80 | 0.83 | Flat/slightly up. |
| Novelty | 1.00 | 0.95 | Essentially flat. |
| Thematic coherence (on-theme share) | 0.68 | 0.63 | Small drop — selecting for subject/action spends tokens on verbs/pronouns. |
| Local grammaticality | 0.76 | 0.75 | Flat — order-2 still owns within-line grammar. |
| **Inter-line continuity** | **0.23** | **0.18** | **Structural drop, see below.** |
| Plan satisfaction (planned concepts realized) | 0.01 | 0.02 | Low, and correctly so — see below. |

Before/after on the same prompts: `Тут астрахань вот стамати` (a bare list of
names) gives way to `Короля султан осаждает` and `Туман ползёт через чело` —
subject + verb (+ object), i.e. actual clauses.

**Two honest limits, both structural rather than tunable:**

- **Continuity dropped (0.23 → 0.18) and it is not a scoring-weight bug.**
  Weighting the continuity term up 3x did not move it. The reason is intrinsic:
  the continuity metric counts *content-noun* graph links between adjacent
  lines, but a line with a strong subject+verb spends tokens on the verb and a
  pronoun — neither of which is a concept-graph node — leaving fewer linkable
  nouns. Making lines more clause-like necessarily thins the noun imagery the
  continuity metric measures. So the two goals (assert an action vs. maximize
  image-to-image links) genuinely trade off; the line-intent experiment buys a
  large subject/action gain for a small, explainable continuity cost.
- **Plan satisfaction stayed near zero (0.02).** The intent can only *score*
  candidates, not *steer generation* toward a planned word — and the
  no-generator-rewrite constraint forbids the latter. If none of a line's 8
  candidates happens to contain the planned subject/object, the reward never
  fires. So the selection layer improved the *shape* of lines (subject+action)
  far more than their *specific content* (this exact image here). Steering
  growth toward plan concepts would need the generator change this experiment
  deliberately excluded — a clean result about where a pure selection layer's
  ceiling is.

This is the **fourth** transferred mechanism (after spreading activation,
fragment context, and discourse salience), and the pattern held once more: the
production `SpeechPlan` gave the shape, the port added one scoring term, and the
gain (clause-like lines) and the cost (thinner image-continuity) both trace to
the same fact — a language layer that plans *what to say* competes for the same
tokens the layer that plans *how images connect* wants.

### Intent-seeded generation: making the plan steer growth, not just ranking

The line-intent layer above hit a hard ceiling: **plan satisfaction stayed at
0.02** because the intent could only *score* candidates, never *steer* them. If
none of a line's 8 candidates happened to contain the planned concept, the
reward never fired. This experiment closes that gap without rewriting the
generator: it lets the intent seed a token *into* candidate growth.

The mechanism is a `must_include` hook on the existing backward/forward walk
(`phrase_model.grow_backward`/`grow_forward`, ~8 lines each). At each step, if a
seed token is a *valid corpus predecessor of the current head*, it is forced in.
Because it must already be in the distribution the walk samples, every forced
hop stays a real corpus transition — trigram grammar, the rhyme endpoint, and
the novelty gate are all untouched. If the seed is never a valid predecessor
along the walk, nothing is forced (the soft fallback). The n-gram order, corpus,
and generator structure are unchanged; growth gained one optional branch.

A/B on the full battery — line-intent ranking **only** vs. ranking **+ seeded
generation**, everything else identical:

| Metric | seeding OFF | seeding ON | Effect |
|---|---|---|---|
| **Plan satisfaction** (planned concepts realized) | **0.02** | **0.11** | **The target. ~5x** — the intent now actually appears in the line, not just in the score. |
| **Inter-line continuity** | 0.18 | **0.28** | +0.10 — seeding the same core images across lines links them; recovers what the line-intent step had cost. |
| **Thematic coherence** (on-theme share) | 0.57 | **0.67** | +0.11 — forced theme concepts pull the whole line on-topic. |
| Local grammaticality | 0.745 | 0.753 | Flat — the predicted risk did **not** materialize (see why below). |
| Rhyme | 0.83 | 0.85 | Flat/up — the other predicted risk did not materialize either. |
| Meter (within ±1) | 0.95 | 0.93 | −0.02, negligible. |
| Novelty | 0.95 | 0.96 | Flat. |
| Proper-name rate / line | 0.013 | **0.000** | Names eliminated (see finding below). |
| Subject/action presence | 0.79 | 0.73 | **−0.06 — the one real cost, see failure cases.** |

Before/after on the autumn prompt: the ranking-only first line was `Я взвиваюсь
звеня кимвалами` (no theme word); seeded it becomes `И осени вершиной
белоснежной` — the theme concept `осени` is now *in* the line.

**The headline finding is a negative one about the brief's own step 3.** The
brief specified a soft fallback: *exact seed → related concept from graph
neighbours → normal*. We implemented that neighbour tier and A/B-swept it — and
**the neighbour tier is counterproductive**. Sweeping seed count × neighbour
count:

| config | plan sat. | subject/action | grammaticality | rhyme | proper-name rate |
|---|---|---|---|---|---|
| 2 seeds + 6 neighbours | 0.130 | 0.675 | 0.710 | 0.800 | 0.050 |
| 2 seeds + 0 neighbours | 0.135 | 0.700 | 0.707 | 0.825 | 0.000 |
| **1 seed + 0 neighbours** | **0.109** | **0.725** | **0.753** | **0.850** | **0.000** |

Forcing only the single *exact* seed is best on nearly every axis. The
neighbour fallback perturbs the walk into name-dense, verb-poor corpus regions
*without* improving plan satisfaction (a neighbour is not the planned concept
the metric counts), so it simultaneously lowers subject/action, lowers
grammaticality, and *raises* proper-name rate. The mechanism from step 3 is
implemented and callable, but the honest result is to default it off. Exact-seed
seeding is the whole win.

**Failure cases (kept, not hidden):**

- **Subject/action dropped 0.79 → 0.73.** Forcing a content noun into the line
  consumes a token slot that might have held the verb. Seeding *what the line is
  about* competes with the line *asserting an action* — the same what-to-say
  vs. how-it-connects tension the previous step showed, now between content and
  predicate. It is a real cost; it is small, and the value stays well above the
  0.45 pre-line-plan baseline.
- **Aggressive seeding (neighbours on) indirectly raised proper-name rate to
  0.05** even though seeds are filtered against the name lexicon — because
  perturbing the walk lands it on name-bearing lines the ranking penalty can't
  always avoid. Fixed by defaulting the neighbour tier off (rate → 0.000), but
  it is a clean example of a change improving its target while quietly
  regressing a distant metric.
- **Plan satisfaction is 0.11, not 1.0.** Seeding only fires when the seed is a
  valid corpus predecessor *somewhere* along a line ending on the required rhyme
  word; for many rhyme/target combinations no such path exists, so the seed
  can't be placed without breaking grammar or rhyme — which the gates forbid.
  0.11 is the honest ceiling of *grammar-respecting* seeding, not a tuning
  failure. Pushing higher would require relaxing a gate the experiment keeps.

Net: intent-seeded generation is the fifth transfer and the first to move
plan-*content* rather than plan-*shape*. It materially hit its target and, at
the exact-seed setting, did so without triggering either risk the brief flagged
— while surfacing that the brief's own neighbour-fallback idea was the wrong
lever.

### Two opposite scaling trends, and why

Coherence keeps climbing with more data; meter and rhyme keep sliding. Both
come from the same cause: a bigger corpus means a bigger, flatter phrase-model
frequency table. That helps **reasoning** (`concept_graph.py` — more edges,
better-informed activation) and hurts **language realization**'s hard
constraints (`generator.py` — the backward/forward walk has more low-frequency
paths competing with the high-frequency one, so it drifts off the syllable
target and off a shared rhyme key slightly more often). This is a real,
mechanism-level trade-off exposed by the scaling test, not a bug: the same
frequency-table traversal that makes meter/rhyme "free" on a small, concentrated
corpus needs a stronger bias term (or a harder syllable-count cutoff) to keep
holding as the table grows. Worth fixing if this experiment continues, left
as-is here since the brief calls for a small, honestly-reported research probe,
not a tuned product.

### Why style separation stays collapsed — and why the answer moved

Per-author line counts are now Пушкин 23840, Блок 10303, Лермонтов 9234,
Ахматова 59, Фет 58, Тютчев 56 — three authors now dominate instead of one.
The signature-vocabulary metric still can't find signal (separation ≈ −0.01,
same as the Lermontov-only run), for the same tf/df-under-imbalance reason.
But notice **which** author now shows a positive gap: Блок (+0.02) instead of
Лермонтов. A plausible reading: Blok's Symbolist, urban-modernist vocabulary
(fonari, apteki — turn-of-the-20th-century imagery) sits further from the
19th-century Romantic register that now dominates the corpus (Pushkin +
Lermontov together are ~76% of all lines), so his signature words stay
distinguishable even diluted, while Lermontov's Romantic vocabulary increasingly
overlaps with Pushkin's now-huge same-era sample and gets absorbed into "the
corpus average" instead of reading as distinctively his. This is a genuinely
interesting side-finding: the metric doesn't fail uniformly — it fails less for
whichever author's vocabulary is furthest, stylistically, from whoever else
dominates the corpus.

### Performance notes from the scaling tests

Dropping in the full Lermontov corpus surfaced a real bug: the stanza-level
co-occurrence pass in `ingest.py` looped over *all pairs* of unique words in a
stanza, and the raw collected edition has long stretches of narrative poems
(Демон, Мцыри) with no blank lines — i.e. pathologically large "stanzas". That
pushed the concept graph to **3.4 million edges** (from ~2,000) and, combined
with `ConceptGraph.neighbors()` doing a linear scan of the edge table on every
call, made a 10-prompt eval run hang indefinitely instead of finishing in
seconds. Fixed by (1) bounding the stanza-level co-occurrence pass to a
sliding window instead of full pairwise, and (2) replacing the linear
`neighbors()` scan with a cached adjacency index.

That fix held up under a second, larger scaling test: adding full Pushkin
(31772 raw lines) and Blok (15157 raw lines) editions pushed the corpus to
43973 verse lines and the concept graph to 2.87 million edges — comparable to
the post-bug Lermontov-only figure despite ~4.6x more source text, confirming
the window bound is doing its job. Ingestion completed in 7.4s (artifact
140MB) and the full eval battery in 2m25s with no hang — slower in proportion
to the data, which is the scaling behaviour a fix like this is supposed to
produce.

## Reverse transfer: fact-bundling back into production QA

Every transfer above ran *production → poetry_lab*. This one ran the other way.
Description mode ("Опиши комнату", "Опиши вечер в Москве") was producing correct
but stunted output — one fact per sentence:

```text
Опиши комнату   →  Комната казалась мрачной. Резная дверь.
Опиши вечер …   →  В Москве был прохладный вечер. Улица тянулась далеко. Изумительный город.
```

The fix was a three-layer **fact bundle**, and it turned out to be a mechanism
production QA lacked, so it was ported *into* `worldpgt/`.

**In poetry_lab.** Three layers, no hand-written vocabulary:

- **facts** (`poemcore/ingest.py`): description relations are extracted
  *clause-locally* (respecting commas — scanning a whole sentence fabricated a
  link across "за город, в рощу"), each tagged with a grammatical role decided
  by Russian morphology, not a word list: `epithet` (agreeing adjective),
  `object_link` (preposition link, "дверь в коридор"), `event_place`
  ("солнце село за строенье"), `property`.
- **reasoning** (`narrative.py::_plan_description_scene`): bundle one primary
  fact + a compatible `epithet` + one `link` about the *same* subject into one
  clause, with explicit rules — a link never becomes the primary fact, at most
  one epithet + one link, no link if the primary already fills the prepositional
  slot.
- **speech** (`_render_description_sentence`): position the bundle only —
  epithet before the subject, link after it.

```text
Опиши комнату   →  Комната казалась мрачной. Через минуту отворилась резная дверь.
Опиши вечер …   →  В Москве был прохладный зимний вечер. Длинная улица тянулась
                    далеко. За заводами кончался изумительный город.
```

**Ported into production (`worldpgt/`).** The same reasoning/speech split, two
landings, both honouring the lab's core discipline — *derive roles from learned
surface, don't hand-list them*:

1. **Learned-fragment fusion class** (`cognition/phrase_graph.py`). Whether two
   adjacent facts coordinate into one sentence ("It is owned by X and is
   headquartered in Y.") is now decided by the *grammatical frame read off each
   fact's learned phrase fragment* (`develops X` → active, `was founded by X` →
   past-passive, `is owned by X` → copular), replacing a hardcoded
   `_FUSIBLE_PREDICATE_BUCKETS` list. A brand-new relation type fuses (or not)
   with zero code edits. This also let three dead predicate frozensets and the
   relative-clause weave's hardcoded list go.
2. **Subject-locative bundle** (`entity_qa/synthesis_engine.py` +
   `phrase_graph.py`). The reasoning layer folds one locative relation into the
   subject noun phrase ("a robotics company **headquartered in Boston**") instead
   of a separate choppy "It is headquartered in Boston." sentence — the direct
   analogue of the lab's `object_link`. The role lives in the facts layer
   (`relation_extraction_v2/types.py::SUBJECT_LOCATIVE_RELATIONS`, mirroring the
   ingest-time `kind` tag); the reasoning layer picks the fold by compatibility
   rules; the speech layer derives the participial surface by stripping the
   learned fragment's leading copula. Never a hand-written locative template.

**Honest limit.** The production overlay currently carries no locative or
second-copular relations, so the bundle is *ready but dormant* — every existing
golden answer (SpaceX, Tesla, Blue Origin, Musk) renders byte-for-byte
unchanged, and the new machinery only activates once the relation extractor
starts emitting `headquartered_in`/`located_in`. Same shape as the lab itself:
the bundling was built before the facts to feed it arrived. Covered by
`worldpgt/tests/test_phrase_graph_relation_fusion_v1.py` and
`test_subject_locative_bundle_v1.py`.

3. **Creative mode — the inverted gate itself**
   (`cognition/creative_generator.py`). The lab's headline finding was that the
   accept/reject gate, not the knowledge source, is what separates factual QA
   from free generation. That gate is now a production layer: a clear creative
   ask ("write a story about…", "imagine…") routes to `creative_request` and a
   token-level generator ported straight from this experiment
   (`poemcore/phrase_model.py`) — order-2 word-transition tables trained on the
   same local prose, seeded deterministic traversal, and the **4-gram novelty
   gate run in reverse polarity**: allow output only when it does *not* recite a
   corpus 4-gram. Factual asks are untouched; every hard-safety screen runs
   first, so a creative framing over private/current material still audits; and
   output is labelled `creative_generated`, never presented as fact. Covered by
   `worldpgt/tests/test_creative_mode_v1.py`. Unlike the two bundling landings,
   this one is *live*, not dormant — it needs no new facts, only the corpus the
   system already ingests.

## Honest conclusion

**Yes, with qualifications.** The same core — typed graph + spreading
activation for reasoning, a learned frequency phrase graph + seeded traversal
for language, JSON artifacts as the layer boundary, and a gate between reasoning
and output — runs unchanged on a poetry corpus and produces structured,
original verse. Two ported functions (`_activate`, `_seeded_weighted_pick`) do
the load-bearing work in both domains without modification. Meter and rhyme,
which QA never needed, emerge cleanly because the language layer was already a
plan-then-traverse renderer, not a template filler.

The limits are equally clear and worth stating:

- **Grammatical fluency needed the QA fragment mechanism, and got it.** A naive
  bigram language layer produced word salad (0.19 real-trigram share). The fix
  was not new machinery but *porting the one thing QA's language layer had that
  this one initially dropped* — multi-word fragment context, realised as an
  order-2 phrase model (0.79 real-trigram share, 4x better). Fluency is now
  acceptable free verse, though still below QA's, because QA also had fixed
  relation frames ("X was founded by Y") that free verse has no equivalent of;
  closing the last gap would need a syntactic constraint
  the QA renderer got for free from its templates.
- **Style separation is weak and sensitive to corpus balance**, not just
  corpus size. It was already thin at 371 balanced lines (+0.02); dropping in
  a single author's full works without rebalancing the others collapsed it
  further (−0.01), because the tf/df-style metric implicitly assumes
  comparable per-author sample sizes. Data-scale and data-balance limit, not
  an architectural one.
- **The inverted gate was necessary.** The architecture transferred, but *only*
  after flipping the support gate. That single change is the honest asterisk on
  "changing only the knowledge source": the knowledge source drove everything
  else, but the accept/reject *polarity* is domain-defining and had to move too.
- **Scaling the corpus 120x (371 → 43973 lines across two rounds) exposed one
  real architectural bug** (unbounded stanza-level co-occurrence — see above),
  not a limit of the approach. Once fixed, ingestion and generation scaled
  proportionally rather than blowing up again at the second, larger round.
- **Reasoning and language scale in opposite directions.** Coherence — the
  reasoning-layer metric — kept improving with more data (0.25 → 0.64 → 0.67).
  Meter and rhyme — the language-layer's hard constraints — gradually degraded
  (89%/0.85 → 84%/0.83 → 78%/0.80) as the phrase-frequency table grew flatter
  and gave the traversal more low-frequency detours. Both effects share one
  cause (a bigger frequency table), which is itself evidence the two layers are
  doing exactly what they're supposed to: activation spreading benefits from
  more graph density without limit, while a frequency-weighted traversal
  chasing a hard target (syllables, rhyme key) needs its bias term strengthened
  as the table grows, or it will drift. A real, fixable, mechanism-level
  finding — not a wall.

So the architecture is genuinely source-agnostic at the mechanism level
(reasoning and language), it scales with corpus size once an
implementation-level bottleneck is fixed (and the reasoning/language halves
scale in opposite directions on quality, for an identifiable reason), and the
**gate policy** — plus, it turns out, per-author corpus balance for any
style-comparison metric — are the components that stay intrinsically tied to
the task and the data. That is itself the most useful result of the
experiment.
