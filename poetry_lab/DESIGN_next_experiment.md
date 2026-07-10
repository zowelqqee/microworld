# Design document — the next mechanism transfer

**Status:** design only. Nothing here is implemented yet.

**Research frame.** We are no longer optimizing poetry. We are testing a
stronger claim: *the same core architecture supports multiple language tasks by
transferring internal mechanisms, not by changing paradigm.* The order-1 →
order-2 phrase-model step already confirmed one transfer (QA's multi-word
fragment context fixed local grammaticality 0.19 → 0.79). This document decides
the **single** next transfer, chosen for sentence-level coherence, maximum
architectural reuse, and minimum new code — no neural nets, no transformers, no
probabilistic language model beyond the existing trigram layer.

---

## Method

I re-read both codebases and inventoried every reusable mechanism that already
exists in production MicroWorld, tagging each with its current status in
`poetry_lab`. Only in-codebase mechanisms are considered; no generic NLP ideas.

Grounding references (production):

- Spreading activation — `worldpgt/cognition/semantic_thought_graph.py::_activate` (lines 330–348).
- Phrase graph traversal — `worldpgt/cognition/phrase_graph.py` (fragments + transitions).
- Explicit semantic state — `worldpgt/entity_qa/semantic_speech_planner.py::SpeechPlan` (persistent `subject`/`reference` + role buckets, lines 44–90).
- Salience ranking over discourse state — `worldpgt/dialogue/salience.py` (`base_salience`/`slot_salience`: integer-breakdown scoring of candidates against `active_topic`, `was_topic`, `recency`, `mention_count`, `sticky_referent`).
- Candidate ranking + self-correction — `worldpgt/cognition/types.py::{AnswerVariant, AnswerVariantEvaluation, AnswerSelectionTrace}` (lines 322–353) and `worldpgt/cognition/self_correction.py::self_correct_answer`.
- Subject/reference threading — `worldpgt/cognition/phrase_graph.py::{_reference,_possessive,_render_capability_run}` (one shared subject across successive clauses).
- Multi-hop path planning — `worldpgt/multihop_qa/path_planner.py`, `relation_graph.py`.
- Typed relation extraction — `worldpgt/relation_extraction_v2/`.

## Mechanism inventory (what is already transferred vs. not)

| # | Mechanism (production) | In `poetry_lab`? | Where |
|---|---|---|---|
| A | Spreading activation | **Ported** | `concept_graph.activate`, used by `planner` |
| B | Phrase-graph traversal | **Ported + extended** | `phrase_model` (order-2), `generator` |
| C | Seeded deterministic pick | **Ported verbatim** | `phrase_model.seeded_weighted_pick` |
| D | Move selection / planning | **Partial** | `planner.plan_poem` picks moves, but shallowly |
| E | Explicit semantic state (`SpeechPlan`, working memory) | **Not ported** | — |
| F | Salience ranking over discourse state | **Not ported** | — |
| G | Candidate ranking + self-correction | **Not ported** | generator accepts *first* valid line |
| H | Subject / reference threading | **Not ported** | each line grown independently |
| I | Typed relation expansion (SVO frames) | **Partial** | only `concept_graph.epithet` (noun→adj) |
| J | Multi-hop path planning | **Not ported** | — |

The unported cluster (E, F, G, H) is exactly the set of mechanisms that, in QA,
operate *above the single clause* — they are why a QA answer reads as one
coherent multi-sentence statement about one subject rather than a pile of
locally-grammatical fragments. That is the precise gap between "local
grammaticality 0.79" (fixed) and "sentence-level coherence" (open).

---

## Candidate analysis

Only the mechanisms that plausibly move **sentence/cross-line coherence** are
analyzed in full (D, E, F, G, H, I, J). A, B, C are already in place.

### E. Explicit semantic state (discourse state across lines)

1. **Why it helped QA.** `SpeechPlan` holds a persistent `subject` and buckets
   every supported fact by semantic role (activity/origin/ownership/…). Every
   rendered sentence draws from, and refers back to, this one state — that is
   what makes the answer *about* one entity.
2. **Why it might help poetry.** Poetry currently has `PoemPlan`, but it is a
   *static* line-by-line schedule with a rotating `focus`; there is no state
   that *evolves* as lines are produced. A running discourse state (which
   images are already in play, how recently, how often) lets later lines build
   on earlier ones instead of each line re-seeding from scratch.
3. **Complexity.** Low-medium: a small mutable `DiscourseState` (a handful of
   counters), a direct structural port of `dialogue/state.py::EntityActivation`.
4. **Expected gain.** Medium on its own — state is only useful if something
   *reads* it during generation (that is F/H).
5. **Risks.** Inert if not wired into selection; adds a field nobody uses.

### F. Salience ranking over the discourse state

1. **Why it helped QA.** `salience.py` scores each candidate referent by an
   integer breakdown over the discourse state (`active_topic`, `was_topic`,
   `recency`, `mention_count`, `sticky_referent`) and picks the highest. The
   breakdown *is* the proof of the choice — deterministic, inspectable.
2. **Why it might help poetry.** The generator already produces multiple
   candidate lines per slot (the 8-attempt retry loop) but throws all but the
   first valid one away. Scoring those candidates by continuity with the
   discourse state — how strongly a candidate's content words connect, in the
   concept graph, to the currently-active images — and keeping the best turns
   an unused byproduct into cross-line coherence.
3. **Complexity.** Low: the candidate loop exists; change "return first valid"
   to "collect valid, score, argmax". Scoring reuses `concept_graph.neighbors`.
4. **Expected gain.** **High** — directly optimizes line-to-line topical
   continuity, the core of sentence-level coherence, at near-zero code.
5. **Risks.** Over-biasing toward continuity could flatten variety (mitigated
   by keeping the seeded pick inside each candidate and scoring only the
   selection *between* candidates); must not fight meter/rhyme (scoring runs
   only over candidates that already passed those gates).

### G. Candidate ranking + self-correction (variant evaluation)

1. **Why it helped QA.** QA rendered several `AnswerVariant`s, scored each with
   `self_correct_answer` (critique + bounded repair) into an
   `AnswerVariantEvaluation`, and selected via `AnswerSelectionTrace`.
2. **Why it might help poetry.** Same shape as F — F is essentially this
   mechanism specialized to a continuity score. Self-correction (dedupe,
   repair) could additionally catch repeated images across a stanza.
3. **Complexity.** Medium: the ranking half overlaps F; the self-correction
   half needs poetry-specific critique rules (new code).
4. **Expected gain.** Medium — the ranking part is F; the repair part is a
   smaller, separate win.
5. **Risks.** Scope creep — bundling repair rules violates "minimum new code".

### H. Subject / reference threading (consistent voice)

1. **Why it helped QA.** `_reference`/`_possessive` thread one grammatical
   subject ("It … It …") across clauses, so successive sentences cohere around
   one referent.
2. **Why it might help poetry.** A consistent lyric subject/voice ("я"/"он"/a
   recurring image) across lines would read as one poem spoken by one voice.
3. **Complexity.** Medium-high in Russian: pronoun choice needs gender/number
   agreement and case, which the corpus does not hand us cleanly.
4. **Expected gain.** Medium — improves the *feel* of one voice, narrower than
   topical continuity.
5. **Risks.** Agreement errors are conspicuous and would *lower* perceived
   fluency; high effort for a narrow slice.

### I. Typed relation expansion (propositional lines)

1. **Why it helped QA.** Typed subject–predicate–object relations let the
   renderer state a proposition ("X develops Y"), the unit of meaning.
2. **Why it might help poetry.** Building a line around a learned SVO frame
   (subject-verb-object) rather than a rhyme-anchored walk could make lines
   *assert* something coherent.
3. **Complexity.** **High.** Requires learning typed relations from figurative,
   inverted, elliptical poetic syntax — the opposite of the clean encyclopedic
   sentences `relation_extraction_v2` was built for.
4. **Expected gain.** High ceiling, low confidence — poetic syntax resists
   reliable SVO extraction.
5. **Risks.** Large new subsystem; violates "minimum new code"; likely noisy.

### J. Multi-hop path planning (associative stanza logic)

1. **Why it helped QA.** `path_planner` chains relations A→B→C to answer
   multi-hop questions with an explicit reasoning path.
2. **Why it might help poetry.** A stanza built as a walk through the concept
   graph (image A leads to associated B leads to C) has associative logic.
3. **Complexity.** Medium — reuses `concept_graph`, but needs a path→line
   binding layer.
4. **Expected gain.** Medium — associative drift can read as coherent *or* as
   wandering; hard to control without a selection score (which is F again).
5. **Risks.** Without a ranking gate it can worsen coherence; depends on F.

### D. Deeper move planning

1/2. Already partially present; enriching move→content binding helps, but only
in combination with a state to bind against (E) and a selector (F).
3. Low. 4. Low-medium alone. 5. Marginal without E/F.

---

## Ranking by expected impact on sentence-level coherence

| Rank | Candidate | Reuse | New code | Coherence gain | Verdict |
|---|---|---|---|---|---|
| **1** | **F — salience-ranked candidate selection over discourse state (E+F+A)** | **Very high** | **Very low** | **High** | **Chosen** |
| 2 | I — typed relation / SVO lines | Medium | High | High (low confidence) | Rejected: too much new code, noisy on poetic syntax |
| 3 | H — subject/reference threading | Medium | Medium-high | Medium | Rejected: Russian agreement cost, narrower gain |
| 4 | J — multi-hop path stanza | Medium | Medium | Medium | Rejected: needs F anyway to not wander |
| 5 | G — full variant + self-correction | Medium | Medium | Medium | Rejected: ranking half ⊂ F; repair half is scope creep |
| 6 | D — deeper move planning | High | Low | Low alone | Deferred: only useful once F exists |

---

## Decision: transfer **salience-ranked candidate selection over an explicit discourse state**

This single change fuses three production mechanisms — **spreading activation
(A)**, **explicit semantic state (E)**, and **salience ranking (F)** — into the
generator, and it is the only candidate that satisfies every constraint:

- **Maximum architectural reuse.** Scoring is a port of `dialogue/salience.py`'s
  integer-breakdown selection; the state is a port of
  `dialogue/state.py::EntityActivation`; the candidate pool already exists in
  the generator's retry loop; the continuity signal reuses
  `concept_graph.neighbors`/`activate`; determinism reuses
  `seeded_weighted_pick`. Four production mechanisms, one small seam.
- **Minimum new code.** The generator already *produces* the candidates
  (currently 8 per line) and discards all but the first valid one. The change is
  to stop discarding: collect the valid candidates, score each against the
  discourse state, keep the argmax, update the state. Estimated ~50–70 lines
  (a `DiscourseState` dataclass + a `line_salience` function + a 4-line change
  to the accept loop).
- **No neural / transformer / new LM.** Scoring is integer/float arithmetic over
  the existing concept graph — exactly the `salience.py` pattern, which is
  itself pure integer scoring with a transparent breakdown.
- **Improves sentence-level, not surface, coherence.** It optimizes *line-to-
  line topical continuity* (does this line develop the images already in play?),
  which is orthogonal to meter and rhyme — those are already gated *before*
  scoring, so they cannot regress.

### Design sketch (not implemented)

```
DiscourseState                        # port of EntityActivation, per poem
  active:   dict[concept -> float]    # currently-salient images
  history:  Counter[concept]          # mention counts (repetition control)
  age:      dict[concept -> int]      # lines since last use (recency)

line_salience(candidate, state, graph) -> (score, breakdown)   # port of base_salience
  + ACTIVE_LINK    for each content word graph-adjacent to an active concept
  + FOCUS_MATCH    if the planned focus concept appears / is reachable
  - REPETITION     penalty for a word already used too many times
  (breakdown is a list[(name, points)] — the proof of the score, as in salience.py)

generator._render_line:                # the only wiring change
  candidates = [c for attempt in range(8) if passes_meter_rhyme_novelty(c)]
  best = argmax(line_salience(c, state, graph)[0] for c in candidates)   # was: candidates[0]
  state.update(best)                   # decay ages, bump active/history
  return best
```

The chosen line's concepts flow into the state, raising the salience of their
neighbours for the *next* line — spreading activation now operating across the
whole poem, not just inside one planning step. That is the mechanism that, in
QA, makes an answer cohere as discourse.

### How we would know it worked

Local grammaticality (order-2's metric) will **not** move — it measures within-
line spans. Sentence-level coherence needs a new metric, itself a reuse of the
same graph: **inter-line continuity** = the average concept-graph connectedness
between consecutive lines' content words (share of adjacent-line word pairs that
are graph neighbours or within two hops). Baseline it on the current order-2
generator, then compare. Secondary check: meter/rhyme/novelty must stay flat
(the gates guarantee it) — if they move, the wiring leaked past the gates.

### What this deliberately does **not** do

No SVO extraction (I), no pronoun agreement (H), no repair rules (G), no path
planner (J). Each is a separate, larger experiment; bundling any of them would
break "minimum new code" and blur the result. The point of this experiment is a
clean, isolated test of one claim: **does porting QA's discourse-state salience
selection, unchanged in mechanism, raise cross-line coherence in a completely
different language task?**
