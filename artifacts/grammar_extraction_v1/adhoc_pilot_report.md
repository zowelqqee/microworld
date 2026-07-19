# Ad-hoc pilot report — methodology / citation / topic predicates on arXiv

**Status:** staged, not committed. Proposal-only run: reads stored records,
writes nothing to accepted / promoted / serving memory.

**Question tested:** does arXiv text actually contain methodology / citation /
topic relations that the narrow extractor simply never looked for — or is the
low yield caused by something deeper than a narrow predicate vocabulary?

**Answer (short):** On this material, broadening the vocabulary did **not**
unlock new relations. The bottleneck is **not** vocabulary narrowness. It is
two structural walls that a wider vocabulary does not move: (1) these relations
are overwhelmingly expressed in non-SPO constructions (participial / discourse
adjuncts) that yield no atomic subject→predicate→object triple, and (2) the few
triples that do form have generic subjects and abstract objects that the
unchanged entity-quality gate rejects — the same wall that produced the earlier
25/25-all-duplicate, 0-new result.

---

## 1. What was changed (ad-hoc, hardcoded — no YAML)

All changes are additive edits in the **same hardcoded-dict style** already used
for the existing 12 verbs. No declarative engine, no refactor, no validator/gate
*logic* change.

- `relation_extraction_v2/spacy_extractor.py`
  - 12 new lemmas in `_VERB_RELATIONS`: `train/fit/calibrate → trained_on`,
    `base/derive/adapt → based_on`, `extend/build/improve → extends`,
    `concern/address/focus → about`.
  - Metadata rows in `_RELATION_META` for the 4 new predicates.
  - `_PILOT_PREP_VERBS` map + `_prep_pobjs()` helper: routes prep-object verbs
    ("trained **on**", "based **on**", "builds **on**", "focuses **on**"),
    mirroring the pre-existing headquarter `prep("in")` special case. Direct-
    object verbs (`extend/concern/address`) ride the existing active nsubj+dobj
    path; the dobj path is disabled for prep-only verbs so "X builds rockets"
    does not misfire as `extends`.
  - One copula branch for "X is about Y".
- `relation_extraction_v2/types.py` — 4 strings added to
  `ALLOWED_SEMI_STABLE_RELATIONS` (`trained_on`, `based_on`, `extends`, `about`).
- `relation_extraction_v2/relation_candidate_validator.py` — `trained_on`,
  `based_on`, `extends` added to `_NON_PERSON_OBJ_RELATIONS` (data only;
  `about` excluded on purpose — "the paper is about Newton" is a legitimate
  person-object topic relation).

These are exactly the data edits `design.md` predicted as the only gate-side
change. **No firewall/validator logic was touched.**

Smoke test (unit sentences) confirms all four families extract correctly and the
existing patterns (`founded_by`, `founded`, `headquartered_in`) are unchanged;
the negative control "SpaceX builds rockets" correctly yields nothing.

## 2. Test setup — same arXiv material

Ran the pilot-extended extractor over the **same stored arXiv source records**
the previous arXiv analysis used (loaded via the existing
`run_arxiv_source_specific_lane_v1` helpers). No new crawl.

- 83 stored arXiv records (title + abstract text), 689 sentences.
- Candidates then passed through the **unchanged** `relation_candidate_validator`.
- Run twice for robustness: once with an empty entity index, once with the real
  production `EntitySurfaceIndex` (158 known surfaces). **Identical outcome.**

## 3. Results

| Predicate | Raw triples formed | Passed precision gate | Net-new independent groups |
|---|---:|---:|---:|
| `trained_on` | 2 | 0 | 0 |
| `based_on` | 0 | 0 | 0 |
| `extends` | 1 | 0 | 0 |
| `about` | 4 | 0 | 0 |
| **Pilot total** | **7** | **0** | **0** |
| (context) all predicates incl. existing | 13 | 0 | — |

- **7 raw pilot triples from 689 sentences.** For comparison, the whole
  extractor (including the existing `founded`/`produces` patterns) formed only
  13 triples total on this material and **0** of any type passed the gate.
- **0 passed the precision gate.** All 7 pilot candidates quarantined
  `missing_explicit_evidence` (generic subject / non-atomic object).
- **0 net-new independent predicate groups.** Same headline as the prior narrow
  run — but now it is clear this is not because the vocabulary was too small.
- Robustness: empty index and real 158-surface index both give 0 passed.

## 4. Root cause — why a wider vocabulary did not help

### 4.1 The relations are mostly expressed in non-SPO syntax

Direct evidence from the text: **"based on" occurs 19 times, yielding 0
triples.** Parsing all 19: every one is `acl` / `prep` / `amod` — a noun-phrase
post-modifier or a sentence-initial discourse frame, never the clean "X is based
on Y" predication:

- *"a novel architecture **based on** Quantum AI…"* → `acl` modifier of
  "architecture" (no `nsubjpass`).
- *"**Based on** these results, we construct…"* → `prep` adverbial (no subject
  at all).
- *"tests **based on** order statistics…"* → `acl` modifier of "tests".

There is simply **no atomic subject→predicate→object triple to extract** in
these forms. The relation is real in the prose, but it lives on a noun phrase or
a discourse connective, not as a predication between two named entities. Cue
frequency vs. extractable triples across the corpus:

| Cue | Occurrences | Clean SPO triples |
|---|---:|---:|
| `based on` | 19 | 0 |
| `focus on` | 5 | 1 |
| `addresses` | 3 | 2 |
| `trained on` | 3 | 2 |
| `extends` | 2 | 0 |
| `improves on` | 1 | 1 |
| `builds upon` | 1 | 0 |

A broader predicate *vocabulary* cannot fix a *construction* mismatch. Catching
"a model based on X" would require new `acl`-attachment parsing patterns — more
extractor logic — and would still hit wall 4.2.

### 4.2 The triples that do form have generic subjects / abstract objects

The 7 pilot triples that did form were all quarantined because their endpoints
are not clean atomic entities — the identical entity-quality discipline that
capped the original narrow lane. Manual spot-check (author read of all 7):

| Subject → pred → object | Evidence (abbrev.) | Verdict |
|---|---|---|
| Artificial intelligence → trained_on → ideal bulk crystals | "AI and ML **models** … are trained on ideal bulk crystals" | Relation correct; subject is a mis-sliced conjunct of "AI and ML models" — generic. Correctly quarantined. |
| machine learning → trained_on → ideal bulk crystals | (same sentence, other conjunct) | Same. Correctly quarantined. |
| REGAI → extends → performance | "REGAI **improves on** the performance of …" | Object "performance" is abstract, not a cited work; "improves on performance" is not a citation. Correctly quarantined. |
| study → about → crucial need | "This **study addresses** the crucial need for fairness…" | Generic subject, abstract object. Correctly quarantined. |
| Existing AML threat evaluation approaches → about → technical attack robustness | "… **focus** primarily **on** technical attack robustness" | 5-word descriptive subject, abstract object. Correctly quarantined. |
| proposed framework → about → fundamental challenges | "the proposed framework **addresses** fundamental challenges…" | Generic subject, abstract object. Correctly quarantined. |
| CryptoBLL → about → tension | "CryptoBLL **addresses** this tension…" | Subject is a clean name, but object "this tension" is anaphoric/abstract. Correctly quarantined. |

Every quarantine is correct. Notice the pattern: even when a subject is a clean
named method (`CryptoBLL`), the object is an abstract common noun. arXiv
abstracts describe *what a method does to concepts*, not *relations between two
named entities* — which is exactly what an atomic knowledge-graph edge needs.

## 5. Honest comparison to `design.md`'s break-even

`design.md` framed the decision as: the declarative engine is worth ~8–9.5 days
if arXiv genuinely contains hidden predicate diversity that a wider vocabulary
would unlock. **This pilot is evidence that, for this material, it does not.**

- The optimistic hypothesis — "the text has these relations, the narrow
  extractor just didn't look" — is **not supported** here. When we looked with a
  4-family-wider vocabulary, we found 7 triples and 0 gate-passes.
- Therefore the break-even argument for the full declarative build gets **weaker**,
  not stronger: a YAML engine over the same text would face the identical two
  walls (non-SPO constructions, generic/abstract endpoints). More predicate
  types do not help when the material does not contain clean, named-entity
  triples of those types.

This is the "yield stays low even with new types → the problem is deeper than
vocabulary" outcome the task asked to record honestly. It is a **negative result
that saves the 8–9.5-day investment** from being made on a false premise.

## 6. Important caveats (do not over-generalize)

- **This corpus is topically diverse, not ML-heavy.** It spans education,
  materials science, statistics, physics — so ML-methodology cues like
  "trained on" are naturally sparse (3 occurrences). An ML-paper-heavy corpus
  would show more `trained_on` / `based_on` surface forms. But wall 4.2 (generic
  "the model" subjects, abstract objects) would still apply, so higher cue
  frequency would not automatically translate into gate-passing triples.
- **The finding is about *extractable atomic SPO relations*, not about whether
  the information exists.** The relations are present in the prose; they are
  just not in the shape a clean entity→predicate→entity graph edge requires.
- The pilot proves the *mechanism* works (unit smoke test extracts all four
  families correctly). The zero yield is a property of the **material**, not a
  bug in the patterns.

## 7. Recommendation

1. **Do not build the declarative YAML engine on the strength of "arXiv has
   hidden diversity."** This pilot does not support that premise for the
   available arXiv material.
2. If methodology/citation/topic relations are still wanted, the higher-leverage
   next probes are about the two real walls, not about vocabulary:
   - Test whether an **ML-paper-heavy** slice yields extractable `trained_on` /
     `based_on` triples with *named* datasets/models as objects (this attacks
     wall 4.1 by picking material where the constructions and named endpoints are
     denser).
   - Decide whether the entity-quality gate *should* admit abstract objects for
     topic-style predicates (`about`) — that is a gate-policy question, and
     loosening it is a precision risk to weigh deliberately, not a vocabulary
     change.
3. Keep these pilot edits **staged for review**. They are harmless (additive,
   proposal-only, existing patterns unchanged) but should not be committed until
   there is a decision, given the zero yield on current material.

## 8. Artifacts

- `artifacts/grammar_extraction_v1/adhoc_pilot_summary.json` — machine-readable
  counts.
- `artifacts/grammar_extraction_v1/adhoc_pilot_spotcheck.json` — the 7 raw
  candidates with evidence spans and gate verdicts.
- Staged code edits: `spacy_extractor.py`, `types.py`,
  `relation_candidate_validator.py` (see §1). Runner kept in scratchpad
  (not committed).
