# Grammar-based relation extraction — design v1

**Status:** design review only. No code in this iteration. Decision to implement
comes after the plan and the trade-off estimate (§5) are reviewed.

**Scope guard (read first).** This is *not* unsupervised relation discovery. The
engine does not invent predicate types from text statistics. It gives a
systematic, extensible way for a **human to declare** a grammatical pattern for
a new predicate type — faster and more auditable than today's ad-hoc per-source
regex — but every new predicate still requires an explicit, human-authored
pattern definition. The only thing that changes is *how* the definition is
expressed: a declarative pattern over the dependency-parse tree, instead of
hardcoded string matching or a hardcoded Python `dict`.

**Zero neural inference.** spaCy's `en_core_web_sm` parser is used strictly as a
*syntactic structure provider* — POS tags and the typed dependency tree. It
never decides *which predicate*; that decision is always an explicit,
deterministic rule over the parse structure. spaCy is already an optional
dependency of `relation_extraction_v2` (see `spacy_extractor.py`), lazily loaded,
and used only in the offline pump/ingestion path — never in the stdlib-only
answer path. This design keeps that boundary: the grammar engine is
ingestion-side only.

---

## 0. What already exists (honest baseline)

This is the single most important framing for the effort estimate. The codebase
already contains ~70–80% of the infrastructure this task describes. The design
is mostly **externalization + generalization**, not greenfield construction.

| Capability the task asks for | Where it already lives | Gap |
| --- | --- | --- |
| Dependency-parse SPO extraction (nsubj / dobj / nsubjpass / agent / prep / conj) | `relation_extraction_v2/spacy_extractor.py` | Works, but verb→predicate map is a **hardcoded Python `dict`** (`_VERB_RELATIONS`, 12 verbs) plus two hardcoded copula branches (`leader_of`, `is_a`). Not declarative; adding a predicate = editing Python. |
| Verb-lemma classes → relation, with active/passive/inversion | `_VERB_RELATIONS` in `spacy_extractor.py` | Same — hardcoded, not a config file. |
| Structural low-confidence reject / quarantine | `relation_candidate_validator.py` | Already implements: max phrase words (7), clean-proper-noun check, fragment-prefix screen (`_FRAGMENT_PREFIXES`), truncated-name check, junk/common-word screen, directionality & entity-type conflict, generic-entity quarantine → `QUARANTINE_REASONS`. Reusable almost verbatim. |
| Evidence citation with exact span | `RelationExtractionEvidence` (`types.py`) + `evidence_span` (source-specific extractor) | Sentence + surfaces + sentence/paragraph index exist; **char offsets not yet captured**. |
| Margin-gated abstain discipline | `knowledge/predicate_centroid_index.py` `find_predicate()` | The *pattern* to imitate: accept only if `best ≥ cutoff AND best − runner_up ≥ margin`, else abstain. Not currently applied to extraction pattern selection. |
| Narrow per-source extractors (the thing being replaced) | `knowledge_pump/source_specific_relation_extractors.py` | `_PREDICATE_CUES` = 6 hardcoded cue types, substring-between-subject-and-object matching, per-lane (`arxiv`/`crossref`/`openalex`). This is the ad-hoc surface the design supersedes. |

**Diagnosed gap that motivates the work** (`artifacts/open_book_qa/extractors/arxiv_predicate_analysis.md`):
the arXiv lane, as implemented, realizes only 5 predicate types
(`uses` 37, `enables` 30, `supports` 12, `provides` 4, `located_in` 3),
dominated by `uses`. The analysis concludes there are *"no extracted
methodology, findings, citation, or topic predicates as distinct schema
types"* and that surfacing them "would be new extraction work." This design is
that work, done once as declarative infrastructure rather than a sixth cue
bolted onto the narrow lane.

---

## 1. Architecture

### 1.1 Pipeline

```
ReadySnapshotDoc / source record text
        │
        ▼  sentence_splitter (existing)
   sentences
        │
        ▼  spaCy parse (existing, lazy, offline)   ── syntactic structure only
   parse trees
        │
        ▼  GRAMMAR ENGINE  (new: declarative pattern interpreter)
   raw candidates  + evidence spans + structural-confidence label
        │
        ▼  structural confidence gate (new: margin-gated abstain, §1.4)
   surviving candidates
        │
        ▼  relation_candidate_validator (existing, +4 predicate rows)
   safe  /  quarantine
        │
        ▼  precision_firewall (existing, unchanged mechanics — §4)
   proposal candidates (requires_review=True, never auto-promoted)
```

The new component is exactly the "GRAMMAR ENGINE" box: a **declarative pattern
interpreter** that replaces the hardcoded `_VERB_RELATIONS` dict and the two
hardcoded copula branches with a config-driven evaluator over the same parse
tree. Everything upstream and downstream is reused.

### 1.2 Pattern definition format (part a)

Patterns live in a human-editable YAML file
(`worldpgt/relation_extraction_v2/patterns/*.yaml`), one entry per grammatical
construction. A pattern is a declarative selector over the dependency tree, not
Python.

**Grammar primitives** a pattern may reference (the fixed vocabulary the
interpreter understands — this is the only Python that new predicates do *not*
require touching):

- `trigger.lemma`: a class of verb (or copula/noun) lemmas, e.g.
  `[train, pretrain, fine-tune, finetune]`.
- `trigger.pos`: `VERB` | `AUX`/copula | `NOUN` (for nominal patterns).
- `trigger.voice`: `active` (verb has `nsubj`) | `passive` (verb has
  `nsubjpass`) | `copula` (root `be` + `attr`). Determined structurally, not by
  keyword.
- `role` selectors, resolved against the trigger token's children:
  - `nsubj`, `nsubjpass`, `dobj`, `attr`
  - `agent:pobj` — the `pobj` under an `agent` dependency ("…by Y").
  - `prep:<text>:pobj` — the `pobj` under a `prep` child whose surface is
    `<text>` (e.g. `prep:on:pobj` for "trained **on** Y", `prep:on:pobj` /
    `prep:upon:pobj` for "based on Y").
  - `attr.prep:<text>:pobj` — object reached through the copula attribute
    ("is the CEO **of** Y" → `attr=CEO`, `prep:of:pobj=Y`).
- `span`: how the entity surface for a role is built. Default reuses
  `_span_text()` (left_edge→head NP, determiner-aware). `conj_expand: true`
  reuses `_conj_chain()` so "X and Y" fan out to two candidates.
- `emit`: `subject → predicate → object`, with `invert: true` when the canonical
  fact reads from the object's perspective (mirrors `invert_for_active` in the
  current dict, e.g. `own`, `publish`).
- Metadata carried onto the candidate: `confidence`, `stability`, `risk`
  (same three-label vocabulary as `types.py`).
- `guards`: per-pattern structural rejection knobs (§1.4).

**Example pattern entries** (proof-of-concept pilots, §2):

```yaml
# "BERT was trained on BookCorpus"  → BERT trained_on BookCorpus
- id: trained_on_passive
  predicate: trained_on
  trigger: { lemma: [train, pretrain, pre-train], pos: VERB, voice: passive }
  subject: { role: nsubjpass }
  object:  { role: prep:on:pobj }
  emit:    { subject: subject, object: object }
  confidence: high
  stability: semi_stable
  risk: low
  guards: { max_object_tokens: 6, reject_nonreferential_subject: true }

# "Our method is based on transformers"  → method based_on transformers
- id: based_on_copula
  predicate: based_on
  trigger: { lemma: [base], pos: VERB, voice: passive }
  subject: { role: nsubjpass }
  object:  { role: prep:on:pobj, alt: prep:upon:pobj }
  emit:    { subject: subject, object: object }
  confidence: high
  stability: semi_stable
  risk: low
  guards: { max_object_tokens: 6, reject_nonreferential_subject: true }

# "This work extends ResNet" / "builds on ResNet"  → work extends ResNet
- id: extends_active
  predicate: extends
  trigger: { lemma: [extend], pos: VERB, voice: active }
  subject: { role: nsubj }
  object:  { role: dobj }
  emit:    { subject: subject, object: object }
  confidence: medium        # citation verbs are noisier — see §1.4
  stability: semi_stable
  risk: low
  guards: { max_object_tokens: 6, reject_nonreferential_subject: true }

- id: builds_on_active
  predicate: builds_on
  trigger: { lemma: [build], pos: VERB, voice: active }
  subject: { role: nsubj }
  object:  { role: prep:on:pobj, alt: prep:upon:pobj }
  emit:    { subject: subject, object: object }
  confidence: medium
  stability: semi_stable
  risk: low
  guards: { max_object_tokens: 6, reject_nonreferential_subject: true }

# "The paper is about diffusion models"  → paper about diffusion models
- id: about_copula
  predicate: about
  trigger: { lemma: [be], pos: AUX, voice: copula }
  subject: { role: nsubj }
  object:  { role: attr.prep:about:pobj }     # "is about Y"
  emit:    { subject: subject, object: object }
  confidence: medium
  stability: semi_stable
  risk: low
  guards: { max_object_tokens: 6, reject_nonreferential_subject: true }

# "This study concerns protein folding"  → study concerns protein folding
- id: concerns_active
  predicate: about                 # normalize concerns→about at emit time
  trigger: { lemma: [concern, address, investigate, study], pos: VERB, voice: active }
  subject: { role: nsubj }
  object:  { role: dobj }
  emit:    { subject: subject, object: object }
  confidence: medium
  stability: semi_stable
  risk: low
  guards: { max_object_tokens: 6, reject_nonreferential_subject: true }
```

The interpreter loads all entries, indexes them by `(trigger.lemma, voice)`, and
for each trigger token in the parse tree evaluates every matching pattern. The
existing 12 hardcoded verbs migrate to this format unchanged in behavior (a
regression fixture pins parity — §3).

### 1.3 Pattern ordering & specificity

Patterns are ordered most-specific-first, exactly as the regex `PATTERNS` list
is today. Specificity is a declared integer `specificity` (default derived from
role-path length: more constrained role paths score higher). Ordering matters
for the ambiguity gate below.

### 1.4 Confidence / audit mechanism (part b)

Same discipline as the centroid fallback: **when the structure matches but with
incomplete certainty, abstain (quarantine), do not guess.** Concrete structural
signals of low confidence, each mapped to an existing or one new
`QUARANTINE_REASONS` entry:

1. **Oversized / non-atomic span.** Subject or object span longer than
   `guards.max_object_tokens` (default 6) *and* not resolving to a clean
   named-entity / proper-noun boundary → `missing_explicit_evidence`. This is
   the entity-prefilter logic already in the validator (`_MAX_PHRASE_WORDS`,
   `_is_clean_entity_phrase`) and in the source-specific
   `_atomic_graph_endpoints` (>6 words, clause markers). Reused, not reinvented.

2. **Nonreferential / discourse / anaphoric subject.** Subject head is a
   demonstrative or pronoun (`this`, `such`, `it`, `these`…) or begins with a
   discourse lead → reject. Reuses `_too_generic`, `_FRAGMENT_PREFIXES`, and
   `_NONREFERENTIAL_SUBJECT_LEADS`. (Directly targets the Round-1 spot-check
   failures in `arxiv_spot_check.md`: "Such unusual magnetotransport…",
   "Disk chemistry also…", "One implementation…".)

3. **Ambiguous prepositional attachment.** If the object's governing `prep`
   could attach to more than one head (e.g. the parser gives the `prep` to a
   noun in the object NP rather than the trigger verb, or two verbs compete),
   the attachment is ambiguous → new reason `ambiguous_attachment`. Detected
   deterministically: the `prep` token's `.head` must be the trigger token (or
   its `attr`); otherwise abstain.

4. **Coordinated / adversative object clause.** Object subtree contains a `cc`
   + `conj` spanning a second clause, or a `mark`/`advcl` introducing
   `but`/`which`/`that`/`including` → reject (`_OBJECT_CLAUSE_MARKERS`, already
   in the source-specific extractor). Targets Round-2 rejects like "…supports →
   different levels of description, **and** has …".

5. **Pattern-tie ambiguity (the direct margin-gate analogue).** If two patterns
   fire on the *same trigger token* with the *same specificity* but *different
   predicates*, that is a near-tie with no separating margin → abstain
   (`ambiguous_relation`). This mirrors `find_predicate`'s
   `best − runner_up ≥ margin` rule: a higher-specificity pattern "wins" (clear
   margin); equal specificity with disagreement is an abstain, never a coin
   flip. Concretely: `specificity(top) − specificity(runner_up) ≥ 1` to accept,
   else quarantine.

Signals 1–4 are per-candidate structural guards; signal 5 is the
selection-level margin gate. All abstain outcomes are quarantined with an
evidence span attached, never silently dropped — so a human can review whether
the pattern needs refinement.

### 1.5 Evidence citation (part c)

Every emitted candidate carries the existing `RelationExtractionEvidence`
(sentence, `pattern_id`, subject/object surface, sentence & paragraph index),
plus **two new fields** so a precision gate can verify *literal* correspondence:

- `subject_char_span: (start, end)` and `object_char_span: (start, end)` —
  character offsets into `evidence_sentence`, taken from the spaCy token
  `.idx`/`.idx+len`. This lets the gate assert that the stored subject/object
  strings are byte-for-byte substrings at the cited offsets, not paraphrases.
- `trigger_lemma` and `trigger_span` — the verb/copula token that fired, so an
  auditor can see *why* the pattern matched.

`evidence_span` (the full proving sentence) is preserved exactly as the
source-specific extractor already does. No candidate is emitted without a
sentence that literally contains subject, trigger, and object in the parsed
order.

### 1.6 Extensibility path (part d)

Adding a new predicate type is a five-step, mostly-declarative procedure:

1. **Author** writes one or more YAML pattern entries (no Python) in a
   `patterns/<predicate>.yaml` file, with `owner` and a one-line rationale.
2. **Register** the predicate in the allowlist: add the string to
   `ALLOWED_SEMI_STABLE_RELATIONS` (or stable/volatile) in `types.py`, and, if
   the predicate has directionality constraints, add rows to the validator
   tables (`_ORG_SUBJ_RELATIONS`, `_NON_PERSON_OBJ_RELATIONS`, …). These are
   one-line data edits, not new code paths.
3. **Spot-check** with the seeded protocol (§3.1) — the same discipline applied
   to every existing source lane (`arxiv_spot_check.md`).
4. **Gate**: a pattern only enters the production set after a Round-2 spot-check
   with 0 incorrect emitted relations. The spot-check artifact
   (`spot_check_<predicate>.md`) is committed alongside the pattern.
5. **Version**: the pattern file carries a `pattern_set_version`; the extraction
   report records which version produced each candidate, so a bad pattern can be
   traced and rolled back.

---

## 2. Pilot predicate types (part / §2)

Four pilot predicates, chosen because the arXiv retrospective explicitly
diagnosed them as **structurally absent** from the current lane, not merely
under-sampled:

| Predicate | Family | Why this one | Example trigger |
| --- | --- | --- | --- |
| `trained_on` | methodology | Ubiquitous in ML abstracts ("trained on ImageNet"); the narrow lane has no methodology predicate at all. | `train`/`pretrain` passive + `on` |
| `based_on` | methodology | Same family; captures "based on / builds upon" method provenance. | `base` passive + `on`/`upon` |
| `extends` / `builds_on` | citation | The analysis found *zero* citation predicates; these are the most literal, lowest-ambiguity citation cues. | `extend` active dobj; `build` + `on` |
| `about` / `concerns` | topic | No topic predicate exists; "is about / concerns / studies X" is a clean copula/verb construction. | copula `be` + `about`; `concern`/`study` active dobj |

These map one-to-one onto the four families the retrospective named as missing:
*methodology, citation, topic* (findings is deferred — it tends to require
multi-clause spans that the atomic-span guard would mostly reject, so it is not
a good proof-of-concept first target).

Deliberately excluded from iteration 1: `findings`/`reports` (non-atomic
objects), any predicate needing cross-sentence coreference, and any volatile
predicate (the pilots are all `semi_stable`, low/medium risk).

---

## 3. Test plan (described, not executed)

### 3.1 Spot-check protocol (per new predicate type)

Identical shape to `arxiv_spot_check.md`:

- Candidate pool: seeded sample of sentences from stored arXiv records that
  contain the trigger lemma. No network fetch.
- Round 1: seed `S1`, 10 candidates. Manual verdict on each: does the structural
  match yield the correct semantic relation with atomic endpoints?
- **Stop condition:** ≥2 extraction errors in Round 1 → pattern is revised
  (tighten guards), Round 1 re-run on a fresh seed. (This is exactly what
  happened to the arXiv lane: Round 1 had 5/10 errors, guards were added, Round
  2 passed.)
- Round 2: seed `S2`, 10 fresh candidates (excluding Round-1 items). **0
  incorrect emitted relations required** to authorize the pattern into
  production. Conservative rejects (quarantines) are expected and count as
  correct.
- Target: 10 examples × 4 predicates = 40 manually verified structural→semantic
  matches, plus their rejects.

### 3.2 Ablation — diversity, not volume

Run the new grammar engine and the old `source_specific_relation_extractors`
arXiv lane on the **same** arXiv quarantine material (the 85/86-row pool the
retrospective already characterized). Report:

- **Predicate-type diversity**: distinct predicate count and the distribution
  (the old lane realizes 5 types, 89% `uses`/`enables`). Success criterion is
  that the new engine surfaces methodology/citation/topic predicates the narrow
  lane *structurally cannot* produce — i.e. new *kinds*, not just more `uses`.
- **Precision on the shared sample**: manual verdict on a seeded subsample, so
  "more diversity" is not bought with more false positives.
- Explicitly *not* a volume race: a new engine that produced 3× as many `uses`
  rows would be a failure by this test.

### 3.3 Precision-gate compatibility

The candidate output shape already matches what the gate consumes — the
source-specific extractor emits `v2_pattern_id`, `evidence_span`,
`confidence_label`, `requires_review=True`, `safe_for_general_runtime=False`, and
the grammar engine will emit the same envelope. So the gate **mechanics** run
unchanged. What is *not* free and must be stated plainly:

- The new predicate strings are absent from `ALLOWED_RELATIONS` (`types.py`);
  the validator will quarantine them as `ambiguous_relation` until added. → 4
  one-line data edits.
- The validator's directionality/entity-type tables (`_ORG_SUBJ_RELATIONS`,
  `_NON_PERSON_SUBJ_RELATIONS`, `_NON_PERSON_OBJ_RELATIONS`) do not know the new
  predicates. `trained_on`/`based_on`/`about` want a non-person object; a few
  table rows encode that. → small data edits, no logic change.
- No firewall *logic* change is anticipated. This must be **confirmed** by
  running the pilot candidates through the existing firewall in dry-run and
  checking the quarantine reasons are the expected structural ones — that
  confirmation is part of iteration 1, not assumed.

### 3.4 Regression parity

A fixture pins that the 12 migrated hardcoded verbs (`found`, `develop`,
`manufacture`, `own`, `publish`, `lead`, `headquarter`, …) produce byte-identical
candidates before and after migration to the declarative format. The refactor is
not allowed to change existing extraction behavior.

---

## 4. Precision-gate note (explicit)

Existing gate logic works **without modification** on the new predicate types,
with two data-only prerequisites (allowlist rows + directionality rows, §3.3).
No new gate code is expected. The one open verification item is §3.3's dry-run
confirmation. If the dry-run surfaces a predicate whose objects legitimately
exceed the atomic-span guard (e.g. some `about <multi-word topic>` cases), the
resolution is to tune that pattern's `max_object_tokens`, **not** to weaken the
gate.

---

## 5. Honest effort estimate & trade-off (part / §4)

### 5.1 Iteration-1 breakdown (infrastructure + 4 pilots)

| Task | Estimate |
| --- | --- |
| Pattern schema + YAML loader + tree interpreter (generalize the hardcoded `spacy_extractor` branches into the primitive vocabulary of §1.2) | 2–3 days |
| Migrate the 12 existing verbs + 2 copula branches to declarative form, with parity fixture (§3.4) | 1 day |
| 4 pilot patterns + guard tuning | 1 day |
| Structural confidence gate: `ambiguous_attachment` detector + pattern-tie margin gate (§1.4 signals 3 & 5); signals 1,2,4 are reuse | 1 day |
| Char-offset evidence fields (§1.5) | 0.5 day |
| Spot-check harness automation + 4× spot-check runs (§3.1) | 1–1.5 days |
| Validator/allowlist wiring for 4 predicates (§3.3) | 0.5 day |
| Ablation experiment + report (§3.2) + gate dry-run (§3.3) | 1 day |
| **Total** | **~8–9.5 working days** (≈2 weeks with review/iteration) |

This is genuinely lower than a greenfield build *because* §0 is true — the
parse-tree walking, the role selectors, the entire quarantine validator, the
evidence dataclass, and the margin-gate reference implementation already exist.
Most of the estimate is externalizing and generalizing, plus the disciplined
spot-check work that any new predicate needs regardless of approach.

### 5.2 The trade-off, stated plainly

**Ad-hoc cost (status quo):** adding *one* new predicate the current way — a new
`_PREDICATE_CUES` entry or a new regex `PatternSpec` + its spot-check — is
roughly **0.5–1 day**.

**Break-even:** the declarative engine is a ~8–9.5-day fixed investment. Against
a ~0.5–1-day marginal ad-hoc cost, break-even is somewhere around **8–14
predicate types**. Below that count, ad-hoc is cheaper in raw hours. Above it,
the declarative set wins, and the gap widens because each ad-hoc addition also
adds an independent code path to maintain, test, and reason about.

**Why the investment is still the right call even before break-even:**

- *Consistency of audit.* Today, arXiv, Crossref, and OpenAlex each have their
  own subtly different subject/object screens. One declarative engine means one
  set of structural guards, one quarantine vocabulary, one place to fix a
  false-positive class for *all* predicates at once.
- *Diversity ceiling.* The narrow lanes are structurally capped at their cue
  lists (arXiv realizes 5 types, 89% `uses`/`enables`). The declarative engine
  removes that ceiling for the cost of a YAML entry, which is the whole point of
  the diagnosed gap.
- *Provenance & rollback.* Versioned pattern sets + per-candidate pattern
  provenance make a bad pattern traceable and reversible; ad-hoc regex edits are
  not individually versioned.

**Honest counter-argument (so the decision is informed):**

- If the realistic near-term need is *only* 3–4 more predicate types ever, the
  ad-hoc path is cheaper and this engine is over-engineering.
- spaCy is a heavier dependency than the regex lanes. It stays strictly
  offline/ingestion-side (never the stdlib-only answer path), so this is
  acceptable — but it is a real added dependency surface for the pump
  environment.
- Parser errors become a new failure mode. A mis-parse produces a wrong role
  assignment; the §1.4 guards catch most, but not all. The spot-check gate is
  the backstop, and it is manual labor per predicate either way.

**Recommendation:** worth doing *if* the roadmap credibly needs the
methodology/citation/topic families (the diagnosed gap suggests it does) and
more than a handful of predicate types over time. If the real need is a single
next predicate, add it ad-hoc and revisit this design when the third or fourth
new type appears.

---

## 6. Explicitly out of scope for iteration 1

Multi-clause `findings` objects; cross-sentence coreference resolution;
volatile-predicate patterns; automatic pattern *induction* from examples (this
would cross into the discovery territory the scope guard forbids); production
routing / auto-promotion (all output stays `requires_review=True`,
`safe_for_general_runtime=False`); non-English grammar.
