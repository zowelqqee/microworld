# Pre-registered holdout validation v1

## Status: COMPLETE HUMAN REVIEW; NAMED-TOKEN VALIDATION COMPLETE

`pre_registered_rules.md` was written before selecting, inspecting,
extracting, grouping, or reviewing holdout material. H1/H2 were applied
unchanged to the resulting candidates before any human verdict exists.

## Immutable rules

- **H1:** `primary_review_group == clean_no_flag` AND `predicate == "uses"`
  AND object contains a capitalized non-common token, acronym, digit, or
  hyphenated token.
- **H2:** `primary_review_group == generic_property_likely` AND
  `predicate == "supports"` AND source sentence contains `tasks such as` OR
  `applications such as` (case-insensitive).

See `pre_registered_rules.md` for the full pre-registered wording.

## Holdout composition and isolation

| Field | Result |
|---|---|
| holdout batch | `runs/arxiv_holdout_74_20260720` |
| official source | arXiv API, one bounded `computer science` query |
| API records returned | 100; no acquisition errors |
| selected sentences | 74 (all available disjoint cue-bearing records; within requested 60–100 range) |
| sentence cap | 1 per source record |
| distinct holdout source records | 74 |
| overlap with prior grouped 120 source records | **0** |
| Gemini model | `gemini-3.1-flash-lite` |
| request spacing | 5 seconds (12 RPM maximum) |
| raw triples | 101 |
| candidates after unchanged node-quality triage | 81 |
| node-quality exclusions | 20 |

The prior 40-sentence batch is a subset of the prior grouped batch's source
set, and all 14 distinct source records behind the 15-sentence pilot map into
that grouped source set. Therefore the zero overlap above covers all three
previous batches by source record ID.

The initial target was 80 sentences. The official query supplied 74 disjoint
cue-bearing records; all 74 were used rather than substituting overlapping or
lower-quality material. This selection constraint did not alter H1/H2.

## Pre-review grouping and mechanical H1/H2 result

| Primary group | Candidates |
|---|---:|
| Anaphora-likely | 3 |
| Temporal-referent-likely | 5 |
| Generic-property-likely | 36 |
| Attachment-shape-flag | 0 |
| Clean / no-flag | 37 |

| Pre-registered rule | Matched candidates |
|---|---:|
| H1 | 0 |
| H2 | 0 |
| H1 OR H2 | 0 |

The match lists and the exact pre-review application are retained in
`pre_review_rule_matches.json`. This is a valid null-selection outcome: there
are no selected candidates for which holdout precision can be measured.

## Human-review completion

All 81 candidates received a supplied human decision: **37 ACCEPT / 44
REJECT (45.7% accept rate)**. The completed decision encoding is
`runs/arxiv_holdout_74_20260720/manual_review_decisions.json`.

H1/H2 remain a null-selection result (0 matches each), so their holdout
precision is undefined. They were not relaxed or widened post hoc.

Nothing was auto-admitted or promoted; the precision gate and serving memory
were not touched, and no commit was created.

## Broader named-token-density validation

The null H1/H2 selection does not test the broader retrospective observation
that accepted candidates carried more named tokens across their endpoints.
Accordingly, `named_token_density_pre_registration.md` freezes one independent
predictor before review: `named_token_count >= 2` across subject and object.

| Named-token count | Holdout candidates |
|---|---:|
| 0 | 35 |
| 1 | 24 |
| 2 | 9 |
| 3 | 6 |
| 4 | 6 |
| 5 | 1 |

The holdout-wide pre-review mean was 1.10 named tokens; 22/81 candidates
(27.2%) met the frozen `>=2` threshold. The threshold and token definition
were not changed before processing the manual ledger.

| Scope | Retrospective ACCEPT / REJECT means | Holdout ACCEPT / REJECT means | Holdout gap |
|---|---|---|---:|
| All 81 candidates | n/a (original analysis was group-stratified) | **1.54 / 0.73** | **+0.81** |
| Generic-property | 1.85 / 0.97 | **1.81 / 1.27** | **+0.54** |
| Clean/no-flag | 1.70 / 1.05 | **1.36 / 0.38** | **+0.98** |

The direction holds in both comparable groups: accepted candidates have more
named tokens than rejected candidates. The generic-property gap is smaller
than retrospective (+0.54 vs +0.88), while the clean/no-flag gap is larger
(+0.98 vs +0.65). Thus this is a replicated directional signal, not an exact
reproduction of the prior effect size.

### Frozen `named_token_count >= 2` rule

| Metric | Result |
|---|---:|
| selected | 22 |
| TP / FP / FN / TN | 15 / 7 / 22 / 37 |
| precision among selected | **68.2%** |
| recall of manual accepts | 40.5% |
| holdout baseline accept rate | 45.7% |
| odds ratio | 3.60 |
| one-sided Fisher exact p-value | 0.0126 |

The broad threshold therefore improves precision by 22.5 percentage points
over the local holdout baseline, with a statistically noticeable association
in this sample. It deliberately misses 22 manual accepts, so it is a
conservative **auto-priority** signal only.

## Verdict

**BROADER UNDERLYING PATTERN HOLDS ON HOLDOUT.** Named-token density
generalizes as a broad review-prioritization predictor independent of the
narrow H1/H2 formulations that did not fire. This validates further
prospective use only as a label/priority cue: it does **not** authorize
automatic admission, a `node_quality_filter.py` modification, precision-gate
execution, or serving-memory promotion.
