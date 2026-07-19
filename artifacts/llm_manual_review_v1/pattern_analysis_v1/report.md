# Retrospective pattern analysis v1

## Scope and guardrail

This is a retrospective analysis of the completed grouped Gemini manual-review
batch only:

- input candidates: `runs/arxiv_grouped_120_20260720/manual_review_candidates.json`
- human labels: `runs/arxiv_grouped_120_20260720/manual_review_decisions.json`
- population: 148 candidates, 36 ACCEPT and 112 REJECT (24.3% global baseline)
- comparison populations: `generic_property_likely` (13/51 ACCEPT, 25.5%) and
  `clean_no_flag` (23/80 ACCEPT, 28.8%)

No new extraction was run. `node_quality_filter.py`, the precision gate, all
proposal selections, and serving memory were left unchanged.

## Feature extraction method

Features were evaluated for every one of the 148 rows, with comparisons below
restricted to the two requested groups.

| Feature | Explicit definition |
|---|---|
| named token | Capitalized alphabetic token, excluding closed-class/common words such as `The`, `This`, `It`, `And`, and prepositions. |
| endpoint type | `both_named`, `subject_named_object_descriptive`, `object_named_subject_descriptive`, or `both_descriptive`, based on named-token presence. |
| acronym | `2+` uppercase letters (optionally with digits/hyphens), e.g. `LLMs`, `RAG`, `DS-KG`. |
| numeric | Any digit in either endpoint. |
| technical surface marker | At least one acronym, digit, hyphenated token, or a transparent technical lexical marker. Used for the comparison only; it is **not** the proposed rule. |
| sentence length | Whitespace-token count of the stored source sentence. |
| predicate class | Exact extracted predicate surface; the operational comparison separates `uses`, `supports`, and copula/state-adjacent forms (`is`, `has`, `shows`, `provides`, `enables`). |

The stored batch contains only each selected sentence, not the full abstract or
the sentence ordinal within it. Therefore **beginning/middle/end of abstract is
not available** and is intentionally not imputed.

There is no shared, closed predicate schema in this LLM manual-review lane to
which the free-form extracted predicates can truthfully be mapped. The analysis
therefore uses exact predicate surfaces, rather than claiming a nonexistent
schema classification.

## Distribution comparison

Percentages in the first two columns are within that group’s human ACCEPT or
REJECT subset; they do not measure a feature's standalone precision.

### Generic-property-likely (13 ACCEPT, 38 REJECT)

| Feature present | ACCEPT | REJECT | Difference / reading |
|---|---:|---:|---|
| subject has named token | 11/13 (84.6%) | 32/38 (84.2%) | No signal. |
| object has named token | 5/13 (38.5%) | 2/38 (5.3%) | Clear positive signal. |
| both endpoints named | 3/13 (23.1%) | 2/38 (5.3%) | Positive, but sparse. |
| both endpoints descriptive | 0/13 (0.0%) | 6/38 (15.8%) | Clear negative signal. |
| any technical surface marker | 11/13 (84.6%) | 15/38 (39.5%) | Clear positive signal, but too broad alone. |
| predicate = `supports` | 6/13 (46.2%) | 5/38 (13.2%) | Strong concentration among accepts. |
| copula/state-adjacent predicate | 4/13 (30.8%) | 23/38 (60.5%) | Negative signal, not a sufficient rejection rule. |
| source sentence <=20 tokens | 10/13 (76.9%) | 28/38 (73.7%) | No useful signal. |

Mean named tokens across both endpoints: **1.85 ACCEPT vs 0.97 REJECT**.
Mean sentence length: **17.1 vs 19.2** tokens, not a material distinction.

### Clean/no-flag (23 ACCEPT, 57 REJECT)

| Feature present | ACCEPT | REJECT | Difference / reading |
|---|---:|---:|---|
| subject has named token | 23/23 (100.0%) | 41/57 (71.9%) | Necessary-looking signal in this sample, but insufficient alone. |
| object has named token | 11/23 (47.8%) | 13/57 (22.8%) | Positive signal. |
| both endpoints named | 11/23 (47.8%) | 10/57 (17.5%) | Positive, but excludes many valid named-to-descriptive relations. |
| both endpoints descriptive | 0/23 (0.0%) | 13/57 (22.8%) | Clear negative signal. |
| any technical surface marker | 20/23 (87.0%) | 30/57 (52.6%) | Positive but insufficient alone. |
| predicate = `uses` | 20/23 (87.0%) | 10/57 (17.5%) | Strongest broad signal. |
| source sentence <=20 tokens | 5/23 (21.7%) | 33/57 (57.9%) | Short sentences skew rejected here. |

Mean named tokens across both endpoints: **1.70 ACCEPT vs 1.05 REJECT**.
Mean sentence length: **26.3 vs 20.5** tokens. Length is observational only;
it is not proposed as a decision feature.

## Proposed narrow heuristics

These are **review-prioritization / prospective-validation hypotheses**, not
automated admission and not changes to the existing node-quality filter.

1. **H1 — technical `uses` shape (clean/no-flag only).** Select a candidate
   only when its primary group is `clean_no_flag`, its exact predicate is
   `uses`, and its object has at least one purely structural technical marker:
   a capitalized non-common token, an acronym, a digit, or a hyphenated token.
   This avoids a hand-built domain vocabulary.
2. **H2 — enumerated `supports` shape (generic-property only).** Select a
   candidate only when its primary group is `generic_property_likely`, its
   exact predicate is `supports`, and its literal source sentence contains
   `tasks such as` or `applications such as`. This is deliberately narrow:
   it requires an explicit enumerative construction rather than treating
   `supports` itself as reliable.

H1 targets the observed `uses` concentration without accepting generic objects
such as `rubrics` or `legal language`. H2 captures the five accepted LogAI and
bi-temporal-imagery rows, while avoiding the rejected generic `supports` rows.

## Retroactive validation on the same labelled batch

| Rule | Selected | TP (manual ACCEPT) | FP (manual REJECT) | FN | Precision among selected | Recall of all 36 ACCEPT |
|---|---:|---:|---:|---:|---:|---:|
| H1 | 15 | 14 | 1 | 22 | 93.3% | 38.9% |
| H2 | 5 | 5 | 0 | 31 | 100.0% | 13.9% |
| H1 OR H2 | 20 | 19 | 1 | 17 | **95.0%** | 52.8% |

The sole H1 false positive is `arxiv-023:0` (`SUMER uses techniques from
semi-supervised learning`): the hyphenated object meets the structural marker
but was manually rejected as an overly broad/generic technique phrase.

For the union rule over all 148 candidates: TN=111 and ordinary accuracy is
130/148 (87.8%), but that number is dominated by the 112 rejections. Precision
among selected is the relevant metric here: **95.0%**, far above the 24.3%
global baseline and both in-group baselines (25.5% generic, 28.8% clean).

## Verdict

**RETROSPECTIVE PASS, PROSPECTIVE UNCONFIRMED.** The two very narrow rules
show a real retrospective precision improvement (95.0%, 19/20) rather than a
small fluctuation around the 25–29% group baselines. They also deliberately
leave 17 known accepts for manual review, preserving a conservative scope.

This is still one labelled batch and the same data was used to discover and
measure the patterns. It is evidence to pre-register a future holdout/manual
review validation, **not** evidence to auto-admit, deploy, modify
`node_quality_filter.py`, run the precision gate, or promote any relation.
