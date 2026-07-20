# Batch-baseline diagnosis v1

## Scope

Compared completed manual-review ledgers only:

| Batch | Candidates | ACCEPT | Rate |
|---|---:|---:|---:|
| `arxiv_grouped_120_20260720` | 148 | 36 | 24.3% |
| `arxiv_holdout_74_20260720` | 81 | 37 | 45.7% |

The extraction prompt, grouping heuristic, node-quality filter, precision gate, and serving-memory paths were unchanged.

## Source and candidate characteristics

| Characteristic | Grouped | Holdout | Interpretation |
|---|---:|---:|---|
| Source sentences | 120 | 74 | Grouped uses 69 records; holdout uses 74 distinct records. |
| Mean sentence tokens | 21.0 | 27.1 | Holdout is 29% longer. |
| Mean sentence characters | 152.5 | 186.5 | Same direction. |
| Raw triples per source sentence | 1.59 | 1.36 | Holdout did not yield more triples. |
| Triaged candidates per source sentence | 1.23 | 1.09 | Holdout did not retain more candidates. |
| Mean named tokens per candidate | 1.13 | 1.10 | Essentially unchanged. |
| Candidates with >=2 named tokens | 30/148 (20.3%) | 22/81 (27.2%) | Higher, but insufficient alone to explain +21.4pp. |

### arXiv categories / topics

The artifacts retain titles, URLs, and source sentences, but not category metadata for each selected source. The holdout manifest records one official API query labelled `computer science`; the grouped batch comes from a stored mixed quarantine pool. No category-level causal claim is made without another metadata acquisition.

## Group composition versus within-group quality

| Primary group | Grouped candidates / accepts / rate | Holdout candidates / accepts / rate |
|---|---:|---:|
| Anaphora | 15 / 0 / 0.0% | 3 / 0 / 0.0% |
| Temporal | 1 / 0 / 0.0% | 5 / 5 / 100.0% |
| Generic-property | 51 / 13 / 25.5% | 36 / 21 / 58.3% |
| Attachment | 1 / 0 / 0.0% | 0 / 0 / n/a |
| Clean/no-flag | 80 / 23 / 28.8% | 37 / 11 / 29.7% |

Clean/no-flag is stable (28.8% versus 29.7%). The difference is concentrated in generic-property (25.5% versus 58.3%) plus five accepted temporal class/member relations.

Applying grouped group-specific rates to the holdout group counts predicts only **24.5%** holdout acceptance. Thus group mix alone does not explain the observed 45.7%; the within-group generic-property change is the main driver.

## Sampling uncertainty

| Batch | Rate | Wilson 95% CI |
|---|---:|---:|
| Grouped | 24.3% | 18.1%–31.8% |
| Holdout | 45.7% | 35.3%–56.5% |

The intervals do not overlap. Two-sided Fisher exact test on `[[36,112],[37,44]]`: **p = 0.00113**. Two-proportion z = 3.32. Under an independent candidate-binomial model, the 21.4-point difference is unlikely to be sampling noise alone.

This independence model is approximate: candidates cluster within source sentences, source pools were selected differently, and these are not random samples from one common arXiv population. Therefore this is evidence of a real difference between these batch constructions, not a claim about all arXiv material.

## Diagnosis

**Material-characteristic difference is the better current explanation, with selection uncertainty retained.** Holdout material did not produce more raw or triaged candidates and had no higher overall named-token density. Instead, it produced far more manually legitimate relations inside generic-property, while clean/no-flag stayed stable.

Operationally, prioritize material with canonical technical concepts, class/member statements, explicit mathematical terminology, and named systems, including candidates initially labelled generic-property. This is not yet a population-wide source-quality ranking; another independently acquired disjoint batch should replicate it.

## Guardrail

The Gemini prompt remains unchanged. Nothing was auto-admitted or promoted; the precision gate and serving memory remain unchanged, and no commit was created.
