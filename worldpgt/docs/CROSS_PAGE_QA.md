# Cross-page Entity QA

Cross-page Entity QA v1 is controlled graph-style multi-hop QA over the isolated
50-page wiki overlay. It tests whether the system can use explicit relation
paths across pages while preserving audit behavior for unsupported paths,
weak-link promotion attempts, and volatile facts.

The cross-page path is:

```text
isolated wiki overlay
-> WikiMemoryOverlayProvider
-> CrossPageQuestionAnalyzer
-> CrossPageAnswerPlanner
-> CrossPageAnswerRenderer
-> CrossPageAnswerValidator / audit
-> deterministic benchmark summary
```

## Benchmark Status

| Metric | Value |
|---|---:|
| prompts | 71 |
| answers | 50 |
| audits | 21 |
| correct_count | 71 |
| wrong_count | 0 |
| answer_precision | 1.0 |
| quality_flagged | 0 |
| relation_edges_used | 35 |
| weak_context_links_used | 31 |
| source_facts_used | 36 |
| safe_for_general_runtime | false |

## Policy Behavior

Cross-page QA answers only when the supporting path is explicit enough for the
requested claim. It can also render a caveated answer for a weak contextual link
when the question asks for link/context behavior rather than a stable fact.

Examples:

- Musk -> SpaceX -> rockets answers because the supporting relation path is
  present.
- Musk -> Starlink audits when there is no explicit stable path.
- SpaceX -> Starlink returns a weak contextual link caveat rather than a stable
  factual relation.

## What It Does Not Do

Cross-page QA does not:

- infer missing links from surface similarity
- promote weak contextual links to stable facts
- convert source-qualified volatile facts into current facts
- answer live/current questions from stale or missing evidence
- use embeddings, neural weights, a GPT renderer, or network calls

The benchmark is a deterministic regression gate over a small isolated overlay,
not evidence of open-domain graph reasoning.
