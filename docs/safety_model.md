# Safety Model

This document collects Microworld's safety and support policy. The central rule
is that unsupported claims must not become answers merely because they are
plausible, fluent, nearby in context, or present in a low-trust artifact.

## Core Policy

Microworld prefers an honest gap over a plausible unsupported answer.

Safety gates block:

- current/live facts without an accepted source-qualified snapshot
- private or sensitive data
- relation inversion
- unsupported universal claims
- weak context promoted to fact
- volatile facts used as stable/current truth
- current-sensitive predicates used as multi-hop bridges
- unsupported entity categories
- ambiguous dialogue references

Temporal policy:

| Class | Meaning |
|---|---|
| `historical` | Stable once verified, e.g. founding events. |
| `semi_stable` | Can change, but usually not minute-to-minute, e.g. ownership or products. |
| `snapshot` | Source-qualified dated value, e.g. net worth estimate. |
| `volatile` | Current-sensitive, must be hedged or audited. |

Optional relation-label similarity exists for parser fallback, but there is no
embedding-based answer generation and no neural renderer.

## Memory Boundary Rules

Proposal artifacts are useful, but they are not silently promoted into trusted
semantic memory.

| Bucket | Artifact | Safety meaning |
|---|---|---|
| Accepted memory | `worldpgt/experiments/accepted_knowledge_memory_v1.json` | Trusted explicit semantic memory. |
| Accepted wiki overlay | `worldpgt/experiments/accepted_wiki_memory_overlay_v1.json` | Isolated wiki semantic-memory overlay. |
| Promoted overlay | `worldpgt/experiments/self_ingestion_v1/promotion/promoted_wiki_memory_overlay_v1.json` | Separate promoted artifact, not accepted memory. |
| Pump dry-run overlay | `worldpgt/experiments/knowledge_pump_v1/pump_dry_run_overlay.json` | Proposal overlay for runtime experiments. |
| Weak context | weak links inside overlays | Contextual association only, never a stable fact. |
| Ontology layer | `wikidata_p279_ontology_layer.json` | Read-only `is_a` traversal support. |
| Dialogue state | `DialogueState` turn records | Session reference state, not factual memory. |
| Community context | Reddit/HN-style artifacts | Speech and cognitive patterns only; no factual support. |
| Live search | web-search result/cache | Volatile source-labelled answer path, never accepted memory. |

## Community Context Boundary

The community layer is deliberately not factual memory. It can shape how an
answer is explained, but it cannot make a factual claim true.

```text
community context may shape how an answer is explained
community context may not make a factual claim true
```

Current community/speech-pattern snapshot, from
`worldpgt/experiments/community_context_v1/reddit_community_summary.json`:

| Artifact | Count / status |
|---|---:|
| Local Reddit/Hacker News-like input records | 371 |
| Accepted community-context items | 371 |
| Cognitive pattern events | 428 |
| Quarantined in final context build | 0 |
| Factual support allowed from community layer | false |
| Accepted/promoted/snapshot overlays modified | false |

The 428 cognitive pattern events are behavior/style patterns, not facts:
`analogy_pattern` 99, `explanation_pattern` 87, `mistake_pattern` 74,
`style_tone_pattern` 69, `question_pattern` 46, `procedure_pattern` 27,
`uncertainty_pattern` 25, and `debugging_pattern` 1.

## Live Search Boundary

Microworld's optional live-search path is intentionally separate from memory:

```text
current/live question
  -> safety route
  -> optional web search provider
  -> answer extraction / relevance filter
  -> rendered with "live web search, volatile" disclosure
  -> never promoted into accepted memory
```

The latest saved WebQuestions-style open benchmark remains experimental:

```text
external_20260706T203034Z.json
total_questions: 250
answer_rate:     42.0%
audit_rate:      58.0%
precision:       28.57% among answered rows
elapsed:         1878.23s
```

The honest status is that live search exists, is safer than a generic fallback,
and is improving, but it is not yet a strong open-domain inference result.

## Known Safety Limits

These limits are part of the safety model:

- Microworld is a bounded semantic-memory runtime, not general intelligence or
  open-domain general inference.
- Coverage is artifact-bound: the system answers only where accepted memory,
  proposal overlays, source-qualified snapshots, or labelled live-search
  results provide explicit support.
- Entity identity is still surface/alias based. The ontology layer can provide
  read-only traversal, but there is no QID-native identity layer yet.
- Cross-sentence extraction coreference remains conservative; dialogue context
  resolves session references, not extraction-time references.
- Current/live facts audit unless they come from a dated source-qualified
  snapshot or the optional volatile live-search path.
- Live web search is optional, labelled volatile, and weak on the saved
  WebQuestions-style evaluation: 28.57% precision among answered rows.
- Weak context and Reddit/HN community context can shape speech or cognitive
  patterns, but cannot support factual claims.
- The language renderer remains bounded and deterministic; cleaner wording does
  not increase factual coverage.

## Conclusion

Microworld's safety model is an artifact-bound support model. It treats memory
source, temporal class, dialogue resolution, community context, and live search
as separate trust domains so the runtime can answer narrowly and audit honestly.
