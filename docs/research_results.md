# Research Results

This document preserves the historical research tracks, demonstrated results,
limitations, next work, and current status for the trimmed Microworld runtime.
It keeps the research framing conservative: this is an experimental semantic AI
runtime, not a claim of general intelligence or general neural-model
replacement.

## Research Tracks Preserved From Earlier Work

The repository also contains earlier Microworld research tracks around:

- ConceptNet-derived relation prediction
- pattern discovery
- transitive and mixed-pattern reasoning
- relation trust
- audit-driven trust learning
- feedback compression
- suppression policy
- name/surname generation
- GPT-2 comparison reports
- risk/coverage benchmarks

Those results remain part of the project history. The current README foregrounds
the `worldpgt` explicit-memory runtime because that is the active path.

## Demonstrated Results

Demonstrated in the current repository and preserved research artifacts:

- Audit-driven trust learning
- Trust transfer on unseen data
- Feedback compression (1598x)
- Deterministic speech renderer
- 1,000-question deterministic benchmark
- 100% honest-gap behavior on the stress suite
- Dialogue context and conservative coreference
- Controlled language generation over explicit support
- Local runtime latency (8.05ms p50 on the speech stress suite)
- Scalable indexed semantic retrieval
- Multi-hop explicit semantic reasoning
- Speech/reasoning surface measured separately from factual coverage
- Reproducible open-book QA comparison with raw evidence spans and a local MLX Qwen baseline
- Failure analysis that isolates parser coverage, resolver coverage, and planner reachability
- Reddit/HN-style cognitive pattern memory that is blocked from factual support
- Optional live-search path with volatile/source-labelled answers

Not demonstrated:

- Open-domain general inference
- General intelligence
- Neural model replacement
- Open-domain live-search precision competitive with modern LLM search tools

## Current Limits

Current limits, from the local artifacts:

- Microworld is a bounded semantic-memory runtime, not general intelligence or
  open-domain general inference.
- Coverage is artifact-bound: the system answers only where accepted memory,
  proposal overlays, source-qualified snapshots, or labelled live-search
  results provide explicit support.
- Extraction recall is limited by deterministic patterns, optional spaCy, and
  the current frontier. The pump currently exposes a 2,836-item answerable-fact
  delta, while the dedicated pump fact QA artifact still covers 570 facts.
- Re-extracting existing Wikipedia snapshots fixed duplicate detection against
  the existing overlay via `_drop_duplicates_of_existing_overlay`, but it does
  not add extraction support for founding dates or mission goals phrased like
  "founded in 2002 with the goal of ...".
- Re-extraction plus manual cleanup promoted clean additional rows into
  `promoted_wiki_memory_overlay_v1.json`; the current file contains 363 items
  and the pre-reextract backup remains beside it as
  `promoted_wiki_memory_overlay_v1.json.backup_before_reextract`.
- Entity identity is still surface/alias based. The ontology layer can provide
  read-only traversal, but there is no QID-native identity layer yet.
- Cross-sentence extraction coreference remains conservative; dialogue context
  resolves session references, not extraction-time references.
- Current/live facts audit unless they come from a dated source-qualified
  snapshot or the optional volatile live-search path.
- Live web search is optional, labelled volatile, and weak on the saved
  WebQuestions-style evaluation: 28.57% precision among answered rows.
- Pump dry-run overlays and weak-context outputs are proposal or experimental
  artifacts. Promotion readiness can pass, but promotion remains an explicit
  artifact and review step; accepted memory is not silently mutated.
- Weak context and Reddit/HN community context can shape speech or cognitive
  patterns, but cannot support factual claims.
- The 1,000-question speech benchmark measures deterministic surface quality
  over controlled categories, not broad conversational generalization.
- The language renderer remains bounded and deterministic; cleaner wording does
  not increase factual coverage.
- The current open-book failure analysis found a hard multi-evidence boundary:
  all 50 measured multi-evidence cases failed entity resolution before planner
  invocation. This is resolver/surface-index coverage, not evidence-plan
  rendering or partial-credit evaluation.
- Paraphrase coverage is incomplete: the measured 42% score includes predicate
  mapping gaps (`make possible`, `provide`, `used by`) and unresolved relation
  subjects. The open-book dataset also exposed eight deictic subjects that
  should have been excluded from a stable-entity benchmark.
- Durable unattended/night-cycle service is not part of the current runtime;
  acquisition and evaluation loops are script-driven.

These limits are not footnotes. They are part of the safety model.

## Next Work

Highest-leverage next steps:

1. Add concrete Starlink-like mechanism evidence roles (`uses`, `works_by`,
   `used_for`) and verify that questions such as `How does Starlink work?` move
   from answer-with-gap to mechanism answer only when supported.
2. Refresh pump fact QA so it covers the current 2,836-item answerable-fact
   delta instead of the older 570-fact QA artifact.
3. Tighten extraction precision around noisy explanatory predicates before
   widening mechanism/purpose extraction further.
4. Improve live-search precision and source relevance before presenting it as a
   serious open-domain result.
5. Add QID-native identity and canonical entity handling to reduce alias and
   homonym collisions.
6. Continue the dialogue-v2 migration from shadow/benchmark validation toward
   serving-path rollout while preserving byte-identical single-turn behavior.
7. Connect community cognitive pattern events more directly to semantic
   reasoning moves while keeping `factual_support_allowed=false`.
8. Add a repeated latency/stability artifact that records workload,
   median/min/max, and environment, beyond single saved benchmark snapshots.
9. Keep promotion explicit: proposal -> fact checks -> review -> promoted
   artifact; never silent accepted-memory mutation.
10. Add deterministic paraphrase predicate mappings and an experimental
    relation-subject resolver fallback, then rerun the same fixed-seed
    open-book dataset before claiming improvement.

## Status

Experimental and local. The active path is a bounded semantic-memory runtime
with deterministic planning, explicit support checks, dialogue state, and
controlled rendering. It is useful where the current artifacts contain support;
outside that boundary it should audit or label volatile sources. The research
target is inspectability of memory, trust, policy, dialogue, and rendering, not
open-ended language-model generality.

## Conclusion

The project history matters because it shows how the current runtime emerged:
from explicit relation experiments, trust learning, audit-driven correction,
and benchmarked comparisons into a semantic-first runtime with controlled
memory, reasoning, dialogue, and rendering boundaries.
