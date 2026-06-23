# Current Implementation Audit

This document records what the current local worldpgt checkout actually
contains, where the requested project description matches the code, and where
the implementation is still incomplete or differs from older README snapshots.

Scope: docs-only audit of the local checkout on 2026-06-21. No runtime code,
tests, accepted memory, accepted overlays, promoted overlays, or pump artifacts
were intentionally regenerated for this document.

## Verified Components

| Component | Current code / artifact |
|---|---|
| Assistant surface | `worldpgt/assistant_surface/`, with CLI entrypoint `worldpgt/experiments/ask_microworld_v1.py` |
| Semantic parser | `worldpgt/entity_qa/semantic_question_parser.py`, returning `SemanticQuery` |
| Multi-hop QA | `worldpgt/multihop_qa/`, including current-sensitive path validation |
| Query engine | `worldpgt/query_engine/`, with `Find`, `Filter`, `Count`, `Compare`, `Traverse`, `Classify`, `Aggregate` |
| Dialogue context | `worldpgt/dialogue/`, in-memory `ConversationContext` and coreference resolver |
| Web/API surface | `worldpgt/api/server.py`, FastAPI `GET /`, `GET /health`, `POST /ask` |
| Knowledge Pump | `worldpgt/knowledge_pump/` plus artifacts under `worldpgt/experiments/knowledge_pump_v1/` |
| Optional spaCy extraction | `worldpgt/relation_extraction_v2/spacy_extractor.py`, enabled by `--enable-spacy` |
| Relation policy | `worldpgt/relation_extraction_v2/relation_policy.py` |
| Staleness detection | `worldpgt/knowledge/staleness_detector.py` |
| Ontology traversal | `worldpgt/knowledge/wikidata_ontology_loader.py` and the read-only P279 layer artifact |

## Confirmed Snapshot Metrics

These are current local artifact values, not general product claims.

| Metric | Value | Source |
|---|---:|---|
| Tests collected | 3239 | `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest --collect-only -q` |
| Full suite in current dirty checkout | 3210 passed, 28 failed, 1 skipped | `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q` |
| Pump batches completed | 25 | `pump_summary.json` |
| Ready docs for ingestion | 2271 | `pump_summary.json` |
| Fetched total | 5535 | `pump_summary.json` |
| Fetch success total | 2486 | `pump_summary.json` |
| Pump dry-run overlay without weak links | 1238 | `pump_summary.json` |
| Pump dry-run overlay with weak links | 6023 | `pump_summary.json` |
| Pump answerable fact delta | 361 | `pump_summary.json` |
| Pump world-model delta | 1011 | `pump_summary.json` |
| Recent ready docs | 175 | `yield_ranked_frontier_v1/incremental_yield_summary.json` |
| Recent new answerable facts | 49 | `yield_ranked_frontier_v1/incremental_yield_summary.json` |
| Recent yield per ready doc | 0.28 | `yield_ranked_frontier_v1/incremental_yield_summary.json` |
| Cumulative answerable facts | 415 | `yield_ranked_frontier_v1/incremental_yield_summary.json` |
| Pump fact QA prompts | 529 | `pump_fact_qa_v1/pump_fact_qa_summary.json` |
| Pump fact QA positive answers | 489 | `pump_fact_qa_v1/pump_fact_qa_summary.json` |
| Pump fact QA wrong answers | 0 | `pump_fact_qa_v1/pump_fact_qa_summary.json` |
| Pump fact QA unsupported answers | 0 | `pump_fact_qa_v1/pump_fact_qa_summary.json` |
| Assistant surface prompts | 45 | `assistant/assistant_surface_summary.json` |
| Assistant surface unsafe answers | 0 | `assistant/assistant_surface_summary.json` |
| Promotion-readiness candidates | 152 | `promotion_readiness_audit_v1/promotion_readiness_summary.json` |
| Promotion-readiness needs review | 85 | `promotion_readiness_audit_v1/promotion_readiness_summary.json` |
| Promotion-readiness reject recommendations | 8 | `promotion_readiness_audit_v1/promotion_readiness_summary.json` |

## What Matches The Intended Architecture

- Answers are deterministic and routed through explicit planners, renderers,
  validators, and safety gates.
- The assistant returns answer, no, or audit-style outcomes rather than using a
  generic fallback.
- Proposal overlays, pump overlays, accepted wiki overlays, and promoted
  overlays remain separate artifacts.
- Weak context links are not counted as stable answerable facts.
- Volatile/current-sensitive predicates are blocked as multi-hop bridge
  predicates.
- Snapshot facts have freshness windows and can be routed into stale/recheck
  frontier signals.
- The gap analyzer separates acquisition candidates from policy-blocked rows.
- Optional spaCy extraction is an extraction path, not a neural answerer.
- The FastAPI server keeps session context in memory and does not write trusted
  memory.

## Known Divergences

- Older README text claimed `2030 passed`; the current checkout collects 3239
  tests. The full suite was run in this dirty working tree and finished with
  `3210 passed, 28 failed, 1 skipped`.
- The task text mentions `worldpgt/night_cycle/`; there is no such package
  directory. The current gap-driven runner is
  `worldpgt/experiments/run_audit_driven_pump_v1.py`.
- The task text says "400+ answerable facts in pump overlay"; current artifacts
  contain several related but different counts: 361 pump answerable fact delta,
  415 cumulative yield-ranked answerable facts, 489 positive QA prompts, and
  1011 world-model delta items. Docs should name the specific artifact count
  instead of collapsing them.
- `pump_summary.json` preserves stale pump fact QA fields
  (`pump_fact_qa_stale=true`, `stale_fact_count_mismatch`). The dedicated pump
  fact QA summary and promotion-readiness summary report current matching QA
  counts. Until summary preservation is repaired, use the dedicated artifacts
  for QA counts.
- The relation-embedding fallback exists, but it is optional `gensim` keyword
  similarity over relation labels. It should not be described as neural QA,
  learned reasoning, or a required runtime dependency.
- The query engine has `Aggregate(earliest/latest)`, but those operations are
  list-order based and do not yet use a timestamp-aware temporal index.

## Missing Or Incomplete

- No autonomous trusted-memory promotion from pump outputs.
- No QID-native entity identity model; canonicalization is still based on
  labels, aliases, redirects, and fallback type classification.
- No cross-sentence extraction coreference. Pronoun-heavy extraction candidates
  are intentionally conservative.
- No live web/current fact answering. Live/current prompts audit unless an
  accepted source-qualified fact exists and policy allows a caveated answer.
- No durable scheduler service for nightly pump cycles.
- No claim of open-domain QA.
- No claim that the pump dry-run overlay is safe for general runtime.
- No accepted-memory mutation from self-ingestion, snapshot ingestion, or the
  pump.

## Run Commands

```bash
# QA through CLI
python3 worldpgt/experiments/ask_microworld_v1.py \
  --overlay pump-dry-run \
  --enable-multihop \
  "Who founded SpaceX?"

# Interactive mode with in-memory dialogue context
python3 worldpgt/experiments/ask_microworld_v1.py \
  --overlay pump-dry-run \
  --enable-multihop \
  --interactive

# Web UI / API
python3 -m worldpgt.api.server --overlay pump-dry-run --port 8000

# Knowledge Pump batch
python3 worldpgt/experiments/run_knowledge_pump_v1.py \
  --enable-spacy --allow-network \
  --frontier-policy yield-ranked

# Gap-driven frontier update
python3 worldpgt/experiments/run_audit_driven_pump_v1.py \
  --period-days 1

# Test inventory
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest --collect-only -q
```

## Documentation Rule

When reporting a metric, name the artifact it came from. Do not merge
proposal-only, pump-dry-run, promoted-overlay, accepted-overlay, and trusted
accepted-memory counts into one number.
