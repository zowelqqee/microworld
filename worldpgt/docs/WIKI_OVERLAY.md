# Wiki Overlay

The wiki overlay is an isolated explicit-memory artifact used by controlled
entity QA and cross-page QA. It is not accepted memory v1, not general runtime
memory, and not a trusted copy of Wikipedia.

The current overlay is built from a 50-page offline local fixture through the
deterministic Wikipedia/Wikidata-style ingestion v2 pipeline:

```text
local curated pages
-> deterministic ingestion candidates
-> WikiCandidateOverlayBuilder
-> accepted_wiki_memory_overlay_v1.json
-> WikiMemoryOverlayProvider
-> controlled QA benchmarks
```

The pipeline makes no network calls and does not query Wikipedia or any API.

## Current Artifact Status

| Metric | Value |
|---|---:|
| pages_total | 50 |
| candidates_total | 283 |
| overlay_items_total | 283 |
| skipped_candidates_total | 0 |
| review_errors_count | 0 |
| review_warnings_count | 0 |
| safe_for_general_runtime | false |
| safe_for_entity_qa_overlay | true |

Overlay item types:

| Type | Count |
|---|---:|
| entity | 50 |
| definition | 50 |
| relation | 53 |
| context_link | 126 |
| source_fact | 4 |

## Weak Links And Volatile Facts

The overlay separates stable relations from weak contextual links and
source-qualified volatile facts.

Weak contextual links:

- 126 links
- all `weak_context_only`
- used as contextual mentions or caveated link explanations
- never promoted to stable factual relations

Volatile source-qualified facts:

- Musk US$1.1T Forbes
- Bezos US$200B Forbes
- Arnault US$180B Bloomberg
- Michael Bloomberg US$100B Forbes

Each volatile fact is stored with `as_of=2026-06`, `requires_recheck=true`, and
`risk=high`. These facts can support source-qualified answers, but they are not
treated as stable or current truth.

## Safety Boundary

The overlay is safe for controlled entity QA because planners and validators
preserve the distinction between stable relations, weak contextual links, and
source-qualified volatile facts.

It remains unsafe for general runtime use:

```text
safe_for_general_runtime=false
```

The overlay does not overwrite:

- `worldpgt/experiments/accepted_knowledge_memory_v1.json`
- `worldpgt/continuation/sense_memory.py`
- the accepted wiki overlay during self-ingestion dry runs

## Limitation

The corpus is small, curated, and offline. The overlay demonstrates controlled
explicit-memory QA behavior over an audited fixture; it does not imply
open-domain QA, autonomous web ingestion, or live/current factual coverage.
