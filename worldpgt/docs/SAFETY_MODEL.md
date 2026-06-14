# Safety Model

Microworld/worldpgt uses explicit memory and deterministic audit gates instead
of implicit model weights. The safety posture is conservative: answer only when
the supporting path is present; audit instead of answer when the requested claim
would require unsupported inference, weak-link promotion, current/live data, or
volatile facts.

## Core Non-Mechanisms

The controlled pipelines use:

- no neural weights
- no backpropagation
- no fine-tuning
- no GPT renderer
- no embeddings
- no network calls
- no generic trusted fallback

## Memory Boundaries

Accepted-memory QA and wiki-overlay QA use separate artifacts.

Accepted memory:

- `worldpgt/experiments/accepted_knowledge_memory_v1.json`
- explicit accepted facts, patterns, senses, and cues
- unchanged by wiki ingestion, overlay builds, and self-ingestion dry runs

Wiki overlay:

- `worldpgt/experiments/accepted_wiki_memory_overlay_v1.json`
- isolated overlay over a 50-page local fixture
- `safe_for_entity_qa_overlay=true`
- `safe_for_general_runtime=false`

Self-ingestion dry-run overlay:

- separate dry-run overlay
- not a promotion into accepted memory
- not a replacement for the current accepted wiki overlay

## Audit Gates

The system audits instead of answering when a prompt requires:

- unsupported inference
- relation inversion
- weak contextual links treated as facts
- current or real-time data
- unsupported private or sensitive data
- invalid universal/generalization claims
- source-qualified volatile facts treated as stable/current truth
- entity category mismatches
- conflicts with existing facts
- ambiguity without enough context

Audit is expected behavior, not a benchmark failure.

## Invariants

Current safety confirmations:

- `accepted_knowledge_memory_v1.json` unchanged
- `sense_memory.py` unchanged
- accepted wiki overlay is not overwritten by self-ingestion
- dry-run overlay is separate
- ingestion extraction unchanged
- overlay builder semantics unchanged
- no thresholds lowered
- validators not weakened
- no generic fallback
- no neural/GPT/training/embedding/network code
- no Wikipedia/API calls
- no live/current claims accepted
- weak links never promoted to facts
- volatile facts never auto-applied as stable
- `nanogpt/` untouched
- `safe_for_general_runtime=false`

## Remaining Limitations

- small curated/offline corpus
- not open-domain
- deterministic/rule-heavy analyzers
- no natural-language generation beyond controlled renderers
- source extraction still narrow
- no autonomous web ingestion yet
- no accepted overlay promotion yet
- volatile facts require human review or recheck
- current facts are not answered as live truth

These limits are part of the current claim. Microworld is a lightweight
deterministic auditable QA/reasoning architecture over explicit memory and
isolated knowledge overlays, not a general language model replacement.
