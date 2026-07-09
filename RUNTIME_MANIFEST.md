# Microworld Runtime Extraction

This folder is the cleaned runtime slice of the larger research repository.
It keeps the pieces needed to run the explicit-memory AI runtime while leaving
old experiments, historical baselines, logs, generated caches, and broad pump
builders in the parent project.

## Included

- `worldpgt/assistant_surface/` - request routing, answer orchestration, trace
  attachment, style handling, and final answer safety metadata.
- `worldpgt/cognition/` - explicit reasoning traces, thought loop,
  support guard, decision speech, phrase graph, and surface selection.
- `worldpgt/dialogue/` - conversation state, salience, reference grammar, and
  conservative coreference/follow-up rewriting.
- `worldpgt/entity_qa/` - semantic parsing, entity inference, synthesis,
  speech planning, and symbolic language rendering.
- `worldpgt/community_context/` - low-trust style/cognitive pattern context.
- `worldpgt/query_engine/`, `worldpgt/multihop_qa/`, `worldpgt/cross_page_qa/`
  - deterministic query, path, and cross-page reasoning support.
- `worldpgt/knowledge/`, `worldpgt/relation_extraction_v2/`,
  `worldpgt/context_pack/` - read-only memory access, type/predicate policy,
  context pack building, and support boundaries.
- `worldpgt/web_search/` - optional volatile live-search path.
- `worldpgt/api/` - FastAPI server and static UI.
- `worldpgt/experiments/ask_microworld_v1.py` - CLI entry point.
- `worldpgt/experiments/benchmark_speech_quality_v1.py` - speech/reasoning
  benchmark entry point.
- Focused tests for the runtime and speech benchmark.
- Current overlay, community-context, ontology, phrase, and benchmark artifacts
  needed by the runtime.

## Deliberately Excluded

- Parent-level `core/`, `data/`, `examples/`, and old root tests.
- GPT-2 baselines and comparison reports.
- Old continuation-only experiments.
- Knowledge-pump builders, self-ingestion machinery, schema induction, broad
  snapshot collection, logs, and historical benchmark outputs.
- `__pycache__`, `.pytest_cache`, and generated local caches.

## Smoke Commands

Run from this folder:

```bash
python3 worldpgt/experiments/ask_microworld_v1.py \
  --overlay pump-dry-run \
  --json \
  "What is Starlink?"

python3 -m worldpgt.experiments.benchmark_speech_quality_v1 \
  --suite large \
  --no-save

python3 -m pytest worldpgt/tests/test_benchmark_speech_quality_v1.py -q
```

Last verified during extraction:

```text
CLI Starlink smoke: answer
large speech benchmark: 50 / 50 passed, 9 / 9 honest gaps
benchmark_speech_quality_v1 tests: 16 passed
```

