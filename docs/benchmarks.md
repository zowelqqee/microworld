# Benchmarks

This document collects the benchmark snapshots, performance notes, validation
commands, and caveats that were previously embedded in the README. The numbers
below are local artifact snapshots, not general product benchmarks.

## Latest Speech/Reasoning Snapshot

Latest speech/reasoning snapshot, from
`worldpgt/experiments/benchmarks/speech_quality_large_20260709T111746Z.json`
and
`worldpgt/experiments/benchmarks/speech_quality_stress_20260709T111906Z.json`:

| Metric | Large suite | Stress suite |
|---|---:|---:|
| Questions | 50 | 1,000 |
| Passed | 50 / 50 | 1,000 / 1,000 |
| Quality rate | 100.0% | 100.0% |
| Honest gap rate | 9 / 9 | 171 / 171 |
| Debug-like output | 0 | 0 |
| Repetitive output | 0 | 0 |
| Decision drift | 0 | 0 |
| Missing required text | 0 | 0 |
| Latency p50 | 3.39 ms | 8.05 ms |
| Latency p95 | 26.98 ms | 29.47 ms |
| Latency p99 | 27.73 ms | 38.90 ms |

The stress suite is a deterministic 1,000-question speech benchmark over known
categories, not 1,000 independent open-domain facts. Its purpose is to measure
the user-facing speech/reasoning surface under load: profiles, thin profiles,
mechanism gaps, direct relations, connection paths, adversarial inversions,
current/live requests, private-info requests, unsupported universal claims, and
style control.

Performance snapshot from the deterministic speech benchmark:

```text
questions:     1,000
passed:        1,000 / 1,000
honest gaps:   171 / 171
p50 latency:   8.05 ms
p95 latency:   29.47 ms
p99 latency:   38.90 ms
max latency:   123.53 ms
hardware:      local CPU, no GPU/model API at answer time
```

These are local snapshots, not general product benchmarks. The useful signal is
the shape: indexed semantic-memory lookup, deterministic reasoning, and
controlled speech rendering stay fast under the tested 1,000-question load, and
the system can run without a GPU or model API on supported memory-backed
questions.

## Speech Quality Benchmark

`benchmark_speech_quality_v1.py` measures the answer surface, not factual
coverage. It treats the semantic planner as an explicit-memory lookup and
checks whether speech stays natural, honest about gaps, non-repetitive, and
free of debug/internal wording.

It records row-level diagnostics:

```text
question
decision / route / support_kind / source_system
answer_text
latency_ms
debug_like
repetitive
honest_gap
decision_mismatch
missing_required_text
flags
```

Current suites:

| Suite | Purpose | Questions | Result |
|---|---|---:|---:|
| `smoke` | fast contract check | 12 | green |
| `large` | broad speech/reasoning baseline | 50 | 50 / 50 |
| `stress` | deterministic load/stability suite | 1,000 | 1,000 / 1,000 |

Stress category coverage:

| Category | Passed |
|---|---:|
| profile | 304 / 304 |
| direct_relation | 162 / 162 |
| mechanism_gap | 114 / 114 |
| adversarial | 72 / 72 |
| missing_or_current | 72 / 72 |
| thin_profile | 57 / 57 |
| style_control | 57 / 57 |
| connection | 54 / 54 |
| private_info | 54 / 54 |
| unsupported_universal | 54 / 54 |

Reproduce:

```bash
python3 -m worldpgt.experiments.benchmark_speech_quality_v1 --suite large
python3 -m worldpgt.experiments.benchmark_speech_quality_v1 --suite stress
python3 -m pytest worldpgt/tests/test_benchmark_speech_quality_v1.py -q
```

Saved report:

```text
worldpgt/experiments/benchmarks/speech_quality_large_20260709T111746Z.json
worldpgt/experiments/benchmarks/speech_quality_stress_20260709T111906Z.json
```

## Performance

Microworld's hot path is mostly indexed semantic-memory lookup, deterministic
planning, dialogue/context routing, and small renderer passes. It has no GPU
dependency and does not call a model API at answer time.

```mermaid
flowchart LR
    Q["Question"] --> SEM["Semantic parse"]
    SEM --> IDX["Semantic entity / relation indexes"]
    IDX --> PLAN["Small deterministic semantic plan"]
    PLAN --> LOOK["O(1)-style semantic-memory lookup"]
    LOOK --> GATE["Safety gate"]
    GATE --> TEXT["Controlled language"]
```

Deterministic speech-renderer benchmark snapshot:

| Metric | Value |
|---|---:|
| questions | 1,000 |
| passed | 1,000 / 1,000 |
| quality_rate | 100.0% |
| honest_gap_rate | 171 / 171 |
| mean latency | 14.71 ms |
| p50 latency | 8.05 ms |
| p95 latency | 29.47 ms |
| p99 latency | 38.90 ms |
| max latency | 123.53 ms |
| debug-like output | 0 |
| repetitive output | 0 |
| decision drift | 0 |

This benchmark measures the answer-surface runtime under deterministic
categories, including profile answers, direct relations, mechanism gaps,
connection paths, adversarial inversions, current/live requests, private-info
requests, unsupported universal claims, and style control. Treat it as a
workload-specific runtime result, not a universal benchmark: it applies when the
question can be handled by Microworld's explicit memory/reasoning/speech path
and does not need open-domain generation.

## Live Search Snapshot

The optional live-search path exists for current or missing information, but the
latest saved WebQuestions-style open benchmark is still weak and intentionally
documented as experimental:

```text
external_20260706T203034Z.json
total_questions: 250
answer_rate:     42.0%
audit_rate:      58.0%
precision:       28.57% among answered rows
elapsed:         1878.23s
```

So the honest status is: live search exists, is safer than a generic fallback,
and is improving, but it is not yet a strong open-domain inference result.

Open WebQuestions-style benchmark with optional live search:

```bash
python3 -m worldpgt.experiments.benchmark_external_v1 \
  --overlay pump-dry-run \
  --web-search
```

## Validation Commands

Focused tests for the most recent runtime/extraction layers:

```bash
python3 -m pytest \
  worldpgt/tests/test_relation_policy_and_patterns.py \
  worldpgt/tests/test_assistant_surface_v1.py \
  worldpgt/tests/test_synthesis_layer_v1.py \
  worldpgt/tests/test_dialogue_coreference.py \
  -q

python3 -m pytest worldpgt/tests/test_knowledge_pump_extraction_yield_v2.py -q
python3 -m pytest worldpgt/tests/test_benchmark_speech_quality_v1.py -q
python3 -m pytest \
  worldpgt/tests/test_dialogue_state.py \
  worldpgt/tests/test_reference_grammar.py \
  worldpgt/tests/test_salience.py \
  worldpgt/tests/test_resolver.py \
  worldpgt/tests/test_dialogue_benchmark_v1.py \
  -q
```

Recent focused validation:

```text
204 passed  # relation policy + assistant surface + synthesis + dialogue coreference
98 passed   # knowledge_pump_extraction_yield_v2
16 passed   # benchmark_speech_quality_v1
52 passed   # dialogue-v2 state/grammar/salience/resolver/benchmark
```

Latest requested validation on this trimmed runtime copy, measured on
2026-07-09:

```text
python3 -m worldpgt.experiments.benchmark_speech_quality_v1 --suite large
50 / 50 passed; honest gaps 9 / 9; latency p50 3.39 ms, p95 26.98 ms, p99 27.73 ms

python3 -m worldpgt.experiments.benchmark_speech_quality_v1 --suite stress
1,000 / 1,000 passed; honest gaps 171 / 171; latency p50 8.05 ms, p95 29.47 ms, p99 38.90 ms

python3 -m worldpgt.benchmarks.dialogue_benchmark
21 / 21 sessions passed; 138 resolver calls; mean 240.6 us/call

python3 -m pytest worldpgt/tests/ -q --ignore=<5 live/network provider tests>
540 passed, 2 skipped, 1 xpassed, 11 failed, 14 errors in 132.28s
```

The pytest failures are expected for this trimmed copy: they reference missing
experiment runners/artifacts such as `run_assistant_surface_v1`,
`run_cross_page_qa_v1`, `run_entity_qa_v1`, `run_answer_planner_v1`,
`run_multihop_qa_v1`, `multihop_qa_summary.json`, `cross_page_qa_v1.csv`, and
`worldpgt/continuation/sense_memory.py`.

## Benchmark Artifacts

```text
worldpgt/experiments/benchmark_speech_quality_v1.py
worldpgt/experiments/benchmarks/speech_quality_large_20260709T111746Z.json
worldpgt/experiments/benchmarks/speech_quality_stress_20260709T111906Z.json
worldpgt/experiments/benchmarks/external_20260706T203034Z.json
worldpgt/benchmarks/dialogue_benchmark.py
worldpgt/benchmarks/fixtures/dialogue_sessions_v1.json
```

## Conclusion

The benchmark evidence supports a narrow claim: on the tested explicit-memory
speech/reasoning path, Microworld remains fast, deterministic, and honest about
gaps. These numbers do not demonstrate open-domain general inference or
general superiority over neural systems.
