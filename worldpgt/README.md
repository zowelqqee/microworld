# worldpgt — Microworld Controlled Continuation Benchmark

Explicit-policy, audit-aware continuation system benchmarked against a GPT-2 baseline on a controlled ambiguity-resolution task.

---

## Purpose

This package measures what happens when a small deterministic system with explicit sense memory attempts controlled continuation of ambiguous prompts, compared to open-ended next-token generation by GPT-2. The benchmark is narrow and intentional: 120 prompts, 6 ambiguous terms, 8 difficulty categories.

---

## Folder Structure

```
worldpgt/
  continuation/        core pipeline modules
    sense_memory.py    explicit lexical cue store; deterministic scoring
    prompt_parser.py   term detection and candidate sense extraction
    continuation_policy.py  thresholded score/margin/anti-cue policy
    continuation_engine.py  pipeline orchestrator
    audit.py           audit row types
    metrics.py         coverage/precision metric helpers
    types.py           shared dataclasses
  baselines/
    gpt2/              GPT-2 inference via local nanoGPT
      run_gpt2_baseline.py       inference runner
      create_gpt2_audit_csv.py   convert outputs to audit-ready CSV
      summarize_gpt2_audit.py    aggregate labeled audit CSV
      compare_microworld_vs_gpt2_audit.py
      parse_outputs.py
  benchmarks/
    full_comparison_report.py    builds full_comparison_report.{json,md}
    runtime_benchmark.py
    rss_benchmark.py
    state_size_benchmark.py
  experiments/
    continuation_prompts_v1.csv               120-row v1 dataset
    microworld_continuation_v1_2_outputs.csv  Microworld v1.2 per-row results
    microworld_continuation_v1_2_summary.json aggregate sense/audit counts
    microworld_continuation_v1_2_risk_coverage.json coverage/precision metrics
    gpt2_continuation_outputs.csv             raw GPT-2 generations
    gpt2_continuation_audit_labeled.csv       human-labeled audit CSV
    gpt2_continuation_audit_summary.json      GPT-2 audit metrics
    full_comparison_report.json               full structured report
    full_comparison_report.md                 human-readable report
    runtime_benchmark_summary.json
    rss_benchmark_summary.json
    state_size_benchmark_summary.json
    ...                                       version comparison JSONs
  tests/                                      1175 unit/integration tests
```

---

## Benchmark Scope

- **Task**: controlled continuation / ambiguity resolution
- **Dataset**: 120 prompts, 6 ambiguous terms (bank, bat, seal, crane, spring, rock)
- **Difficulty types**: cue_rich, delayed_cue, weak_cue, conflicting_cue, negation, misleading_surface_cue, no_clear_answer, no_known_term
- **Answerable rows**: 110 (excludes 5 no_known_term + 5 no_clear_answer)
- **Not tested**: open-domain generation quality, factual accuracy, fluency at scale

---

## Architecture Summary

### Microworld Pipeline

```
prompt
  → prompt_parser     detects ambiguous term, extracts candidate senses
  → sense_memory      scores senses by lexical cue overlap
                      applies negation window (3-token lookahead)
                      fires anti-cue overrides and guard rules
  → continuation_policy  thresholded decision:
                          continue  (score ≥ min_score, margin ≥ min_margin,
                                     no anti-cue conflict, no guard failure)
                          audit     (score too low, conflict, negated evidence,
                                     anti-cue fired, guard failed)
                          suppress  (banned surface pattern in candidate)
  → realization       template lookup per sense_id
  → surface-risk gate (suppression check on emitted text)
  → ContinuationResult with decision, selected_sense, confidence, reasons, memory_hits
```

All scoring is deterministic. No neural weights, no sampling. Every decision has an explicit audit trail.

### GPT-2 Baseline

GPT-2 base (124M parameters) loaded via local nanoGPT. Temperature 0.8, top-k 40, max 32 new tokens. No fine-tuning. No native sense selection or audit decision. Output quality judged post-hoc (good / bad / unclear + judged_sense).

---

## Dataset

Source: `worldpgt/experiments/continuation_prompts_v1.csv`

| Column         | Description                                          |
|----------------|------------------------------------------------------|
| id             | row identifier (e1-001 … v1-120)                     |
| prompt         | incomplete sentence with one ambiguous term          |
| ambiguous_term | the lexically ambiguous word                         |
| expected_sense | ground-truth intended sense (may be blank)           |
| difficulty_type | one of 8 categories                                 |

Dataset is frozen. Do not modify it.

---

## Headline Results

### Quality

| System    | Continued / Generated | Wrong | Precision on Emitted |
|-----------|-----------------------|-------|----------------------|
| Microworld v1.2 | 38 / 120        | 0     | 1.000                |
| GPT-2 base      | 76 good / 11 bad / 33 unclear | 7 wrong sense | 0.8736 (audited) |

GPT-2 good/bad/unclear refers to post-hoc human-assigned labels. Microworld precision is on the 38 prompts where it emitted a continuation; the other 82 were routed to audit.

### Risk / Coverage

| System    | Continue | Audit | Coverage Rate | Answerable Recall |
|-----------|----------|-------|---------------|-------------------|
| Microworld v1.2 | 38  | 82    | 0.3167        | 0.3455            |
| GPT-2 base      | 120 (all) | — (none) | 1.000  | —                 |

GPT-2 has no native audit path. All 120 prompts received generated output.

---

## Runtime / Memory / State Size

| Metric                    | Microworld v1.2 | GPT-2 base     |
|---------------------------|-----------------|----------------|
| Avg time / prompt         | ~0.000046 s     | ~0.422 s       |
| Peak RSS                  | ~22.9 MB        | ~1348.7 MB     |
| Explicit state / weights  | 7,882 bytes     | ~548 MB        |
| Trainable parameters      | 0               | 123.65 M       |

RSS is approximate and environment-dependent (measured via `resource.ru_maxrss` in subprocess; macOS reports bytes, Linux reports kilobytes). GPT-2 RSS includes model-load overhead.

---

## Reproducing Results

All commands run from the `worldmvp/` directory.

### Run Microworld v1.2

```bash
python3 -m worldpgt.experiments.run_v1_2_microworld_continuation \
    --input worldpgt/experiments/continuation_prompts_v1.csv \
    --output worldpgt/experiments/microworld_continuation_v1_2_outputs.csv
```

### Summarize v1.2 sense/audit counts

```bash
python3 -m worldpgt.experiments.summarize_v1_microworld_results \
    --input worldpgt/experiments/microworld_continuation_v1_2_outputs.csv \
    --output worldpgt/experiments/microworld_continuation_v1_2_summary.json
```

### Run risk/coverage summary

```bash
python3 -m worldpgt.experiments.summarize_risk_coverage \
    --input worldpgt/experiments/microworld_continuation_v1_2_outputs.csv \
    --output worldpgt/experiments/microworld_continuation_v1_2_risk_coverage.json
```

### Run GPT-2 baseline (requires local nanoGPT)

```bash
python3 -m worldpgt.baselines.gpt2.run_gpt2_baseline \
    --input worldpgt/experiments/continuation_prompts_v1.csv \
    --output worldpgt/experiments/gpt2_continuation_outputs.csv \
    --nanogpt-dir nanogpt \
    --device mps
```

`nanogpt/` is not committed. Obtain nanoGPT and a cached `gpt2` checkpoint separately.

### Create GPT-2 audit CSV

```bash
python3 -m worldpgt.baselines.gpt2.create_gpt2_audit_csv \
    --input worldpgt/experiments/gpt2_continuation_outputs.csv \
    --output worldpgt/experiments/gpt2_continuation_audit.csv
```

Apply labels (good / bad / unclear) and judged_sense manually. Save as `gpt2_continuation_audit_labeled.csv`.

### Summarize GPT-2 audit

```bash
python3 -m worldpgt.baselines.gpt2.summarize_gpt2_audit \
    --input worldpgt/experiments/gpt2_continuation_audit_labeled.csv \
    --output worldpgt/experiments/gpt2_continuation_audit_summary.json
```

### Run full comparison report

```bash
python3 -m worldpgt.benchmarks.full_comparison_report \
    --microworld-output worldpgt/experiments/microworld_continuation_v1_2_outputs.csv \
    --microworld-risk worldpgt/experiments/microworld_continuation_v1_2_risk_coverage.json \
    --gpt2-audit worldpgt/experiments/gpt2_continuation_audit_labeled.csv \
    --gpt2-summary worldpgt/experiments/gpt2_continuation_audit_summary.json \
    --runtime worldpgt/experiments/runtime_benchmark_summary.json \
    --state-size worldpgt/experiments/state_size_benchmark_summary.json \
    --rss worldpgt/experiments/rss_benchmark_summary.json \
    --json-output worldpgt/experiments/full_comparison_report.json \
    --md-output worldpgt/experiments/full_comparison_report.md
```

### Run tests

```bash
python3 -m pytest worldpgt/tests -q
python3 -m pytest -q
```

---

## Limitations

- 120-row benchmark. Results do not generalize beyond this controlled task.
- GPT-2 is an old base model (2019), not an instruction-tuned assistant.
- GPT-2 audit labels were assigned by a human pass after generation, not by a native model gate.
- Microworld realization is template-based; generated text quality is not representative of fluent generation.
- Microworld coverage is low (38/120 = 31.7%). The system abstains on ambiguous or weak-cue prompts.
- RSS figures are approximate and environment-dependent.
- Sense memory covers 6 terms. Performance on out-of-vocabulary terms is undefined.

---

## Non-Claims

- No claim that Microworld beats neural networks generally.
- No claim that this benchmark predicts open-domain generation quality.
- No claim that GPT-2 represents modern instruction-tuned assistants (ChatGPT, Claude, etc.).
- No claim that zero wrong continuations on 38 rows implies zero errors at scale.

---

## Generated Reports

The following files are generated artifacts, not source:

```
worldpgt/experiments/full_comparison_report.md
worldpgt/experiments/full_comparison_report.json
worldpgt/experiments/microworld_continuation_v1_2_outputs.csv
worldpgt/experiments/microworld_continuation_v1_2_summary.json
worldpgt/experiments/microworld_continuation_v1_2_risk_coverage.json
worldpgt/experiments/gpt2_continuation_outputs.csv
worldpgt/experiments/gpt2_continuation_audit.csv
worldpgt/experiments/gpt2_continuation_audit_labeled.csv
worldpgt/experiments/gpt2_continuation_audit_summary.json
worldpgt/experiments/runtime_benchmark_summary.json
worldpgt/experiments/rss_benchmark_summary.json
worldpgt/experiments/state_size_benchmark_summary.json
```
