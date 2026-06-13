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
    semantic_frame.py  coarse actor/intent/connector frame
    semantic_renderer.py  weightless renderer v2 (phrase compose + rank)
    phrase_library.py  curated per-sense continuation phrases
    surface_validator.py  banned-pattern surface check
    surface_repair.py  deterministic post-render grammar/role/coref repair
    subject_action_validator.py  role-aware subject/action check
    connector_rewriter.py  clause-boundary comma repair
    coreference_repair.py  repeated-object / prey coreference repair
    audit.py           audit row types
    metrics.py         coverage/precision metric helpers
    types.py           shared dataclasses
  qa/                  QA layer (question answering over accepted memory)
    question_analyzer.py      detects QA intent from surface form
    answer_planner.py         selects response strategy from accepted memory
    answer_renderer.py        composes semantic answer forms
    answer_validator.py       checks correctness and flags quality issues
    audit_renderer.py         helpful audit text for ambiguous questions
    semantic_language_realizer.py  clause-level language realization
    contrast_realizer.py      contrast explanations for distinguish_senses
    accepted_memory_provider.py   accepted knowledge memory provider
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
  tests/                                      unit/integration tests (incl. surface-repair gate)
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
  → realization       template lookup per sense_id (semantic renderer v2)
  → surface repair     deterministic grammar/role/coreference fixes, then
                       re-validation; unsafe repairs route to audit
  → surface-risk gate (suppression check on emitted text)
  → ContinuationResult with decision, selected_sense, confidence, reasons, memory_hits
```

All scoring is deterministic. No neural weights, no sampling. Every decision has an explicit audit trail.

### Surface Repair Layer

The surface repair layer runs **after** the semantic renderer selects a single
candidate and **before** final emission. It does not generate text, score senses,
or change any decision threshold — it only applies small, fixed string transforms
to remove residual grammar / role / coreference bugs the template library could
still produce, then re-validates the result.

**Why it exists.** The renderer composes continuations from a curated phrase
library. A few compositions were grammatically appended but awkward: a missing
comma at a clause boundary, a body part as the subject of a whole-animal action,
or a repeated object noun. These were visible surface bugs, not reasoning errors.

**What it fixes** (all deterministic, rule-based):

- **Connector grammar** (`connector_rewriter.py`) — inserts a comma at a
  subordinate-clause boundary before a new main clause:
  `before the swing he steadied himself` → `before the swing, he steadied himself`.
- **Subject/action role** (`subject_action_validator.py`) — rejects a body part
  performing a whole-animal action and substitutes a body-part-appropriate
  clause: `its wings searched for insects` → `its wings spread wide`. Allowed
  body-part actions (`wings spread`) pass untouched.
- **Coreference / repetition** (`coreference_repair.py`) — replaces a repeated
  object noun with a pronoun or `its prey`: `close the envelope` → `close it`;
  `catch another fish` → `catch its prey`.
- **Attachment drift** (`surface_repair.py`) — detects a positional clause that
  would attach across an intervening actor (ambiguous subject) and routes the row
  to **audit** rather than emit it: `... until the climber saw it on the cliff and
  lay near the river` → `audit_reason=no_safe_repaired_candidate`.

**Why it does not change policy or reasoning.** Sense scoring, negation/anti-cue
handling, guard rules, and the continue/audit/suppress thresholds are untouched.
Every repaired candidate is re-run through the existing surface validator, the
role validator, drift checks, and a repeated-object check; if a repair cannot
produce a clean candidate, the layer emits an **audit**, never a risky
continuation. The bias is unchanged: prefer audit/abstain over emitting a risky
continuation.

**Measured benchmark impact (v1.2, 120 prompts).** After the deterministic
surface repair layer, Microworld emits one fewer continuation on v1.2 but removes
the remaining measured semantic-render-quality flags while preserving zero
measured wrong continuations.

| Metric                       | Before repair | After repair |
|------------------------------|---------------|--------------|
| continue_count               | 38            | 37           |
| audit_count                  | 82            | 83           |
| wrong_continue_count         | 0             | 0            |
| precision_on_continued       | 1.000         | 1.000        |
| semantic-quality flagged     | 1 / 38        | 0 / 37       |
| coverage_rate                | 0.3167        | 0.3083       |

Repaired rows: `v1-007` (connector comma), `v1-009` (prey coreference), `v1-011`
(object repetition), `v1-043` (body-part subject/action), `v1-008` (object
repetition). Audited row: `v1-051` (attachment drift). The numbers are locked by
`worldpgt/tests/test_surface_repair_benchmark_gate.py`.

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
| Microworld v1.2 (with surface repair) | 37 / 120 | 0 | 1.000          |
| GPT-2 base      | 76 good / 11 bad / 33 unclear | 7 wrong sense | 0.8736 (audited) |

GPT-2 good/bad/unclear refers to post-hoc human-assigned labels. Microworld precision is on the 37 prompts where it emitted a continuation; the other 83 were routed to audit. (Pre-repair the split was 38 / 82; the surface repair layer audits one attachment-drift row — see [Surface Repair Layer](#surface-repair-layer).)

### Risk / Coverage

| System    | Continue | Audit | Coverage Rate | Answerable Recall |
|-----------|----------|-------|---------------|-------------------|
| Microworld v1.2 (with surface repair) | 37 | 83 | 0.3083 | 0.3364       |
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
python3 -m pytest worldpgt/tests/test_answer_planner_v1.py -q  # 95 passed
python3 -m pytest worldpgt/tests -q                             # 702 passed
python3 -m pytest -q                                            # 1745 passed
```

---

## QA Layer

worldpgt includes a small controlled QA assistant over explicit accepted
memory. It answers, distinguishes, explains, or safely audits ambiguous-term
questions through a transparent planner/renderer/validator pipeline — no
neural weights, no model-based generation.

### Architecture

```
question text
  → QuestionAnalyzer     detects QA intent
                         (define_sense / classify_context / explain_cue /
                          distinguish_senses / unknown_or_ambiguous)
  → AcceptedMemoryProvider  loads accepted facts, patterns, senses
  → AnswerPlanner        selects response strategy from accepted memory
  → AnswerRenderer       composes semantic answer forms
                         (common clues, contexts, signs, location phrases,
                          action phrases, contrast explanations)
  → AnswerValidator      checks correctness; flags quality issues
  → result: answer text or helpful audit text
```

All decisions are deterministic and based on accepted memory. No weights.
No backpropagation. No GPT renderer. No generic fallback.

### Accepted Memory Provider

```text
total items:     221
fact items:      163
pattern items:    58
ambiguous terms:   6  (bank, bat, crane, rock, seal, spring)
senses:           12
```

### Main QA Benchmark (48 controlled questions)

```text
qa_total:          48
answer_count:      42
audit_count:        6
correct_count:     48
wrong_count:        0
accuracy:          1.0
answer_precision:  1.0
quality_flagged:    0
```

### Generalization Benchmark (24 novel phrasings)

```text
qa_total:          24
correct_count:     12
wrong_count:       12
accuracy:          0.5
answer_count:       8
audit_count:       16
answer_precision:  0.875
quality_flagged:    0
```

The bottleneck is the `QuestionAnalyzer` — the system audits conservatively
on unseen phrasings rather than forcing a wrong answer. The renderer and
planner are not the failure point.

### Example Outputs

```text
A baseball bat is a club used to hit a ball in sports such as baseball.
Common clues are pitchers, balls, batters and plates.
It is used to swing, hit or strike the ball, and it is often found at home plate.

Spring is the season that follows winter.
Common signs are thaw, flowers, warmer mornings and rain.
In spring, flowers bloom and snow thaws.

Rock music is a music genre linked to bands, concerts and crowds.
A rock is a solid mineral object found near cliffs, boulders and trails.
```

Helpful audit (safe abstention):

```text
"Seal" is ambiguous: it can mean a marine animal or a wax/document seal.
I need context to choose the right meaning.
```

### Reproduce

```bash
# Main QA benchmark
python3 -m worldpgt.experiments.run_answer_planner_v1 \
  --qa-input worldpgt/experiments/qa_prompts_v1.csv \
  --accepted-memory worldpgt/experiments/accepted_knowledge_memory_v1.json \
  --output-csv worldpgt/experiments/answer_planner_v1_outputs.csv \
  --output-json worldpgt/experiments/answer_planner_v1_summary.json

# Generalization benchmark
python3 -m worldpgt.experiments.run_answer_planner_v1 \
  --qa-input worldpgt/experiments/qa_generalization_test_v1.csv \
  --accepted-memory worldpgt/experiments/accepted_knowledge_memory_v1.json \
  --output-csv worldpgt/experiments/qa_generalization_test_v1_outputs.csv \
  --output-json worldpgt/experiments/qa_generalization_test_v1_summary.json
```

---

## Limitations

- 120-row benchmark. Results do not generalize beyond this controlled task.
- GPT-2 is an old base model (2019), not an instruction-tuned assistant.
- GPT-2 audit labels were assigned by a human pass after generation, not by a native model gate.
- Microworld realization is template-based; generated text quality is not representative of fluent generation.
- Microworld coverage is low (37/120 = 30.8%). The system abstains on ambiguous or weak-cue prompts, and the surface repair layer audits one further row it cannot repair without subject drift.
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
