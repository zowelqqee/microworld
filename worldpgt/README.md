# worldpgt - Microworld Controlled Continuation and QA

Explicit-policy, audit-aware continuation and QA system over deterministic
memory, accepted facts, and isolated knowledge overlays.

---

## Purpose

This package measures what happens when a small deterministic system with
explicit sense memory and accepted knowledge attempts controlled continuation,
ambiguity-resolution QA, and entity QA over an isolated wiki candidate overlay.
The benchmark scope is narrow and intentional. It is not a general-purpose
language model, and it does not claim LLM-level open-domain performance.

Microworld/worldpgt explores whether useful controlled QA, memory, reasoning,
and knowledge ingestion can be built without neural weights, backpropagation,
fine-tuning, GPT-style next-token rendering, embeddings, GPU, or network calls.
LLMs learn to speak and world understanding emerges as a side effect.
Microworld tries to build explicit world memory first, then use language as an
interface to that world.

---

## Current Status

Microworld currently demonstrates a lightweight, deterministic, auditable QA
architecture over explicit memory and isolated knowledge overlays. It is strong
on controlled benchmark domains, explicit memory, source-aware facts, safe
abstention/audit, and low runtime cost. It is limited by narrow scope, curated
inputs, rule/curriculum-based analyzers, and surface renderer quality.

Microworld currently demonstrates a narrow but useful property: on controlled
explicit-memory QA benchmarks, it can answer when the supporting path is present
and audit when the answer would require unsupported inference, weak-link
promotion, current/live data, or volatile facts.

Across the current controlled QA benchmark layers, worldpgt handles 350 prompts
with 0 wrong decisions: 219 answered and 131 safely audited. Answer precision is
1.0 on each benchmark family. These results are scoped to supported controlled
domains and isolated overlays; they are not open-domain or live-fact claims.

| Benchmark | Input / artifact | Items | Correct / errors | Answered | Audited | Precision | Time | Peak RSS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Main QA | `experiments/qa_prompts_v1.csv` / `answer_planner_v1_summary.json` | 48 | 48 correct, 0 wrong | 42 | 6 | 1.0 | ~0.070 s | ~24.73 MB |
| Generalization QA | `experiments/qa_generalization_test_v1.csv` / `qa_generalization_test_v1_summary.json` | 24 | 24 correct, 0 wrong | 19 | 5 | 1.0 | ~0.140 s | ~24.61 MB |
| Entity QA | `experiments/entity_qa_prompts_v1.csv` / `entity_qa_v1_summary.json` | 28 | 28 correct, 0 wrong | 23 | 5 | 1.0 | ~0.060 s | ~23.61 MB |
| Entity QA expansion | `experiments/entity_qa_expansion_v1_summary.json` | 111 | 111 correct, 0 wrong | 79 | 32 | 1.0 | - | - |
| Adversarial Entity QA | `experiments/entity_qa_adversarial_v1_summary.json` | 68 | 68 correct, 0 wrong | 6 | 62 | 1.0 | - | - |
| Cross-page Entity QA | `experiments/cross_page_qa_v1_summary.json` | 71 | 71 correct, 0 wrong | 50 | 21 | 1.0 | ~0.160 s | ~24 MB |
| Wiki ingestion v2 | `experiments/wiki_pages_curated_v2.json` / `wiki_ingestion_v2_summary.json` | 283 candidates | 0 review errors | - | - | - | ~0.060 s | ~23.8 MB |
| Wiki overlay v1 | `experiments/accepted_wiki_memory_overlay_v1_summary.json` | 283 overlay items | 0 skipped | - | - | - | ~0.050 s | ~23.7 MB |
| Self-ingestion v1 dry run | dry-run 310-item overlay | regressions green | - | - | - | - | ~0.210 s | ~25.8 MB |

The current Python implementation runs these small controlled benchmark batches
in about 0.05-0.21 seconds with roughly 24-26 MB peak RSS. These are single-run
local measurements from `/usr/bin/time -l` on macOS and should be treated as
order-of-magnitude efficiency indicators, not final benchmark claims.

Current local test status:

```text
python3 -m pytest worldpgt/tests/test_wiki_ingestion_v2.py -q       -> 34 passed
python3 -m pytest worldpgt/tests/test_wiki_memory_overlay_v1.py -q  -> 26 passed
python3 -m pytest worldpgt/tests/test_entity_qa_v1.py -q            -> 33 passed
python3 -m pytest -q                                                -> 2007 passed
```

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
  knowledge/           deterministic offline wiki-style ingestion and overlay
    wiki_ingestion_v2_types.py
    wiki_page_reader.py
    wiki_entity_extractor.py
    wiki_claim_extractor.py
    wiki_claim_normalizer.py
    wiki_ingestion_v2.py
    wiki_memory_overlay_types.py
    wiki_candidate_overlay_builder.py
    wiki_memory_overlay_provider.py
  entity_qa/           controlled QA over isolated wiki overlay
    types.py
    entity_question_analyzer.py
    entity_answer_planner.py
    entity_answer_renderer.py
    entity_answer_validator.py
  cross_page_qa/       controlled graph-style QA over isolated wiki overlay
    cross_page_question_analyzer.py
    cross_page_answer_planner.py
    cross_page_answer_renderer.py
    cross_page_answer_validator.py
  self_ingestion/      offline self-feeding pipeline workspace (dry-run only)
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
    accepted_knowledge_memory_v1.json
    qa_prompts_v1.csv
    answer_planner_v1_summary.json
    qa_generalization_test_v1.csv
    qa_generalization_test_v1_summary.json
    wiki_pages_curated_v2.json
    wiki_ingestion_v2_candidates.{json,csv}
    wiki_ingestion_v2_summary.json
    wiki_ingestion_v2_review.json
    accepted_wiki_memory_overlay_v1.json
    accepted_wiki_memory_overlay_v1_summary.json
    accepted_wiki_memory_overlay_v1_skipped.json
    entity_qa_prompts_v1.csv
    entity_qa_v1_outputs.csv
    entity_qa_v1_summary.json
    entity_qa_expansion_v1_outputs.csv
    entity_qa_expansion_v1_summary.json
    entity_qa_adversarial_v1.csv
    entity_qa_adversarial_v1_outputs.csv
    entity_qa_adversarial_v1_summary.json
    cross_page_qa_v1.csv
    cross_page_qa_v1_outputs.csv
    cross_page_qa_v1_summary.json
    ...                                       version comparison JSONs
  docs/
    WIKI_OVERLAY.md
    CROSS_PAGE_QA.md
    SELF_INGESTION.md
    SAFETY_MODEL.md
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

### Controlled QA and Knowledge Overlay Pipeline

Accepted-memory QA:

```text
accepted memory artifact
  -> AcceptedMemoryProvider
  -> QuestionAnalyzer / GeneralizedQuestionAnalyzer
  -> AnswerPlanner
  -> AnswerRenderer
  -> AnswerValidator / AuditRenderer
  -> benchmark summary
```

Wiki-style entity QA:

```text
local curated source pages
  -> deterministic ingestion candidates
  -> isolated wiki candidate memory overlay
  -> WikiMemoryOverlayProvider
  -> EntityQuestionAnalyzer
  -> EntityAnswerPlanner
  -> EntityAnswerRenderer
  -> EntityAnswerValidator / audit
  -> entity QA benchmark
```

Accepted memory v1 and the wiki overlay are separate artifacts. Wiki ingestion
v2 is deterministic, offline, local-fixture only, candidate-generation only, and
does not modify accepted memory or runtime memory.

Cross-page entity QA uses the same isolated overlay, but requires an explicit
stable relation path or an explicitly caveated weak contextual link. Weak links
are never treated as stable facts, and volatile source-qualified facts are never
treated as stable/current facts.

Self-ingestion v1 is an offline dry-run pipeline: local Wikipedia-like documents
are converted to wiki-like pages, passed through the unchanged ingestion and
overlay builders, classified as duplicates/deltas/conflicts/quarantine, and
checked by deterministic QA regressions before any promotion step. It does not
write raw source text into accepted memory.

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
python3 -m pytest worldpgt/tests/test_wiki_ingestion_v2.py -q       # 34 passed
python3 -m pytest worldpgt/tests/test_wiki_memory_overlay_v1.py -q  # 26 passed
python3 -m pytest worldpgt/tests/test_entity_qa_v1.py -q            # 33 passed
python3 -m pytest -q                                                # 2007 passed
```

---

## QA Layer

worldpgt includes controlled QA assistants over explicit accepted memory and an
isolated wiki candidate overlay. They answer, distinguish, explain, or safely
audit supported questions through transparent analyzer/planner/renderer/
validator pipelines - no neural weights, no embeddings, no network calls, and
no model-based generation.

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

All decisions are deterministic and based on accepted memory or the isolated
overlay. No weights. No backpropagation. No GPT renderer. No generic fallback.

### Accepted Memory Provider

```text
total items:     221
fact items:      163
pattern items:    58
ambiguous terms:   6  (bank, bat, crane, rock, seal, spring)
senses:           12
positive cues:   104
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
correct_count:     24
wrong_count:        0
accuracy:          1.0
answer_count:      19
audit_count:        5
answer_precision:  1.0
quality_flagged:    0
```

Supported generalized forms include:

```text
Is a bat with wings an animal or sports equipment?
The seal was swimming near the coast. What kind of seal is it?
The crane had a hook and lifted a load. What does crane mean?
The band played rock on stage. What does rock mean?
Why do wings point to bat as an animal?
```

Conflicting cue prompts audit rather than force an answer.

### Wiki Ingestion v2

Wikipedia/Wikidata-style ingestion v2 is deterministic, offline, and local
fixture only. It creates reviewable candidates and does not modify accepted
memory or runtime memory.

```text
pages_total:              50
candidates_total:        283
entity_card:              50
definition_claim:         50
relation_claim:           53
context_link:            126
source_qualified_fact:     4
stable:                   50
semi_stable:              53
volatile:                  4
low risk:                100
medium risk:              53
high risk:                 4
review_errors_count:       0
review_warnings_count:     0
safe_for_runtime_memory: false
```

### Wiki Candidate Memory Overlay v1

The wiki candidate overlay is isolated from general runtime memory and from
accepted memory v1. It is safe for the controlled entity QA overlay, not for
general runtime use.

```text
source_candidates_total:     283
overlay_items_total:         283
skipped_candidates_total:      0
overlay_entity:               50
overlay_definition:           50
overlay_relation:             53
overlay_context_link:        126
overlay_source_fact:           4
source_facts_count:            4
weak_context_links_count:    126
safe_for_general_runtime:  false
safe_for_entity_qa_overlay: true
```

The four volatile source-qualified facts are stored with `as_of=2026-06`,
`requires_recheck=true`, and `risk=high`. They are never converted into stable
relations. The 126 contextual links remain `weak_context_only`.

### Entity QA Benchmark v1

Entity QA is controlled QA over the isolated wiki overlay. Supported intents:
`define_entity`, `relation_lookup`, `link_explanation`, `source_fact_lookup`,
and `unknown_or_unsupported`.

```text
qa_total:                  28
correct_count:             28
wrong_count:                0
answer_count:              23
audit_count:                5
quality_flagged:            0
accuracy:                 1.0
answer_precision:         1.0
source_facts_used:          3
weak_context_links_used:    4
safe_for_general_runtime: false
```

### Entity QA Expansion v1

The expanded entity QA benchmark uses the same isolated 50-page overlay and
broader phrasings over entity definitions, relations, weak contextual links,
source-qualified facts, and unsupported/current questions.

```text
qa_total:                  111
correct_count:             111
wrong_count:                 0
answer_count:               79
audit_count:                32
quality_flagged:             0
accuracy:                  1.0
answer_precision:          1.0
source_facts_used:          12
weak_context_links_used:    15
safe_for_general_runtime: false
```

### Adversarial Entity QA v1

The adversarial benchmark is designed to verify audit behavior under prompts
that try to force unsupported inference or unsafe promotion.

```text
qa_total:                  68
correct_count:             68
wrong_count:                0
answer_count:               6
audit_count:               62
quality_flagged:            0
accuracy:                 1.0
answer_precision:         1.0
safe_for_general_runtime: false
```

Adversarial categories include relation inversion, weak-link-as-fact,
current/real-time requests, category confusion, invalid universal or
generalization claims, source-qualified volatility, and unsupported
private/personal data.

### Cross-page Entity QA v1

Cross-page QA is controlled graph-style multi-hop QA over the isolated 50-page
overlay. It answers when a stable supporting path is present, gives a caveat
for weak contextual links, and audits when a requested path is unsupported.

```text
qa_total:                  71
correct_count:             71
wrong_count:                0
answer_count:              50
audit_count:               21
quality_flagged:            0
accuracy:                 1.0
answer_precision:         1.0
relation_edges_used:       35
weak_context_links_used:   31
source_facts_used:         36
safe_for_general_runtime: false
```

Examples: Musk -> SpaceX -> rockets can answer; Musk -> Starlink audits when no
explicit stable path exists; SpaceX -> Starlink returns a weak contextual link
caveat instead of a stable factual relation.

### Wikipedia Self-Ingestion v1

Self-ingestion v1 is a safe offline self-feeding pipeline. It reads local
Wikipedia-like documents, converts them to wiki-like pages, reuses unchanged
`WikiIngestionV2` and `WikiCandidateOverlayBuilder`, classifies
deltas/conflicts/quarantine, proposes an overlay delta, creates a dry-run
overlay, and runs QA/adversarial/cross-page regressions.

It never writes raw text directly into accepted memory. The dry-run overlay is
separate from the accepted wiki overlay.

```text
sources_total:             14
url_sources_rejected:       1
documents_read:            14
read_errors:                0
candidates_total:          39
new_candidates:            28
duplicate_existing:         8
conflicts:                  2
overlay_delta_items:       27
quarantined_total:          9
rejected_total:             4
dry_run_overlay_items:    310
safe_to_apply_overlay_delta: true
safe_for_general_runtime: false
```

Dry-run QA against the 310-item overlay remains green for Entity QA v1,
Entity QA expansion, Adversarial Entity QA, and Cross-page Entity QA.

Example entity QA outputs:

```text
What does SpaceX develop?
SpaceX develops rockets and spacecraft.

Why is Forbes linked to Elon Musk?
Forbes is linked from the Elon Musk page as a weak contextual mention. It is
not treated as a stable factual relation by this overlay.

What does Forbes estimate about Elon Musk?
According to Forbes as of 2026-06, Elon Musk's estimated net worth is US$1.1
trillion. This is a volatile source-qualified estimate and should be rechecked.
```

Current/real-time questions audit unless the needed fact is present as an
accepted source-qualified fact. For example, `What is Tesla's current stock
price?` should audit.

Known caveat: renderer polish is ongoing. A directional relation verbalization
bug can make `Who is Elon Musk?` say `founded by SpaceX`; this is a surface
verbalization issue, not a knowledge, planner, or overlay issue.

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

# Wiki ingestion v2
python3 -m worldpgt.experiments.run_wiki_ingestion_v2

# Wiki overlay v1
python3 -m worldpgt.experiments.build_wiki_memory_overlay_v1

# Entity QA v1
python3 -m worldpgt.experiments.run_entity_qa_v1
```

---

## Safety

- Audit is a safe decision, not a failure, when evidence is missing,
  conflicting, unsupported, or volatile.
- Source-qualified volatile facts require recheck before operational use.
- Weak context links are contextual mentions, not stable factual relations.
- Weak links are never promoted to stable facts.
- Volatile facts are never auto-applied as stable/current facts.
- Current unsupported questions audit rather than using a generic fallback.
- Accepted memory v1 is not modified by wiki ingestion or wiki overlay builds.
- The accepted wiki overlay is not overwritten by self-ingestion.
- The self-ingestion dry-run overlay is separate.
- `safe_for_general_runtime` remains false for the wiki overlay.
- Validators and planner thresholds are part of the safety boundary and should
  not be weakened to increase coverage.

---

## Limitations

- Controlled benchmark results only; not open-domain QA.
- Narrow scope: 6 ambiguous terms in accepted-memory QA and 50 local pages in
  the current wiki fixture.
- Curated corpus and accepted memory are doing important work.
- Analyzers are rule/curriculum-based and need explicit expansion.
- Renderer surface quality is still being polished.
- The wiki overlay is isolated and is not accepted memory v1.
- Weak context links are not facts.
- Source-qualified volatile facts require recheck.
- Source extraction is still narrow.
- There is no autonomous web ingestion.
- There is no accepted overlay promotion yet.
- Current facts are not answered as live truth.
- No claim that Microworld is a general-purpose language model.
- 120-row benchmark. Results do not generalize beyond this controlled task.
- GPT-2 is an old base model (2019), not an instruction-tuned assistant.
- GPT-2 audit labels were assigned by a human pass after generation, not by a native model gate.
- Microworld realization is template-based; generated text quality is not representative of fluent generation.
- Microworld coverage is low (37/120 = 30.8%). The system abstains on ambiguous or weak-cue prompts, and the surface repair layer audits one further row it cannot repair without subject drift.
- RSS figures are approximate and environment-dependent.
- Sense memory covers 6 terms. Performance on out-of-vocabulary terms is undefined.

---

## Next Steps

1. Promote Overlay Delta v1 - validate and promote the safe self-ingestion
   overlay delta into a separate promoted overlay artifact, without modifying
   trusted accepted memory or the current accepted overlay.
2. Add a repeated efficiency benchmark with median/min/max.
3. Compare with a GPT-style baseline on the same controlled questions.

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
worldpgt/experiments/answer_planner_v1_summary.json
worldpgt/experiments/qa_generalization_test_v1_summary.json
worldpgt/experiments/wiki_ingestion_v2_candidates.json
worldpgt/experiments/wiki_ingestion_v2_candidates.csv
worldpgt/experiments/wiki_ingestion_v2_summary.json
worldpgt/experiments/wiki_ingestion_v2_review.json
worldpgt/experiments/accepted_wiki_memory_overlay_v1.json
worldpgt/experiments/accepted_wiki_memory_overlay_v1_summary.json
worldpgt/experiments/accepted_wiki_memory_overlay_v1_skipped.json
worldpgt/experiments/entity_qa_v1_outputs.csv
worldpgt/experiments/entity_qa_v1_summary.json
worldpgt/experiments/entity_qa_expansion_v1_outputs.csv
worldpgt/experiments/entity_qa_expansion_v1_summary.json
worldpgt/experiments/entity_qa_adversarial_v1.csv
worldpgt/experiments/entity_qa_adversarial_v1_outputs.csv
worldpgt/experiments/entity_qa_adversarial_v1_summary.json
worldpgt/experiments/cross_page_qa_v1.csv
worldpgt/experiments/cross_page_qa_v1_outputs.csv
worldpgt/experiments/cross_page_qa_v1_summary.json
```
