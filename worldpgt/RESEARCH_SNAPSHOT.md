# Research Snapshot - Microworld / worldpgt

---

## Current QA and Knowledge Overlay Snapshot (2026-06-14)

worldpgt implements controlled QA over explicit accepted memory and an isolated
wiki candidate overlay. It is an experimental explicit-memory, graph, trust,
policy, and audit based reasoning and QA system. It explores whether useful
controlled QA, memory, reasoning, safe abstention, and knowledge ingestion can
be built without neural weights, backpropagation, fine-tuning, GPT-style
next-token rendering, embeddings, GPU, or network calls.

Correct framing: Microworld demonstrates a lightweight, deterministic,
auditable QA architecture over explicit memory and isolated knowledge overlays.
It is currently strong on controlled benchmark domains, explicit memory,
source-aware facts, safe abstention/audit, and low runtime cost. It is currently
limited by narrow scope, curated inputs, rule/curriculum-based analyzers, and
surface renderer quality.

LLMs learn to speak and world understanding emerges as a side effect.
Microworld tries to build explicit world memory first, then use language as an
interface to that world.

### QA Layer Components

| Component | Role |
|---|---|
| `QuestionAnalyzer` / generalized analyzer | detects QA intent from surface form |
| `AnswerPlanner` | selects response strategy from accepted memory |
| `AnswerRenderer` | composes semantic answer forms |
| `AnswerValidator` | checks correctness; flags quality issues |
| `AuditRenderer` | helpful abstention text for ambiguous questions |
| `SemanticLanguageRealizer` | clause-level language realization |
| `ContrastRealizer` | contrast explanations for distinguish_senses |
| `AcceptedMemoryProvider` | loads accepted facts, patterns, senses (221 items) |
| `wiki_ingestion_v2` | creates reviewable candidates from local curated pages |
| `WikiMemoryOverlayProvider` | serves isolated wiki overlay memory |
| `EntityQuestionAnalyzer` | detects entity QA intent |
| `EntityAnswerPlanner` | plans entity overlay answers or audits |
| `EntityAnswerRenderer` | renders entity, relation, link, and source-fact answers |
| `EntityAnswerValidator` | validates entity QA output quality and safety |
| `cross_page_qa` components | controlled graph-style multi-hop QA over the isolated wiki overlay |
| self-ingestion pipeline | offline dry-run overlay delta proposal with quarantine and regression gates |

Supported QA intents: `define_sense`, `classify_context`, `explain_cue`,
`distinguish_senses`, `unknown_or_ambiguous`

Supported entity QA intents: `define_entity`, `relation_lookup`,
`link_explanation`, `source_fact_lookup`, `unknown_or_unsupported`

### Accepted Memory Provider

Artifact: `worldpgt/experiments/accepted_knowledge_memory_v1.json`

```text
total items:     221
fact items:      163
pattern items:    58
ambiguous terms:   6  (bank, bat, crane, rock, seal, spring)
senses:           12
positive cues:   104
```

Provider load summary:

```text
items loaded:     221
terms loaded:       6
senses loaded:     12
positive cues:    104
```

### Pipeline

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

### Benchmark Summary

| Benchmark | File / output | Total | Correct | Wrong | Answered | Audited | Precision |
|---|---|---:|---:|---:|---:|---:|---:|
| Main QA | `qa_prompts_v1.csv` / `answer_planner_v1_summary.json` | 48 | 48 | 0 | 42 | 6 | 1.0 |
| Generalization QA | `qa_generalization_test_v1.csv` / `qa_generalization_test_v1_summary.json` | 24 | 24 | 0 | 19 | 5 | 1.0 |
| Entity QA | `entity_qa_prompts_v1.csv` / `entity_qa_v1_summary.json` | 28 | 28 | 0 | 23 | 5 | 1.0 |
| Entity QA expansion | `entity_qa_expansion_v1_summary.json` | 111 | 111 | 0 | 79 | 32 | 1.0 |
| Adversarial Entity QA | `entity_qa_adversarial_v1_summary.json` | 68 | 68 | 0 | 6 | 62 | 1.0 |
| Cross-page Entity QA | `cross_page_qa_v1_summary.json` | 71 | 71 | 0 | 50 | 21 | 1.0 |

Across the current controlled QA benchmark layers, Microworld handles 350
prompts with 0 wrong decisions: 219 answered and 131 safely audited. Answer
precision is 1.0 on each benchmark family. These results are scoped to
controlled domains and isolated overlays; they are not open-domain or live-fact
claims.

### Main QA Benchmark - 48 Controlled Questions

```text
qa_total:          48
answer_count:      42
audit_count:        6
correct_count:     48
wrong_count:        0
accuracy:          1.0
answer_precision:  1.0
quality_flagged:    0
"associated with" in outputs: 0
```

The renderer no longer emits flat `associated with` lists. All outputs use
semantic answer forms: common clues, common contexts, common signs,
location-aware phrases, action/agency-aware phrases, compact contrast
explanations.

### Example QA Outputs

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

### Generalization Benchmark - 24 Novel Phrasings

Source: `worldpgt/experiments/qa_generalization_test_v1.csv`

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

### Wikipedia/Wikidata-style Ingestion v2

Deterministic, offline, local curated fixture only, no network, and
candidate-generation only. It does not modify accepted memory or runtime
memory.

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

Files:

```text
worldpgt/knowledge/wiki_ingestion_v2_types.py
worldpgt/knowledge/wiki_page_reader.py
worldpgt/knowledge/wiki_entity_extractor.py
worldpgt/knowledge/wiki_claim_extractor.py
worldpgt/knowledge/wiki_claim_normalizer.py
worldpgt/knowledge/wiki_ingestion_v2.py
worldpgt/experiments/run_wiki_ingestion_v2.py
worldpgt/experiments/wiki_pages_curated_v2.json
worldpgt/tests/test_wiki_ingestion_v2.py
```

Artifacts:

```text
worldpgt/experiments/wiki_ingestion_v2_candidates.json
worldpgt/experiments/wiki_ingestion_v2_candidates.csv
worldpgt/experiments/wiki_ingestion_v2_summary.json
worldpgt/experiments/wiki_ingestion_v2_review.json
```

### Wiki Candidate Memory Overlay v1

The overlay is isolated from wiki ingestion candidates. It is not general
runtime memory and not accepted memory v1.

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

Volatile source-qualified facts are stored with `as_of=2026-06`,
`requires_recheck=true`, and `risk=high`. They are never converted to stable
relations. Weak contextual links are `weak_context_only` and are never promoted
to facts.

Files:

```text
worldpgt/knowledge/wiki_memory_overlay_types.py
worldpgt/knowledge/wiki_candidate_overlay_builder.py
worldpgt/knowledge/wiki_memory_overlay_provider.py
worldpgt/experiments/build_wiki_memory_overlay_v1.py
worldpgt/tests/test_wiki_memory_overlay_v1.py
```

Artifacts:

```text
worldpgt/experiments/accepted_wiki_memory_overlay_v1.json
worldpgt/experiments/accepted_wiki_memory_overlay_v1_summary.json
worldpgt/experiments/accepted_wiki_memory_overlay_v1_skipped.json
```

### Entity QA Benchmark v1

Controlled QA over the isolated wiki overlay.

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

Examples:

```text
What does SpaceX develop?
SpaceX develops rockets and spacecraft.

Why is Forbes linked to Elon Musk?
Forbes is linked from the Elon Musk page as a weak contextual mention. It is
not treated as a stable factual relation by this overlay.

What does Forbes estimate about Elon Musk?
According to Forbes as of 2026-06, Elon Musk's estimated net worth is US$1.1
trillion. This is a volatile source-qualified estimate and should be rechecked.

What is Tesla's current stock price?
Audits because current/real-time data is not available as an accepted
source-qualified fact.
```

Known caveat: renderer polish is ongoing. `Who is Elon Musk?` may say
`founded by SpaceX`. This is a directional relation verbalization bug, not a
knowledge, planner, or overlay bug.

Files:

```text
worldpgt/entity_qa/__init__.py
worldpgt/entity_qa/types.py
worldpgt/entity_qa/entity_question_analyzer.py
worldpgt/entity_qa/entity_answer_planner.py
worldpgt/entity_qa/entity_answer_renderer.py
worldpgt/entity_qa/entity_answer_validator.py
worldpgt/experiments/run_entity_qa_v1.py
worldpgt/experiments/entity_qa_prompts_v1.csv
worldpgt/tests/test_entity_qa_v1.py
```

Artifacts:

```text
worldpgt/experiments/entity_qa_v1_outputs.csv
worldpgt/experiments/entity_qa_v1_summary.json
```

### Entity QA Expansion v1

Expanded controlled entity QA over the same isolated 50-page overlay.

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

Adversarial entity QA verifies audit behavior for relation inversion,
weak-link-as-fact prompts, current/real-time prompts, category confusion,
invalid universal/generalization claims, source-qualified volatility, and
unsupported private/personal data.

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

### Cross-page Entity QA v1

Controlled graph-style multi-hop QA over the isolated 50-page overlay.

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

Examples: Musk -> SpaceX -> rockets answers; Musk -> Starlink audits because no
explicit stable path exists; SpaceX -> Starlink returns a weak contextual link
caveat rather than a stable factual relation.

### Wikipedia Self-Ingestion v1

Safe offline self-feeding pipeline under `worldpgt/self_ingestion/`. It reads
local Wikipedia-like docs, converts them to wiki-like pages, reuses unchanged
`WikiIngestionV2` and `WikiCandidateOverlayBuilder`, classifies
deltas/conflicts/quarantine, proposes an overlay delta, creates a dry-run
overlay, and runs QA/adversarial/cross-page regressions. It never writes raw
text directly into accepted memory.

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

Dry-run QA against the 310-item overlay:

```text
Entity QA v1:             28/28
Entity QA expansion:     111/111
Adversarial Entity QA:    68/68
Cross-page Entity QA:     71/71
```

### Efficiency Snapshot

Measured using `/usr/bin/time -l` on macOS:

| Batch | Items | Result | Time | Peak RSS | Approx throughput |
|---|---:|---|---:|---:|---:|
| Main QA | 48 | 48 correct, 0 wrong, 42 answers, 6 audits | ~0.070 s | ~24.73 MB | ~686 questions/sec |
| Generalization QA | 24 | 24 correct, 0 wrong, 19 answers, 5 audits | ~0.140 s | ~24.61 MB | ~171 questions/sec |
| Wiki ingestion | 283 | 0 review errors | ~0.060 s | ~23.8 MB | order-of-magnitude only |
| Wiki overlay | 283 | 0 skipped | ~0.050 s | ~23.7 MB | order-of-magnitude only |
| Entity QA | 28 | 28 correct, 0 wrong, 23 answers, 5 audits | ~0.060 s | ~23.61 MB | ~467 questions/sec |
| Cross-page QA | 71 | 71 correct, 0 wrong, 50 answers, 21 audits | ~0.160 s | ~24 MB | order-of-magnitude only |
| Self-ingestion dry run | 4 regressions | all green on 310-item dry-run overlay | ~0.210 s | ~25.8 MB | order-of-magnitude only |

The current Python implementation runs these small controlled benchmark batches
in about 0.05-0.21 seconds with roughly 23.7-25.8 MB peak RSS. These are
single-run local measurements and should be treated as order-of-magnitude
efficiency indicators, not final benchmark claims.

### Safety Constraints

* no neural weights
* no backpropagation
* no fine-tuning
* no GPT renderer
* no embeddings
* no GPU or network calls in the controlled pipelines
* no generic trusted fallback
* no threshold weakening
* no forced answers on ambiguity
* audit is a safe decision, not a failure
* source-qualified volatile facts require recheck
* weak context links are not facts
* weak links are never promoted to stable facts
* volatile facts are never auto-applied as stable/current facts
* current unsupported questions audit
* accepted memory v1 is not modified by wiki overlay
* accepted wiki overlay is not overwritten by self-ingestion
* dry-run overlay is separate
* ingestion extraction unchanged
* overlay builder semantics unchanged
* `safe_for_general_runtime` remains false for wiki overlay
* `sense_memory.py` not modified
* `nanogpt/` not touched

### Test Status

Latest requested verification target:

```text
python3 -m pytest -q  ->  2007 passed
```

### Next Steps

1. Promote Overlay Delta v1 - validate and promote the safe self-ingestion
   overlay delta into a separate promoted overlay artifact, without modifying
   trusted accepted memory or the current accepted overlay.
2. Add a repeated efficiency benchmark with median/min/max.
3. Compare with a GPT-style baseline on the same controlled questions.

---

## Controlled Continuation v1.2 — Surface Repair Layer (2026-06-13)

Benchmark: Controlled Continuation v1 (120 prompts)
Policy version: v1.2 (anti-cue guarded)

A deterministic, rule-based **surface repair layer** was added after the semantic
renderer and before final emission. It does not generate text and does not change
sense scoring, risk policy, or any decision threshold. It applies fixed string
fixes for residual grammar / role / coreference bugs and re-validates; if a
candidate cannot be repaired cleanly it routes to audit. No neural weights, no
GPT renderer, no training are involved (the `nanogpt/` baseline is untouched).

Honest summary: after the deterministic surface repair layer, Microworld emits
one fewer continuation on v1.2 but removes the remaining measured
semantic-render-quality flags while preserving zero measured wrong continuations.

| Metric                       | Old (pre-repair) | New (with repair) |
|------------------------------|------------------|-------------------|
| continue_count               | 38               | 37                |
| audit_count                  | 82               | 83                |
| wrong_continue_count         | 0                | 0                 |
| precision_on_continued       | 1.000            | 1.000             |
| semantic-quality flagged     | 1 / 38           | 0 / 37            |
| coverage_rate                | 0.3167           | 0.3083            |
| answerable_recall            | 0.3455           | 0.3364            |

Repaired rows:

- `v1-007` — connector comma (`before the swing he steadied himself` → `before the swing, he steadied himself`)
- `v1-009` — prey coreference (`catch another fish` → `catch its prey`)
- `v1-011` — object repetition (`close the envelope` → `close it`)
- `v1-043` — body-part subject/action (`its wings searched for insects` → `its wings spread wide`)
- `v1-008` — object repetition (`dropped the bat` → `dropped it`)

Audited row:

- `v1-051` — attachment / subject drift, not repairable without inventing a
  subject → `audit_reason=no_safe_repaired_candidate`

The numbers above are locked by `worldpgt/tests/test_surface_repair_benchmark_gate.py`.
The body of this snapshot below reflects the pre-repair v1.2 measurement and is
retained for history; current emitted counts are 37 continue / 83 audit.

---

## What Was Tested

A 120-row controlled continuation benchmark where each prompt contains one lexically ambiguous term (bank, bat, seal, crane, spring, rock). Each row has an expected sense and a difficulty label (cue_rich, delayed_cue, weak_cue, conflicting_cue, negation, misleading_surface_cue, no_clear_answer, no_known_term).

Two systems were evaluated:

1. **Microworld v1.2** — explicit policy system with inspectable sense memory, deterministic cue scoring, anti-cue overrides, guard rules, and a native audit/suppress path.
2. **GPT-2 base (124M)** — inference via local nanoGPT. No fine-tuning. Next-token generation, no native sense selection or audit decision.

The task is purely ambiguity resolution and continuation plausibility within this controlled set. It is not a general NLU or generation benchmark.

---

## Why GPT-2 Is a Useful but Limited Baseline

GPT-2 is the simplest available neural baseline with known public weights. It was trained on web text (WebText) and can generate fluent English continuations. Using it here gives a concrete point of comparison: what does a system with 123M parameters and no task-specific design produce on the same prompts?

Limitations that must be stated clearly:

- GPT-2 (2019) is not representative of modern LLMs. ChatGPT, Claude, Gemini, and comparable instruction-tuned models are qualitatively different.
- GPT-2 has no native sense selection, no confidence score on the relevant axis, and no audit gate. It generates text; quality must be assessed externally.
- GPT-2 labels (good / bad / unclear) were applied in a single human audit pass after generation. This is not the same as the deterministic audit Microworld performs at inference time.
- GPT-2 runtime on CPU is slow (~0.422 s/prompt) relative to Microworld. This reflects hardware constraints, not an inherent architectural ceiling.

---

## What Microworld Does Differently

Microworld makes its decisions explicit and auditable at inference time rather than after the fact.

**Pipeline:**
1. Parse the prompt for a known ambiguous term.
2. Score candidate senses by lexical cue overlap (non-negated cues matched against the prompt).
3. Apply negation detection: cues preceded by a negation word within a 3-token window are counted as negated rather than positive evidence.
4. Apply anti-cue rules: known phrases that contradict a sense zero out that sense's score.
5. Apply guard rules: weak cues (e.g., "cash" alone for "bank:financial_institution", "player" alone for "bat:sports_equipment") are insufficient without a strong corroborating cue.
6. Policy decision: continue if top sense score ≥ threshold and margin ≥ threshold and no conflict/anti-cue/guard failure; audit otherwise.
7. Realization: pick the first template continuation for the selected sense and append it to the prompt.

The system emits an empty continuation when it audits. It never generates text for a prompt it cannot confidently resolve.

**Tradeoff:** low coverage, high precision within emitted continuations.

---

## Main Result

On this 120-row benchmark:

- Microworld emitted continuations for 38 prompts (31.7% coverage). All 38 had the correct sense. Zero wrong continuations.
- GPT-2 generated output for all 120 prompts. 76 were labeled good, 11 bad, 33 unclear by post-hoc audit. 7 had an incorrect sense judgment.

The systems are not interchangeable. Microworld achieves higher precision on its emitted subset by refusing to emit continuations when evidence is insufficient. GPT-2 always emits, producing more usable output overall but also more errors and ambiguous output.

---

## Quality Comparison

| Metric                        | Microworld v1.2 | GPT-2 base   |
|-------------------------------|-----------------|--------------|
| Continued / generated rows    | 38              | 120          |
| Wrong sense count             | 0               | 7            |
| Precision on emitted output   | 1.000           | 0.8736 (audited) |
| Correct sense identified      | 38              | 101          |

GPT-2 "precision" is computed as good / (good + bad) = 76 / 87 = 0.8736. Unclear outputs (33) are excluded from the precision denominator. This audit method is not directly comparable to Microworld's native precision since the audit was applied externally and after the fact.

---

## Risk / Coverage Comparison

| Metric                    | Microworld v1.2 | GPT-2 base         |
|---------------------------|-----------------|--------------------|
| Continue count            | 38              | 120 (all)          |
| Audit count               | 82              | 0 (no native gate) |
| Coverage rate             | 0.3167          | 1.000              |
| Answerable recall         | 0.3455          | —                  |
| Wrong continue rate       | 0.000           | —                  |

Coverage rate = continued / total (38/120). Answerable recall = continued / answerable rows (38/110, excluding no_known_term and no_clear_answer rows). GPT-2 coverage is 1.0 by construction; it does not abstain.

The central observation: Microworld trades coverage for a zero measured wrong-continue rate on this dataset. GPT-2 maximizes coverage but incurs bad and unclear outputs. Neither outcome is inherently better — it depends on downstream tolerance for errors versus gaps.

---

## Efficiency Comparison

| Metric                    | Microworld v1.2   | GPT-2 base        |
|---------------------------|-------------------|-------------------|
| Avg time / prompt         | ~0.000046 s       | ~0.422 s          |
| Peak RSS                  | ~22.9 MB          | ~1348.7 MB        |
| State / model size        | 7,882 bytes       | ~548 MB           |
| Trainable parameters      | 0                 | 123.65 M          |

Microworld is ~9,000× faster per prompt and ~59× smaller in RSS. These differences are expected: Microworld is a lookup + threshold system; GPT-2 is a transformer running forward passes. The comparison is informative for resource-constrained deployment contexts, not as evidence of architectural superiority.

---

## Failure Modes

**Microworld failure modes observed or expected:**

- **Weak cues**: prompts with common but non-diagnostic words (e.g., "the bank manager") score zero and go to audit. High audit rate on weak_cue rows (20/20 audited in v1.2).
- **Conflicting cues**: prompts with lexical cues from multiple senses trigger conflict detection and audit. v1.2 audits 18/20 conflicting_cue rows.
- **Negation leakage**: the 3-token negation window is a heuristic; complex negation structures (e.g., "not the kind of bank that holds money") may not be fully captured.
- **Out-of-vocabulary terms**: Microworld returns audit for any prompt where no known term is found. It does not attempt to generalize.
- **Template quality**: emitted continuations are picked from a fixed template list. They are grammatically appended but may be awkward in context.

**GPT-2 failure modes observed:**

- **Sense drift**: GPT-2 sometimes resolves the initial sense correctly but then continues with off-topic text (e.g., v1-002: "asked about credit to a friend's computer...").
- **Truncation / fragment**: some outputs are a single punctuation mark or one-word fragment with no usable content.
- **Hallucination**: generated text is internally fluent but factually wrong or contextually inconsistent.
- **No abstention**: GPT-2 generates output even for prompts that are genuinely ambiguous or unanswerable.

---

## Limitations

- **Dataset size**: 120 rows is too small to draw statistically reliable conclusions. The zero wrong-continue count is consistent with Microworld's design but is not a guarantee at scale.
- **GPT-2 is not representative**: modern instruction-tuned models would likely perform substantially better on this task.
- **Single audit pass**: GPT-2 labels were applied in one pass. Inter-annotator agreement was not measured.
- **Template realization**: Microworld continuations are template-derived; they do not represent generative fluency.
- **Coverage gap**: 68.3% of prompts go to audit in Microworld. This is a practical limitation for any downstream use.
- **RSS measurement**: peak RSS is measured via `resource.ru_maxrss` in subprocesses; figures are approximate and environment-dependent.
- **Vocabulary coverage**: Microworld covers 6 terms with 12 senses. Scaling requires manual curation of additional entries.

---

## Next Steps

In rough priority order:

1. **Strict GPT-2 audit pass** — apply a second independent annotator pass to GPT-2 outputs and compute inter-annotator agreement. Current labels are a single pass.
2. **Larger benchmark** — more prompts per term, more seeds, broader ambiguous-term vocabulary to reduce per-category sample variance.
3. **tinyGPT / nanoGPT trained-from-scratch baseline** — compare Microworld against a smaller neural model trained specifically on this task family rather than an off-the-shelf general model.
4. **Human audit instead of single-annotator audit** — use multiple annotators with a majority-vote or IRR threshold for GPT-2 labels.
5. **Better realization layer** — replace fixed templates with a small constrained generation step or graph-based composer to produce more natural continuations without abandoning explicit sense selection.
6. **SQLite / disk-backed explicit state** — replace the in-memory `ExplicitSenseMemory` dict with a persistent store for larger-scale term coverage and reproducible snapshot versioning.
7. **Subprocess-isolated memory benchmark** — run RSS measurement in process-isolated subprocesses across repeated runs to reduce OS memory-management noise and report median/spread rather than single-run peak.
