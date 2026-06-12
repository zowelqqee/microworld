# Research Snapshot — Microworld Controlled Continuation v1.2

Date: 2026-06-13  
Benchmark: Controlled Continuation v1 (120 prompts)  
Policy version: v1.2 (anti-cue guarded)

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
