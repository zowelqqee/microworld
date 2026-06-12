# worldmvp / Microworld

Microworld is an experimental graph-based memory, reasoning, and learning
system. It explores whether useful behavioral learning can happen through
explicit graph state, audit feedback, and trust calibration without neural
weights, backpropagation, or fine-tuning.

The project is intentionally research-oriented. It does not claim that symbolic
graphs beat neural networks. It explores a complementary path: compact explicit
memory and trust learning for graph reasoning, where behavior can be audited,
compressed, transferred, and corrected without updating neural weights.

Current test status:

```text
870 passing tests
```

## Why This Exists

Modern AI systems are powerful, but much of their knowledge and behavior is
stored implicitly inside model weights or long context histories. That makes it
hard to inspect why a behavior changed, compress feedback into durable memory,
or debug errors at the level of relations, nodes, and policies.

Microworld asks a narrower question:

```text
Can some memory, reasoning, and learning behavior be achieved more efficiently
and more transparently than by simply growing neural models wider, expanding
context, or relying on backpropagation?
```

The current answer is partially yes. For graph-based symbolic reasoning tasks,
Microworld shows that audit feedback can be compressed into a tiny explicit
trust state, that the state can transfer to unseen data, and that behavior can
change without retraining a neural model.

## Current Architecture

Knowledge is represented as explicit graph relations:

```text
source --relation_type--> target
```

The current system includes:

* graph memory
* ConceptNet import
* pattern discovery
* transitive reasoning
* mixed-pattern reasoning
* structural similarity
* concept discovery
* relation proposal
* hub penalty
* relation trust
* node quality
* relation drift
* relation blacklist
* audit pipeline
* audit-driven trust learning
* trust transfer experiment
* feedback compression benchmark
* suppression audit
* quality-aware suppression policy
* full pipeline demo

The main reasoning path is:

```text
graph memory
-> pattern discovery
-> prediction
-> baseline confidence
-> learned trust confidence
-> suppression candidate
-> quality-aware policy
-> final decision
```

The key design choice is separation of concerns:

* trust memory estimates whether relation/rule families have been reliable
* decision policy decides what to suppress or keep
* normalization should repair source and target spelling/canonicalization issues

## Latest Experimental Results

### Human Audit Baseline

Manual audit of ConceptNet-derived graph predictions:

```text
reviewed: 104
useful overall: 78.8%
made_of useful: 86.2%
part_of useful: 76.7%
is_a useful: 75.6%
mixed reasoning useful: 76.7%
```

`useful` means the prediction was labeled correct or plausible. These are small
exploratory audits, not formal benchmark claims.

### Audit-Driven Trust Learning

Microworld can turn manual audit feedback into a compact trust profile. That
profile changes behavior on unseen TEST data without backpropagation.

Trust transfer experiment:

```text
baseline accepted: 195
learned accepted: 99
suppressed: 96
```

Interpretation:

```text
feedback -> trust profile -> changed behavior on unseen split
```

This confirms that Microworld can learn behavioral preferences from audit
feedback without updating neural weights.

### Feedback Compression

Feedback compression benchmark with 10,000 audit rows:

```text
raw audit history: ~500,360 tokens
trust state: ~313 tokens
compression: ~1598.6x
```

This is one of the strongest current results. It shows an alternative to simply
expanding context, memory logs, or model size: large feedback histories can be
compressed into a tiny, explicit, inspectable state.

### Suppression Audit

The first suppression rule was intentionally simple:

```text
baseline_confidence >= threshold
AND learned_confidence < threshold
```

Manual audit of that naive rule:

```text
total reviewed: 50
should_suppress: 11
should_keep: 38
unclear: 1
suppression_precision: 0.224
```

The learned trust signal changed behavior, but it was too aggressive as the
final suppression decision. It suppressed many useful predictions.

Delta calibration did not solve the issue. Useful and harmful suppressions had
similar confidence drops, so confidence delta alone did not separate bad
predictions from useful ones.

### Quality-Aware Suppression

A separate suppression policy layer was added:

```text
graph prediction
-> baseline confidence
-> learned trust confidence
-> suppression candidate
-> quality-aware policy
-> final suppression
```

Quality-aware v1 exported 12 rows. Manual audit found:

```text
should_suppress: 11
should_keep: 1
suppression_precision: 0.917
```

The only false suppression was:

```text
talbe --made_of--> wood
```

That prediction is almost certainly useful. The problem is a source typo for
`table`, so it belongs in normalization/canonicalization rather than
suppression.

Quality-aware v2 no longer lets source noise trigger suppression. Target noise
still triggers suppression. The output contained 11 rows, all with:

```text
target = oxegen
```

The `talbe --made_of--> wood` row disappeared.

Current interpretation:

* bad target -> suppress
* bad source -> normalize later
* bad relation or pattern -> lower trust
* clean prediction with a source typo -> keep after normalization

## What Was Learned

The main research conclusion is deliberately modest:

* audit feedback can be compressed into a tiny explicit trust state
* that state can transfer to unseen data
* behavior can change without backpropagation
* errors can be debugged directly
* small explicit policy layers can sharply improve behavior
* trust memory, decision policy, and normalization should be separate components

Microworld does not prove general superiority over neural networks. It shows
that explicit graph memory and audit-driven trust learning are useful research
tools for a bounded class of symbolic reasoning problems.

## Running

From this directory:

```bash
pytest -q
```

Example demos:

```bash
python3 examples/full_pipeline_demo.py
python3 examples/trust_transfer_experiment.py
python3 examples/feedback_scaling_benchmark.py
python3 examples/suppression_audit_export.py
```

## Documentation

Start with:

* `docs/index.md`
* `docs/architecture.md`
* `docs/experiments.md`
* `docs/suppression_policy.md`

## Limitations

* Current results are exploratory and based on bounded graph reasoning tasks.
* The ConceptNet work uses filtered samples, not a full benchmark.
* Manual audits are still small.
* Mixed-pattern reasoning is conservative and manually allowlisted.
* Trust learning is useful as a signal but should not be the final decision layer.
* Delta-only suppression calibration did not separate useful from harmful cases.
* Normalization/canonicalization is not yet a mature component.
* There is no perception layer, reinforcement learning loop, or neural training.

## Next Steps

* normalization candidate export
* typo/canonicalization layer
* target normalization with semantic re-evaluation
* larger audit sample
* relation-specific suppression policies
* compare against LLM-only memory baselines

## Status

Experimental. The goal is not to build a production knowledge graph or a
replacement for LLMs, but to create a controlled environment where hypotheses
about explicit memory, reasoning, feedback compression, and inspectable learning
can be tested.
