# Experiments

Microworld experiments are exploratory research checks for explicit graph
reasoning and audit-driven learning. They should be read as bounded evidence,
not as claims of general superiority over neural systems.

The current research question is:

```text
Can feedback over symbolic graph predictions be compressed into explicit state
that changes future behavior without neural retraining?
```

Current answer: partially yes.

## Human Audit Baseline

Manual audit of ConceptNet-derived predictions:

```text
reviewed: 104
useful overall: 78.8%
made_of useful: 86.2%
part_of useful: 76.7%
is_a useful: 75.6%
mixed reasoning useful: 76.7%
```

`useful` means `correct + plausible`.

Interpretation:

* relation-specific graph reasoning can produce useful predictions
* mixed reasoning is viable but still conservative
* manual audit is necessary because automatic graph completion metrics can mark
  plausible missing edges as false positives

Limit:

* this is a small exploratory audit, not a formal benchmark

## Audit-Driven Trust Learning

Microworld maps manual audit feedback into explicit trust state. The trust state
can lower or raise confidence for relation families, rules, drift categories,
and evidence sources.

Learning path:

```text
manual audit rows
-> compact trust profile
-> prediction confidence adjustment
-> changed behavior on future data
```

This does not use gradient descent, backpropagation, fine-tuning, or hidden
weight updates.

## Trust Transfer Experiment

Trust learned from audit data was evaluated on an unseen TEST split.

```text
baseline accepted: 195
learned accepted: 99
suppressed: 96
```

Interpretation:

```text
feedback -> trust profile -> changed behavior on unseen split
```

The learned trust profile transferred to new rows and changed acceptance
behavior without neural retraining.

## Feedback Compression Benchmark

Benchmark result for 10,000 audit rows:

```text
raw audit history: ~500,360 tokens
trust state: ~313 tokens
compression: ~1598.6x
```

This is a central result because it separates memory from transcript length. The
system does not need to keep the full audit history in context to apply what was
learned. It compresses repeated feedback into explicit reusable buckets.

Interpretation:

```text
large feedback history
-> tiny explicit state
-> future behavior change
```

## Naive Suppression Audit

Initial suppression used a direct trust-threshold rule:

```text
baseline_confidence >= threshold
AND learned_confidence < threshold
```

Manual audit:

```text
total reviewed: 50
should_suppress: 11
should_keep: 38
unclear: 1
suppression_precision: 0.224
```

Interpretation:

* learned trust changed behavior
* trust alone was too aggressive as a final suppression decision
* many useful predictions were suppressed

Conclusion:

```text
trust learning is useful as a signal, not as the final decision layer
```

## Delta Calibration

The next attempt used confidence drop magnitude as a calibration signal.

Result:

* useful suppressions and harmful suppressions had similar confidence drops
* delta magnitude alone did not separate bad suppressions from useful predictions

Interpretation:

```text
confidence delta is diagnostic context, not a complete suppression policy
```

## Quality-Aware Suppression

A separate suppression policy layer was added:

```text
graph prediction
-> baseline confidence
-> learned trust confidence
-> suppression candidate
-> quality-aware policy
-> final suppression
```

Quality-aware v1:

```text
exported rows: 12
should_suppress: 11
should_keep: 1
suppression_precision: 0.917
```

The only false suppression was:

```text
talbe --made_of--> wood
```

Reason:

* the prediction is useful
* `talbe` is almost certainly a typo for `table`
* this should be handled by normalization/canonicalization, not suppression

Quality-aware v2:

* source noise no longer triggers suppression
* target noise still triggers suppression
* output contained 11 rows
* all 11 rows had `target = oxegen`
* `talbe --made_of--> wood` disappeared

Interpretation:

* bad target -> suppress
* bad source -> normalize later
* bad relation or pattern -> lower trust
* clean prediction with source typo -> keep after normalization

## Main Research Conclusion

Microworld supports a narrow but useful result:

* audit feedback can be compressed into a tiny explicit trust state
* that state can transfer to unseen data
* behavior can change without backpropagation
* errors can be debugged directly
* adding explicit policy layers can sharply improve behavior
* trust memory, decision policy, and normalization should be separate components

It does not prove that graph systems are generally superior to neural networks.
It shows that explicit memory and trust learning can be useful for graph-based
symbolic reasoning tasks where auditability and compact feedback state matter.

## Suggested Comparisons

Next research comparisons should include:

* larger audit samples
* relation-specific suppression policies
* normalization candidate export
* typo/canonicalization layer
* target normalization with semantic re-evaluation
* LLM-only memory baselines using long context or transcript history
