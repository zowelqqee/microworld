# Audit And Evaluation

Microworld uses human audit as a calibration signal, not as hidden training.

The main audit labels are:

```text
correct
plausible
wrong
unclear
```

`correct + plausible` is reported as useful.

## Current Audit Highlights

ConceptNet human audit:

```text
104 reviewed
78.8% useful
```

By relation:

```text
made_of 86.2%
part_of 76.7%
is_a    75.6%
```

Mixed reasoning:

```text
30 reviewed
76.7% useful
```

These are small exploratory audits, not formal benchmark claims.

## Audit-Driven Trust Learning

Manual audit feedback can be compressed into a trust profile and applied to
future predictions.

Trust transfer experiment on unseen TEST data:

```text
baseline accepted: 195
learned accepted: 99
suppressed: 96
```

Interpretation:

```text
feedback -> trust profile -> changed behavior on unseen split
```

This changes symbolic prediction behavior without neural weights,
backpropagation, or fine-tuning.

## Feedback Compression

Feedback compression benchmark:

```text
10,000 audit rows
raw audit history: ~500,360 tokens
trust state: ~313 tokens
compression: ~1598.6x
```

The feedback history grows with every audit row. The trust state stays compact
because repeated feedback is aggregated into explicit reliability buckets.

## Suppression Audit

Naive suppression rule:

```text
baseline_confidence >= threshold
AND learned_confidence < threshold
```

Manual result:

```text
total reviewed: 50
should_suppress: 11
should_keep: 38
unclear: 1
suppression_precision: 0.224
```

This showed that learned trust is a useful signal but not a final decision
layer. It suppressed many useful predictions.

Delta calibration did not solve the issue because useful and harmful
suppressions had similar confidence drops.

Quality-aware suppression improved the manual audit sharply:

```text
exported rows: 12
should_suppress: 11
should_keep: 1
suppression_precision: 0.917
```

The only false suppression was `talbe --made_of--> wood`, which should be
handled by normalization because the source is likely a typo for `table`.

Quality-aware v2 no longer suppresses solely because of source noise. It still
allows target noise to trigger suppression.

## Exporting Audit Rows

```bash
python3 examples/pattern_audit_export.py \
  --input data/conceptnet_sample.csv \
  --output data/pattern_audit.csv \
  --mode transitive \
  --limit 100
```

Modes:

```text
transitive
mixed
all
```

Useful flags:

```bash
--use-relation-trust
--use-node-quality
--use-relation-drift
--max-intermediate-degree 20
--include-disabled-relations
```

The output includes:

```text
source
relation_type
target
path_length
confidence
reason
evidence
manual_label
notes
```

## Summarizing Audit Results

```bash
python3 examples/pattern_audit_summary.py \
  --input data/pattern_audit_filtered.csv
```

## Why Manual Audit Matters

ConceptNet samples are incomplete and noisy.  Automatic precision can mark a
plausible missing edge as a false positive.  Manual audit lets the project
separate:

- genuinely useful predictions
- relation drift
- overgeneralization
- sense ambiguity
- dataset noise
