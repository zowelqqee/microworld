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

