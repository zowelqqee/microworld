# Demos

Run all commands from the project root:

```bash
cd /Users/arseniyabramidze/mini-worldgrad/worldmvp
```

## Application Demo

Best starting point:

```bash
python3 examples/application_demo.py
```

This prints four sections:

1. Strong predictions
2. Weak but useful predictions
3. Rejected / risky predictions
4. Summary

It demonstrates Microworld as an explainable graph reasoning engine, not just
a benchmark runner.

## Pattern Prediction Demo

```bash
python3 examples/pattern_prediction_demo.py
```

Shows transitive pattern prediction, hub penalties, and explanations.

## Mixed Pattern Demo

```bash
python3 examples/mixed_pattern_demo.py
```

Shows manually allowed mixed rules:

```text
is_a + capable_of => capable_of
is_a + has_property => has_property
is_a + used_for => used_for
is_a + has_a => has_a
part_of + made_of => made_of
```

## Relation Proposal Demo

```bash
python3 examples/relation_proposal_demo.py
```

Learns candidate output relation labels from existing graph closures.

Useful when the system finds a meaningful A-C connection but the output
relation label is too crude.

## Drift-Aware Prediction Demo

```bash
python3 examples/drift_aware_prediction_demo.py
```

Shows before/after confidence:

```text
song made_of sounds          unchanged
book made_of wood            raw material penalty
blood made_of iron           atomic component penalty
community made_of ideals     abstract component penalty
```

## Relation Drift Report

```bash
python3 examples/relation_drift_report.py \
  --input data/conceptnet_sample.csv \
  --audit data/audit_made_of.csv
```

Prints relation support, drift support, audit accuracy, and examples.

## Audit Export

Export predictions for manual review:

```bash
python3 examples/pattern_audit_export.py \
  --input data/conceptnet_sample.csv \
  --output data/pattern_audit.csv \
  --mode transitive \
  --limit 100 \
  --use-relation-drift
```

Include mixed predictions:

```bash
python3 examples/pattern_audit_export.py \
  --input data/conceptnet_sample.csv \
  --output data/pattern_audit_all.csv \
  --mode all \
  --limit 200
```

Export relation proposals:

```bash
python3 examples/relation_proposal_audit_export.py \
  --input data/conceptnet_sample.csv \
  --output data/relation_proposal_audit.csv
```

