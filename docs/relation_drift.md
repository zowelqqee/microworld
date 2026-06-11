# Relation Drift

Relation drift happens when a relation chain is meaningful but the output
relation label changes semantic level.

## Simple Good Case

```text
table --made_of--> tree
tree --made_of--> wood
=> table --made_of--> wood
```

This is broadly useful.

## Drift Case

```text
blood --made_of--> haemoglobin
haemoglobin --made_of--> iron
=> blood --made_of--> iron
```

The A-C connection is meaningful, but the relation has drifted.  A better
interpretation is closer to:

```text
blood contains_element iron
```

Microworld keeps the prediction explainable but reduces confidence.

## Material Categories

`core/relation_drift.py` currently uses coarse material categories:

```text
direct_material
raw_material
atomic_component
abstract_component
```

Examples:

```text
paper      direct_material
wood       raw_material
iron       atomic_component
ideals     abstract_component
```

## Drift Penalties

Default penalties:

```text
direct_material    1.00
raw_material       0.85
atomic_component   0.65
abstract_component 0.70
```

These are explicit heuristics, not learned model weights.

## Running The Report

```bash
python3 examples/relation_drift_report.py \
  --input data/conceptnet_sample.csv \
  --audit data/audit_made_of.csv
```

Example report fields:

```text
relation
support
drift
reviewed
wrong
accuracy
examples
```

## Drift-Aware Prediction

Use:

```python
predictor.predict_from_bigrams(
    use_relation_drift=True,
)
```

Reason strings include:

```text
drift=atomic_component, drift_penalty=0.65
```

