# Reasoning Layers

Microworld confidence is intentionally interpretable.  It is not a learned
linear model.  Each layer has a readable reason and can be turned on or off.

## 1. Pattern Support

For transitive prediction, the base confidence comes from the discovered
bigram count:

```text
A --r--> B
B --r--> C
=> A --r--> C
```

Current base formula:

```text
base_confidence = min(0.95, 0.5 + 0.05 * log(count + 1))
```

## 2. Hub Penalty

High-degree intermediate nodes create many weak bridges.  The hub penalty
reduces confidence through broad nodes:

```text
hub_factor = sqrt(10 / max(10, degree_of_B))
```

Example:

```text
A --part_of--> Andes
Andes --part_of--> Bolivia/Chile/Peru/...
```

This may be structurally valid but semantically overgeneralized.

## 3. Relation Trust

Relation trust priors are derived from human audit results.

Current priors:

```text
made_of     0.862
part_of     0.767
is_a        0.756
at_location 0.100
```

Relations not in the table use an uncertainty default.

## 4. Node Quality

Node quality filters noisy ConceptNet entries.

Examples of low-quality nodes:

```text
sister_naked
epic_fail
tu_hermana_en_bolas
```

Node quality can either filter predictions or reduce their confidence.

## 5. Relation Drift

Relation drift handles cases where a transitive chain is meaningful but the
relation label changes semantic level.

Example:

```text
blood --made_of--> haemoglobin
haemoglobin --made_of--> iron
```

The useful interpretation is closer to:

```text
blood contains_element iron
```

So drift-aware scoring keeps the prediction explainable but reduces confidence.

Default drift penalties:

```text
direct_material    1.00
raw_material       0.85
atomic_component   0.65
abstract_component 0.70
```

## 6. Disabled Relations

Some relations are too noisy for default reasoning.  Currently:

```text
at_location
```

Audit/demo commands can opt in with:

```bash
--include-disabled-relations
```

This is useful for diagnostics, but not for normal prediction.

