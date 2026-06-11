# Architecture

## Core Data Model

Microworld stores knowledge as `Relation` objects:

```python
Relation(source="book", relation_type="made_of", target="paper")
```

The main graph container is `World`:

```python
from core.datasets import load_relations_csv, build_world_from_relations

rows = load_relations_csv("data/conceptnet_sample.csv")
world = build_world_from_relations(rows)
relations = world.get_relations()
```

## Important Modules

### `core/datasets.py`

Loads `source,relation_type,target` CSV files and builds a `World`.

### `core/patterns.py`

Discovers frequent relation bigrams and trigrams:

```text
part_of -> part_of
made_of -> made_of
is_a -> is_a
```

This is descriptive analysis: it asks what structures exist in the graph.

### `core/pattern_prediction.py`

Generates explainable predictions from discovered patterns.

Supported modes:

- same-relation transitive prediction
- mixed manually allowed rules

Example:

```text
A --part_of--> B
B --part_of--> C
=> A --part_of--> C
```

Mixed example:

```text
A --is_a--> B
B --has_property--> C
=> A --has_property--> C
```

### `core/relation_proposal.py`

Learns possible output relation labels from existing direct closures.

Instead of forcing:

```text
A --r1--> B --r2--> C
=> A --r2--> C
```

it learns:

```text
(r1, r2) => r_out
```

from observed graph examples.

### `core/relation_drift.py`

Detects semantic-level drift in `made_of` chains.

Material categories:

- `direct_material`
- `raw_material`
- `atomic_component`
- `abstract_component`

Example:

```text
blood --made_of--> haemoglobin --made_of--> iron
```

The target `iron` is an `atomic_component`, so the original relation
`made_of` is useful but too crude.

### `core/reasoning_relations.py`

Defines relation policy:

```python
DEFAULT_REASONING_RELATIONS = {"made_of", "part_of", "is_a"}
DEFAULT_DISABLED_RELATIONS = {"at_location"}
```

Disabled relations are skipped by default because they are noisy.

### `core/node_quality.py`

Detects low-quality graph nodes, including noise tokens from ConceptNet-like
data.

Examples:

```text
sister_naked
epic_fail
tu_hermana_en_bolas
```

