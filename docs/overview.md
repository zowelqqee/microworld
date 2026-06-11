# Overview

Microworld explores explicit reasoning over a small knowledge graph.  Instead
of storing all knowledge inside model weights, the system keeps objects,
relations, patterns, predictions, and explanations inspectable.

The current ConceptNet sample is loaded from:

```text
data/conceptnet_sample.csv
```

Each row has:

```text
source,relation_type,target
```

Example:

```text
blood,made_of,haemoglobin
haemoglobin,made_of,iron
```

The graph can then produce a candidate prediction:

```text
blood --made_of--> iron
```

Microworld does not simply accept this as equally strong as every other
`made_of` prediction.  It can explain:

- the evidence chain
- the pattern count
- hub penalty
- relation trust
- node quality
- relation drift

For the example above, relation drift marks `iron` as an `atomic_component`,
so confidence is reduced and the report explains that the relation has drifted
toward something like `contains_element`.

## What The Project Is For

Microworld is useful for:

- testing symbolic reasoning hypotheses
- auditing graph predictions
- finding error classes in ConceptNet-style data
- showing explainable reasoning reports
- studying where transitive assumptions fail

It is not meant to be:

- a production knowledge graph
- a replacement for LLMs
- a benchmark-only project
- a black-box ML scorer

## Current Research Questions

- Which relation chains are safely transitive?
- When does a same-relation chain change semantic level?
- Which relations are too noisy for default reasoning?
- Can audit labels improve confidence without losing interpretability?
- Can the system diagnose failures instead of hiding them?

