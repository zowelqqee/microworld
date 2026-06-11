# worldmvp

A lightweight research sandbox for experimenting with symbolic world models, concept formation, structural similarity, and explainable link prediction.

## Motivation

Modern AI systems are extremely effective at extracting patterns from large amounts of data.

However, much of their knowledge is stored implicitly inside model weights, making it difficult to:

* inspect reasoning
* explain predictions
* update knowledge incrementally
* study memory consolidation separately from learning

worldmvp explores an alternative idea:

> Can part of reasoning be represented explicitly as objects, relations, concepts, and structural patterns instead of being hidden entirely inside neural network weights?

This project does **not** attempt to replace LLMs.

Instead, it serves as a small experimental environment for testing hypotheses about:

* symbolic memory
* concept formation
* structural generalization
* knowledge consolidation ("sleep")
* sample-efficient reasoning

---

## Core Architecture

### Objects

Knowledge is represented as entities:

```text
яблоня
яблоко
семя
```

### Relations

Entities are connected through typed relations:

```text
яблоня --produces_result--> яблоко
яблоко --contains--> семя
семя --grows_into--> яблоня
```

### Causal Reasoning

The system can:

* trace chains
* explain paths
* estimate consequences of removing nodes

Example:

```text
яблоня
↓
яблоко
↓
семя
↓
яблоня
```

### Concept Formation

Repeated structures are consolidated into concepts.

Example:

```text
яблоко contains семя
груша_плод contains семя
апельсин contains семя
```

becomes:

```text
concept_rel_contains_семя
```

with membership relations:

```text
яблоко      member_of concept_rel_contains_семя
груша_плод  member_of concept_rel_contains_семя
апельсин    member_of concept_rel_contains_семя
```

### Structural Similarity

Instead of manually specifying similarity:

```python
world.add_similarity("слива", "персик", 0.9)
```

the system derives similarity from graph structure.

Example:

```text
слива contains косточка
персик contains косточка
```

↓

```text
similarity(слива, персик)
```

computed automatically using structural profile overlap.

### Prediction

The system attempts to recover missing relations using:

1. Majority templates
2. Structural similarity
3. Concept membership
4. Hybrid reasoning

Every prediction remains explainable.

Example:

```text
слива contains косточка

Reason:
concept match:
concept_rel_contains_косточка
```

---

## ConceptNet Integration

worldmvp can now import filtered ConceptNet data and build a `World` directly from relation CSV files.

The current import path supports:

* extracting a filtered ConceptNet relation sample
* loading `source,relation_type,target` CSV files
* building a world graph without the natural-language parser
* discovering concepts from external graphs
* computing structural similarities between entities
* discovering frequent relation patterns

One current example run over `data/conceptnet_sample.csv` contains roughly:

* ~5000 relations
* 9 relation types
* 443 discovered concepts
* 18k+ structural similarity pairs

These numbers describe the checked-in sample run, not a benchmark.

---

## Pattern Discovery

worldmvp no longer relies only on hand-written lifecycle patterns.

The system can discover common relation chains directly from a graph. On the current ConceptNet sample, examples include:

```text
part_of -> part_of
made_of -> made_of
is_a -> is_a
```

Pattern discovery is exploratory. It is used to understand the structure of a knowledge graph before prediction, and to see which relation chains actually occur often enough to study.

---

## Pattern-Based Prediction

A second prediction engine now exists alongside the lifecycle/template-based predictor.

Instead of using lifecycle templates, it uses discovered graph patterns. The current implementation supports transitive same-relation reasoning:

```text
A part_of B
B part_of C

=>

A part_of C
```

The pattern-based predictor produces explanations, evidence chains, and confidence scores. It supports same-relation transitive chains and a small set of explicitly allowed mixed-relation rules such as:

```text
A is_a B
B capable_of C

=>

A capable_of C
```

Mixed-relation reasoning is intentionally conservative: noisy relations can be disabled, intermediate hubs can be penalized, relation trust can lower confidence, and relation drift can annotate cases where composition changes semantic level.

### Human Audit

Pattern predictions can be exported to CSV and reviewed manually. The current audit labels are:

* `correct`
* `plausible`
* `wrong`
* `unclear`

ConceptNet-derived pattern predictions were manually reviewed:

```text
reviewed predictions: 104

correct:   43.3%
plausible: 35.6%
wrong:     21.2%

useful (correct + plausible): 78.8%
```

By relation:

```text
made_of: 86.2%
part_of: 76.7%
is_a:    75.6%
```

These results come from a small exploratory audit and should not be interpreted as a formal benchmark.

---

## Audit-Driven Trust Learning

worldmvp can learn simple trust profiles from human audit CSV files without neural backpropagation.

Audit labels are mapped to scores:

```text
correct   -> 1.0
plausible -> 0.7
unclear   -> 0.4
wrong     -> 0.0
```

The system averages those scores by relation, rule, and drift type, then reruns symbolic prediction with the learned trust profile.

Learning loop result:

```text
threshold: 0.40

baseline accepted predictions:      253
learned-trust accepted predictions: 161
suppressed after audit learning:     92
newly promoted:                       0
```

Interpretation:

Human audit lowered trust for weak relations and made the system more conservative without neural retraining. This is not a weights-and-biases learning loop; it is explicit trust calibration over inspectable symbolic reasoning rules.

---

## Sample Efficiency Experiment

A synthetic benchmark evaluates whether structural consolidation improves learning efficiency.

Four lifecycle families are generated:

```text
tree → fruit → seed → tree
tree → fruit → pit → tree
animal → egg → offspring → animal
business → product → revenue → business
```

Prediction quality is measured while reducing available observations.

Results:

| Budget | Majority | Structural | Concept | Hybrid |
| ------ | -------- | ---------- | ------- | ------ |
| 20%    | 0.18     | 0.18       | 0.18    | 0.18   |
| 40%    | 0.18     | 0.40       | 0.40    | 0.40   |
| 60%    | 0.18     | 0.67       | 0.67    | 0.67   |
| 80%    | 0.18     | 1.00       | 1.00    | 1.00   |
| 100%   | 0.18     | 1.00       | 1.00    | 1.00   |

Interpretation:

* Majority reasoning never learns family-specific structure.
* Structural and concept-based reasoning generalize after observing only a small number of examples.
* Consolidation improves sample efficiency on this synthetic benchmark.

---

## What This Project Is

* Research sandbox
* Symbolic reasoning playground
* World-model experiment
* Concept formation experiment
* Explainable prediction system

---

## What This Project Is Not

* AGI
* A replacement for LLMs
* A production knowledge graph
* A biologically accurate brain simulation
* Evidence that symbolic systems outperform transformers

---

## Current Limitations

* ConceptNet work currently uses a filtered sample, not a full benchmark
* No perception layer
* No vision
* No reinforcement learning
* No neural learning
* No full-scale dataset pipeline yet
* Structural similarity still depends on observable graph structure
* Mixed-pattern reasoning is manually allowlisted and still exploratory
* Audit-driven trust is based on small reviewed samples, not large-scale validation

---

## Current Direction

Current work is moving toward broader ConceptNet evaluation, relation-specific trust estimation, automatic pattern discovery, graph consolidation and concept formation, and reasoning without neural training.

The emphasis is still engineering/research: make the symbolic state inspectable, measure behavior on small cases first, and avoid treating exploratory results as proof.

---

## Status

Experimental.

The goal of worldmvp is not to build a new intelligence system, but to create a controlled environment where hypotheses about memory, concepts, consolidation, and reasoning can be tested and measured.
