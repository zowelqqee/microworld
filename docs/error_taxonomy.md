# Error Taxonomy

Microworld does not try to hide bad predictions.  It treats them as diagnostic
signals for improving symbolic reasoning.

## 1. Relation Drift

The A-C connection is meaningful, but the output relation is too crude.

Example:

```text
blood --made_of--> haemoglobin
haemoglobin --made_of--> iron
=> blood --made_of--> iron
```

This is better interpreted as:

```text
blood contains_element iron
```

## 2. Source Component Leakage

The chain walks from an object to a source entity, then leaks source components
back onto the original object.

Example:

```text
table --made_of--> tree
tree --made_of--> leaves
=> table --made_of--> leaves
```

The source entity has leaves, but the table does not.

## 3. Sense Ambiguity

A node has multiple meanings and the chain crosses senses.

Example:

```text
arm_bone --part_of--> arm
arm --part_of--> armchair
=> arm_bone --part_of--> armchair
```

The first `arm` is anatomical; the second is furniture.

## 4. Overgeneralized Geography

Large geographic regions bridge too broadly.

Example:

```text
aconcagua --part_of--> andes
andes --part_of--> bolivia
=> aconcagua --part_of--> bolivia
```

The Andes span multiple countries, but that does not mean every part of the
Andes belongs to each country.

## 5. Dataset Noise

ConceptNet-like data can contain junk, memes, profanity, or malformed nodes.

Examples:

```text
internet --made_of--> sister_naked
troll --made_of--> epic_fail
```

Node quality is designed to expose and suppress this class.
