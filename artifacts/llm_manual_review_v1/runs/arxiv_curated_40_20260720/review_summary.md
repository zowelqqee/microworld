# Completed manual review — Gemini 3.1 Flash Lite curated batch

Status: **human-reviewed; accepted relations remain proposal-only**.

## Admission numbers

| Measure | Count |
|---|---:|
| node-quality-triaged review candidates | 54 |
| manual accepts | 10 |
| manual rejects | 44 |
| manual accept rate | 18.5% |
| literal support: yes | 53 / 54 (98.1%) |
| clean entities: yes | 10 / 54 (18.5%) |
| accepted existing-predicate relations | 9 |
| accepted new-predicate relations | 1 |

The 10 accepts are present in `manual_accepted_proposal_overlay.json`. This is
not a serving overlay, was not passed through the precision gate, and must not
be promoted automatically.

## Manually accepted relations

| ID | Relation | Predicate class |
|---|---|---|
| arxiv-003:0 | GCR energy density → is → 0.9+/-0.3 eV/cm3 | existing |
| arxiv-006:0 | KRAKENS → uses → detector technology | existing |
| arxiv-006:1 | detector technology → is → MKIDs | existing |
| arxiv-011:0 | SciServer → builds upon → SkyServer | **new** |
| arxiv-011:1 | SciServer → extends → SkyServer | existing |
| arxiv-016:0 | MAVIS → uses → two deformable mirrors | existing |
| arxiv-025:0 | AutoGrad → uses → Processing | existing |
| arxiv-025:1 | Processing → is used in → Interactive Media Arts | existing |
| arxiv-028:1 | NB2Slides → uses → example-based prompts | existing |
| arxiv-038:0 | ProbCT → uses → a neural-field representation | existing |

## Review conclusion

Literal extraction was strong, but entity admission remained the binding
constraint: 44 reviewed triples were rejected for anaphora, temporary
referents, generic endpoints, abstractions, process/activity phrases, or a
non-relational attachment. The result is observational evidence only; it does
not authorize a new automatic admission rule.
