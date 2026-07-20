# Targeted anti-coercion prompt v1

Frozen before independent-batch acquisition. This profile is targeted prompt v1
unchanged, with the following addendum appended:

```text
IMPORTANT: Only extract a relation if the EXACT predicate and its direction
are directly and unambiguously stated in the text. Do NOT convert the
following into a class/member or system-property relation:
- Possessive constructions ('X's Y') — do not force these into 'is_a_type_of'.
- Purpose/goal clauses ('X for Y', 'X in order to Y') — do not convert into a
  system-property statement unless the text explicitly states what X IS, not
  merely what X is FOR.
- Satellite/incidental events mentioned alongside the main entity — do not
  attach them as if they were direct properties.
If the exact relation type is unclear or would require inference beyond what
is literally stated, return an empty result for that sentence rather than
forcing a fit.
```

This is an independent manual-review experiment, not a production prompt
change or an automatic-admission mechanism.
