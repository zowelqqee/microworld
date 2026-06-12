# Suppression Policy

Suppression is the final decision layer for predictions that changed after
audit-driven trust learning. It should not be confused with trust learning
itself.

Current architecture:

```text
graph prediction
-> baseline confidence
-> learned trust confidence
-> suppression candidate
-> quality-aware policy
-> final suppression
```

## Why A Separate Policy Exists

The first suppression rule was intentionally simple:

```text
baseline_confidence >= threshold
AND learned_confidence < threshold
```

Manual audit showed that this rule changed behavior but over-suppressed useful
predictions:

```text
total reviewed: 50
should_suppress: 11
should_keep: 38
unclear: 1
suppression_precision: 0.224
```

Conclusion:

```text
trust learning is useful as a signal, but should not be the final decision layer
```

## Delta Calibration

Confidence-drop calibration did not fix the issue. Useful and harmful
suppressions had similar confidence drops, so delta magnitude alone did not
separate bad suppressions from useful predictions.

The delta can still be useful diagnostic context, but it is not a complete
policy.

## Quality-Aware V1

The first quality-aware policy added node-quality information to the final
suppression decision.

Output:

```text
exported rows: 12
```

Manual audit:

```text
should_suppress: 11
should_keep: 1
suppression_precision: 0.917
```

The only false suppression was:

```text
talbe --made_of--> wood
```

This is almost certainly a typo for:

```text
table --made_of--> wood
```

The prediction itself is useful. The typo is in the source node, so it should be
handled by normalization/canonicalization, not suppression.

## Quality-Aware V2

The second version changed the policy:

* source noise no longer triggers suppression
* target noise still triggers suppression

Result:

```text
output rows: 11
all rows had target = oxegen
talbe --made_of--> wood disappeared
```

## Current Interpretation

The current error policy is:

```text
bad target -> suppress
bad source -> normalize later
bad relation or pattern -> lower trust
clean prediction with source typo -> keep after normalization
```

This suggests three separate components:

* trust memory for relation/rule reliability
* suppression policy for final keep/suppress decisions
* normalization for typo and canonicalization repair

## Next Steps

* export normalization candidates
* add typo/canonicalization layer
* re-evaluate target normalization semantically
* collect a larger suppression audit sample
* add relation-specific suppression policies
