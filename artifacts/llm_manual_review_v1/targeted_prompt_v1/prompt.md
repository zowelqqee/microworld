# Targeted Gemini extraction prompt v1

This prompt profile is frozen before targeted-pilot source acquisition.

```text
Extract ONLY relations of these specific types from the text:
1. Category/class membership: 'X is a type of Y', 'X is an instance of Y',
   'X belongs to the category of Y'.
2. Named technical system properties: a specifically named system, tool,
   method, or framework (proper noun, not a generic description) and what it
   does, uses, or is based on.

DO NOT extract:
- Statements about the paper itself ('this paper', 'we propose', 'this study').
- Generic properties or activities without a specifically named subject.
- Relations where subject or object is a pronoun or vague reference.

Return as JSON: [{subject, predicate, object, evidence_span}]. If no relation
of the above types exists in the text, return an empty list — do not force an
extraction.
```

The target is manual-review yield, not automated admission. Node-quality
filtering stays unchanged; any retained candidate remains manual-review-only.
