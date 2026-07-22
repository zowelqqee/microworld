# Targeted legal extraction prompt v1

Profile id: `targeted_legal_provision_relations_v1`. Frozen before extraction.

Derived from `artifacts/llm_manual_review_v1/targeted_prompt_v1/prompt.md` by
swapping only the relation-type vocabulary. The structure — a closed
enumerated list of permitted types, an explicit DO-NOT list, verbatim-span
requirement, and explicit permission to return `[]` — is preserved exactly.

```text
You are reading one numbered provision of a United States statute. The
provision is given with its citation, any governing stem text, and the
provision text itself.

Extract ONLY relations of these specific types:
1. Definition: the provision defines a term. Subject is the defined term as
   written; object is the definition as written.
2. Cross-reference: the provision text explicitly cites another numbered
   provision (a section, subsection, paragraph, or subparagraph). Subject is
   the citation of THIS provision; object is the cited provision exactly as it
   is written in the text.
3. Scope / applicability: the provision states what class of thing, person, or
   situation it governs. Subject is the citation of THIS provision; object is
   the governed class as written.
4. Legal consequence: the provision states that a stated condition produces a
   stated legal result. Subject is the citation of THIS provision; predicate
   states the result including its polarity; object is the result as written.

DO NOT extract:
- Anything you know about patent law that this provision text does not state.
- Relations whose subject or object is a pronoun or a vague reference.
- A consequence stripped of the condition that governs it.
- A relation whose direction or negation you are not certain of.

Preserve negation exactly. If the text says something "shall not" be or "is
not" something, the predicate must say so.

Subject, object, and evidence_span must each be verbatim contiguous spans of
the provision text as given, except that the citation of THIS provision may be
used as a subject.

Return as JSON: [{"subject":string,"predicate":string,"object":string,"evidence_span":string}].
If no relation of the above types exists in the provision, return an empty
list — do not force an extraction.

PROVISION:
```
