# Legal-domain pilot v1 — source selection

Written **before** extraction. It records which single narrow legal source was
chosen and, explicitly, why it was chosen for *structural* pilot-friendliness
rather than legal importance.

## Selected source

**35 U.S.C. Chapter 10 — Patentability of Inventions** (sections 100–105),
United States Code, 2023 Edition, as published by the U.S. Government
Publishing Office.

- Retrieval URL: `https://www.govinfo.gov/content/pkg/USCODE-2023-title35/html/USCODE-2023-title35-partII-chap10.htm`
- Sections present: §100 (Definitions), §101 (Inventions patentable),
  §102 (Conditions for patentability; novelty), §103 (Conditions for
  patentability; non-obvious subject matter), §105 (Inventions in outer space).
  §104 is repealed and carries no operative text.

## Why this section, structurally

The choice is a *structural* one. Each property below was checked against the
retrieved document before the pilot was run.

| Pilot-friendly property | How this chapter satisfies it |
|---|---|
| Fixed, machine-readable numbering | GPO HTML tags every operative paragraph as `statutory-body`, `statutory-body-1em`, or `statutory-body-2em`, and every section as `section-head`. Subdivision hierarchy is recoverable deterministically, with no NLP. |
| Operative text separable from commentary | GPO wraps operative text in `field-start:statute` … `field-end:statute`. Editorial notes, amendment history, and source credits are outside that span and are dropped by construction, not by heuristic. |
| Small, closed unit count | 28 leaf provisions total — inside the 20–30 pilot budget without sampling, so there is **no selection bias**: the pilot runs on the entire chapter. |
| Low interpretive/discretionary language | §100 is pure lexical definition. §102 and §105 are rule statements with explicit conditions. The only genuinely discretionary standard in the chapter is §103's "person having ordinary skill in the art", which is one unit out of 28. |
| Cross-references mostly *internal* | §102 references §102(a)(1), (a)(2), (b)(2)(C) — i.e. itself. External references are a bounded, enumerable set (§119, §120, §121, §122(b), §151, §302, §365, §386) that point elsewhere in the same title and are never dereferenced during this pilot. |
| Public domain | U.S. Code is not subject to copyright. No licensing question blocks storing verbatim spans in an artifact. |

Rejected alternatives and why:

- **GDPR (Regulation 2016/679)** — structurally attractive, but EUR-Lex
  returned an empty body to a plain programmatic request (HTTP 202, 0 bytes),
  and GDPR articles carry far heavier conditional and cross-regulation
  referencing. Not a first pilot.
- **17 U.S.C. Chapter 1** — right structure, wrong size: §107–§122 are long,
  exception-dense sections that would have forced sampling and therefore
  selection bias.

## Honest scope limitation recorded up front

This chapter contains **no violation → penalty provisions.** It is a
patentability chapter, not an enforcement chapter. The requested
`results_in_consequence` relation type therefore cannot be probed here in its
"violation of X results in penalty Y" form. It is probed only in its weaker
form — *condition → legal consequence* ("a person shall be entitled to a
patent unless …", "a disclosure shall not be prior art … if …").

Any verdict this pilot reaches about consequence relations is a verdict about
conditional legal consequences, **not** about penalty provisions. A separate
source would be required for the penalty form.

## Guardrail

This is source selection only. It authorizes no promotion, no serving-memory
write, and no claim that the chapter is representative of legal text in
general.
