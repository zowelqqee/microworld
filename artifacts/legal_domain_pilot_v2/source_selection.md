# Legal-domain pilot v2 — source selection

Written **before** extraction. Records the single narrow legal source chosen
for pilot v2 and why it was chosen for *structural* difference from pilot v1,
not for legal importance.

## Selected source

**18 U.S.C. Chapter 41 — Extortion and Threats** (sections 871–880), United
States Code, 2023 Edition, U.S. Government Publishing Office.

- Retrieval URL: `https://www.govinfo.gov/content/pkg/USCODE-2023-title18/html/USCODE-2023-title18-partI-chap41.htm`
- Sections: §871 (threats against the President), §872 (extortion by federal
  officers), §873 (blackmail), §874 (kickbacks), §875–§878 (interstate/mailed
  threats, threats against foreign officials), §879 (threats against former
  Presidents; includes a definitions subsection), §880 (receiving proceeds of
  extortion).

## Why this chapter — the deliberate contrast with pilot v1

Pilot v1 (35 U.S.C. ch. 10) was a *patentability* chapter: definitions,
scope, and conditional entitlement. Its own report recorded a scope limit it
could not test:

> "This chapter contains **no violation → penalty provisions.** … The
> requested `results_in_consequence` relation type therefore cannot be probed
> here in its 'violation of X results in penalty Y' form."

Chapter 41 is chosen precisely to close that gap. It is a **criminal offense
chapter**: almost every section is the canonical form
*"Whoever [does X] shall be fined under this title or imprisoned not more than
N years, or both."* This exercises the one relation type v1 could not — a
**violation → penalty** consequence — under the same discipline.

| Pilot-friendly property | How ch. 41 satisfies it |
|---|---|
| Fixed, machine-readable numbering | Same GPO `statutory-body` / `section-head` markup as v1; subdivision hierarchy recoverable deterministically. |
| Operative text separable from commentary | Same `field-start:statute … field-end:statute` fencing; editorial/amendment notes dropped by construction. |
| Small, closed unit count | **28 leaf provisions** — inside the 20–30 pilot budget, so the pilot runs on the **entire chapter with no sampling and no selection bias.** |
| Target relation type present | **25 of 28 units carry an explicit penalty clause** ("shall be fined … or imprisoned …"). This is the violation→penalty form by design. |
| Structurally different from v1 | v1 was definitions + conditions; v2 is offense-definition + penalty. Different consequence shape, different subject shape (an actor "whoever …" rather than a defined term). |
| Cross-references mostly internal / bounded | Penalties reference "this title" (Title 18) and a bounded set of sibling sections (§112, §871(b), §1116, §1201, §3056). Never dereferenced during the pilot. |
| Public domain | U.S. Code is not subject to copyright. |

## The one structural outlier, declared up front

**§879** is the single atypical section (14 paragraphs). It differs from the
rest of the chapter in two ways the pilot must handle honestly:

1. Its penalty is a **trailing "hanging" clause** — "shall be fined under this
   title or imprisoned not more than 5 years, or both." appears *after* the
   enumerated list of protected persons (a)(1)–(a)(4), closing subsection (a)
   as a whole.
2. It contains a **nested definitional subsection** (b) reaching a 4-level
   hierarchy `(b)(1)(B)(i)`, with lowercase-roman markers — deeper than
   anything in ch. 10.

This is the direct analogue of §100 (Definitions) being the outlier in ch. 10.
§879 is **kept, not excluded**: the v2 segmenter is extended to fold trailing
hanging penalty clauses into their governing subsection and to recognise a
4th (roman) hierarchy level, so the penalty is captured rather than orphaned.
The segmenter change is documented in `pre_registration.md` and is confined to
the pilot's own experiment script — no production module is touched.

## Content note

Chapter 41 defines threat and extortion offenses. The pilot extracts the
statute's own structural relations (what each offense is, what penalty
attaches); it neither reproduces threatening content nor provides any
operational detail. The text is quoted only as verbatim evidence spans for
relation verification, exactly as in v1.

## Guardrail

Source selection only. Authorizes no promotion, no serving-memory write, and
no claim that this chapter is representative of criminal statutes in general.
