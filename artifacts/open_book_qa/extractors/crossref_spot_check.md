# Crossref source-specific extractor: pre-diagnostic and spot-check

**Scope:** proposal-only re-parsing of the existing Crossref quarantine slice.
No accepted memory, promoted overlay, or serving memory was modified.

## Predicate novelty before extraction

The bounded slice contains 59 unique relations for 54 current held-out-pool
subjects. Before extraction, each candidate predicate was compared with the
subject's existing predicate groups in the clean relation baseline: 53
candidates duplicate an existing group; 6 are potential new groups. This is a
diagnostic only; it does not make a candidate admissible.

The first manual sample used deterministic seed `20260717`. It contained nine
duplicates and one potential novelty:

| # | Candidate | Existing groups | Pre-extraction novelty | Parser verdict |
|---:|---|---|---|---|
| 1 | Evaluation — uses — quantitative and qualitative approaches | uses | duplicate | reject: generic subject |
| 2 | Digital agriculture — supports — farmers' decision making… | supports | duplicate | reject: non-atomic object |
| 3 | Blockchain — enables — secure and tamper-proof recording… | enables | duplicate | reject: non-atomic object |
| 4 | Klebanov — uses — a folklore quotation… | uses | duplicate | reject: non-atomic object |
| 5 | Contemporary psychological research increasingly — supports — effectiveness of meditation… | supports | duplicate | reject: non-atomic endpoint |
| 6 | Conventional particleboard industry — uses — synthetic, mostly formaldehyde-based adhesives… | uses | duplicate | reject: non-atomic object |
| 7 | Artificial Intelligence — located_in — AI integration | used_for | potential new | reject: no explicit source support |
| 8 | Monetary policy — uses — the money supply and interest rates | uses | duplicate | accept |
| 9 | Predictive analytics — uses — machine learning, statistical modeling | uses | duplicate | accept |
| 10 | Progressivism — supports — changes for the better | supports | duplicate | accept |

The stored Crossref records were inspected for all ten cases. The first
implementation incorrectly accepted the single-word topic label `Evaluation`;
it was the only spot-check error. The general `_GENERIC_SUBJECTS` guard was
added (not a source- or candidate-specific exception), then the sample was
rerun. The final sample had zero incorrect accept/reject decisions, so the
full bounded lane was allowed to run.

The full result is in
[`crossref_full_run_report.json`](crossref_full_run_report.json).
