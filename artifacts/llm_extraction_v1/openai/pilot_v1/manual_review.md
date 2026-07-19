# Manual review — OpenAI extraction pilot v1

Every raw triple in `raw_responses.json` was reviewed against its source sentence.
`Literal` means that the asserted direction and relation are explicitly licensed by
the sentence. `Clean` requires two non-generic, resolvable endpoints. `New` means
the relation would be a genuinely new predicate rather than an existing generic
one. No result is promoted by this review.

| ID | Extracted triple | Literal | Clean | New | Verdict / reason |
|---|---|:---:|:---:|:---:|---|
| 00:0 | GAs — optimizes — models | yes | no | no | object is generic |
| 00:1 | probabilistic reasoning — lowers — statistical uncertainties | yes | no | no | generic endpoint |
| 01:0 | this paper — presents — NB2Slides | yes | no | no | authorial self-reference |
| 01:1 | NB2Slides — facilitates — users composing presentations | yes | no | no | generic activity/object |
| 02:0 | physicist approach — is — generic path | **no** | no | no | sentence is an elliptical fragment; it does not assert `is` |
| 03:0 | SciServer — builds upon and extends — SkyServer | yes | yes | **yes*** | one clean candidate, but must split/canonicalize two predicates before any gate |
| 03:1 | SkyServer — introduced — astronomical community to SQL | yes | no | no | group/activity endpoint is not a clean entity |
| 03:2 | SkyServer — serves — SDSS catalog data to public | yes | no | no | object/activity is generic |
| 04:0 | REGAI — improves on — performance of LLM techniques | yes | no | no | abstract performance endpoint |
| 05:0 | we — propose — tutor model | yes | no | no | authorial self-reference |
| 06:0 | fuzz-testing techniques — focus on — identifying vulnerabilities | yes | no | no | generic class/activity |
| 06:1 | vulnerabilities — allow — adversaries to achieve outcomes | yes | no | no | generic class/activity |
| 07:0 | AML approaches — focus on — attack robustness | yes | no | no | generic class/property |
| 07:1 | AML approaches — overlook — real-world factors | yes | no | no | generic endpoint |
| 07:2 | real-world factors — include — environments/dependencies/feasibility | yes | no | no | generic list endpoint |
| 08:0 | this study — addresses — need for fairness/explainability | yes | no | no | authorial self-reference |
| 09:0 | AutoSlim — uses — Random Forest classification | yes | yes | no | clean, but `uses` is not a new predicate |
| 09:1 | Random Forest classification — prunes — transitions | **no** | no | no | wrong agent: the sentence assigns pruning to AutoSlim |
| 09:2 | transitions — are based on — edge scores/features | **no** | no | no | attachment error; `based on` modifies the pruning operation |
| 09:3 | AutoSlim — reduces — automata graph density | yes | no | no | generic metric endpoint |
| 09:4 | AutoSlim — preserves — semantic correctness | yes | no | no | generic property endpoint |
| 10:0 | we — design — ProbCT | yes | no | no | authorial self-reference |
| 11:0 | methods — can achieve — wave packets | **no** | no | no | source says wave packets *can be achieved*, not that methods achieve them |
| 12:0 | we — construct — debiased estimators | yes | no | no | authorial self-reference |
| 12:1 | debiased estimators — are — asymptotically Gaussian | yes | no | no | generic mathematical property |
| 12:2 | debiased estimators — allow — hypothesis testing | yes | no | no | generic activity/object |
| 13:0 | this paper — proposes — architecture | yes | no | no | authorial self-reference |
| 13:1 | architecture — enables — quantum and AI in vehicles | yes | no | no | generic subject/object |
| 13:2 | QNN — are used for — sensor fusion | **no** | no | no | colon list is nominal/elliptical; no explicit relation verb |
| 13:3 | Nav-Q — is used for — reinforcement learning | **no** | no | no | colon list is nominal/elliptical; no explicit relation verb |
| 13:4 | protocols — are used for — secure communication | **no** | no | no | colon list is nominal/elliptical; no explicit relation verb |
| 14:0 | proposed framework — addresses — challenges | yes | no | no | generic self-referential subject/object |
| 14:1 | proposed framework — provides — performance/security | yes | no | no | generic self-referential subject/object |

\* `builds upon` is the sole clean, literal new-predicate proposal. It is still
proposal-only: the existing validator does not accept arbitrary new predicate
labels, so it remains quarantined rather than admitted to an overlay.
