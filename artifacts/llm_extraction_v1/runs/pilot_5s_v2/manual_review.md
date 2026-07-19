# Manual review — Gemini extraction pilot v2 (5-second pacing)

Every raw triple was checked against its full source sentence. `Literal` means
that the exact relation and direction are explicitly licensed; `Clean` requires
two non-generic, resolvable endpoints; `New` means a predicate not already in
the current schema. This is proposal-only review, not promotion.

| ID | Extracted triple | Literal | Clean | New | Verdict / reason |
|---|---|:---:|:---:|:---:|---|
| 00:0 | GAs — optimizes — models | yes | no | no | generic object |
| 00:1 | probabilistic reasoning — lowers — uncertainties | yes | no | no | generic abstract endpoint |
| 01:0 | this paper — presents — NB2Slides | yes | no | no | authorial self-reference |
| 01:1 | NB2Slides — facilitates — users composing presentations | yes | no | no | generic activity/object |
| 03:0 | SciServer — builds upon — SkyServer | yes | yes | **yes** | clean new-predicate proposal; quarantine pending schema admission |
| 03:1 | SciServer — extends — SkyServer | yes | yes | no | clean existing-predicate proposal |
| 03:2 | SkyServer — introduced — astronomical community | yes | no | no | generic group endpoint |
| 03:3 | SkyServer — serves — SDSS catalog data | yes | no | no | generic resource/activity endpoint |
| 04:0 | REGAI — improves on — LLM performance | yes | no | no | abstract performance endpoint |
| 05:0 | we — propose — tutor model | yes | no | no | authorial self-reference |
| 05:1 | tutor model — considers — student proficiency | yes | no | no | generic concept/activity |
| 06:0 | fuzz-testing techniques — focus on — vulnerabilities | yes | no | no | generic class/activity |
| 06:1 | vulnerabilities — allow — adversarial outcomes | yes | no | no | generic class/activity |
| 07:0 | AML approaches — focus on — robustness | yes | no | no | generic property |
| 07:1 | AML approaches — overlook — real-world factors | yes | no | no | generic/list endpoint |
| 08:0 | this study — addresses — need for fairness/explainability | yes | no | no | authorial self-reference |
| 09:0 | AutoSlim — uses — Random Forest classification | yes | yes | no | clean existing-predicate proposal |
| 09:1 | AutoSlim — prunes — low-impact transitions | yes | no | no | object is generic |
| 09:2 | Random Forest classification — prunes — transitions | **no** | no | no | wrong semantic agent; source assigns pruning to AutoSlim |
| 10:0 | we — design — ProbCT | yes | no | no | authorial self-reference |
| 10:1 | ProbCT — is a — learning-based model | yes | no | no | apposition is explicit, but type object is generic |
| 10:2 | CT — is of — clouds | **no** | no | no | invented copular relation from noun phrase |
| 10:3 | CT of clouds — based on — images | **no** | no | no | attachment is not explicitly licensed for this subject |
| 11:0 | we — benchmark — methods | yes | no | no | authorial self-reference |
| 11:1 | we — show — wave packets can be achieved | yes | no | no | authorial self-reference/claim fragment |
| 12:0 | we — construct — estimators | yes | no | no | authorial self-reference |
| 12:1 | estimators — are — asymptotically Gaussian | yes | no | no | mathematical property, not graph entity knowledge |
| 12:2 | estimators — allow — hypothesis testing | yes | no | no | generic activity endpoint |
| 13:0 | we — propose — architecture | yes | no | no | authorial self-reference |
| 13:1 | architecture — based on — Quantum Artificial Intelligence | yes | no | no | literal ACL construction, but generic subject |
| 13:2 | architecture — enables — quantum and AI in vehicles | yes | no | no | generic subject/object |
| 13:3 | QNN — are for — sensor fusion | **no** | no | no | nominal-list completion, not an explicit predicate |
| 13:4 | Nav-Q — is for — reinforcement learning | **no** | no | no | nominal-list completion, not an explicit predicate |
| 13:5 | protocols — are for — secure communication | **no** | no | no | nominal-list completion, not an explicit predicate |
| 14:0 | framework — addresses — challenges | yes | no | no | self-referential generic endpoints |
| 14:1 | framework — provides — performance/security | yes | no | no | self-referential generic endpoints |

## Count

- Literal-support failures: **6 / 36 (16.7%)**.
- Unclean/generic endpoints: **33 / 36 (91.7%)**.
- Clean, literal, new-predicate proposals: **1 / 36 (2.8%)** (`builds upon`).
- Clean, literal, existing-predicate proposals: **2 / 36** (`extends`, `uses`).
