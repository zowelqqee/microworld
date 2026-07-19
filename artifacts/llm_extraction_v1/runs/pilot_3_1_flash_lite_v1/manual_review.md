# Manual review — Gemini 3.1 Flash-Lite extraction pilot v1

`Literal` requires the direction and predicate to be stated by the full source
sentence. `Clean` requires two non-generic, graph-resolvable endpoints. `New`
means a predicate not currently in the schema. Nothing in this review is
admitted to an overlay.

| ID | Extracted triple | Literal | Clean | New | Verdict / reason |
|---|---|:---:|:---:|:---:|---|
| 00:0 | GAs — optimizes — models | yes | no | no | generic object |
| 00:1 | probabilistic reasoning — lowers — uncertainties | yes | no | no | generic abstract endpoint |
| 01:0 | this paper — presents — NB2Slides | yes | no | no | authorial self-reference |
| 01:1 | NB2Slides — is — AI system | yes | no | no | generic type object |
| 01:2 | NB2Slides — facilitates — user presentations | yes | no | no | generic activity/object |
| 02:0 | physicist approach — is — generic path | **no** | no | no | invented copula from an elliptical fragment |
| 03:0 | SciServer — builds upon — SkyServer | yes | yes | **yes** | clean new-predicate proposal; remains quarantined |
| 03:1 | SciServer — extends — SkyServer | yes | yes | no | clean existing-predicate proposal |
| 03:2 | SkyServer — introduced — astronomical community to SQL | yes | no | no | group/activity endpoint |
| 03:3 | SkyServer — serves — SDSS catalog data | yes | no | no | generic resource/activity endpoint |
| 04:0 | REGAI — improves performance of — classical LLMs | yes | no | no | abstract performance endpoint |
| 04:1 | REGAI — improves performance of — RAG techniques | yes | no | no | abstract performance endpoint |
| 05:0 | we — propose — tutor model | yes | no | no | authorial self-reference |
| 05:1 | tutor model — considers — student proficiency | yes | no | no | generic concept/activity |
| 06:0 | fuzz-testing techniques — focus on — vulnerabilities | yes | no | no | generic class/activity |
| 06:1 | vulnerabilities — allow — adversarial outcomes | yes | no | no | generic class/activity |
| 07:0 | AML approaches — focus on — robustness | yes | no | no | generic class/property |
| 08:0 | this study — addresses — fairness/explainability need | yes | no | no | authorial self-reference |
| 09:0 | AutoSlim — uses — Random Forest classification | yes | yes | no | clean existing-predicate proposal |
| 09:1 | AutoSlim — prunes — low-impact transitions | yes | no | no | generic object |
| 09:2 | AutoSlim — reduces — graph density | yes | no | no | generic metric endpoint |
| 10:0 | we — design — ProbCT | yes | no | no | authorial self-reference |
| 10:1 | ProbCT — achieves — CT of clouds | yes | no | no | generic event/object, not a stable graph entity |
| 11:0 | we — benchmark — methods | yes | no | no | authorial self-reference |
| 11:1 | wave packets — can be achieved in — experiments | yes | no | no | event statement rather than an entity relation |
| 12:0 | estimators — are — asymptotically Gaussian | yes | no | no | mathematical property, not graph knowledge |
| 13:0 | QNN — are used for — sensor fusion | **no** | no | no | nominal-list completion, not an explicit predicate |
| 13:1 | Nav-Q — is used for — reinforcement learning | **no** | no | no | nominal-list completion, not an explicit predicate |
| 13:2 | protocols — are used for — secure communication | **no** | no | no | nominal-list completion, not an explicit predicate |
| 14:0 | framework — addresses — challenges | yes | no | no | self-referential generic endpoints |
| 14:1 | framework — provides — quantum performance | yes | no | no | self-referential generic endpoint |
| 14:2 | framework — provides — future-proof security | yes | no | no | self-referential generic endpoint |

## Count

- Literal-support failures: **4 / 32 (12.5%)**.
- Unclean/generic endpoints: **29 / 32 (90.6%)**.
- Clean, literal, new-predicate proposals: **1 / 32 (3.1%)**.
- Clean, literal, existing-predicate proposals: **2 / 32**.
