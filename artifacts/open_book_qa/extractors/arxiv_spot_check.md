# arXiv source-specific extractor spot check

## Protocol

- Candidate pool: 85 unique arXiv relations already present in the quarantine
  audit; no new source records were fetched.
- Source material: stored arXiv record title and abstract text, matched by
  source URL.
- Round 1 seed: `20260715`, 10 candidates.
- Stop condition: two or more extraction errors. Round 1 had five, so the full
  lane was not run.
- Revision: reject nonreferential/discourse subjects and unbounded clause
  objects before the unchanged precision gate.
- Round 2 seed: `20260716`, sampled from the remaining 75 candidates (the
  original ten were excluded). Round 2 had zero incorrect emitted relations;
  it is the gating check that authorizes the subsequent full run.

## Round 1 — stopped (5 / 10 errors)

| # | Source snippet | Candidate relation | Manual verdict |
|---:|---|---|---|
| 1 | “Confidence aware abstention enables reliable real world deployment by favoring conservative and trustworthy decisions over forced classification.” | `Confidence aware abstention — enables → reliable real world deployment by favoring …` | Correct sentence relation, but endpoint is an overlong clause; this is not an atomic graph fact. **Incorrect extraction shape.** |
| 2 | “Finite simulation supports the conjecture.” | `Finite simulation — supports → the conjecture` | Correct. |
| 3 | “Automated formalization of mathematics enables mechanical verification but remains limited to isolated theorems and short snippets.” | `Automated formalization of mathematics — enables → mechanical verification but remains …` | Object captured an adversative clause. **Incorrect.** |
| 4 | “Such unusual magnetotransport enables PrSbxTe2-x topological materials as a promising platform for device applications.” | `Such unusual magnetotransport — enables → PrSbxTe2-x …` | Demonstrative subject has no stable antecedent in the relation. **Incorrect.** |
| 5 | “The DPoEV uses the knowledge map algorithm to accurately assess the economic value of digital assets.” | `DPoEV — uses → the knowledge map algorithm` | Correct. |
| 6 | “Disk chemistry also enables unique probes of disk structures and dynamics …” | `Disk chemistry also — enables → unique probes …` | Discourse adverb leaked into the subject. **Incorrect.** |
| 7 | “Early detection of intrapartum risks enables timely interventions to prevent or mitigate adverse labor outcomes …” | `Early detection of intrapartum risks — enables → timely interventions …` | Correct sentence relation, but object is an unbounded outcome clause; rejected by the revised atomic-span rule. |
| 8 | “MalGPT enables cybersecurity researchers to develop advanced defense capabilities by emulating large-scale realistic AMG.” | `MalGPT — enables → cybersecurity researchers to develop …` | Correct sentence relation, but object is an unbounded clause; rejected by the revised atomic-span rule. |
| 9 | “FRAME enables organizations to prioritize AML risks, supporting secure AI deployment in real-world environments.” | `FRAME — enables → organizations to prioritize …` | Correct sentence relation, but object is an unbounded clause; rejected by the revised atomic-span rule. |
| 10 | “One implementation uses surface codes created by forming defects in a highly-entangled cluster state.” | `One implementation — uses → surface codes …` | Anaphoric subject, not a stable entity. **Incorrect.** |

## Round 2 — passed (0 / 10 incorrect emitted relations)

| # | Source snippet / result | Candidate relation | Manual verdict |
|---:|---|---|---|
| 1 | `MathNet — supports → three tasks: …` | Object is a list/overlong span. | Correctly rejected. |
| 2 | `Debiasing group graphical lasso estimates — enables → statistical inference when …` | Object is a clause. | Correctly rejected. |
| 3 | `COVID-19 — provides → a high-scale case study, where …` | Object is a clause. | Correctly rejected. |
| 4 | `MCBSG — enables → structured modeling … and quantitative …` | Object is overlong. | Correctly rejected. |
| 5 | `LogAI — supports → tasks such as …` | Object contains uncontrolled enumeration. | Correctly rejected. |
| 6 | “CYSEC uses assessment questions and recommendations to communicate cybersecurity knowledge to the end-user SMBs …” | `CYSEC — uses → assessment questions and recommendations` | Correctly extracted. |
| 7 | `Physiological computing — uses → human physiological data as system inputs in real time` | Object exceeds the bounded atomic span. | Correctly rejected. |
| 8 | `Mat3ra-2D — enables → systematic creation and organization …` | Object is overlong. | Correctly rejected. |
| 9 | “TrackOR uses 3D geometric signatures to achieve state-of-the-art online tracking performance …” | `TrackOR — uses → 3D geometric signatures` | Correctly extracted. |
| 10 | `MetaChem — supports → different levels of description, and has …` | Object contains a conjunction/second clause. | Correctly rejected. |

Round 2 contains two emitted facts, both supported by their stored arXiv text,
and eight conservative rejections. No relation was promoted; all full-run
outputs remain proposal-only and must traverse the existing precision gates.
