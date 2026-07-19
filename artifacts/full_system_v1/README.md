# Full-system v1 — router-driven Qwen-3B comparison

Scope: 50 existing cases, comprising 10 frozen held-out QA cases, 13 reflective
pilot cases, and all 27 constrained-creative A/B subjects. The router selects
the MicroWorld branch automatically. Both systems receive matched per-question
evidence without an explicit branch label; Qwen is
`mlx-community/Qwen2.5-3B-Instruct-4bit`, temperature 0.

| Category | MicroWorld | Qwen-3B |
|---|---:|---:|
| QA accuracy (n=10) | 1.00 | 0.90 |
| Reflective correct audits (n=11) | 11/11 | 11/11 |
| Reflective admitted speculative (n=2) | 2/2 answered | 0/2 `UNKNOWN` |
| Constrained inclusion | 0.963 | 0.988 |
| Constrained fidelity | 0.889 | 0.790 |
| Constrained hallucination proxy | 0.037 | 0.449 |

The `IoT` entity-recognition routing gap is included in the constrained score.
This is a matched-scale result, not a general comparison with LLMs.
