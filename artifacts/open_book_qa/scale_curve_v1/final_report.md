# Qwen scale curve v1 — 0.5B → 3B → 7B on the frozen held-out sets

Goal: turn the single Qwen 0.5B comparison point into a scale curve
(0.5B → 3B → 7B) on the *same* frozen held-out material, to see where the
MicroWorld↔Qwen gap moves with model size. Same pipeline, same datasets, same
generation settings — only the model id changed.

## Method — reused pipeline, nothing new

- Runner: the existing `worldpgt/benchmarks/open_book_qa/qwen_runner.py`,
  unchanged. It already accepts `model_name`; the two larger ids were passed
  through `run_file(..., model_name=...)`. No production code was edited.
- Models: `mlx-community/Qwen2.5-3B-Instruct-4bit`,
  `mlx-community/Qwen2.5-7B-Instruct-4bit` (mlx-lm 0.31.3).
- Generation settings **identical** to the 0.5B run: temperature 0 (greedy),
  the same system prompt (incl. "If the evidence does not contain the answer,
  output exactly UNKNOWN"), the same per-category `max_tokens` (128, or 256 for
  multi-evidence). No chain-of-thought.
- Metrics: the shipped `evaluate._measure` / `percentile`, reused verbatim, so
  numbers are directly comparable to `comparison_table.csv` for 0.5B.
- Datasets: the frozen held-out sets the 0.5B was finally measured on —
  `heldout_v2`, `heldout_v3`, `independent_paraphrase_v1` — plus the
  `main_dataset` (250-case) tuning set. Qwen answers from the evidence contexts
  in each `dataset.jsonl`; the graph is irrelevant to Qwen, so the comparison
  is fair as long as the dataset is identical, which it is.

### Two honest protocol notes

1. **Reduced repeats.** The 0.5B protocol used 50 warm-ups / 5 repeats; the
   3B/7B runs used 2 / 3 to stay practical on this machine. At temperature 0
   the answer is deterministic, so accuracy / unsupported / recall / precision
   are **unaffected** by repeat count — only latency-percentile stability is
   reduced.
2. **Machine.** 8 GB RAM Mac. 4-bit quantization + MLX unified memory kept even
   7B resident without OOM or swap thrash (load ~7 s, ~5–8 s/query). Latency
   here is machine-specific and not portable; read it as order-of-magnitude,
   not a spec.

### Coverage limit — 7B main_dataset not completed

7B finished all three **frozen held-out sets** (v2, v3, independent — the
task's stated focus) but the `main_dataset` (250-case) 7B run was stopped
before completion and is not reported. 0.5B and 3B cover all four sets; 7B
covers the three held-out sets. This is a practical limitation, recorded rather
than worked around.

## Scale curve — answer accuracy / unsupported-claim rate

| Set / category | MicroWorld | Qwen-0.5B | Qwen-3B | Qwen-7B |
| --- | --- | --- | --- | --- |
| heldout_v2 / paraphrase | 1.00 / 0.00 | 0.70 / 0.00 | 0.50 / 0.00 | **0.15** / 0.00 |
| heldout_v2 / multi-evidence implicit | 1.00 / 0.00 | 0.70 / 0.00 | **0.90** / 0.00 | 0.70 / 0.00 |
| heldout_v2 / multi-evidence explicit | 1.00 / 0.00 | 0.60 / 0.00 | 0.70 / 0.00 | **0.00** / 0.00 |
| heldout_v3 / paraphrase | 1.00 / 0.05 | 0.80 / 0.05 | 0.55 / 0.00 | 0.50 / 0.00 |
| heldout_v3 / multi-evidence implicit | 1.00 / 0.00 | 0.40 / 0.10 | **0.90** / 0.00 | 0.80 / 0.00 |
| heldout_v3 / multi-evidence explicit | 1.00 / 0.00 | 1.00 / 0.00 | 0.80 / 0.00 | **0.00** / 0.00 |
| independent_v1 / paraphrase | 0.88 / 0.06 | 0.69 / 0.00 | 0.44 / 0.00 | 0.38 / 0.00 |
| independent_v1 / negative | 1.00 / 0.00 | **0.00** / 0.00 | **1.00** / 0.00 | **1.00** / 0.00 |
| main_dataset / direct | 0.98 / 0.09 | 0.65 / 0.01 | 0.68 / 0.00 | — |
| main_dataset / negative | 1.00 / 0.00 | **0.08** / 0.04 | **0.98** / 0.00 | — |
| main_dataset / paraphrase | 0.94 / 0.10 | 0.58 / 0.02 | 0.46 / 0.00 | — |
| main_dataset / multi-evidence | 0.76 / 0.68 | 0.12 / 0.06 | 0.18 / 0.00 | — |

Latency (median p50 across category rows, this 8 GB machine): MicroWorld ~23 ms;
Qwen 0.5B ~0.48 s; 3B ~2.3 s; 7B ~4.6 s per query.

## The single mechanism behind the curve: scale buys abstention

The dominant, consistent effect across every set is that **larger Qwen models
abstain more** — they output `UNKNOWN` more readily. This one behavioural shift
explains both the gains and the losses, and it runs *opposite* to the
hypothesis that more parameters would narrow the paraphrase gap.

Direct evidence — paraphrase `UNKNOWN` rate rose sharply with size:

| heldout paraphrase | 0.5B | 3B |
| --- | --- | --- |
| v2 UNKNOWN / 20 | 1 | 10 |
| v3 UNKNOWN / 20 | 1 | 8 |

The held-out sets deliberately phrase questions in passive / nominalized forms
("By whom was X **engineered**?", "For what application is X **employed**?")
while the evidence says *developed* / *used for*. The strict system prompt says
to emit `UNKNOWN` when the evidence "does not contain the answer." Larger models
follow that instruction more literally: when the question's predicate verb
diverges from the evidence wording, they judge the answer absent and abstain.
0.5B was loose enough to answer anyway (and, on these answerable cases, was
right). So the accuracy drop is not the big model being "worse at language" —
it is the big model **more faithfully declining when surface wording diverges**,
penalised by an answerable-case gold label.

### Per-category reading

- **paraphrase — gap WIDENS with scale (opposite of the hypothesis).**
  MicroWorld 1.00 / 1.00 / 0.88 vs Qwen 0.70→0.50→0.15 (v2), 0.80→0.55→0.50
  (v3), 0.69→0.44→0.38 (indep). Every held-out paraphrase set gets *worse* for
  Qwen as it grows, because the sets stress exactly the passive/nominal
  phrasings that trigger large-model abstention. MicroWorld's structural +
  semantic predicate resolution maps those phrasings to the right relation, so
  its lead grows rather than shrinks. This directly contradicts the going-in
  expectation that scale would close the paraphrase gap; the measurement says
  it does the reverse under this prompt + evaluator.

- **multi-evidence explicit — collapses to 0.00 at 7B.** Same mechanism,
  amplified: these questions use the "By whom was X engineered, and for what
  application…" passive template, and 7B returns `UNKNOWN` on 10/10. Not an
  evaluator artifact — the outputs are literally `UNKNOWN`. The 0.5B→3B step is
  flat-to-up (0.60→0.70, 1.00→0.80); the 7B step is the abstention cliff.

- **multi-evidence implicit — non-monotonic, peaks at 3B.** 0.70→0.90→0.70 (v2),
  0.40→0.90→0.80 (v3). The implicit phrasing is closer to the evidence wording,
  so abstention bites less; here scale genuinely helps up to 3B, then 7B gives
  a little back. MicroWorld stays at 1.00.

- **negative detection — IMPROVES sharply with scale (also opposite of the
  hypothesis).** The going-in guess was that hallucination on unanswerable
  questions might not fix itself with size. The opposite happened: main negative
  0.08 (0.5B) → 0.98 (3B); independent negative 0.00 (0.5B) → 1.00 (3B) → 1.00
  (7B). The same rising abstention that hurts paraphrase is exactly right here —
  larger models correctly decline genuinely unanswerable questions. 0.5B
  hallucinated answers to negatives; 3B/7B say `UNKNOWN`. MicroWorld was already
  at 1.00 by construction (its audit gate), so this is the one axis where scale
  lets Qwen *reach* MicroWorld rather than fall further behind.

- **direct lookup — roughly flat (0.65 → 0.68 at 3B).** Plain "Who founded X?"
  phrasing matches the evidence, so abstention does not trigger and scale barely
  moves it. MicroWorld 0.98.

- **unsupported-claim rate — falls to ~0 by 3B everywhere.** The flip side of
  abstention: larger models assert fewer unsupported objects. 0.5B carried
  small unsupported rates (0.04–0.10 on several categories); 3B and 7B are 0.00
  almost throughout. Fewer hallucinated claims, at the cost of more
  over-abstention on answerable paraphrases.

## Bottom line

On this frozen held-out material, scaling Qwen 0.5B → 3B → 7B does **not** close
the paraphrase gap with MicroWorld — it widens it, because larger models abstain
more when a question's wording diverges from the evidence, and these sets are
built from exactly such paraphrases. The same scale-driven abstention makes
larger Qwen much better at declining true negatives (0.08 → 0.98), which is the
one category where scale lets it catch up. Multi-evidence-implicit peaks at 3B;
direct lookup is flat; unsupported rate falls toward zero throughout.

No extrapolation beyond the three measured points is made. 7B on the 250-case
`main_dataset` was not completed (8 GB machine; run stopped) and is left blank
rather than estimated.

## Files

- `comparison_table.csv` — all rows (MicroWorld + Qwen 0.5B/3B/7B × every
  set/category actually measured).
- `raw/qwen3b/*`, `raw/qwen7b/*` — per-query results and run metadata for the
  new runs (3B: all four sets; 7B: the three held-out sets). 0.5B and MicroWorld
  numbers are sourced from the committed frozen artifacts.
- No production code changed; this was a read-only measurement over the existing
  runner. Note: the runner hardcodes the string `"Qwen2.5-0.5B-Instruct 4-bit"`
  in its metadata `system` field regardless of model; the authoritative model id
  is the `model` field, which is correct in every `qwen_run_metadata.json`.
