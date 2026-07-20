# Microworld from zero vs LLM pretraining from zero

## Like-for-like definition

These are not the same kind of training:

- **Microworld from zero**: no foundation model is trained. Existing Gemini 3.1
  Flash-Lite converts official source sentences into automated graph proposals;
  the existing node-quality filter removes some candidates. This is the cost to
  fill a fresh relation store.
- **LLM from zero**: pretraining a new foundation model from raw corpora. It
  requires GPU clusters, a web-scale corpus, distributed-training engineering,
  evaluation, and alignment. The public dollar figures below are estimated
  training-compute costs, not total company R&D cost.

The Microworld figures deliberately exclude manual review, as requested.
They are a cost per automated candidate, not a quality-guaranteed graph fact.

## Measured Microworld pump cost

Source run: `arxiv_targeted_anti_coercion_100_20260720`.

| Measured quantity | Value |
|---|---:|
| Source sentences | 100 |
| Input / output tokens | 40,230 / 1,340 |
| Filter-passed candidates | 14 |
| Cost per source sentence | $0.0001207 standard; $0.0000603 Batch |
| Cost per candidate | $0.000862 standard; $0.000431 Batch |

Extrapolation at the same measured yield:

| Automated relation candidates | Source sentences needed | Standard Gemini API | Batch API |
|---|---:|---:|---:|
| 1M | ~7.14M | **$862** | **$431** |
| 10M | ~71.4M | **$8,620** | **$4,310** |
| 100M | ~714M | **$86,200** | **$43,100** |

This assumes the current targeted anti-coercion prompt, the current filter,
and no search grounding or human-review cost. It is linear API spend; it does
not require owning GPUs or training a base model.

## Published cost estimates for LLM pretraining from zero

Stanford AI Index 2025 republishes Epoch AI's 2024 estimates for selected
models' training compute:

| Foundation model | Estimated training-compute cost |
|---|---:|
| GPT-4 | **$79M** |
| Gemini Ultra | **$192M** |
| Llama 3.1 405B | **$170M** |

The scale is independently consistent with Meta's technical disclosure for
Llama 3.1 405B: more than 15T training tokens and more than 16,000 H100 GPUs.
These figures do not include all of the data collection, research, failed
runs, evaluation, product, and staff costs; total program cost is therefore
higher and not publicly knowable precisely.

## Direct scale comparison

| Comparison | Microworld automated candidate pump | Frontier LLM pretraining |
|---|---:|---:|
| 1M relations/candidates | $431 Batch / $862 standard | $79M–$192M reference models |
| 100M relations/candidates | $43k Batch / $86k standard | $79M–$192M reference models |
| Compute ownership | No; pay per token | Large GPU cluster / cloud reservation |
| Knowledge update | Add provenance-bearing relation proposals immediately | Retrain or fine-tune; model weights do not expose per-fact provenance |
| Current exactness | Filtered candidate, not guaranteed fact | Unknown without a separate factuality evaluation |

At the 1M-candidate point, frontier pretraining is roughly **92,000x to
445,000x** the standard Microworld API spend. Even a 100M-candidate pump is
roughly **900x to 2,200x** cheaper in API spend than these published frontier
pretraining estimates.

## Decision implication

If the goal is a bounded, inspectable, updatable factual store, training an
LLM from zero is economically the wrong substitute: the Microworld is orders
of magnitude cheaper and preserves evidence/provenance. Pretraining only makes
sense when the goal is to create a general-purpose language model capable of
many tasks outside the bounded world.

Fine-tuning is a third option, not "LLM from zero." It reuses an existing base
model and can lower future extraction-prompt cost, but still needs a training
dataset and does not itself become a provenance-preserving factual store.

## Sources (accessed 2026-07-20)

- Stanford HAI, *AI Index 2025*, Figure 1.3.24, estimates based on Epoch AI
  2024: GPT-4 $79M, Gemini Ultra $192M, Llama 3.1-405B $170M.
- Meta, *Introducing Llama 3.1*: 15T+ tokens and 16,000+ H100 GPUs for the
  405B-scale training run.
- Google Gemini Developer API pricing: Gemini 3.1 Flash-Lite standard and
  Batch rates used in the measured pump calculation.
