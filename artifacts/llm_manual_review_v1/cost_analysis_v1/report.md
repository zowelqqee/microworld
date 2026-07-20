# API-only cost model: one-time Microworld pump vs fine-tuning

## Scope

This deliberately excludes human review. Costs are USD list prices and include
only model API / tuning tokens, not storage, engineering, or serving
infrastructure.

The source measurement is `arxiv_targeted_anti_coercion_100_20260720`:

| Observed quantity | Value |
|---|---:|
| Source sentences | 100 |
| Gemini input tokens | 40,230 |
| Gemini output tokens | 1,340 |
| Raw triples | 20 |
| Candidates passing unchanged node-quality filter | 14 |
| Mean source length | 28.6 words |

Gemini 3.1 Flash-Lite standard text pricing is $0.25/M input and $1.50/M
output tokens; Batch API is half price. The observed one-time pump therefore
costs **$0.01207 per 100 source sentences** standard, or **$0.00603** via
Batch API.

## Automated-pump unit economics

| Unit | Standard API | Batch API |
|---|---:|---:|
| One source sentence | $0.0001207 | $0.0000603 |
| One raw emitted triple | $0.000603 | $0.000302 |
| One filter-passed candidate | $0.000862 | $0.000431 |
| 100k source sentences | $12.07 | $6.03 |
| 1M source sentences | $120.68 | $60.34 |

At the measured yield, 1M source sentences produce about **200k raw triples**
and **140k filter-passed candidates**. This is the cost of a one-time
automated proposal pump; official arXiv acquisition is free here.

## Fine-tuning is additive, not an alternative pump

A fine-tuned extractor still needs an input/output dataset. With no human
review, it learns the generator-plus-filter behavior, including residual
errors. It also cannot replace Microworld relation IDs, literal evidence,
provenance, or the proposal overlay; it can only replace or shorten future
extraction calls.

Vertex AI lists Gemini 3.1 Flash-Lite supervised fine-tuning at **$3 per 1M
training tokens**. Training tokens equal dataset tokens multiplied by epochs.
Tuned Gemini 3 endpoints cost 1.5x base prediction price.

Assume a compact SFT record: 80 system-instruction tokens + 37 source tokens
+ 13 output tokens = **130 tokens/example**, at three epochs. To learn when to
return `[]`, the dataset must include all source sentences, not only the 14%
that produced candidates.

| Training corpus | Pump cost (standard) | SFT tokens (130 x 3) | SFT cost | One-time total |
|---|---:|---:|---:|---:|
| 100k source sentences | $12.07 | 39.0M | $117.00 | **$129.07** |
| 1M source sentences | $120.68 | 390.0M | $1,170.00 | **$1,290.68** |

Using only filter-passed rows lowers the training bill by about 7.1x, but
removes the negative/empty-output distribution. That is a candidate classifier
dataset, not a general extractor training set.

## Future inference and break-even

Current targeted anti-coercion inference uses 402.3 input and 13.4 output
tokens/sentence: **$0.0001207/sentence** at standard pricing. If SFT permits
the compact 80-token instruction, tuned inference is estimated as
`(117 x $0.375 + 13.4 x $2.25) / 1M = $0.0000740/sentence`, about **39% less**.
This uses the documented 1.5x tuned-endpoint multiplier.

At that optimistic prompt-shortening assumption, a 100k-sentence SFT job
breaks even after about **2.50M subsequent source sentences**. If the prompt
cannot be shortened, SFT inference is 1.5x more expensive and has no API-cost
break-even.

## Decision

For one-off or sub-million-scale pumps, direct Gemini 3.1 Flash-Lite plus the
existing filter is cheaper: **$6–$60 via Batch API** (or **$12–$121 standard**)
per 100k–1M sentences, versus approximately **$129–$1,291** for pump +
full-source SFT at three epochs. SFT is economically justified only after
several million future sentences and safe prompt compression.

## Quality boundary

The latest audit found 3 rejected relations among 14 filter-passed candidates.
Therefore these are costs per automated candidate/proposal, not per verified
graph fact. This analysis does not auto-admit, promote, or modify serving
memory.

## Pricing sources (accessed 2026-07-20)

- Google Gemini Developer API pricing: Gemini 3.1 Flash-Lite standard and
  batch rates.
- Google Cloud Agent Platform pricing: Gemini 3.1 Flash-Lite SFT is $3/M
  training tokens; training tokens equal dataset tokens x epochs; Gemini 3
  tuned endpoints are 1.5x base prediction price.
