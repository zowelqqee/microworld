# Microworld vs GPT-2 Controlled Continuation Report

## Benchmark Scope
Controlled continuation / ambiguity resolution.

This compares explicit audit-aware continuation with GPT-2 next-token generation on the v1 prompt set.

## Dataset
- Rows: 120
- Expected answerable: 110
- Expected no-answer: 10

## Architecture
- Microworld: Explicit sense memory, deterministic cue scoring, anti-cues/guards, conservative policy, template realization, and surface-risk audit gate.
- GPT-2: GPT-2 base model inference through local nanoGPT; no training or fine-tuning performed here.
- Comparison: Microworld is designed for abstention and auditability; GPT-2 is designed for open-ended language modeling.

## Quality Comparison
- Microworld correct continuations: 38
- Microworld wrong continuations: 0
- Microworld precision on continued: 1.0
- GPT-2 good/bad/unclear: 76 / 11 / 33
- GPT-2 audited precision: 0.8736
- GPT-2 wrong sense count: 7

## Risk/Coverage Comparison
- Microworld continue/audit: 38 / 82
- Microworld coverage rate: 0.3167
- Microworld answerable recall: 0.3455
- GPT-2 has no native audit path; GPT-2 quality labels were assigned after generation.

## Runtime Comparison
- Microworld timed run total sec: 0.00559
- Microworld avg sec/prompt: 4.6e-05
- GPT-2 total generation sec from CSV: 50.6932
- GPT-2 avg generation sec/prompt from CSV: 0.422443

## Memory/RSS Comparison
- Microworld peak RSS MB: 22.906
- GPT-2 peak RSS MB: 1348.672

## State/Model Size Comparison
- Microworld explicit state bytes: 7882
- GPT-2 parameter count: 123650000
- GPT-2 model state size bytes: 548105171

## Examples
### Microworld Continue Examples
- `v1-001` The customer reached the bank teller with cash to -> The customer reached the bank teller with cash to open an account
- `v1-002` At the bank she opened an account and asked about credit to -> At the bank she opened an account and asked about credit to speak with the teller
- `v1-003` The fisherman sat on the muddy bank by the river to -> The fisherman sat on the muddy bank by the river to watch the current
- `v1-004` The boat drifted toward the bank as the current -> The boat drifted toward the bank as the current carried it downstream
- `v1-005` The bat flew from the cave at night with its wings -> The bat flew from the cave at night with its wings and searched for insects

### Microworld Audit Examples
- `v1-021` The client waited at the bank counter to -> 
- `v1-022` Near closing time the bank manager motioned for -> 
- `v1-023` The children climbed down the bank below the bridge to -> 
- `v1-024` A narrow path crossed the bank below the reeds and -> 
- `v1-025` The ranger dimmed the attic light so the bat could -> 

### Gpt2 Good Examples
- `v1-001` The customer reached the bank teller with cash to -> make a deposit to pay for the car after he had left the store.
- `v1-003` The fisherman sat on the muddy bank by the river to -> catch the fish.
- `v1-004` The boat drifted toward the bank as the current -> was low.
- `v1-005` The bat flew from the cave at night with its wings -> flapping and its tail flapping.
- `v1-006` The tiny bat hung in the cave after night -> fall. After the dark, the man told his story, then went to sleep.

### Gpt2 Bad Unclear Examples
- `v1-002` At the bank she opened an account and asked about credit to -> a friend's computer, and when he turned around she went to her boyfriend's house with him and
- `v1-007` The baseball player lifted the bat before the swing -> , then turned it into a pitch and the ball turned to a ball and had to be hit before
- `v1-008` The bat cracked when he hit it during the game -> .
- `v1-013` The crane spread its wings above the marsh and -> over the lake.
- `v1-014` The bird expert watched the crane build a nest near the lake -> side.

## Limitations
- This is a small 120-row controlled continuation benchmark.
- GPT-2 is an old base model, not ChatGPT or an instruction-tuned model.
- GPT-2 has no native audit path; labels were audited after generation.
- Microworld has low coverage and template-based realization.
- RSS is approximate and environment-dependent.

## Non-Claims
- No claim that Microworld beats neural networks generally.
- No claim that this small benchmark predicts open-domain generation quality.
- No claim that GPT-2 represents modern instruction-tuned assistants.

## Framing
Microworld emits fewer continuations but had zero measured wrong continuations among emitted rows in this benchmark; GPT-2 produced more usable continuations overall but also bad/unclear continuations and has no native audit path.

The result exposes a risk/coverage tradeoff between explicit policy continuation and open-ended next-token generation.
