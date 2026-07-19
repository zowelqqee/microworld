#!/usr/bin/env python3
"""Throwaway A/B driver: run Qwen on the constrained-creative prompts and score
BOTH systems with the shared verifier. Imports production reasoning code
read-only; modifies nothing. Not committed, not production.

Protocol matches prior Qwen comparisons: mlx-community/Qwen2.5-0.5B-Instruct-4bit,
make_sampler(temp=0.0) (deterministic), Qwen chat template, a short warm-up.
Each prompt is generated TWICE to check run-to-run determinism at temp=0.

Usage:  python3 artifacts/constrained_creative_v1/qwen_ab.py
"""

from __future__ import annotations

import json
from pathlib import Path

from worldpgt.reasoning.constrained_creative_v1 import (
    generate_constrained, proxy_fluency, select_facts, verify,
)
from worldpgt.experiments.run_constrained_creative_v1 import _llm_prompt, _attested_trigrams
from worldpgt.benchmarks.open_book_qa.qwen_runner import _load, MODEL

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT / "artifacts" / "compositional_grammar_v1" / "capability_overlay.json"
OUT = ROOT / "artifacts" / "constrained_creative_v1"
N = 3
SYSTEM = "You follow the user's writing instructions precisely and literally."


def subjects_with_min_facts(overlay, k):
    from collections import defaultdict
    c = defaultdict(int)
    for r in overlay:
        if isinstance(r, dict) and r.get("overlay_type") == "overlay_relation":
            c[r["subject"]] += 1
    return sorted(s for s, n in c.items() if n >= k)


def main():
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))
    subjects = subjects_with_min_facts(overlay, N)
    attested = _attested_trigrams()

    model, tokenizer, generate, sampler = _load(MODEL)

    def ask(prompt: str, max_tokens: int = 160) -> str:
        apply = getattr(tokenizer, "apply_chat_template", None)
        text = (
            apply([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                  tokenize=False, add_generation_prompt=True)
            if apply else f"{SYSTEM}\n\n{prompt}\n"
        )
        return generate(model, tokenizer, prompt=text, max_tokens=max_tokens,
                        sampler=sampler, verbose=False).strip()

    # warm-up
    for _ in range(3):
        ask("Write one sentence about testing.", max_tokens=16)

    specs = {s: select_facts(overlay, s, n=N) for s in subjects}
    prompts = {s: _llm_prompt(specs[s]) for s in subjects}

    qwen_run1, qwen_run2 = {}, {}
    for s in subjects:
        qwen_run1[s] = ask(prompts[s])
    for s in subjects:
        qwen_run2[s] = ask(prompts[s])

    # deterministic drop-in for the production runner (run 1)
    (OUT / "qwen_outputs.json").write_text(json.dumps(qwen_run1, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    determinism_mismatches = 0
    for s in subjects:
        spec = specs[s]
        mw_text = generate_constrained(spec)
        mw = verify(mw_text, spec)
        q1 = verify(qwen_run1[s], spec)
        identical = qwen_run1[s] == qwen_run2[s]
        if not identical:
            determinism_mismatches += 1
        rows.append({
            "subject": s, "n_facts": spec.n,
            "facts": [f"{f.predicate} {f.object}" for f in spec.facts],
            "prompt": prompts[s],
            "microworld": {"text": mw_text, **mw.to_dict(), "proxy_fluency": proxy_fluency(mw_text, attested)},
            "qwen": {"text": qwen_run1[s], **q1.to_dict(), "proxy_fluency": proxy_fluency(qwen_run1[s], attested)},
            "qwen_run2_text": qwen_run2[s],
            "qwen_deterministic": identical,
        })

    def mean(key, sysname):
        vals = [r[sysname][key] for r in rows if r[sysname][key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "n_subjects": len(subjects), "N_facts": N, "model": MODEL, "sampler": "temp=0.0",
        "determinism_mismatches_run1_vs_run2": determinism_mismatches,
        "microworld": {k: mean(k, "microworld") for k in
                       ("inclusion_rate", "fidelity_rate", "hallucination_token_rate", "proxy_fluency")},
        "qwen": {k: mean(k, "qwen") for k in
                 ("inclusion_rate", "fidelity_rate", "hallucination_token_rate", "proxy_fluency")},
    }
    (OUT / "qwen_ab_results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
