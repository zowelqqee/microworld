#!/usr/bin/env python3
"""Constrained-creative v1 experiment.

For each eligible subject in the overlay: select N facts, generate constrained
text, and score it with the shared post-hoc verifier. Also computes the proxy
fluency metric from the existing Creative model's trigram tables.

The LLM (Qwen) side of the A/B is wired but not executed here (no model in this
session): the exact constrained prompt is emitted per subject, and if a
``qwen_outputs.json`` file is present the SAME verifier scores it. Otherwise the
LLM columns are marked pending.

Usage:
    python3 -m worldpgt.experiments.run_constrained_creative_v1
"""

from __future__ import annotations

import json
from pathlib import Path

from worldpgt.reasoning.constrained_creative_v1 import (
    generate_constrained,
    proxy_fluency,
    select_facts,
    verify,
)

_ROOT = Path(__file__).resolve().parents[2]
_OVERLAY = _ROOT / "artifacts" / "compositional_grammar_v1" / "capability_overlay.json"
_OUT_DIR = _ROOT / "artifacts" / "constrained_creative_v1"
_QWEN_OUTPUTS = _OUT_DIR / "qwen_outputs.json"   # optional drop-in for the LLM side

_N = 3
_SUBJECTS = ["SpaceX", "Blue Origin", "Tesla", "Neuralink", "Elon Musk"]


def _llm_prompt(spec) -> str:
    facts = "; ".join(f"{spec.subject} {f.predicate} {f.object}" for f in spec.facts)
    return (
        f"Write a short piece about {spec.subject} using only these facts: {facts}. "
        "Use all of them. Do not add anything not listed."
    )


def _attested_trigrams() -> set[tuple[str, str, str]]:
    try:
        from worldpgt.cognition.creative_generator import default_creative_model
        model = default_creative_model()
    except Exception:
        return set()
    tri: set[tuple[str, str, str]] = set()
    for (w1, w2), nexts in model.forward2.items():
        for w3 in nexts:
            tri.add((w1, w2, w3))
    return tri


def main() -> None:
    overlay = json.loads(_OVERLAY.read_text(encoding="utf-8"))
    attested = _attested_trigrams()
    qwen_outputs = {}
    if _QWEN_OUTPUTS.is_file():
        qwen_outputs = json.loads(_QWEN_OUTPUTS.read_text(encoding="utf-8"))

    rows = []
    for subject in _SUBJECTS:
        spec = select_facts(overlay, subject, n=_N)
        if spec.n < 2:
            rows.append({"subject": subject, "skipped": "fewer than 2 facts available"})
            continue
        ours = generate_constrained(spec)
        ours_report = verify(ours, spec)
        row = {
            "subject": subject,
            "n_facts": spec.n,
            "facts": [f"{f.predicate} {f.object}" for f in spec.facts],
            "llm_prompt": _llm_prompt(spec),
            "microworld": {
                "text": ours,
                **ours_report.to_dict(),
                "proxy_fluency": proxy_fluency(ours, attested),
            },
        }
        if subject in qwen_outputs:
            llm_text = qwen_outputs[subject]
            llm_report = verify(llm_text, spec)
            row["qwen"] = {
                "text": llm_text,
                **llm_report.to_dict(),
                "proxy_fluency": proxy_fluency(llm_text, attested),
            }
        else:
            row["qwen"] = {"status": "pending — no qwen_outputs.json in this session"}
        rows.append(row)

    scored = [r for r in rows if "microworld" in r]
    summary = {
        "overlay": str(_OVERLAY.relative_to(_ROOT)),
        "n_subjects": len(scored),
        "N_facts_per_subject": _N,
        "attested_trigrams_available": bool(attested),
        "microworld_mean_inclusion": _mean(r["microworld"]["inclusion_rate"] for r in scored),
        "microworld_mean_fidelity": _mean(r["microworld"]["fidelity_rate"] for r in scored),
        "microworld_mean_hallucination": _mean(r["microworld"]["hallucination_token_rate"] for r in scored),
        "microworld_mean_proxy_fluency": _mean(
            r["microworld"]["proxy_fluency"] for r in scored
            if r["microworld"]["proxy_fluency"] is not None
        ),
        "qwen_status": "pending — drop qwen_outputs.json to score the LLM side symmetrically",
    }
    out = {"summary": summary, "rows": rows}
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "build_v1_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print()
    for r in scored:
        mw = r["microworld"]
        print(f'--- {r["subject"]} (N={r["n_facts"]}) ---')
        print(f'   text: {mw["text"]}')
        print(f'   inclusion={mw["inclusion_rate"]} fidelity={mw["fidelity_rate"]} '
              f'hallucination={mw["hallucination_token_rate"]} proxy_fluency={mw["proxy_fluency"]}')


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


if __name__ == "__main__":
    main()
