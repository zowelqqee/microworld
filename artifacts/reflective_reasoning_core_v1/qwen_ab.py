#!/usr/bin/env python3
"""Throwaway A/B driver: run Qwen on the SAME 13 counterfactual + 14 why-might
questions from the pilots, giving it the graph premises, and compare with the
MicroWorld reflective plan. Imports production reasoning read-only; modifies
nothing. Not committed.

Protocol: mlx-community/Qwen2.5-0.5B-Instruct-4bit, temp=0.0, chat template,
short warm-up; each prompt generated twice to check determinism.

Defensibility of Qwen free-text reasoning is NOT fully auto-judgeable. We compute
structural proxies (hallucinated entities beyond premises; expressed uncertainty)
and align each case with MicroWorld's decision, but the final reasoning-quality
verdict on the 11 MW-admitted cases is explicitly FLAGGED FOR MANUAL REVIEW —
the texts are saved verbatim for that.

Usage:  PYTHONPATH=. python3 artifacts/reflective_reasoning_core_v1/qwen_ab.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from worldpgt.reasoning.reflective_reasoning_v1 import load_edges, reflect
from worldpgt.benchmarks.open_book_qa.qwen_runner import _load, MODEL

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT / "artifacts" / "compositional_grammar_v1" / "capability_overlay.json"
OUT = ROOT / "artifacts" / "reflective_reasoning_core_v1"
SYSTEM = ("You reason from the given facts. If the facts do not support an answer, "
          "say clearly that you are not certain rather than guessing.")

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
_UNCERTAIN = ("not certain", "uncertain", "unclear", "cannot", "can't", "do not know",
              "don't know", "no information", "not enough", "unknown", "not sure",
              "difficult to say", "hard to say", "not possible to", "no direct")


def counterfactual_questions():
    d = json.loads((OUT / "pilot_traces.json").read_text(encoding="utf-8"))
    out = []
    for c in d["cases"]:
        if "error" in c:
            continue
        s, p, o = [x.strip() for x in c["focal_fact"].rsplit("  [", 1)[0].split(" | ")]
        out.append({"kind": "counterfactual", "question": c["question"],
                    "entities": [s, o], "focal": (s, p, o)})
    return out


def whymight_questions():
    d = json.loads((OUT / "pilot_abduction_traces.json").read_text(encoding="utf-8"))
    return [{"kind": "why_might", "question": c["question"], "entities": [c["start"], c["goal"]]}
            for c in d["cases"]]


def premises_for(edges, entities):
    ents = {e.strip().lower() for e in entities}
    facts = []
    for e in edges:
        if e.s in ents or e.o in ents:
            facts.append(f"{e.subject} {e.predicate} {e.object}")
    # de-dup, keep order
    seen, uniq = set(), []
    for f in facts:
        if f not in seen:
            seen.add(f); uniq.append(f)
    return uniq


def mw_decision(question, overlay):
    plan = reflect(question, overlay)
    if plan is None:
        return "unrouted", None
    return plan.decision, plan


def structural_flags(answer, premises, question):
    low = answer.lower()
    prem_tokens = set()
    for p in premises + [question]:
        prem_tokens.update(w.lower() for w in _WORD_RE.findall(p))
    ans_tokens = [w for w in _WORD_RE.findall(answer) if len(w) >= 4]
    # proper-noun-ish tokens (capitalized mid-text) not present in premises
    extra_named = sorted({w for w in ans_tokens
                          if w[0].isupper() and w.lower() not in prem_tokens
                          and w.lower() not in {"the", "this", "that", "based", "given", "however"}})
    return {
        "expressed_uncertainty": any(u in low for u in _UNCERTAIN),
        "extra_named_tokens": extra_named,
        "n_extra_named": len(extra_named),
    }


def main():
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))
    edges = load_edges(overlay)
    cases = counterfactual_questions() + whymight_questions()

    model, tokenizer, generate, sampler = _load(MODEL)

    def ask(prompt, max_tokens=200):
        apply = getattr(tokenizer, "apply_chat_template", None)
        text = (apply([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                      tokenize=False, add_generation_prompt=True)
                if apply else f"{SYSTEM}\n\n{prompt}\n")
        return generate(model, tokenizer, prompt=text, max_tokens=max_tokens,
                        sampler=sampler, verbose=False).strip()

    for _ in range(3):
        ask("Reason from: Sky is blue. Question: why might rain be wet?", 16)

    rows = []
    mismatches = 0
    for c in cases:
        prem = premises_for(edges, c["entities"])
        prompt = ("Given what is known:\n- " + "\n- ".join(prem) +
                  f"\n\n{c['question']}\nAnswer, and if you are not certain of something, say so clearly.")
        a1 = ask(prompt)
        a2 = ask(prompt)
        if a1 != a2:
            mismatches += 1
        decision, plan = mw_decision(c["question"], overlay)
        rows.append({
            "kind": c["kind"], "question": c["question"], "entities": c["entities"],
            "n_premises": len(prem),
            "microworld_decision": decision,
            "microworld_rendered": (
                __import__("worldpgt.reasoning.reflective_reasoning_v1", fromlist=["render_reflective_plan"]).render_reflective_plan(plan)
                if plan else None),
            "qwen_answer": a1,
            "qwen_deterministic": a1 == a2,
            **structural_flags(a1, prem, c["question"]),
        })

    # Aggregate: how does Qwen behave on MW-admitted (speculative) vs MW-declined (audit).
    admitted = [r for r in rows if r["microworld_decision"] == "speculative"]
    declined = [r for r in rows if r["microworld_decision"] == "audit"]
    deferred = [r for r in rows if r["microworld_decision"] == "grounded_deferral"]

    summary = {
        "n_cases": len(rows), "model": MODEL, "sampler": "temp=0.0",
        "determinism_mismatches": mismatches,
        "mw_admitted_speculative": len(admitted),
        "mw_declined_audit": len(declined),
        "mw_grounded_deferral": len(deferred),
        "qwen_on_mw_declined": {
            "expressed_uncertainty": sum(r["expressed_uncertainty"] for r in declined),
            "confidently_answered_no_uncertainty": sum(not r["expressed_uncertainty"] for r in declined),
            "mean_extra_named_tokens": round(sum(r["n_extra_named"] for r in declined) / len(declined), 2) if declined else None,
        },
        "qwen_on_mw_admitted": {
            "mean_extra_named_tokens": round(sum(r["n_extra_named"] for r in admitted) / len(admitted), 2) if admitted else None,
            "expressed_uncertainty": sum(r["expressed_uncertainty"] for r in admitted),
        },
        "MANUAL_REVIEW_REQUIRED": "reasoning-quality/defensibility of Qwen answers on the 11 MW-admitted cases needs human judgment; texts saved verbatim.",
    }
    (OUT / "qwen_ab_results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
