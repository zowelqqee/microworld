"""Build and run the frozen, router-driven full-system v1 comparison.

Usage: python3 -m worldpgt.experiments.full_system_v1 {build|microworld|qwen|score}
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "full_system_v1"
MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def build():
    """52 existing cases: first 10 frozen v3 QA, all 15 first reflective pilot
    cases, and all 27 existing constrained-creative subjects (no new questions)."""
    from worldpgt.benchmarks.open_book_qa.dataset import read_jsonl
    from worldpgt.reasoning.reflective_reasoning_v1 import load_edges
    qa = read_jsonl(ROOT / "artifacts/open_book_qa/heldout_v3/dataset.jsonl")[:10]
    overlay = json.loads((ROOT / "artifacts/compositional_grammar_v1/capability_overlay.json").read_text())
    frozen = json.loads((ROOT / "artifacts/open_book_qa/heldout_v3/frozen_relations.json").read_text())
    # The router must see the exact frozen QA evidence as well as the capability
    # overlay used by the reflective/constrained cases.  Qwen receives the
    # per-question evidence in its prompt, never this branch label or overlay.
    _write(OUT / "composed_evidence_overlay.json", [*overlay, *frozen])
    edges = load_edges(overlay)
    pilot = json.loads((ROOT / "artifacts/reflective_reasoning_core_v1/pilot_traces.json").read_text())["cases"]
    reflective = []
    for i, row in enumerate(pilot[:15]):
        q = row["question"]; entities = [x.strip() for x in row["focal_fact"].split("  [")[0].split(" | ")[::2]]
        facts = [f"{e.subject} {e.predicate} {e.object}" for e in edges if e.subject in entities or e.object in entities]
        reflective.append({"id": f"reflective-{i:02d}", "category": "reflective", "question": q, "evidence": facts, "expected": row.get("decision")})
    old = json.loads((ROOT / "artifacts/constrained_creative_v1/qwen_ab_results.json").read_text())["rows"]
    constrained = [{"id": f"constrained-{i:02d}", "category": "constrained_creative", "question": r["prompt"], "evidence": r["facts"], "subject": r["subject"]} for i,r in enumerate(old)]
    qa_rows = [{"id": r["id"], "category": "qa", "question": r["question"], "evidence": r["contexts"], "expected_objects": r["expected_objects"], "expected_decision": r["expected_decision"]} for r in qa]
    _write(OUT / "unified_test_set.json", {"scope": "52 existing cases: 10 frozen QA + 15 reflective pilot + 27 constrained-creative A/B", "cases": qa_rows + reflective + constrained})

def _prompt(c):
    evidence = "\n".join(f"- {x}" for x in c["evidence"])
    return ("Use only the evidence below. Do not add facts. If it cannot support an answer, say UNKNOWN. "
            "When the wording asks about a hypothetical or possible association, state uncertainty rather than presenting an inference as fact.\n\n"
            f"Evidence:\n{evidence}\n\nQuestion:\n{c['question']}\n\nAnswer:")

def microworld():
    from worldpgt.reasoning.integrated_answer_router import IntegratedAnswerRouter
    cases = json.loads((OUT / "unified_test_set.json").read_text())["cases"]
    router = IntegratedAnswerRouter(
        overlay_path=str(OUT / "composed_evidence_overlay.json"),
        qa_experimental_graph_paths=[ROOT / "artifacts/open_book_qa/heldout_v3/frozen_relations.json"],
    )
    rows = []
    for i, c in enumerate(cases, 1):
        rows.append({"id": c["id"], **router.answer(c["question"]).to_dict()})
        print(f"MicroWorld {i}/{len(cases)}: {c['id']}", flush=True)
    _write(OUT / "microworld_results.json", rows)

def qwen():
    from worldpgt.benchmarks.open_book_qa.qwen_runner import _load
    cases = json.loads((OUT / "unified_test_set.json").read_text())["cases"]
    model, tok, generate, sampler = _load(MODEL)
    apply = getattr(tok, "apply_chat_template", None)
    def ask(c):
        user = _prompt(c); text = apply([{"role":"system","content":"Answer directly and concisely."},{"role":"user","content":user}], tokenize=False, add_generation_prompt=True) if apply else user
        return generate(model, tok, prompt=text, max_tokens=192, sampler=sampler, verbose=False).strip()
    for c in cases[:3]: ask(c)
    rows = []
    for i, c in enumerate(cases, 1):
        rows.append({"id": c["id"], "answer": ask(c)})
        print(f"Qwen-3B {i}/{len(cases)}: {c['id']}", flush=True)
    _write(OUT / "qwen3b_results.json", rows)

def score():
    from worldpgt.reasoning.constrained_creative_v1 import ConstraintSpec, Fact, verify
    cases = json.loads((OUT / "unified_test_set.json").read_text())["cases"]; mw={r["id"]:r for r in json.loads((OUT/"microworld_results.json").read_text())}; qw={r["id"]:r for r in json.loads((OUT/"qwen3b_results.json").read_text())}
    rows=[]
    for c in cases:
        def hit(text): return all(x.casefold() in text.casefold() for x in c.get("expected_objects",[]))
        if c["category"]=="qa": rows.append({"category":"qa","id":c["id"],"microworld_correct":hit(mw[c["id"]]["answer_text"]),"qwen_correct":hit(qw[c["id"]]["answer"])})
        elif c["category"]=="reflective": rows.append({"category":"reflective","id":c["id"],"microworld_decision":mw[c["id"]]["decision"],"qwen_mentions_uncertainty":any(x in qw[c["id"]]["answer"].casefold() for x in ("uncertain","unknown","cannot","can't","not enough"))})
        else:
            facts = tuple(Fact(x.split(" ", 1)[0], x.split(" ", 1)[1]) for x in c["evidence"])
            spec = ConstraintSpec(c["subject"], facts)
            a, b = verify(mw[c["id"]]["answer_text"], spec), verify(qw[c["id"]]["answer"], spec)
            rows.append({"category":"constrained_creative","id":c["id"],
                         "microworld":a.to_dict(), "qwen":b.to_dict(),
                         "microworld_branch":mw[c["id"]]["branch"]})
    _write(OUT / "comparison_rows.json", rows)
    summary={k:[r for r in rows if r["category"]==k] for k in ("qa","reflective","constrained_creative")}
    _write(OUT / "comparison_summary.json", summary)
    (OUT/"README.md").write_text("# Full-system v1\n\n`comparison_rows.json` contains automatic QA and reflective proxies. Constrained outputs require the existing shared verifier; reflective defensibility requires manual review of saved answers.\n",encoding="utf-8")

if __name__ == "__main__":
    a=argparse.ArgumentParser(); a.add_argument("command",choices=("build","microworld","qwen","score")); x=a.parse_args(); globals()[x.command]()
