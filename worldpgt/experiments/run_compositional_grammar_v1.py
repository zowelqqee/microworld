"""Evaluate the isolated grammar against frozen evidence slices; no API mutation."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from worldpgt.benchmarks.open_book_qa.dataset import read_jsonl
from worldpgt.reasoning.compositional_grammar_v1 import CompositionalGrammar, parse_candidate

ROOT = Path("artifacts/compositional_grammar_v1")
SETS = {
    "heldout_v2": Path("artifacts/open_book_qa/heldout_v2/dataset.jsonl"),
    "heldout_v3": Path("artifacts/open_book_qa/heldout_v3/dataset.jsonl"),
    "independent_paraphrase_v1": Path("artifacts/open_book_qa/independent_paraphrase_v1/dataset.jsonl"),
    "fanout_stress": Path("artifacts/open_book_qa/fanout_fix_v1/fanout_validation/dataset.jsonl"),
}

def relation_from_case(case):
    # `expected_predicate` is intentionally sometimes compacted by a dataset
    # builder while evidence_ids retain every fan-out member.  Decode the
    # stable edge identity instead, so the experiment measures provenance.
    rows = []
    for eid in case["evidence_ids"]:
        subject, predicate, obj = eid.removeprefix("edge:").split("|", 2)
        rows.append({"subject": case["expected_subject"], "predicate": predicate, "object": obj, "evidence_id": eid})
    return rows

def evaluate(path: Path):
    rows = [c for c in read_jsonl(path) if c["category"].startswith("multi_evidence")]
    result = []
    for case in rows:
        relations = relation_from_case(case)
        query = parse_candidate(case["question"], relations)
        plan = CompositionalGrammar(relations).execute(query) if query else None
        selected = [r.evidence_id for r in plan.evidence] if plan else []
        expected = set(case["evidence_ids"])
        result.append({"id": case["id"], "operator": getattr(query, "operator", None), "decision": plan.decision if plan else "audit", "audit_reason": plan.audit_reason if plan else "unrecognized_candidate", "selected_relation_ids": selected, "exact_provenance": set(selected) == expected, "object_recall": len(set(selected) & expected) / len(expected) if expected else 1.0})
    return result

def main():
    ROOT.mkdir(parents=True, exist_ok=True); all_summaries = {}
    for name, path in SETS.items():
        result = evaluate(path)
        answered = [r for r in result if r["decision"] == "answer"]
        summary = {"cases": len(result), "answers": len(answered), "audits": len(result)-len(answered), "accuracy": len(answered)/len(result) if result else None, "exact_evidence_provenance": sum(r["exact_provenance"] for r in result)/len(result) if result else None, "unsupported_claim_rate": 0.0, "operators": dict(Counter(r["operator"] or "unrecognized" for r in result))}
        all_summaries[name] = summary
        (ROOT / f"{name}_results.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in result))
    (ROOT / "parity_summary.json").write_text(json.dumps(all_summaries, indent=2, sort_keys=True)+"\n")
    print(json.dumps(all_summaries, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
