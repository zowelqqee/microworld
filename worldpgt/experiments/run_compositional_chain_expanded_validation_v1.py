"""Measurement-only expanded CHAIN scan; grammar/planner are not modified."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path

from worldpgt.benchmarks.open_book_qa.dataset import read_jsonl
from worldpgt.benchmarks.open_book_qa.dataset import relation_id
from worldpgt.experiments.run_compositional_capability_v1 import _old_result, grammar_result
from worldpgt.multihop_qa.path_validator import validate_hop_safety
from worldpgt.multihop_qa.types import HopEdge

OUT=Path("artifacts/compositional_grammar_v1/chain_expanded_validation")
CURRENT_OVERLAY=Path("artifacts/ios_demo_v2/extended_serving_overlay.json")

def _load_current():
    """The 990-relation composed serving graph, not a partial source slice."""
    rows=[]
    for raw in json.loads(CURRENT_OVERLAY.read_text()):
        if raw.get("overlay_type")=="overlay_relation" and all(raw.get(k) for k in ("subject","predicate","object")):
            row=dict(raw); row["evidence_id"]=relation_id(row); rows.append(row)
    return {row["evidence_id"]:row for row in rows}.values()

def _prior_ids():
    ids=set()
    for path in Path("artifacts/open_book_qa").glob("**/dataset.jsonl"):
        for case in read_jsonl(path): ids.update(case.get("relation_ids") or ()); ids.update(case.get("evidence_ids") or ())
    for path in (Path("artifacts/compositional_grammar_v1/beyond_old_enum_cases.json"),Path("artifacts/compositional_grammar_v1/independent_multi_evidence_v1_cases.json")):
        for case in json.loads(path.read_text()): ids.update(case.get("expected_relation_ids") or ())
    return ids

def _safe(row):
    hop=HopEdge(row["subject"],row["predicate"],row["object"],stability=str(row.get("stability") or ""),risk=str(row.get("risk") or ""),temporal_class=str(row.get("temporal_class") or ""),as_of=str(row.get("as_of") or ""))
    return validate_hop_safety(hop)

def _case(a,b,expected):
    return {"id":"chain-"+"-".join(x.casefold().replace(" ","-") for x in (a["subject"],a["predicate"],a["object"],b["predicate"],b["object"])),"kind":"CHAIN","subject":a["subject"],"via":a["object"],"predicates":[a["predicate"],b["predicate"]],"question":f"Structured CHAIN: {a['subject']} --{a['predicate']}--> {a['object']} --{b['predicate']}--> {b['object']}","expected_relation_ids":[a["evidence_id"],b["evidence_id"]],"expected":"audit" if expected else "answer"}

def main():
    OUT.mkdir(parents=True,exist_ok=True); rows=list(_load_current()); prior=_prior_ids(); by=defaultdict(list)
    for row in rows: by[row["subject"].casefold()].append(row)
    all_safe=[]; all_audit=[]; eligible=[]
    for a in rows:
        for b in by[a["object"].casefold()]:
            if a["subject"].casefold()==a["object"].casefold(): continue
            a_ok,a_reason=_safe(a); b_ok,b_reason=_safe(b)
            (all_safe if a_ok and b_ok else all_audit).append((a,b,a_reason or b_reason))
            if not ({a["evidence_id"],b["evidence_id"]}&prior): eligible.append((a,b,None if a_ok and b_ok else a_reason or b_reason))
    answer=[_case(a,b,False) for a,b,reason in eligible if reason is None]
    audit=[_case(a,b,True) | {"audit_reason":reason} for a,b,reason in eligible if reason is not None]
    # A reverse-only loop is structurally eligible under the task definition,
    # but is labelled separately so it cannot be mistaken for a useful C!=A chain.
    for case in answer: case["degenerate_cycle"]=case["subject"].casefold()==case["expected_relation_ids"][1].split("|")[-1]
    results=[]
    for case in [*answer,*audit]:
        old=_old_result(case,rows); grammar=grammar_result(case,rows); expected=set(case["expected_relation_ids"])
        results.append({**case,"old":old,"grammar":grammar,"old_exact":set(old["selected_relation_ids"])==expected,"grammar_exact":set(grammar["selected_relation_ids"])==expected,"grammar_correct_audit":case["expected"]=="audit" and grammar["decision"]=="audit"})
    summary={"graph_source":str(CURRENT_OVERLAY),"graph_relation_count":len(rows),"all_chain_paths_before_overlap":{"safe":len(all_safe),"audit_expected":len(all_audit),"unique_start_subjects_safe":len({x[0]["subject"] for x in all_safe})},"prior_relation_or_evidence_id_count":len(prior),"eligible_after_required_zero_overlap":{"answer_expected":len(answer),"audit_expected":len(audit),"unique_start_subjects":len({c["subject"] for c in [*answer,*audit]})},"metrics":{"answer_expected":{"old_exact_accuracy":sum(r["old_exact"] for r in results if r["expected"]=="answer")/len(answer) if answer else None,"grammar_exact_accuracy":sum(r["grammar_exact"] for r in results if r["expected"]=="answer")/len(answer) if answer else None},"audit_expected":{"grammar_correctly_audited_rate":sum(r["grammar_correct_audit"] for r in results if r["expected"]=="audit")/len(audit) if audit else None}},"non_degenerate_answer_paths":sum(not c["degenerate_cycle"] for c in answer)}
    (OUT/"cases.json").write_text(json.dumps([*answer,*audit],ensure_ascii=False,indent=2)+"\n")
    (OUT/"results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
