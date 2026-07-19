"""Frozen beyond-parity capability check; read-only against production code."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path

from worldpgt.benchmarks.open_book_qa.dataset import relation_id
from worldpgt.benchmarks.open_book_qa.dataset import read_jsonl
from worldpgt.reasoning.compositional_grammar_v1 import AndQuery, ChainQuery, CompositionalGrammar, RelationRequest
from worldpgt.reasoning.answer_behavior import build_answer_plan, prepare_evidence_graph

OUT = Path("artifacts/compositional_grammar_v1")
SOURCES = (
    Path("worldpgt/experiments/accepted_wiki_memory_overlay_v1.json"),
    Path("worldpgt/experiments/self_ingestion_v1/promotion/promoted_wiki_memory_overlay_v1.json"),
    Path("artifacts/open_book_qa/crossref_doi_seed_v1/precision_gate/accepted_proposal_overlay.json"),
    Path("artifacts/open_book_qa/wikidata_seed_v1/precision_gate/accepted_proposal_overlay.json"),
    Path("artifacts/open_book_qa/openalex_seed_v1/precision_gate/accepted_proposal_overlay.json"),
)
AND_SUBJECTS = ("DeepSeek-R1", "Tianhe-2", "HDF5", "IoT", "Large Language Model")
CHAINS = (("Elon Musk", "founded", "develops", "SpaceX"), ("Elon Musk", "leader_of", "produces", "Tesla"),
          ("Elon Musk", "founded", "develops", "Neuralink"), ("Jeff Bezos", "founded", "develops", "Blue Origin"),
          ("Elon Musk", "leader_of", "develops", "SpaceX"), ("SpaceX", "produces", "uses", "Dragon spacecraft"))

def _load():
    rows=[]
    for path in SOURCES:
        for row in json.loads(path.read_text()):
            if all(row.get(k) for k in ("subject", "predicate", "object")):
                row=dict(row); row["overlay_type"]="overlay_relation"; row["experimental_tier"]="evidence_grounded_capability_v1"
                row["evidence_id"]=relation_id(row); rows.append(row)
    unique={r["evidence_id"]:r for r in rows}; return list(unique.values())

def _old_result(case, rows):
    # The existing multi-evidence planner only expands edges anchored at the
    # target subject.  This invokes that unchanged planner directly, rather
    # than conflating its result with API dialogue/shadow routing.
    graph = prepare_evidence_graph(rows)
    if case["kind"] == "AND":
        plan = build_answer_plan(case["question"], [], targets=[case["subject"]], predicate_filter=frozenset(case["predicates"]), max_blocks=4, prepared_edges=graph)
    else:
        plan = build_answer_plan(case["question"], [], targets=[case["subject"]], max_blocks=4, prepared_edges=graph)
    payload = plan.to_dict() if plan else None
    selected = [edge.get("evidence_id") for block in (payload or {}).get("blocks", []) for edge in [*(block.get("object_slots") or []), (block.get("step") or {}).get("edge", {})] if edge.get("evidence_id")]
    return {"decision":"answer" if plan else "audit", "support":"existing_answer_behavior_planner", "selected_relation_ids":selected, "trace":payload}

def build_cases(rows):
    by_subject=defaultdict(list)
    for row in rows: by_subject[row["subject"].casefold()].append(row)
    cases=[]
    for subject in AND_SUBJECTS:
        values=by_subject[subject.casefold()]; predicates=sorted({r["predicate"] for r in values})[:3]
        selected=[r for r in values if r["predicate"] in predicates]
        cases.append({"id":"and-"+subject.casefold().replace(" ","-"),"kind":"AND","subject":subject,"predicates":predicates,
                      "question":f"For {subject}, what are its " + ", ".join(p.replace("_"," ") for p in predicates[:-1]) + f", and {predicates[-1].replace('_',' ')} relations?",
                      "expected_relation_ids":[r["evidence_id"] for r in selected]})
    for subject,first,second,via in CHAINS:
        first_edges=[r for r in by_subject[subject.casefold()] if r["predicate"]==first and r["object"].casefold()==via.casefold()]
        second_edges=[r for r in by_subject[via.casefold()] if r["predicate"]==second]
        if first_edges and second_edges:
            cases.append({"id":f"chain-{subject.casefold().replace(' ','-')}-{first}-{via.casefold().replace(' ','-')}-{second}","kind":"CHAIN","subject":subject,"via":via,"predicates":[first,second],
                          "question":f"What does {subject} {first.replace('_',' ')} that {second.replace('_',' ')} something?",
                          "expected_relation_ids":[first_edges[0]["evidence_id"],second_edges[0]["evidence_id"]]})
    assert len(cases)>=10, len(cases)
    return cases[:10]

def grammar_result(case, rows):
    local=[r for r in rows if r["evidence_id"] in case["expected_relation_ids"]]
    q = AndQuery(case["subject"],tuple(RelationRequest(p) for p in case["predicates"])) if case["kind"]=="AND" else ChainQuery(case["subject"],case["predicates"][0],case["predicates"][1])
    plan=CompositionalGrammar(local).execute(q)
    return {"decision":plan.decision,"audit_reason":plan.audit_reason,"selected_relation_ids":[r.evidence_id for r in plan.evidence],"plan":plan.to_dict()}

def independent_cases(rows):
    """Ten subject-disjoint, previously unseen multi-evidence cases."""
    specs=(("Elon Musk",("founded","known_for")),("Jeff Bezos",("founded","estimated_net_worth")),("SpaceX",("develops","located_in")),
           ("Bernard Arnault",("estimated_net_worth","leader_of")),("Bloomberg News",("located_in","publishes")),("Blue Origin",("develops","located_in")),
           ("Larry Ellison",("leader_of","located_in")),("Marc Tarpenning",("founded","known_for")),("Martin Eberhard",("founded","known_for")),("Neuralink",("develops","founded_by")))
    seen={case.get("expected_subject","").casefold() for path in Path("artifacts/open_book_qa").glob("**/dataset.jsonl") for case in read_jsonl(path)}
    by=defaultdict(list)
    for r in rows: by[r["subject"].casefold()].append(r)
    cases=[]
    for subject,predicates in specs:
        assert subject.casefold() not in seen, subject
        selected=[r for r in by[subject.casefold()] if r["predicate"] in predicates]
        assert {r["predicate"] for r in selected} == set(predicates), subject
        cases.append({"id":"independent-"+subject.casefold().replace(" ","-"),"kind":"AND","subject":subject,"predicates":list(predicates),"expected_relation_ids":[r["evidence_id"] for r in selected],"question":f"For {subject}, what are its {predicates[0].replace('_',' ')} and {predicates[1].replace('_',' ')} relations?"})
    return cases

def main():
    OUT.mkdir(parents=True,exist_ok=True); rows=_load(); cases=build_cases(rows)
    overlay=OUT/"capability_overlay.json"; overlay.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n")
    results=[]
    for case in cases:
        old=_old_result(case,rows); old_ids=sorted(old["selected_relation_ids"]); new=grammar_result(case,rows); expected=set(case["expected_relation_ids"])
        results.append({**case,"old":old,"grammar":new,
                        "old_exact":set(old_ids)==expected,"grammar_exact":set(new["selected_relation_ids"])==expected})
    for lane in ("old","grammar"):
        score=[r[f"{lane}_exact"] for r in results]; print(lane,sum(score),len(score))
    (OUT/"beyond_old_enum_cases.json").write_text(json.dumps(cases,ensure_ascii=False,indent=2)+"\n")
    (OUT/"beyond_old_enum_results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
    summary={lane:{"cases":len(results),"exact_accuracy":sum(r[f"{lane}_exact"] for r in results)/len(results),"by_kind":{kind:sum(r[f"{lane}_exact"] for r in results if r["kind"]==kind)/sum(r["kind"]==kind for r in results) for kind in ("AND","CHAIN")}} for lane in ("old","grammar")}
    (OUT/"beyond_old_enum_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    independent=independent_cases(rows)
    independent_results=[{**case,"grammar":grammar_result(case,rows)} for case in independent]
    for row in independent_results: row["grammar_exact"]=set(row["grammar"]["selected_relation_ids"])==set(row["expected_relation_ids"])
    (OUT/"independent_multi_evidence_v1_cases.json").write_text(json.dumps(independent,ensure_ascii=False,indent=2)+"\n")
    (OUT/"independent_multi_evidence_v1_results.json").write_text(json.dumps(independent_results,ensure_ascii=False,indent=2)+"\n")
    (OUT/"independent_multi_evidence_v1_summary.json").write_text(json.dumps({"cases":len(independent),"subject_disjoint_from_prior_open_book_datasets":True,"grammar_exact_accuracy":sum(r["grammar_exact"] for r in independent_results)/len(independent_results),"grammar_audits":sum(r["grammar"]["decision"]=="audit" for r in independent_results)},indent=2)+"\n")
if __name__=="__main__": main()
