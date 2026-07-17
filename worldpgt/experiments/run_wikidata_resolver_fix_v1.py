"""Measure conservative alias/P31 disambiguation on the 290 prior misses."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from worldpgt.experiments.run_wikidata_density_recon_v1 import _Client
from worldpgt.benchmarks.open_book_qa.dataset import _norm
from worldpgt.knowledge_pump.heldout_density_frontier import attach_wikidata_exact_resolution

def main() -> int:
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default='artifacts/open_book_qa/wikidata_resolver_fix_v1');p.add_argument('--delay-seconds',type=float,default=.15);p.add_argument('--allow-network',action='store_true');a=p.parse_args()
    if not a.allow_network: raise SystemExit('refusing to fetch without --allow-network')
    ua=os.environ.get('MICROWORLD_WIKI_USER_AGENT','')
    if not ua: raise SystemExit('MICROWORLD_WIKI_USER_AGENT is required')
    root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
    all_rows=json.loads(Path('artifacts/open_book_qa/wikidata_density_recon/resolution_manifest.json').read_text())
    failed=[r for r in all_rows if 'original_331' in r['cohorts'] and not r.get('canonical_qid')]
    cache_path=root/'search_cache.json'; cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    client=_Client(user_agent=ua,delay_seconds=a.delay_seconds)
    for row in failed:
        key=_norm(row['subject'])
        if key not in cache:
            cache[key]=client.search(row['subject'])
            cache_path.write_text(json.dumps(cache,ensure_ascii=False,indent=2)+'\n')
    # Only candidates that can be selected by the conservative resolver need
    # P31 claims.  Fetching all ten search hits per subject would turn this
    # small resolution audit into needless API load.
    candidate_qids=sorted({
        str(h.get('id')) for row in failed for h in cache.get(_norm(row['subject']), [])
        if str(h.get('id','')).startswith('Q') and (
            _norm(str(h.get('label') or '')) == _norm(row['subject']) or
            (_norm(str((h.get('match') or {}).get('text') or '')) == _norm(row['subject'])
             and (h.get('match') or {}).get('type') == 'alias')
        )
    })
    entities=client.entities(candidate_qids,properties='claims') if candidate_qids else {}
    resolved=attach_wikidata_exact_resolution(failed,cache,entities=entities,enable_alias_disambiguation=True)
    new=[r for r in resolved if r.get('canonical_qid')]
    root.joinpath('resolved_candidates.json').write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n')
    reviews=json.loads(root.joinpath('manual_review.json').read_text()) if root.joinpath('manual_review.json').exists() else []
    approved={str(row['canonical_qid']) for row in reviews if row.get('review_status') == 'approved'}
    reviewed={str(row['canonical_qid']) for row in reviews}
    vetted=[row for row in new if str(row['canonical_qid']) in approved]
    summary={'version':'wikidata_resolver_fix_v1','production_resolver_modified':True,'extraction_performed':False,'precision_gate_run':False,'accepted_memory_modified':False,'serving_overlay_modified':False,'prior_exact_resolved_original':'41/331','prior_failure_count':len(failed),'automatic_candidate_count':len(new),'manually_reviewed_candidate_count':len(reviewed),'manually_confirmed_new_resolution_count':len(vetted),'manual_false_positive_count':len(new)-len(vetted) if len(reviewed)==len(new) else None,'after_resolved_original_count':41+len(vetted) if len(reviewed)==len(new) else None,'after_resolved_original_rate':(41+len(vetted))/331 if len(reviewed)==len(new) else None,'new_resolution_methods':{m:sum(r.get('canonical_resolution_method')==m for r in vetted) for m in sorted({r.get('canonical_resolution_method') for r in vetted})},'search_queries_cached':len(cache),'network_calls_this_invocation':client.calls,'manual_review_required':'Every automatic candidate must be manually reviewed before use outside this resolver measurement. Only approved entries are eligible for a proposal-only extractor.'}
    root.joinpath('summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False));return 0
if __name__=='__main__': raise SystemExit(main())
