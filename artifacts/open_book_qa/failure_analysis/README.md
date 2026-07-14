# MicroWorld open-book QA failure analysis

This report joins the completed dataset with five warm repeats per case. Parser
and planner diagnostics were rerun only for failed paraphrase and multi-evidence
cases with the same EntitySurfaceIndex and persistent evidence graph used by
serving; production API behavior was not changed. `failure_analysis_cases.jsonl`
contains the per-case earliest evidenced stage. Empty candidate groups in the
representative report mean that no such case existed, rather than being omitted.
